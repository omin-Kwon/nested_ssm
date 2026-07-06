"""T3: real-language nested GDN on wikitext-103.
~55M params: d=512, 6 blocks, 8 heads, head_k 64 (key total 512), expand_v 2.
Nested widths {8,16,32,64} per-sample; dedicated-64 control via --fixed_only.
"""
import argparse, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from gdn_a4 import Block

class GDNLanguageModel(nn.Module):
    def __init__(self, vocab=32000, d=512, n_layers=6, heads=8, head_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, heads, head_dim) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.head_dim = head_dim

    def set_width(self, m):
        for b in self.blocks:
            b.attn.nest_width = m

    def forward(self, ids, width=None):
        if width is not None:
            self.set_width(width)
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm_f(x))

def batches(data, batch, seqlen, gen, device):
    ix = torch.randint(len(data) - seqlen - 1, (batch,), generator=gen, device=device)
    rows = torch.stack([torch.from_numpy(data[i:i + seqlen + 1].astype(np.int64))
                        for i in ix.tolist()]).to(device)
    return rows[:, :-1], rows[:, 1:]

@torch.no_grad()
def val_ppl(model, val, widths, seqlen=1024, n_batches=8, device="cuda"):
    model.eval()
    out = {}
    for w in widths:
        tot, cnt = 0.0, 0
        for b in range(n_batches):
            s = b * seqlen * 4
            row = torch.from_numpy(val[s:s + seqlen + 1].astype(np.int64))[None].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(row[:, :-1], width=w)
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                                   row[:, 1:].reshape(-1))
            tot += loss.item(); cnt += 1
        out[w] = math.exp(tot / cnt)
    model.train()
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--fixed_only", type=int, default=0)
    p.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--save", default="gdn_lm_nested.pt")
    p.add_argument("--log_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--data", default="wt103_train.npy")
    args = p.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    train = np.load(args.data, mmap_mode="r")
    val = np.load("wt103_val.npy")
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    model = GDNLanguageModel(d=args.d, n_layers=args.layers, heads=args.heads).to(device)  # fp32 master
    n_par = sum(p_.numel() for p_ in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    widths = args.widths if not args.fixed_only else [args.fixed_only]
    tag = "nested" if not args.fixed_only else f"fixed{args.fixed_only}"
    print(f"[{tag}] {n_par/1e6:.1f}M params | train {len(train)/1e6:.0f}M tok | "
          f"budget {args.steps*args.batch*args.seqlen/1e6:.0f}M tok", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batches(train, args.batch, args.seqlen, gen, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if len(widths) > 1:
                ws = torch.tensor(widths, device=device)[
                    torch.randint(len(widths), (x.shape[0],), generator=gen, device=device)]
                logits = model(x, width=ws)
            else:
                logits = model(x, width=widths[0])
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0:
            ppls = val_ppl(model, val, widths, args.seqlen, device=device)
            el = time.time() - t0
            print(f"[{tag}] step {step+1} loss {loss.item():.3f} ({el:.0f}s) val_ppl: "
                  + " ".join(f"k{w}:{v:.2f}" for w, v in ppls.items()), flush=True)
    ppls = val_ppl(model, val, widths, args.seqlen, n_batches=16, device=device)
    print(f"[{tag}] FINAL val_ppl: " + " ".join(f"k{w}:{v:.2f}" for w, v in ppls.items()),
          flush=True)
    torch.save(model.state_dict(), args.save)
    print(f"[{tag}] saved {args.save}", flush=True)

if __name__ == "__main__":
    main()
