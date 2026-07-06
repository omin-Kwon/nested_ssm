"""Generality run: nested LM with fla KimiDeltaAttention (KDA = channel-wise
decay + DELTA rule — the Kimi Linear family) on wikitext-103. Same shapes as
GDN-35M / GLA-35M pairs (hidden 512, 8 heads, head_k 64, head_v 128).
Nesting recipe identical to GDN: mask q/k prefix; in-kernel l2norm normalizes
the active prefix, so truncation semantics stay clean. Channel gate g lives on
k-channels; masked channels have k=0 so their state rows stay zero."""
import argparse, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from fla.layers.kda import KimiDeltaAttention

def masked_kda_forward(self, hidden_states, **kw):
    from fla.ops.kda import chunk_kda
    if self.use_short_conv:
        q, _ = self.q_conv1d(x=self.q_proj(hidden_states), output_final_state=False)
        k, _ = self.k_conv1d(x=self.k_proj(hidden_states), output_final_state=False)
        v, _ = self.v_conv1d(x=self.v_proj(hidden_states), output_final_state=False)
    else:
        q = F.silu(self.q_proj(hidden_states))
        k = F.silu(self.k_proj(hidden_states))
        v = F.silu(self.v_proj(hidden_states))
    g = self.f_proj(hidden_states)
    beta = self.b_proj(hidden_states)
    q, k = (rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim) for x in (q, k))
    g = rearrange(g, '... (h d) -> ... h d', d=self.head_k_dim)
    v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
    m = self.nest_width
    if torch.is_tensor(m):
        idx = torch.arange(self.head_k_dim, device=q.device)
        wm = (idx[None, :] < m[:, None]).to(q.dtype)[:, None, None, :]
        q = q * wm; k = k * wm
    elif m < self.head_k_dim:
        q = torch.cat([q[..., :m], torch.zeros_like(q[..., m:])], -1)
        k = torch.cat([k[..., :m], torch.zeros_like(k[..., m:])], -1)
    o, _ = chunk_kda(q=q, k=k, v=v, g=g, beta=beta,
                     A_log=self.A_log, dt_bias=self.dt_bias,
                     initial_state=None, output_final_state=False,
                     use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                     use_beta_sigmoid_in_kernel=True,
                     allow_neg_eigval=self.allow_neg_eigval,
                     safe_gate=self.safe_gate, lower_bound=self.lower_bound,
                     state_v_first=True)
    o = self.o_norm(o, rearrange(self.g_proj(hidden_states),
                                 '... (h d) -> ... h d', d=self.head_v_dim))
    o = rearrange(o, 'b t h d -> b t (h d)')
    return self.o_proj(o)

class KDABlock(nn.Module):
    def __init__(self, d, heads, head_dim):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = KimiDeltaAttention(hidden_size=d, num_heads=heads,
                                       head_dim=head_dim, expand_v=2.0, mode='chunk')
        assert self.attn.head_k_dim == head_dim, self.attn.head_k_dim
        self.attn.nest_width = head_dim
        self.attn.forward = masked_kda_forward.__get__(self.attn)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

class KDALanguageModel(nn.Module):
    def __init__(self, vocab=32000, d=512, n_layers=6, heads=8, head_dim=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([KDABlock(d, heads, head_dim) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

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

def main():
    from gdn_lm import batches
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--fixed_only", type=int, default=0)
    p.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--save", default="kda_lm_nested.pt")
    p.add_argument("--log_every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    train = np.load("wt103_train.npy", mmap_mode="r")
    val = np.load("wt103_val.npy")
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    model = KDALanguageModel().to(device)                 # fp32 master
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    widths = args.widths if not args.fixed_only else [args.fixed_only]
    tag = "kda-nested" if not args.fixed_only else f"kda-fixed{args.fixed_only}"
    print(f"[{tag}] {sum(x.numel() for x in model.parameters())/1e6:.1f}M params", flush=True)

    @torch.no_grad()
    def vppl(w, n=8):
        model.eval(); tot = 0.0
        for b in range(n):
            s = b * args.seqlen * 4
            row = torch.from_numpy(val[s:s + args.seqlen + 1].astype(np.int64))[None].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(row[:, :-1], width=w)
            tot += F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                                   row[:, 1:].reshape(-1)).item()
        model.train(); return math.exp(tot / n)

    t0 = time.time()
    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g_ in opt.param_groups: g_["lr"] = lr
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
            print(f"[{tag}] step {step+1} loss {loss.item():.3f} ({time.time()-t0:.0f}s) "
                  + " ".join(f"k{w}:{vppl(w):.2f}" for w in widths), flush=True)
    print(f"[{tag}] FINAL " + " ".join(f"k{w}:{vppl(w, 16):.2f}" for w in widths), flush=True)
    torch.save(model.state_dict(), args.save)
    print(f"[{tag}] saved {args.save}", flush=True)

if __name__ == "__main__":
    main()
