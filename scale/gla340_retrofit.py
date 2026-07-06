"""T9: nested-FT on a REAL PUBLIC pretrained checkpoint (fla-hub/gla-340M-15B).
This is the deployment leg: channel-decay family (the path Kimi/KDA needs —
rotation shortcut doesn't apply, general FT does). Protocol:
  1. validation gate: our masked forward == model's own forward at full width
  2. pre-FT truncation baseline (expect collapse — pretrained has no ordering)
  3. FT with per-sample width menu {16,32,64,128} (nested) or fixed 128 (control)
  4. post-FT ppl vs width
NOTE: unlike our from-scratch GLA-35M, NO qk-norm is added here — we must
preserve the pretrained function exactly; warm weights don't need the init fix."""
import argparse, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

def masked_gla_forward(self, hidden_states, **kw):
    from fla.ops.gla import chunk_gla
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    gk = self.gk_proj(hidden_states)
    q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
    k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
    gk = rearrange(gk, '... (h d) -> ... h d', d=self.head_k_dim)
    v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
    gk = F.logsigmoid(gk.float()) / self.gate_logit_normalizer
    m = self.nest_width
    if torch.is_tensor(m):
        idx = torch.arange(self.head_k_dim, device=q.device)
        wm = (idx[None, :] < m[:, None]).to(q.dtype)[:, None, None, :]
        q = q * wm; k = k * wm
    elif m < self.head_k_dim:
        q = torch.cat([q[..., :m], torch.zeros_like(q[..., m:])], -1)
        k = torch.cat([k[..., :m], torch.zeros_like(k[..., m:])], -1)
    gk = gk.to(v.dtype)
    o, _ = chunk_gla(q=q, k=k, v=v, g=gk, initial_state=None,
                     output_final_state=False, state_v_first=True)
    g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
    o = self.g_norm_swish_gate(o, g)
    o = rearrange(o, 'b t h d -> b t (h d)')
    return (self.o_proj(o), None, None)

def get_attn_layers(model):
    layers = model.model.layers
    return [l.attn for l in layers]

def set_width(model, m):
    for a in get_attn_layers(model):
        a.nest_width = m

def batches(train, bsz, seqlen, gen, device):
    ix = torch.randint(0, len(train) - seqlen - 1, (bsz,), generator=gen, device=device).cpu()
    x = torch.stack([torch.from_numpy(train[i:i + seqlen].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(train[i + 1:i + 1 + seqlen].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def vppl(model, val, w, seqlen=1024, n=8, device="cuda"):
    model.eval(); tot = 0.0
    set_width(model, w)
    for b in range(n):
        s = b * seqlen * 4
        row = torch.from_numpy(val[s:s + seqlen + 1].astype(np.int64))[None].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(row[:, :-1]).logits
        tot += F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                               row[:, 1:].reshape(-1)).item()
    model.train(); return math.exp(tot / n)

def main():
    import fla.models  # registers 'gla' etc. with transformers Auto classes
    from transformers import AutoModelForCausalLM
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--fixed_only", type=int, default=0)
    p.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--save", default="gla340_nested.pt")
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    train = np.load("wt103_train.npy", mmap_mode="r")
    val = np.load("wt103_val.npy")
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    tag = "gla340-nested" if not args.fixed_only else f"gla340-fixed{args.fixed_only}"
    model = AutoModelForCausalLM.from_pretrained(
        "fla-hub/gla-340M-15B", torch_dtype=torch.float32).to(device)  # fp32 master
    # baseline ppl with the model's OWN forward (pre-patch)
    model.eval()
    with torch.no_grad():
        tot = 0.0
        for b in range(8):
            s = b * args.seqlen * 4
            row = torch.from_numpy(val[s:s + args.seqlen + 1].astype(np.int64))[None].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(row[:, :-1]).logits
            tot += F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                                   row[:, 1:].reshape(-1)).item()
    p_orig = math.exp(tot / 8)
    # patch masked forward
    for a in get_attn_layers(model):
        assert a.head_k_dim == 128, a.head_k_dim
        a.nest_width = 128
        a.forward = masked_gla_forward.__get__(a)
    p_patch = vppl(model, val, 128, device=device)
    print(f"[{tag}] VALIDATION original={p_orig:.2f} patched-full={p_patch:.2f}", flush=True)
    print(f"[{tag}] PRE-FT ppl vs width: " +
          " ".join(f"k{w}:{vppl(model, val, w, device=device):.2f}" for w in args.widths),
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    widths = args.widths if not args.fixed_only else [args.fixed_only]
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g_ in opt.param_groups: g_["lr"] = lr
        x, y = batches(train, args.batch, args.seqlen, gen, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if len(widths) > 1:
                ws = torch.tensor(widths, device=device)[
                    torch.randint(len(widths), (x.shape[0],), generator=gen, device=device)]
                set_width(model, ws)
            else:
                set_width(model, widths[0])
            logits = model(x).logits
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0:
            print(f"[{tag}] step {step+1} loss {loss.item():.3f} ({time.time()-t0:.0f}s) "
                  + " ".join(f"k{w}:{vppl(model, val, w, device=device):.2f}" for w in args.widths),
                  flush=True)
    print(f"[{tag}] FINAL " +
          " ".join(f"k{w}:{vppl(model, val, w, device=device):.2f}" for w in args.widths), flush=True)
    torch.save(model.state_dict(), args.save)
    print(f"[{tag}] saved {args.save}", flush=True)

if __name__ == "__main__":
    main()
