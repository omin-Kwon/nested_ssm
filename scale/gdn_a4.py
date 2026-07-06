"""A4: nested multi-width training of a REAL fla GatedDeltaNet (multi-head,
short-conv, output gate, in-kernel qk L2 norm) on interleaved MQAR.

Core questions (pinned):
 Q1  does v4's correction-exact / readout-stale separation hold per head?
 Q2  does the nested hot tier rescue RECALL under cold staleness
     (the A3v2 real-LM warning: query-time write-read resonance)?

This script trains (nested multi-width | dedicated fixed-width) models and
saves checkpoints; the verdict eval lives in gdn_a4_eval.py.
Width semantics: zero the tail of q,k per head BEFORE the kernel — the
in-kernel L2 norm then normalizes over the active prefix only (clean
per-width truncation, no norm-then-slice artifact).
"""
import argparse, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/omin/TTT-PNM/poc")
from nested_delta_mqar import make_imqar
from fla.layers.gated_deltanet import GatedDeltaNet

def masked_gdn_forward(self, hidden_states, **kw):
    """Copy of fla GatedDeltaNet.forward (chunk path, no cache) with per-head
    key/query width masking (self.nest_width)."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from einops import rearrange
    q, _ = self.q_conv1d(x=self.q_proj(hidden_states))
    k, _ = self.k_conv1d(x=self.k_proj(hidden_states))
    v, _ = self.v_conv1d(x=self.v_proj(hidden_states))
    q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
    v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
    if getattr(self, 'rot', None) is not None:   # E8-style per-head key rotation
        q = torch.einsum('bthd,hde->bthe', q, self.rot)
        k = torch.einsum('bthd,hde->bthe', k, self.rot)
    m = self.nest_width
    if torch.is_tensor(m):                       # per-sample widths (training)
        idx = torch.arange(self.head_k_dim, device=q.device)
        wm = (idx[None, :] < m[:, None]).to(q.dtype)[:, None, None, :]
        q = q * wm
        k = k * wm
    elif m < self.head_k_dim:                    # uniform width (eval)
        q = torch.cat([q[..., :m], torch.zeros_like(q[..., m:])], -1)
        k = torch.cat([k[..., :m], torch.zeros_like(k[..., m:])], -1)
    beta = self.b_proj(hidden_states)
    o, _ = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=self.a_proj(hidden_states), beta=beta,
        A_log=self.A_log, dt_bias=self.dt_bias,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, allow_neg_eigval=self.allow_neg_eigval,
        state_v_first=True, cu_seqlens=None)
    if self.use_gate:
        g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
        o = self.o_norm(o, g)
    else:
        o = self.o_norm(o)
    o = rearrange(o, 'b t h d -> b t (h d)')
    return self.o_proj(o)

class Block(nn.Module):
    def __init__(self, d, heads, head_dim):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = GatedDeltaNet(hidden_size=d, num_heads=heads, head_dim=head_dim,
                                  expand_v=2, mode='chunk')
        self.attn.nest_width = head_dim
        self.attn.forward = masked_gdn_forward.__get__(self.attn)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

class GDNLM(nn.Module):
    def __init__(self, vocab, d=256, n_layers=2, heads=4, head_dim=64, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
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
        x = self.emb(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm_f(x))

def eval_grid(model, Ds, widths, nk, nv, nq, device, gen, batch=512):
    model.eval()
    out = {}
    with torch.no_grad():
        for D in Ds:
            inp, tgt = make_imqar(batch, D, nq, nk, nv, device, gen)
            row = {}
            for w in widths:
                logits = model(inp, width=w)
                mask = tgt != -100
                row[w] = (logits.argmax(-1)[mask] == tgt[mask]).float().mean().item()
            out[D] = row
    model.train()
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fixed_only", type=int, default=0)
    p.add_argument("--Ds", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--n_keys", type=int, default=256)
    p.add_argument("--n_vals", type=int, default=128)
    p.add_argument("--n_query", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="")
    p.add_argument("--log_every", type=int, default=1000)
    args = p.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    vocab = 1 + args.n_keys + args.n_vals
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)
    model = GDNLM(vocab).to(device).bfloat16()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    widths = args.widths if not args.fixed_only else [args.fixed_only]
    print(f"training {'nested ' + str(widths) if not args.fixed_only else f'fixed-{args.fixed_only}'}"
          f" | Ds={args.Ds} vocab={vocab}", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        D = args.Ds[torch.randint(len(args.Ds), (1,), generator=gen, device=device).item()]
        inp, tgt = make_imqar(args.batch, D, args.n_query, args.n_keys, args.n_vals, device, gen)
        opt.zero_grad()
        if len(widths) > 1:                       # nested: per-sample random width
            ws = torch.tensor(widths, device=device)[
                torch.randint(len(widths), (inp.shape[0],), generator=gen, device=device)]
            logits = model(inp, width=ws)
        else:
            logits = model(inp, width=widths[0])
        loss = F.cross_entropy(logits.float().view(-1, vocab), tgt.view(-1),
                               ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0:
            g = eval_grid(model, args.Ds, widths, args.n_keys, args.n_vals,
                          args.n_query, device, gen, batch=256)
            print(f"step {step+1} loss {loss.item():.3f}", flush=True)
            print("D\\k  " + " ".join(f"{w:>6d}" for w in widths))
            for D in args.Ds:
                print(f"{D:<4d} " + " ".join(f"{g[D][w]:6.3f}" for w in widths), flush=True)
    g = eval_grid(model, args.Ds, widths, args.n_keys, args.n_vals, args.n_query,
                  device, gen, batch=1024)
    print(f"FINAL ({time.time()-t0:.0f}s):")
    print("D\\k  " + " ".join(f"{w:>6d}" for w in widths))
    for D in args.Ds:
        print(f"{D:<4d} " + " ".join(f"{g[D][w]:6.3f}" for w in widths), flush=True)
    if args.save:
        torch.save(model.state_dict(), args.save)
        print("saved", args.save)

if __name__ == "__main__":
    main()
