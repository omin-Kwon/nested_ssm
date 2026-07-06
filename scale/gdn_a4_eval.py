"""A4 verdict eval: per-head v4/staleness fingerprints on the trained REAL
multi-head GatedDeltaNet (checkpoints from gdn_a4.py).

Q1  correction-exact/readout-stale separation per head (heterogeneity)
Q2  does the nested hot tier rescue recall under cold staleness
    (A3v2 real-LM warning)

Arms (mirroring the toy verdict semantics, [K,V] state, per-head scalar decay):
  fresh    exact recurrence (VALIDATION GATE: must match fused kernel grid)
  a        stale cold correction + stale cold readout (hot fresh)
  c1       exact correction, ALL readout stale (honest Config B)
  v4       cold cols exact-write (PNM replay), hot cols stale-corrected write;
           hot readout fresh, cold readout stale
"""
import argparse, json, sys
import torch, torch.nn.functional as F
import numpy as np
from einops import rearrange
sys.path.insert(0, "/home/omin/TTT-PNM/poc")
from nested_delta_mqar import make_imqar
from gdn_a4 import GDNLM
from fla.ops.gated_delta_rule.gate import naive_gdn_gate

CFG = {"mode": "fresh", "c": 8, "pb": 16, "head_sel": None, "grow": None}

def eval_gdn_forward(self, hidden_states, **kw):
    B, T, _ = hidden_states.shape
    q, _ = self.q_conv1d(x=self.q_proj(hidden_states))
    k, _ = self.k_conv1d(x=self.k_proj(hidden_states))
    v, _ = self.v_conv1d(x=self.v_proj(hidden_states))
    K, V = self.head_k_dim, self.head_v_dim
    q = rearrange(q, 'b t (h d) -> b t h d', d=K).float()
    k = rearrange(k, 'b t (h d) -> b t h d', d=K).float()
    v = rearrange(v, 'b t (h d) -> b t h d', d=V).float()
    m = self.nest_width
    if m < K:
        q = torch.cat([q[..., :m], torch.zeros_like(q[..., m:])], -1)
        k = torch.cat([k[..., :m], torch.zeros_like(k[..., m:])], -1)
    q = F.normalize(q, p=2, dim=-1)                     # in-kernel l2norm equiv
    k = F.normalize(k, p=2, dim=-1)
    beta = torch.sigmoid(self.b_proj(hidden_states)).float()          # (B,T,H)
    g = naive_gdn_gate(self.a_proj(hidden_states), self.A_log, self.dt_bias)  # (B,T,H) log-decay
    scale = K ** -0.5
    H = self.num_heads
    mode, c, pb = CFG["mode"], CFG["c"], min(CFG["pb"], m)
    hsel = CFG["head_sel"]                              # None = all heads staled
    hmask = torch.ones(H, dtype=torch.bool, device=q.device)
    if hsel is not None:
        hmask[:] = False; hmask[hsel] = True            # only this head staled
    h = q.new_zeros(B, H, K, V)                         # state, k-major
    Snap = h.clone()
    G = q.new_zeros(B, H)                               # log decay since snapshot
    outs = []
    for t in range(T):
        if CFG["grow"] is not None:                     # E4: per-timestep width
            w0, w1, t0 = CFG["grow"]
            wt = w0 if t < t0 else w1
            if wt < K:
                k_cur = torch.cat([k[:, t][..., :wt],
                                   torch.zeros_like(k[:, t][..., wt:])], -1)
                q_cur = torch.cat([q[:, t][..., :wt],
                                   torch.zeros_like(q[:, t][..., wt:])], -1)
                k_cur = F.normalize(k_cur, p=2, dim=-1)
                q_cur = F.normalize(q_cur, p=2, dim=-1)
                k[:, t] = k_cur; q[:, t] = q_cur
        a_t = g[:, t].exp()                             # (B,H)
        h = h * a_t[..., None, None]                    # decay (ref semantics)
        if mode != "fresh":
            if t % c == 0:
                Snap = h.clone(); G = torch.zeros_like(G)
            else:
                G = G + g[:, t]
        k_t, q_t, v_t, b_t = k[:, t], q[:, t] * scale, v[:, t], beta[:, t]
        corr_exact = torch.einsum('bhk,bhkv->bhv', k_t, h)
        if mode in ("a", "v4"):
            corr_hot = torch.einsum('bhk,bhkv->bhv', k_t[..., :pb].contiguous(),
                                    h[:, :, :pb])
            corr_cold_stale = G.exp()[..., None] * torch.einsum(
                'bhk,bhkv->bhv', k_t[..., pb:].contiguous(), Snap[:, :, pb:])
            corr_stale = corr_hot + corr_cold_stale
            corr_stale = torch.where(hmask[None, :, None], corr_stale, corr_exact)
        w_exact = b_t[..., None] * (v_t - corr_exact)
        if mode == "fresh" or mode == "c1":
            h = h + k_t.unsqueeze(-1) * w_exact.unsqueeze(-2)
        elif mode == "a":
            w = b_t[..., None] * (v_t - corr_stale)
            h = h + k_t.unsqueeze(-1) * w.unsqueeze(-2)
        elif mode == "v4":                               # tier-local writes
            w_gpu = b_t[..., None] * (v_t - corr_stale)
            h_hot = h[:, :, :pb] + k_t[..., :pb].unsqueeze(-1) * w_gpu.unsqueeze(-2)
            h_cold = h[:, :, pb:] + k_t[..., pb:].unsqueeze(-1) * w_exact.unsqueeze(-2)
            h = torch.cat([h_hot, h_cold], dim=2)
        # readout
        if mode == "fresh":
            y = torch.einsum('bhk,bhkv->bhv', q_t, h)
        elif mode == "c1":
            y_st = G.exp()[..., None] * torch.einsum('bhk,bhkv->bhv', q_t, Snap)
            y_fr = torch.einsum('bhk,bhkv->bhv', q_t, h)
            y = torch.where(hmask[None, :, None], y_st, y_fr)
        else:                                            # a / v4: hot fresh + cold stale
            y_hot = torch.einsum('bhk,bhkv->bhv', q_t[..., :pb].contiguous(), h[:, :, :pb])
            y_cold = G.exp()[..., None] * torch.einsum(
                'bhk,bhkv->bhv', q_t[..., pb:].contiguous(), Snap[:, :, pb:])
            y_st = y_hot + y_cold
            y_fr = torch.einsum('bhk,bhkv->bhv', q_t, h)
            y = torch.where(hmask[None, :, None], y_st, y_fr)
        outs.append(y)
    o = torch.stack(outs, 1).to(hidden_states.dtype)
    gate = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', d=V)
    o = self.o_norm(o, gate)
    o = rearrange(o, 'b t h d -> b t (h d)')
    return self.o_proj(o)

AGE_BINS = [(1, 4), (5, 8), (9, 32), (33, 10**6)]

def probe(model, D, nq, nk, nv, w, device, gen, batch=768, by_age=True):
    model.eval()
    inp, tgt, age = make_imqar(batch, D, nq, nk, nv, device, gen, return_age=True)
    with torch.no_grad():
        logits = model(inp, width=w)
    ok = logits.argmax(-1) == tgt
    msk = tgt != -100
    res = {"all": ok[msk].float().mean().item()}
    if by_age:
        for lo, hi in AGE_BINS:
            mm = msk & (age >= lo) & (age <= hi)
            res[f"{lo}-{hi if hi<10**6 else '+'}"] = ok[mm].float().mean().item() if mm.any() else float("nan")
    return res

def use_eval_path(model):
    for b in model.blocks:
        b.attn.forward = eval_gdn_forward.__get__(b.attn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="gdn_a4_nested.pt")
    ap.add_argument("--tag", default="nested")
    ap.add_argument("--pb", type=int, default=16)
    ap.add_argument("--cs", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--D", type=int, default=64)
    args = ap.parse_args()
    device = "cuda"
    nk, nv, nq = 256, 128, 24
    vocab = 1 + nk + nv
    model = GDNLM(vocab).to(device).bfloat16()
    model.load_state_dict(torch.load(args.ckpt))
    gen = torch.Generator(device=device); gen.manual_seed(777)

    # validation gate: fused vs naive-fresh
    r_fused = probe(model, args.D, nq, nk, nv, 64, device, gen, by_age=False)
    use_eval_path(model); CFG["mode"] = "fresh"
    gen.manual_seed(777)
    r_naive = probe(model, args.D, nq, nk, nv, 64, device, gen, by_age=False)
    print(f"[{args.tag}] VALIDATION fused={r_fused['all']:.3f} naive={r_naive['all']:.3f}", flush=True)

    out = {}
    def show(name, res):
        print(f"[{args.tag}] {name:16s} " + " ".join(f"{k}:{v:.2f}" for k, v in res.items()), flush=True)
        out[name] = res
    CFG["head_sel"] = None
    CFG["mode"] = "fresh"
    for w, nm in [(64, "fresh-k64"), (args.pb, f"fresh-k{args.pb}(hot-alone)")]:
        gen.manual_seed(778); show(nm, probe(model, args.D, nq, nk, nv, w, device, gen))
    for mode in ["a", "c1", "v4"]:
        for c in args.cs:
            CFG.update(mode=mode, c=c, pb=args.pb)
            gen.manual_seed(778)
            show(f"{mode}-c{c}", probe(model, args.D, nq, nk, nv, 64, device, gen))
    # Q1: per-head heterogeneity of correction-staleness damage (mode a, worst c)
    print(f"[{args.tag}] Q1 per-head (a)-staleness damage (c={max(args.cs)}):", flush=True)
    for hsel in range(4):
        CFG.update(mode="a", c=max(args.cs), pb=args.pb, head_sel=hsel)
        gen.manual_seed(778)
        r = probe(model, args.D, nq, nk, nv, 64, device, gen, by_age=False)
        print(f"[{args.tag}]   head {hsel} only: all={r['all']:.3f}", flush=True)
        out[f"head{hsel}"] = r
    CFG["head_sel"] = None
    json.dump(out, open(f"gdn_a4_eval_{args.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
