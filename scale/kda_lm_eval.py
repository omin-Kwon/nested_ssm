"""KDA-family staleness verdict (Kimi Linear generality): ppl under arms on the
language-trained nested KDA. KDA = per-CHANNEL decay + DELTA rule, so all four
arms exist: fresh / a (stale correction — the taboo) / c1 (all-stale readout) /
v4 (correction exact + hot-fresh readout + cold-stale readout).
Per-channel G compensation. Validation gate: naive==fused."""
import argparse, math, json
import numpy as np
import torch, torch.nn.functional as F
from einops import rearrange
from fla.ops.kda.gate import naive_kda_gate
from kda_lm import KDALanguageModel

CFG = {"mode": "fresh", "c": 16, "pb": 16}

def naive_masked_kda_forward(self, hidden_states, **kw):
    B, T, _ = hidden_states.shape
    K, V = self.head_k_dim, self.head_v_dim
    if self.use_short_conv:
        q, _ = self.q_conv1d(x=self.q_proj(hidden_states), output_final_state=False)
        k, _ = self.k_conv1d(x=self.k_proj(hidden_states), output_final_state=False)
        v, _ = self.v_conv1d(x=self.v_proj(hidden_states), output_final_state=False)
    else:
        q = F.silu(self.q_proj(hidden_states))
        k = F.silu(self.k_proj(hidden_states))
        v = F.silu(self.v_proj(hidden_states))
    q = rearrange(q, 'b t (h d) -> b t h d', d=K).float()
    k = rearrange(k, 'b t (h d) -> b t h d', d=K).float()
    v = rearrange(v, 'b t (h d) -> b t h d', d=V).float()
    g = naive_kda_gate(rearrange(self.f_proj(hidden_states), 'b t (h d) -> b t h d', d=K),
                       self.A_log, self.dt_bias).float()       # (B,T,H,K) log-decay
    beta = torch.sigmoid(self.b_proj(hidden_states)).float()   # (B,T,H)
    m = self.nest_width
    if m < K:
        q = torch.cat([q[..., :m], torch.zeros_like(q[..., m:])], -1)
        k = torch.cat([k[..., :m], torch.zeros_like(k[..., m:])], -1)
    q = F.normalize(q, p=2, dim=-1)                     # in-kernel l2norm equiv
    k = F.normalize(k, p=2, dim=-1)
    scale = K ** -0.5
    H = q.shape[2]
    mode, c, pb = CFG["mode"], CFG["c"], min(CFG["pb"], m)
    h = q.new_zeros(B, H, K, V)                         # k-major (naive_recurrent_kda layout)
    Snap = h.clone()
    G = q.new_zeros(B, H, K)                            # per-CHANNEL log decay since snap
    outs = []
    for t in range(T):
        a_t = g[:, t].exp()                             # (B,H,K)
        h = h * a_t[..., None]                          # decay-then-correct (ref semantics)
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
            corr_cold_stale = torch.einsum(
                'bhk,bhkv->bhv', k_t[..., pb:].contiguous(),
                Snap[:, :, pb:] * G[..., pb:].exp()[..., None])
            corr_stale = corr_hot + corr_cold_stale
        w_exact = b_t[..., None] * (v_t - corr_exact)
        if mode in ("fresh", "c1"):
            h = h + k_t.unsqueeze(-1) * w_exact.unsqueeze(-2)
        elif mode == "a":                               # stale correction persists
            w = b_t[..., None] * (v_t - corr_stale)
            h = h + k_t.unsqueeze(-1) * w.unsqueeze(-2)
        elif mode == "v4":                              # tier-local writes
            w_gpu = b_t[..., None] * (v_t - corr_stale)
            h_hot = h[:, :, :pb] + k_t[..., :pb].unsqueeze(-1) * w_gpu.unsqueeze(-2)
            h_cold = h[:, :, pb:] + k_t[..., pb:].unsqueeze(-1) * w_exact.unsqueeze(-2)
            h = torch.cat([h_hot, h_cold], dim=2)
        if mode == "fresh":
            y = torch.einsum('bhk,bhkv->bhv', q_t, h)
        elif mode == "c1":
            y = torch.einsum('bhk,bhkv->bhv', q_t, Snap * G.exp()[..., None])
        else:                                           # a / v4: hot fresh + cold stale
            y_hot = torch.einsum('bhk,bhkv->bhv', q_t[..., :pb].contiguous(), h[:, :, :pb])
            y_cold = torch.einsum('bhk,bhkv->bhv', q_t[..., pb:].contiguous(),
                                  Snap[:, :, pb:] * G[..., pb:].exp()[..., None])
            y = y_hot + y_cold
        outs.append(y)
    o = torch.stack(outs, 1).to(hidden_states.dtype)
    o = self.o_norm(o, rearrange(self.g_proj(hidden_states),
                                 'b t (h d) -> b t h d', d=V))
    o = rearrange(o, 'b t h d -> b t (h d)')
    return self.o_proj(o)

@torch.no_grad()
def ppl(model, val, w, seqlen=1024, n=6, device="cuda"):
    tot = 0.0
    for b in range(n):
        s = b * seqlen * 4
        row = torch.from_numpy(val[s:s + seqlen + 1].astype(np.int64))[None].to(device)
        logits = model(row[:, :-1], width=w)
        tot += F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                               row[:, 1:].reshape(-1)).item()
    return math.exp(tot / n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="kda_lm_nested.pt")
    ap.add_argument("--tag", default="kda-nested")
    ap.add_argument("--pb", type=int, default=16)
    ap.add_argument("--cs", type=int, nargs="+", default=[16, 64])
    args = ap.parse_args()
    device = "cuda"
    val = np.load("wt103_val.npy")
    model = KDALanguageModel().to(device).bfloat16()
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    p_fused = ppl(model, val, 64, device=device)
    for b in model.blocks:
        b.attn.forward = naive_masked_kda_forward.__get__(b.attn)
    CFG["mode"] = "fresh"
    p_naive = ppl(model, val, 64, device=device)
    print(f"[{args.tag}] VALIDATION fused={p_fused:.2f} naive={p_naive:.2f}", flush=True)
    out = {"fresh": p_naive}
    for mode in ["a", "c1", "v4"]:
        for c in args.cs:
            CFG.update(mode=mode, c=c, pb=args.pb)
            v = ppl(model, val, 64, device=device)
            print(f"[{args.tag}] ppl {mode}-c{c}: {v:.2f}", flush=True)
            out[f"{mode}_c{c}"] = v
    json.dump(out, open(f"kda_lm_eval_{args.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
