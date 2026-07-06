"""GLA-family staleness verdict (generality 2/2): ppl under arms on the
language-trained nested GLA. GLA is ADDITIVE (no delta correction) so the arms
are pure readout-staleness: fresh / c1 (ALL dims stale) / v4 (hot fresh + cold
stale). Per-DIM decay compensation (G_j vector). Validation gate: naive==fused."""
import argparse, math, json
import numpy as np
import torch, torch.nn.functional as F
from einops import rearrange
from gla_lm import GLALanguageModel

CFG = {"mode": "fresh", "c": 16, "pb": 16}

def naive_masked_gla_forward(self, hidden_states, **kw):
    B, T, _ = hidden_states.shape
    K, V = self.head_k_dim, self.head_v_dim
    q = rearrange(self.q_proj(hidden_states), 'b t (h d) -> b t h d', d=K).float()
    k = rearrange(self.k_proj(hidden_states), 'b t (h d) -> b t h d', d=K).float()
    v = rearrange(self.v_proj(hidden_states), 'b t (h d) -> b t h d', d=V).float()
    gk = rearrange(self.gk_proj(hidden_states), 'b t (h d) -> b t h d', d=K).float()
    gk = F.logsigmoid(gk) / self.gate_logit_normalizer
    m = self.nest_width
    if m < K:
        q = torch.cat([q[..., :m], torch.zeros_like(q[..., m:])], -1)
        k = torch.cat([k[..., :m], torch.zeros_like(k[..., m:])], -1)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    scale = K ** -0.5
    H = q.shape[2]
    mode, c, pb = CFG["mode"], CFG["c"], CFG["pb"]
    S = q.new_zeros(B, H, V, K)                         # v-major, per-dim decay on K
    Snap = S.clone()
    G = q.new_zeros(B, H, K)                            # per-DIM log decay since snap
    outs = []
    for t in range(T):
        a = gk[:, t].exp()                              # (B,H,K)
        S = S * a.unsqueeze(2) + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(2)
        if mode != "fresh":
            if t % c == 0:
                Snap = S.clone(); G = torch.zeros_like(G)
            else:
                G = G + gk[:, t]
        qt = q[:, t] * scale
        if mode == "fresh":
            y = torch.einsum('bhvk,bhk->bhv', S, qt)
        elif mode == "c1":
            y = torch.einsum('bhvk,bhk->bhv', Snap * G.exp().unsqueeze(2), qt)
        else:                                           # v4: hot fresh + cold stale
            y_hot = torch.einsum('bhvk,bhk->bhv', S[..., :pb], qt[..., :pb].contiguous())
            y_cold = torch.einsum('bhvk,bhk->bhv',
                                  Snap[..., pb:] * G[..., pb:].exp().unsqueeze(2),
                                  qt[..., pb:].contiguous())
            y = y_hot + y_cold
        outs.append(y)
    o = torch.stack(outs, 1).to(hidden_states.dtype)
    g = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', d=V)
    o = self.g_norm_swish_gate(o, g)
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
    ap.add_argument("--ckpt", default="gla_lm_nested.pt")
    ap.add_argument("--tag", default="gla-nested")
    ap.add_argument("--pb", type=int, default=16)
    ap.add_argument("--cs", type=int, nargs="+", default=[16, 64])
    args = ap.parse_args()
    device = "cuda"
    val = np.load("wt103_val.npy")
    model = GLALanguageModel().to(device).bfloat16()
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    p_fused = ppl(model, val, 64, device=device)
    for b in model.blocks:
        b.attn.forward = naive_masked_gla_forward.__get__(b.attn)
    CFG["mode"] = "fresh"
    p_naive = ppl(model, val, 64, device=device)
    print(f"[{args.tag}] VALIDATION fused={p_fused:.2f} naive={p_naive:.2f}", flush=True)
    out = {"fresh": p_naive}
    for mode in ["c1", "v4"]:
        for c in args.cs:
            CFG.update(mode=mode, c=c, pb=args.pb)
            v = ppl(model, val, 64, device=device)
            print(f"[{args.tag}] ppl {mode}-c{c}: {v:.2f}", flush=True)
            out[f"{mode}_c{c}"] = v
    json.dump(out, open(f"gla_lm_eval_{args.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
