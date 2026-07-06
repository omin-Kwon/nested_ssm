"""Mamba2/SSD-family staleness verdict (Nemotron generality): ppl under arms on
the language-trained nested M2-core LM. ADDITIVE + per-head SCALAR decay: no
correction term, so arms are pure readout staleness: fresh / c1 / v4.
Scalar-G compensation. Validation gate: naive==fused."""
import argparse, math, json
import numpy as np
import torch, torch.nn.functional as F
from einops import rearrange
from m2_lm import M2LanguageModel

CFG = {"mode": "fresh", "c": 16, "pb": 16}

def naive_m2_forward(self, x):
    B, T, _ = x.shape
    K, V = self.head_k_dim, self.head_v_dim
    q = rearrange(self.q_proj(x), 'b t (h d) -> b t h d', d=K).float()
    k = rearrange(self.k_proj(x), 'b t (h d) -> b t h d', d=K).float()
    v = rearrange(self.v_proj(x), 'b t (h d) -> b t h d', d=V).float()
    g = F.logsigmoid(self.a_proj(x).float())            # (B,T,H) log-decay
    m = self.nest_width
    if m < K:
        q = torch.cat([q[..., :m], torch.zeros_like(q[..., m:])], -1)
        k = torch.cat([k[..., :m], torch.zeros_like(k[..., m:])], -1)
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    scale = K ** -0.5
    H = q.shape[2]
    mode, c, pb = CFG["mode"], CFG["c"], min(CFG["pb"], m)
    S = q.new_zeros(B, H, K, V)
    Snap = S.clone()
    G = q.new_zeros(B, H)                               # scalar log decay since snap
    outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None] \
            + k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
        if mode != "fresh":
            if t % c == 0:
                Snap = S.clone(); G = torch.zeros_like(G)
            else:
                G = G + g[:, t]
        q_t = q[:, t] * scale
        if mode == "fresh":
            y = torch.einsum('bhk,bhkv->bhv', q_t, S)
        elif mode == "c1":
            y = G.exp()[..., None] * torch.einsum('bhk,bhkv->bhv', q_t, Snap)
        else:                                           # v4: hot fresh + cold stale
            y_hot = torch.einsum('bhk,bhkv->bhv', q_t[..., :pb].contiguous(), S[:, :, :pb])
            y_cold = G.exp()[..., None] * torch.einsum(
                'bhk,bhkv->bhv', q_t[..., pb:].contiguous(), Snap[:, :, pb:])
            y = y_hot + y_cold
        outs.append(y)
    o = torch.stack(outs, 1).to(x.dtype)
    gate = rearrange(self.g_proj(x), 'b t (h d) -> b t h d', d=V)
    o = self.o_norm(o) * F.silu(gate)
    return self.o_proj(rearrange(o, 'b t h d -> b t (h d)'))

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
    ap.add_argument("--ckpt", default="m2_lm_nested.pt")
    ap.add_argument("--tag", default="m2-nested")
    ap.add_argument("--pb", type=int, default=16)
    ap.add_argument("--cs", type=int, nargs="+", default=[16, 64])
    args = ap.parse_args()
    device = "cuda"
    val = np.load("wt103_val.npy")
    model = M2LanguageModel().to(device).bfloat16()
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    p_fused = ppl(model, val, 64, device=device)
    for b in model.blocks:
        b.attn.forward = naive_m2_forward.__get__(b.attn)
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
    json.dump(out, open(f"m2_lm_eval_{args.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
