"""A3: staleness-by-decay-group on pretrained GLA-340M (zero training).

Replaces each GLA layer's fused recurrence with a naive per-token recurrence
(fp32 state) supporting chunk-refresh STALE READOUT on a per-dim mask, with
per-dim decay compensation (G_j cumprod). GLA is additive (no delta correction)
so this isolates pure readout staleness — the correction-taboo axis doesn't
apply here by construction.

Arms: none | slow-stale (tau>=TH dims) | fast-stale (count-matched tau<8 dims)
Pre-registered: slow-stale ~ baseline (ppl + old-needle recall); fast-stale
breaks broadly. If so: the model's own gates identify PNM-eligible dims,
zero training, on a third architecture.
"""
import argparse, math, json
import torch, torch.nn.functional as F
import numpy as np, fla
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "fla-hub/gla-340M-15B"

def naive_gla_forward(self, hidden_states, attention_mask=None, past_key_values=None,
                      use_cache=False, output_attentions=False, **kw):
    B, T, _ = hidden_states.shape
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)
    gk = self.gk_proj(hidden_states)
    dk, dv = self.head_k_dim, self.head_v_dim
    q = rearrange(q, 'b t (h d) -> b t h d', d=dk).float()
    k = rearrange(k, 'b t (h d) -> b t h d', d=dk).float()
    v = rearrange(v, 'b t (h d) -> b t h d', d=dv).float()
    gk = rearrange(gk, 'b t (h d) -> b t h d', d=dk).float()
    gk = F.logsigmoid(gk) / self.gate_logit_normalizer          # log alpha (B,T,H,dk)
    scale = dk ** -0.5
    H = self.num_heads
    S = q.new_zeros(B, H, dv, dk)                                # state (v-major)
    cfg = STALE_CFG
    mask = cfg["masks"][self.layer_idx].to(q.device) if cfg["masks"] else None  # (H,dk) True=stale
    c = cfg["c"]
    Snap = S.clone()
    G = q.new_zeros(B, H, dk)                                    # log-decay since snapshot
    outs = []
    for t in range(T):
        a = gk[:, t].exp()                                       # (B,H,dk)
        S = S * a.unsqueeze(2) + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(2)
        if mask is not None:
            if t % c == 0:
                Snap = S.clone(); G = torch.zeros_like(G)
            else:
                G = G + gk[:, t]
            # fresh dims read S; stale dims read decay-compensated snapshot
            Sread = torch.where(mask[None, :, None, :],
                                Snap * G.exp().unsqueeze(2), S)
        else:
            Sread = S
        y = torch.einsum('bhvk,bhk->bhv', Sread, q[:, t] * scale)
        outs.append(y)
    o = torch.stack(outs, 1)                                     # (B,T,H,dv)
    o = o.to(hidden_states.dtype)
    g = self.g_proj(hidden_states)
    g = rearrange(g, 'b t (h d) -> b t h d', d=dv)
    o = self.g_norm_swish_gate(o, g)
    o = rearrange(o, 'b t h d -> b t (h d)')
    o = self.o_proj(o)
    return o, None, past_key_values

STALE_CFG = {"masks": None, "c": 64}

def set_arm(model, arm, tau, c, slow_th=64.0, fast_th=8.0, seed=0):
    STALE_CFG["c"] = c
    if arm == "none":
        STALE_CFG["masks"] = None
        return 0.0
    masks, tot, n = [], 0, 0
    rng = np.random.default_rng(seed)
    for li in range(tau.shape[0]):
        t = tau[li]                                              # (512,) = H*dk
        if arm == "slow":
            m = t >= slow_th
        elif arm == "fast":
            nslow = int((t >= slow_th).sum())
            idx = np.where(t < fast_th)[0]
            pick = rng.choice(idx, size=min(nslow, len(idx)), replace=False)
            m = np.zeros_like(t, dtype=bool); m[pick] = True
        elif arm == "all":
            m = np.ones_like(t, dtype=bool)
        masks.append(torch.tensor(m.reshape(4, 128)))
        tot += m.sum(); n += m.size
    STALE_CFG["masks"] = masks
    return tot / n

def perplexity(model, ids):
    with torch.no_grad():
        out = model(ids, labels=ids)
    return math.exp(out.loss.item())

def needle_prompts(tok, n_prompts=24, n_pairs=6, seed=0):
    """Induction-style recall: pairs scattered through filler; query one pair.
    Returns input_ids list, target first-token ids, needle age (tokens)."""
    rng = np.random.default_rng(seed)
    fill_src = open("pp.txt", encoding="utf-8").read()[20000:400000].split(". ")
    names = ["Falcon", "Zephyr", "Quartz", "Nebula", "Ostrich", "Lantern", "Mango", "Cobalt"]
    prompts, targets, ages = [], [], []
    for p in range(n_prompts):
        pairs = [(names[i], rng.integers(100, 999)) for i in rng.choice(len(names), n_pairs, replace=False)]
        segs = []
        qi = int(rng.integers(0, n_pairs))                       # which pair to query
        for i, (nm, val) in enumerate(pairs):
            segs.append(" ".join(rng.choice(fill_src, 2)) + ". ")
            segs.append(f"The secret code of {nm} is {val}. ")
        segs.append(" ".join(rng.choice(fill_src, 2)) + ". ")
        qname, qval = pairs[qi]
        query = f"The secret code of {qname} is"
        text = "".join(segs) + query
        ids = tok(text, return_tensors="pt").input_ids[0]
        # age = distance from the value's occurrence to the end
        vpos_txt = "".join(segs).rfind(f"of {qname} is {qval}")
        pre = tok("".join(segs)[:vpos_txt], return_tensors="pt").input_ids.shape[1]
        ages.append(int(ids.shape[0] - pre))
        prompts.append(ids)
        targets.append(tok(f" {qval}", add_special_tokens=False).input_ids[0])
    return prompts, targets, ages

def needle_recall(model, tok, prompts, targets, device):
    hits = []
    for ids, tgt in zip(prompts, targets):
        with torch.no_grad():
            logits = model(ids[None].to(device)).logits[0, -1]
        hits.append(int(logits.argmax().item() == tgt))
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["none", "slow", "fast"])
    ap.add_argument("--cs", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev).eval()
    text = open("pp.txt", encoding="utf-8").read()[10000:60000]
    ids = tok(text, return_tensors="pt").input_ids[:, :1024].to(dev)

    if args.validate:
        ppl_fused = perplexity(model, ids)
        for li, layer in enumerate(model.model.layers):
            layer.attn.layer_idx = li
            layer.attn.forward = naive_gla_forward.__get__(layer.attn)
        STALE_CFG["masks"] = None
        ppl_naive = perplexity(model, ids)
        print(f"VALIDATION  ppl fused={ppl_fused:.3f}  naive-fresh={ppl_naive:.3f}  "
              f"(match => recurrence semantics correct)")
        return

    for li, layer in enumerate(model.model.layers):
        layer.attn.layer_idx = li
        layer.attn.forward = naive_gla_forward.__get__(layer.attn)

    tau = np.load("gla340m_tau_pp.npy")                          # (24, 512)
    prompts, targets, ages = needle_prompts(tok)
    print("needle ages (tok): min", min(ages), "median", int(np.median(ages)), "max", max(ages))
    results = {}
    for arm in args.arms:
        for c in (args.cs if arm != "none" else [0]):
            frac = set_arm(model, arm, tau, max(c, 1))
            ppl = perplexity(model, ids)
            hits = needle_recall(model, tok, prompts, targets, dev)
            ages_np = np.array(ages); hits_np = np.array(hits)
            old = hits_np[ages_np > c].mean() if (ages_np > c).any() else float("nan")
            yng = hits_np[ages_np <= c].mean() if (ages_np <= c).any() else float("nan")
            print(f"arm={arm:5s} c={c:>4d} staled={frac*100:4.1f}% | ppl={ppl:8.3f} | "
                  f"recall all={hits_np.mean():.2f} old(age>c)={old:.2f} young={yng:.2f}", flush=True)
            results[f"{arm}_c{c}"] = dict(ppl=ppl, recall=float(hits_np.mean()),
                                          old=float(old), young=float(yng), frac=float(frac))
    json.dump(results, open("gla_stale_results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
