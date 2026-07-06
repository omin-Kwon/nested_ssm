"""A3 v2: staleness-by-decay-group on pretrained gla-1.3B (zero training).
Fixes vs v1: working induction probe (teacher-forced digit scoring), model-
general shapes, refined slow thresholds (64 vs 256)."""
import argparse, math, json
import torch, torch.nn.functional as F
import numpy as np, fla
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer

STALE_CFG = {"masks": None, "c": 64}

def naive_gla_forward(self, hidden_states, attention_mask=None, past_key_values=None,
                      use_cache=False, output_attentions=False, **kw):
    B, T, _ = hidden_states.shape
    dk, dv, H = self.head_k_dim, self.head_v_dim, self.num_heads
    q = rearrange(self.q_proj(hidden_states), 'b t (h d) -> b t h d', d=dk).float()
    k = rearrange(self.k_proj(hidden_states), 'b t (h d) -> b t h d', d=dk).float()
    v = rearrange(self.v_proj(hidden_states), 'b t (h d) -> b t h d', d=dv).float()
    gk = rearrange(self.gk_proj(hidden_states), 'b t (h d) -> b t h d', d=dk).float()
    gk = F.logsigmoid(gk) / self.gate_logit_normalizer
    scale = dk ** -0.5
    S = q.new_zeros(B, H, dv, dk)
    cfg = STALE_CFG
    mask = cfg["masks"][self.layer_idx].to(q.device) if cfg["masks"] else None
    c = cfg["c"]
    Snap = S.clone()
    G = q.new_zeros(B, H, dk)
    outs = []
    for t in range(T):
        a = gk[:, t].exp()
        S = S * a.unsqueeze(2) + v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(2)
        if mask is not None:
            if t % c == 0:
                Snap = S.clone(); G = torch.zeros_like(G)
            else:
                G = G + gk[:, t]
            Sread = torch.where(mask[None, :, None, :], Snap * G.exp().unsqueeze(2), S)
        else:
            Sread = S
        outs.append(torch.einsum('bhvk,bhk->bhv', Sread, q[:, t] * scale))
    o = torch.stack(outs, 1).to(hidden_states.dtype)
    g = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', d=dv)
    o = rearrange(self.g_norm_swish_gate(o, g), 'b t h d -> b t (h d)')
    return self.o_proj(o), None, past_key_values

def set_arm(model, arm, tau, c, H, dk, seed=0):
    STALE_CFG["c"] = max(c, 1)
    if arm == "none":
        STALE_CFG["masks"] = None; return 0.0
    th = {"slow64": 64.0, "slow256": 256.0}.get(arm)
    rng = np.random.default_rng(seed)
    masks, tot, n = [], 0, 0
    for li in range(tau.shape[0]):
        t = tau[li]
        if th is not None:
            m = t >= th
        elif arm == "fast":                    # count-matched to slow64
            nslow = int((t >= 64.0).sum())
            idx = np.where(t < 8.0)[0]
            m = np.zeros_like(t, dtype=bool)
            m[rng.choice(idx, size=min(nslow, len(idx)), replace=False)] = True
        elif arm == "all":
            m = np.ones_like(t, dtype=bool)
        masks.append(torch.tensor(m.reshape(H, dk)))
        tot += m.sum(); n += m.size
    STALE_CFG["masks"] = masks
    return tot / n

def build_probes(tok, n_prompts, n_pairs=4, nfill=6, seed=0):
    rng = np.random.default_rng(seed)
    fill = open("pp.txt", encoding="utf-8").read()[20000:400000].split(". ")
    names = ["Falcon", "Zephyr", "Quartz", "Nebula", "Ostrich", "Lantern"]
    probes = []
    for p in range(n_prompts):
        pairs = [(nm, rng.integers(100, 999)) for nm in rng.choice(names, n_pairs, replace=False)]
        qi = int(rng.integers(0, n_pairs))
        s = ""
        vmarks = {}
        for nm, val in pairs:
            s += " ".join(rng.choice(fill, nfill)) + ". "
            vmarks[nm] = len(s)
            s += f"The secret code of {nm} is {val}. "
        s += " ".join(rng.choice(fill, 2)) + ". "
        qname, qval = pairs[qi]
        prompt = s + f"The secret code of {qname} is"
        full = prompt + f" {qval}"
        ids = tok(full, return_tensors="pt").input_ids[0]
        plen = tok(prompt, return_tensors="pt").input_ids.shape[1]
        ans = ids[plen:]                                # answer tokens (space+digits)
        # digit positions = answer tokens that decode to digits
        dpos = [plen + i for i, tid in enumerate(ans.tolist())
                if tok.decode([tid]).strip().isdigit()]
        pre = tok(s[:vmarks[qname]], return_tensors="pt").input_ids.shape[1]
        age = int(ids.shape[0] - pre)
        probes.append(dict(ids=ids, dpos=dpos, age=age))
    return probes

def run_probes(model, probes, dev):
    hits, ages = [], []
    for pr in probes:
        with torch.no_grad():
            logits = model(pr["ids"][None].to(dev)).logits[0]
        ok = all(logits[p - 1].argmax().item() == pr["ids"][p].item() for p in pr["dpos"])
        hits.append(int(ok)); ages.append(pr["age"])
    return np.array(hits), np.array(ages)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fla-hub/gla-1.3B-100B")
    ap.add_argument("--tau", default="gla13b_tau_pp.npy")
    ap.add_argument("--arms", nargs="+", default=["none", "slow64", "slow256", "fast"])
    ap.add_argument("--cs", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--n_prompts", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev).eval()
    H = model.model.layers[0].attn.num_heads
    dk = model.model.layers[0].attn.head_k_dim
    text = open("pp.txt", encoding="utf-8").read()[10000:60000]
    ids = tok(text, return_tensors="pt").input_ids[:, :1024].to(dev)

    # sanity: fused vs naive-fresh ppl
    with torch.no_grad():
        ppl_fused = math.exp(model(ids, labels=ids).loss.item())
    for li, layer in enumerate(model.model.layers):
        layer.attn.layer_idx = li
        layer.attn.forward = naive_gla_forward.__get__(layer.attn)
    STALE_CFG["masks"] = None
    with torch.no_grad():
        ppl_naive = math.exp(model(ids, labels=ids).loss.item())
    print(f"validation: fused {ppl_fused:.3f} vs naive {ppl_naive:.3f}", flush=True)

    tau = np.load(args.tau)
    probes = build_probes(tok, args.n_prompts)
    print("ages:", sorted(p["age"] for p in probes), flush=True)
    results = {}
    for arm in args.arms:
        for c in (args.cs if arm != "none" else [0]):
            frac = set_arm(model, arm, tau, c, H, dk)
            with torch.no_grad():
                ppl = math.exp(model(ids, labels=ids).loss.item())
            hits, ages = run_probes(model, probes, dev)
            old = hits[ages > c].mean() if (ages > c).any() else float("nan")
            yng = hits[ages <= c].mean() if (ages <= c).any() else float("nan")
            print(f"arm={arm:8s} c={c:>4d} staled={frac*100:4.1f}% | ppl={ppl:8.3f} | "
                  f"recall all={hits.mean():.2f} old={old:.2f} young={yng:.2f}", flush=True)
            results[f"{arm}_c{c}"] = dict(ppl=ppl, recall=float(hits.mean()),
                                          old=float(old), young=float(yng))
    json.dump(results, open("gla_stale2_results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
