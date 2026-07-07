"""T10 full verdict suite on retrofitted Nemotron-Nano-9B-v2 (REAL public 9B):
(1) validation gate (naive per-token == native torch_forward at full width)
(2) ppl under arms: fresh / c1 / v4 (additive family -> no correction taboo;
    staleness axis on d_state, hot prefix pb in the ROTATED basis)
(3) needle recall under arms (9B has real capability, unlike 35M/120M)
Run with ~/nemo_env/bin/python3."""
import argparse, math, json
import numpy as np
import torch, torch.nn.functional as F
from nemotron_retrofit import ActRotMask, get_wrappers, set_width

CFG = {"mode": "fresh", "c": 16, "pb": 32, "lag": 0, "cold_bf16": 0}

def naive_mixer_forward(self, input_states, cache_params=None, cache_position=None,
                        attention_mask=None, **kw):
    B_, T, _ = input_states.shape
    proj = self.in_proj(input_states)
    d_mlp = (proj.shape[-1] - 2 * self.intermediate_size
             - 2 * self.n_groups * self.ssm_state_size - self.num_heads) // 2
    _, _, gate, hBC, dt = proj.split(
        [d_mlp, d_mlp, self.intermediate_size,
         self.intermediate_size + 2 * self.n_groups * self.ssm_state_size,
         self.num_heads], dim=-1)
    hBC = hBC.transpose(1, 2)
    hBC = self.act(self.conv1d(hBC)[..., :hBC.shape[-1]].transpose(1, 2))  # act = ActRotMask (rot+mask)
    x, B, C = torch.split(
        hBC, [self.intermediate_size, self.n_groups * self.ssm_state_size,
              self.n_groups * self.ssm_state_size], dim=-1)
    H, P, N, G = self.num_heads, self.head_dim, self.ssm_state_size, self.n_groups
    rep = H // G
    x = x.reshape(B_, T, H, P).float()
    B = B.reshape(B_, T, G, N).repeat_interleave(rep, dim=2).float()
    C = C.reshape(B_, T, G, N).repeat_interleave(rep, dim=2).float()
    dt = F.softplus(dt.float() + self.dt_bias.float())
    dt = torch.clamp(dt, self.time_step_min)                      # (B,T,H)
    A = -torch.exp(self.A_log.float())                            # (H,)
    mode, c, pb, lag = CFG["mode"], CFG["c"], CFG["pb"], CFG.get("lag", 0)
    S = x.new_zeros(B_, H, P, N)
    Snap = S.clone()
    SnapL = S.clone()                                             # lagged snapshot (async flush)
    Gtot = x.new_zeros(B_, H)                                     # running scalar/head log decay
    gSnap = Gtot.clone(); gSnapL = Gtot.clone()
    outs = []
    for t in range(T):
        dA_log = dt[:, t] * A                                     # (B,H) <= 0
        S = S * dA_log.exp()[..., None, None] \
            + (dt[:, t][..., None] * x[:, t])[..., None] * B[:, t][:, :, None, :]
        Gtot = Gtot + dA_log
        if mode != "fresh" and t % c == 0:
            SnapL = Snap; gSnapL = gSnap                          # readers lag one chunk
            # cold_bf16: snapshot stored bf16 (values rounded, kept fp32 for einsum)
            Snap = (S.to(torch.bfloat16).float() if CFG.get("cold_bf16")
                    else S.clone())
            gSnap = Gtot.clone()
        Rd, gRd = (SnapL, gSnapL) if lag else (Snap, gSnap)
        Glog = Gtot - gRd                                         # decay since read snapshot
        Ct = C[:, t]
        if mode == "fresh":
            y = torch.einsum('bhpn,bhn->bhp', S, Ct)
        elif mode == "c1":
            y = torch.einsum('bhpn,bhn->bhp', Rd, Ct) * Glog.exp()[..., None]
        else:                                                     # v4
            y_hot = torch.einsum('bhpn,bhn->bhp', S[..., :pb], Ct[..., :pb].contiguous())
            y_cold = torch.einsum('bhpn,bhn->bhp', Rd[..., pb:],
                                  Ct[..., pb:].contiguous()) * Glog.exp()[..., None]
            y = y_hot + y_cold
        outs.append(y)
    y = torch.stack(outs, 1)                                      # (B,T,H,P)
    y = y + x * self.D.float()[None, None, :, None]
    y = y.reshape(B_, T, -1).to(input_states.dtype)
    y = self.norm(y, gate)
    return self.out_proj(y)

@torch.no_grad()
def ppl(model, val, seqlen=1024, n=4, device="cuda"):
    tot = 0.0
    for b in range(n):
        s = b * seqlen * 4
        row = torch.from_numpy(val[s:s + seqlen + 1].astype(np.int64))[None].to(device)
        logits = model(row[:, :-1]).logits
        tot += F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                               row[:, 1:].reshape(-1)).item()
    return math.exp(tot / n)

def build_probes(tok, n_prompts=24, n_pairs=4, nfill=6, seed=0):
    rng = np.random.default_rng(seed)
    fill = open("pp.txt", encoding="utf-8").read()[20000:400000].split(". ")
    names = ["Falcon", "Zephyr", "Quartz", "Nebula", "Ostrich", "Lantern"]
    probes = []
    for p in range(n_prompts):
        pairs = [(nm, rng.integers(100, 999)) for nm in rng.choice(names, n_pairs, replace=False)]
        qi = int(rng.integers(0, n_pairs))
        s = ""
        vmark = {}
        for nm, val_ in pairs:
            s += " ".join(rng.choice(fill, nfill)) + ". "
            vmark[nm] = len(s)
            s += f"The secret code of {nm} is {val_}. "
        s += " ".join(rng.choice(fill, 2)) + ". "
        qname, qval = pairs[qi]
        full = s + f"The secret code of {qname} is {qval}"
        ids = tok(full, return_tensors="pt").input_ids[0]
        plen = tok(s + f"The secret code of {qname} is", return_tensors="pt").input_ids.shape[1]
        dpos = [plen + i for i, tid in enumerate(ids[plen:].tolist())
                if tok.decode([tid]).strip().isdigit()]
        pre = tok(s[:vmark[qname]], return_tensors="pt").input_ids.shape[1]
        probes.append(dict(ids=ids, dpos=dpos, age=int(ids.shape[0] - pre)))
    return probes

@torch.no_grad()
def needle(model, probes, device):
    hits, ages = [], []
    for pr in probes:
        logits = model(pr["ids"][None].to(device)).logits[0]
        ok = all(logits[p - 1].argmax().item() == pr["ids"][p].item() for p in pr["dpos"])
        hits.append(int(ok)); ages.append(pr["age"])
    return np.array(hits), np.array(ages)

def agestr(h, a, c):
    old = h[a > c].mean() if (a > c).any() else float("nan")
    yng = h[a <= c].mean() if (a <= c).any() else float("nan")
    return f"all={h.mean():.2f} old={old:.2f} young={yng:.2f}"

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="nemo9b_rot_qr.pt")
    ap.add_argument("--tag", default="nemo9b-qr1k")
    ap.add_argument("--pb", type=int, default=32)
    ap.add_argument("--cs", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--skip_needle", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    val = np.load("wt103_val_nemo.npy")
    tok = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    model = AutoModelForCausalLM.from_pretrained(
        "nvidia/NVIDIA-Nemotron-Nano-9B-v2", dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    mixers = [m for m in model.modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
    for m in mixers:
        m.act = ActRotMask(m.act, m.intermediate_size, m.n_groups, m.ssm_state_size).to(device)
    saved = torch.load(args.ckpt)
    for i, m in enumerate(mixers):
        m.act.R.data.copy_(saved[i].to(device).float())
    if "decay" in saved:                                  # tune_decay ckpts
        for i, m in enumerate(mixers):
            m.A_log.data.copy_(saved["decay"]["A_log"][i].to(device))
            m.dt_bias.data.copy_(saved["decay"]["dt_bias"][i].to(device))
    model.eval()
    set_width(model, 128)
    p_fused = ppl(model, val, device=device)
    for m in mixers:
        m.forward = naive_mixer_forward.__get__(m)
    CFG["mode"] = "fresh"
    p_naive = ppl(model, val, device=device)
    print(f"[{args.tag}] VALIDATION torch_forward={p_fused:.2f} naive={p_naive:.2f}", flush=True)
    out = {"fresh": p_naive}
    for mode in ["c1", "v4"]:
        for c in args.cs:
            CFG.update(mode=mode, c=c, pb=args.pb)
            v = ppl(model, val, device=device)
            print(f"[{args.tag}] ppl {mode}-c{c} (hot {args.pb}/128): {v:.2f}", flush=True)
            out[f"{mode}_c{c}"] = v
    if not args.skip_needle:
        probes = build_probes(tok)
        CFG.update(mode="fresh")
        h, a = needle(model, probes, device)
        print(f"[{args.tag}] needle fresh-k128: {agestr(h, a, max(args.cs))} "
              f"(med age {int(np.median(a))})", flush=True)
        out["needle_fresh"] = float(h.mean())
        set_width(model, args.pb)                      # hot-alone: cold DISCARDED
        h, a = needle(model, probes, device)
        print(f"[{args.tag}] needle hot-alone-k{args.pb}: all={h.mean():.2f}", flush=True)
        out["needle_hotalone"] = float(h.mean())
        set_width(model, 128)
        for mode in ["c1", "v4"]:
            for c in args.cs:
                CFG.update(mode=mode, c=c, pb=args.pb)
                h, a = needle(model, probes, device)
                print(f"[{args.tag}] needle {mode}-c{c}: {agestr(h, a, c)}", flush=True)
                out[f"needle_{mode}_c{c}"] = float(h.mean())
    json.dump(out, open(f"nemo9b_eval_{args.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
