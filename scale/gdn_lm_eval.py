"""T3 verdict: language-trained nested GDN under staleness arms.
Metrics: (1) val ppl vs width (elasticity on real language)
         (2) val ppl under arms fresh/a/c1/v4 (pb=16, c in {16,64})
         (3) age-resolved needle recall under arms (definitive A3v2-warning test)
Validation gate: fused vs naive ppl equality."""
import argparse, math, json, sys
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, "/home/omin/TTT-PNM/poc")
from gdn_lm import GDNLanguageModel
from gdn_a4_eval import eval_gdn_forward, CFG
from transformers import AutoTokenizer

def use_eval_path(model):
    for b in model.blocks:
        b.attn.forward = eval_gdn_forward.__get__(b.attn)

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

def build_probes(tok, n_prompts=32, n_pairs=4, nfill=6, seed=0):
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
def needle(model, probes, device, w=64):
    hits, ages = [], []
    for pr in probes:
        logits = model(pr["ids"][None].to(device), width=w)[0]
        ok = all(logits[p - 1].argmax().item() == pr["ids"][p].item() for p in pr["dpos"])
        hits.append(int(ok)); ages.append(pr["age"])
    return np.array(hits), np.array(ages)

def agestr(hits, ages, c):
    old = hits[ages > c].mean() if (ages > c).any() else float("nan")
    yng = hits[ages <= c].mean() if (ages <= c).any() else float("nan")
    return f"all={hits.mean():.2f} old={old:.2f} young={yng:.2f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="gdn_lm_nested.pt")
    ap.add_argument("--tag", default="lm-nested")
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--pb", type=int, default=16)
    ap.add_argument("--cs", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()
    device = "cuda"
    val = np.load("wt103_val.npy")
    tok = AutoTokenizer.from_pretrained("fla-hub/gla-340M-15B")
    model = GDNLanguageModel(d=args.d, n_layers=args.layers, heads=args.heads).to(device).bfloat16()
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    out = {}
    # (1) elasticity: ppl vs width (fused)
    row = {w: ppl(model, val, w, device=device) for w in args.widths}
    print(f"[{args.tag}] ppl vs width: " + " ".join(f"k{w}:{v:.2f}" for w, v in row.items()), flush=True)
    out["ppl_vs_width"] = row
    # validation gate
    p_fused = row[64]
    use_eval_path(model); CFG.update(mode="fresh", grow=None, head_sel=None)
    p_naive = ppl(model, val, 64, device=device)
    print(f"[{args.tag}] VALIDATION fused={p_fused:.2f} naive={p_naive:.2f}", flush=True)
    # (2) ppl under arms
    for mode in ["a", "c1", "v4"]:
        for c in args.cs:
            CFG.update(mode=mode, c=c, pb=args.pb)
            v = ppl(model, val, 64, device=device)
            print(f"[{args.tag}] ppl {mode}-c{c}: {v:.2f}", flush=True)
            out[f"ppl_{mode}_c{c}"] = v
    # (3) needle under arms
    probes = build_probes(tok)
    CFG.update(mode="fresh")
    h, a = needle(model, probes, device)
    print(f"[{args.tag}] needle fresh-k64: {agestr(h, a, max(args.cs))} "
          f"(ages {int(np.median(a))} med)", flush=True)
    out["needle_fresh"] = float(h.mean())
    CFG_pb = args.pb
    h16, _ = needle(model, probes, device, w=args.pb)
    print(f"[{args.tag}] needle fresh-k{args.pb}(hot alone): all={h16.mean():.2f}", flush=True)
    for mode in ["a", "c1", "v4"]:
        for c in args.cs:
            CFG.update(mode=mode, c=c, pb=CFG_pb)
            h, a = needle(model, probes, device)
            print(f"[{args.tag}] needle {mode}-c{c}: {agestr(h, a, c)}", flush=True)
            out[f"needle_{mode}_c{c}"] = float(h.mean())
    json.dump(out, open(f"gdn_lm_eval_{args.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
