"""Placement-dial experiment: accuracy vs hot/cold boundary pb under v4.
Sweeps pb (hot prefix size) at fixed total width 64 and staleness c.
pb=64 == fresh (all hot); pb=0 == all-cold readout (~c1 with exact writes)."""
import argparse, math
import numpy as np
import torch, torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", choices=["gdn", "m2", "kda"], default="gdn")
    ap.add_argument("--ckpt", default="gdn_lm_nested.pt")
    ap.add_argument("--tag", default="gdn35-nested")
    ap.add_argument("--c", type=int, nargs="+", default=[16])
    ap.add_argument("--pbs", type=int, nargs="+", default=[0, 4, 8, 16, 32, 48, 64])
    args = ap.parse_args()
    device = "cuda"
    val = np.load("wt103_val.npy")
    if args.fam == "gdn":
        from gdn_lm import GDNLanguageModel
        from gdn_a4_eval import eval_gdn_forward, CFG
        from gdn_lm_eval import ppl
        model = GDNLanguageModel().to(device).bfloat16()
        model.load_state_dict(torch.load(args.ckpt)); model.eval()
        for b in model.blocks:
            b.attn.forward = eval_gdn_forward.__get__(b.attn)
        CFG.update(head_sel=None, grow=None)
    elif args.fam == "m2":
        from m2_lm import M2LanguageModel
        from m2_lm_eval import naive_m2_forward, CFG, ppl
        model = M2LanguageModel().to(device).bfloat16()
        model.load_state_dict(torch.load(args.ckpt)); model.eval()
        for b in model.blocks:
            b.attn.forward = naive_m2_forward.__get__(b.attn)
    else:
        from kda_lm import KDALanguageModel
        from kda_lm_eval import naive_masked_kda_forward, CFG, ppl
        model = KDALanguageModel().to(device).bfloat16()
        model.load_state_dict(torch.load(args.ckpt)); model.eval()
        for b in model.blocks:
            b.attn.forward = naive_masked_kda_forward.__get__(b.attn)
    CFG.update(mode="fresh")
    base = ppl(model, val, 64, device=device)
    print(f"[{args.tag}] fresh(=pb64): {base:.2f}", flush=True)
    for c in args.c:
        for pb in args.pbs:
            if pb >= 64:
                continue
            CFG.update(mode="v4", c=c, pb=pb)
            v = ppl(model, val, 64, device=device)
            print(f"[{args.tag}] v4 c={c} pb={pb} (hot {pb}/64={pb/64:.0%}): "
                  f"{v:.2f} (+{(v/base-1)*100:.1f}%)", flush=True)

if __name__ == "__main__":
    main()
