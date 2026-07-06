"""E4: mid-stream width growth/shrink on the trained nested real GDN.
P4 claim: k can change during a sequence with no recompute; grown dims start
empty (valid); shrink keeps a valid prefix memory."""
import torch, sys, numpy as np
sys.path.insert(0, "/home/omin/TTT-PNM/poc")
from nested_delta_mqar import make_imqar
from gdn_a4 import GDNLM
from gdn_a4_eval import use_eval_path, CFG, probe
device = "cuda"
nk, nv, nq = 256, 128, 24
model = GDNLM(1 + nk + nv).to(device).bfloat16()
model.load_state_dict(torch.load("gdn_a4_nested.pt"))
use_eval_path(model)
gen = torch.Generator(device=device); gen.manual_seed(999)
D = 64
CFG.update(mode="fresh", grow=None)
for w, nm in [(16, "fixed-16"), (64, "fixed-64")]:
    gen.manual_seed(999)
    r = probe(model, D, nq, nk, nv, w, device, gen)
    print(f"{nm:14s} " + " ".join(f"{k}:{v:.2f}" for k, v in r.items()), flush=True)
L = 2 * D + nq   # ~152 tokens; grow/shrink at midpoint
for w0, w1 in [(16, 64), (64, 16)]:
    CFG["grow"] = (w0, w1, L // 2)
    gen.manual_seed(999)
    r = probe(model, D, nq, nk, nv, 64, device, gen)
    print(f"grow {w0}->{w1:2d}    " + " ".join(f"{k}:{v:.2f}" for k, v in r.items()), flush=True)
CFG["grow"] = None
