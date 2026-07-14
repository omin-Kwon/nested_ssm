"""E-T2 (redesigned): domain-wise elasticity curves of a FIXED trained R.

Claim under test (generalization/transfer): the trained rotation's hot
subspace is task-SHARED — the vppl-vs-width curve keeps its shape across
domains the R was not specifically fit to (wiki prose / math word problems /
code). Controls: identity R (raw-basis truncation) and the GHOST-lite
calibration permutation — if those inflate much harder AND less uniformly
across domains, "learned basis transfers, statistic basis doesn't" is a
direct measurement (companion to E-T1's cos≈random verdict).

Protocol: vppl(seqlen 512, n=6 strided windows) per (ckpt, domain, width).
Curves reported raw and normalized by the same config's full-width (k128)
ppl, so per-domain difficulty cancels and only the truncation TAX remains.

Run: CUDA_VISIBLE_DEVICES=<G> ~/nemo_env/bin/python3 probe_domain_elasticity.py
"""
import json, os
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemotron_retrofit import ActRotMask, vppl
import v4_native_decode as V

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
HUB = "/NHNHOME/ARC/arclab/shared/hub"
CKPTS = {"trained": "nemo9b_rot_longcot2.pt",
         "identity": "nemo9b_identity.pt",
         "ghost": "nemo9b_ghost_perm.pt"}
WIDTHS = [8, 16, 32, 64, 96, 128]

tok = AutoTokenizer.from_pretrained(MID, cache_dir=HUB)

def _tokenize_corpus(texts, path):
    ids = []
    for t in texts:
        ids.extend(tok(t + "\n\n", add_special_tokens=False)["input_ids"])
    arr = np.array(ids, dtype=np.int32)
    np.save(path, arr)
    print(f"[prep] {path}: {len(arr):,} tokens", flush=True)
    return arr

def get_domains():
    doms = {"wiki": np.load("wt103_val_nemo.npy", mmap_mode="r")}
    if not os.path.exists("math_val_nemo.npy"):
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        _tokenize_corpus([r["question"] + "\n" + r["answer"] for r in ds],
                         "math_val_nemo.npy")
    doms["math"] = np.load("math_val_nemo.npy", mmap_mode="r")
    if not os.path.exists("code_val_nemo.npy"):
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        _tokenize_corpus([r["prompt"] + r["canonical_solution"] for r in ds],
                         "code_val_nemo.npy")
    doms["code"] = np.load("code_val_nemo.npy", mmap_mode="r")
    for k, v in doms.items():
        need = 512 * 4 * 6 + 513
        assert len(v) >= need, f"{k}: {len(v)} < {need} tokens"
    return doms

doms = get_domains()

model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, cache_dir=HUB).to("cuda")
model.config.use_cache = True
V.M.is_fast_path_available = False          # silent-bypass trap (see HANDOFF)
mixers = [m for m in model.modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
for m in mixers:
    m.act = ActRotMask(m.act, m.intermediate_size, m.n_groups,
                       m.ssm_state_size).to("cuda")
    m.chunk_size = 64                        # shared-GPU sliver memory
model.eval()
wrappers = [m.act for m in mixers]

out = {}
for name, ck in CKPTS.items():
    saved = torch.load(ck)
    for i, w in enumerate(wrappers):
        w.R.data.copy_(saved[i].to("cuda").float())
    out[name] = {}
    for dom, val in doms.items():
        curve = {}
        for k in WIDTHS:
            with torch.no_grad():
                curve[k] = vppl(model, val, k)
            model.eval()                     # vppl leaves train mode
        base = curve[128]
        out[name][dom] = {"ppl": curve,
                          "tax": {k: curve[k] / base for k in WIDTHS}}
        line = "  ".join(f"k{k}:{curve[k]:.2f}({curve[k]/base:.2f}x)"
                         for k in WIDTHS)
        print(f"[{name}/{dom}] {line}", flush=True)

json.dump(out, open("results/domain_elasticity.json", "w"), indent=1, default=str)
print("DOMAIN ELASTICITY DONE", flush=True)
