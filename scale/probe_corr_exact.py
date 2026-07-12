"""Gate for rank-c exact correction: with fp32 cold (no quant rounding),
v4-c32-pb32 +corr must generate TOKEN-IDENTICAL text to fresh (pb=128).
Additive/mamba2 exactness proof by construction.
Run: CUDA_VISIBLE_DEVICES=<G> ~/nemo_env/bin/python3 probe_corr_exact.py"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemotron_retrofit import ActRotMask
import v4_native_decode as V

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
HUB = "/NHNHOME/ARC/arclab/shared/hub"
CKPT = "nemo9b_rot_longcot.pt"

tok = AutoTokenizer.from_pretrained(MID, cache_dir=HUB)
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, cache_dir=HUB).to("cuda")
model.config.use_cache = True
V.M.is_fast_path_available = False
mixers = [m for m in model.modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
for m in mixers:
    m.act = ActRotMask(m.act, m.intermediate_size, m.n_groups,
                       m.ssm_state_size).to("cuda")
saved = torch.load(CKPT)
for i, m in enumerate(mixers):
    m.act.R.data.copy_(saved[i].to("cuda").float())
if "decay" in saved:
    for i, m in enumerate(mixers):
        m.A_log.data.copy_(saved["decay"]["A_log"][i].to("cuda"))
        m.dt_bias.data.copy_(saved["decay"]["dt_bias"][i].to("cuda"))
model.eval()

prompt = ("Q: A farmer has 17 sheep and buys 23 more each week for 4 weeks, "
          "then sells half. How many remain? Think step by step.\nA:")
ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")

def gen(label, **cfg):
    V.install(model, **cfg)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    print(f"[{label}] ...{tok.decode(out[0, -40:])!r}"[:150], flush=True)
    return out[0, ids.shape[1]:].tolist()

fresh = gen("fresh(pb128)", pb=128, c=16, cold_bf16=0)
stale = gen("v4-c32 stale", pb=32, c=32, cold_bf16=0, corr=0)
corr = gen("v4-c32 +corr", pb=32, c=32, cold_bf16=0, corr=1)

def agree(a, b):
    n = min(len(a), len(b))
    same = next((i for i in range(n) if a[i] != b[i]), n)
    return same, n

s, n = agree(fresh, stale)
print(f"stale vs fresh: first divergence at token {s}/{n}")
s, n = agree(fresh, corr)
print(f"corr  vs fresh: first divergence at token {s}/{n}")
print("GATE " + ("PASS — correction is exact" if s == n else "FAIL"))
