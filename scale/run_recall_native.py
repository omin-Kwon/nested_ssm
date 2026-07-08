"""Recall-intensive suite through the NATIVE engine (coherent generation),
on the retrofitted 9B (R + tuned decay from ckpt).  Arms:
  fresh : retrofitted model, native forward (deploy reference)
  v4    : + v4_native_decode (prefill fresh, decode tiered pb/c, bf16 cold)
Usage: run_recall_native.py <fresh|v4> [ckpt] [limit]"""
import sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from nemotron_retrofit import ActRotMask
import v4_native_decode as V

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
mode = sys.argv[1]
ckpt = sys.argv[2] if len(sys.argv) > 2 else "nemo9b_rot_p4long.pt"
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 300
TASKS = ["fda", "swde", "squad_completion", "triviaqa", "nq_open", "drop"]

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to("cuda")
model.config.use_cache = True
mixers = [m for m in model.modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
for m in mixers:
    m.act = ActRotMask(m.act, m.intermediate_size, m.n_groups, m.ssm_state_size).to("cuda")
saved = torch.load(ckpt)
for i, m in enumerate(mixers):
    m.act.R.data.copy_(saved[i].to("cuda").float())
if "decay" in saved:
    for i, m in enumerate(mixers):
        m.A_log.data.copy_(saved["decay"]["A_log"][i].to("cuda"))
        m.dt_bias.data.copy_(saved["decay"]["dt_bias"][i].to("cuda"))
model.eval()
tag = f"nat_{mode}"
if mode == "v4":
    n = V.install(model, pb=32, c=16, cold_bf16=1)
    print(f"[{tag}] v4 installed on {n} mixers (pb=32 c=16 bf16cold)", flush=True)
else:
    # rotation caveat: native decode branch calls act on 2D -> ActRotMask would
    # skip R. Route ALL mixers through the v4 dispatcher with pb=128 (all-hot),
    # which mirrors native semantics but applies R in decode as well.
    n = V.install(model, pb=128, c=16, cold_bf16=0)
    print(f"[{tag}] fresh via pb=128 dispatcher on {n} mixers (R applies in decode)",
          flush=True)
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1)
res = lm_eval.simple_evaluate(model=lm, tasks=TASKS, limit=limit, bootstrap_iters=0)
out = {}
for t, m in res["results"].items():
    out[t] = {k.split(",")[0]: v for k, v in m.items()
              if isinstance(v, (int, float)) and "stderr" not in k}
    print(f"[{tag}] {t}: " + " ".join(f"{k}={round(v,4)}" for k, v in out[t].items()),
          flush=True)
json.dump({"mode": mode, "ckpt": ckpt, "limit": limit, "results": out},
          open(f"nemo9b_recall_{tag}.json", "w"), indent=1)
print(f"[{tag}] DONE", flush=True)
