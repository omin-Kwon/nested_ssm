"""Gate test: v4_native_decode with pb=128 (all-hot) must reproduce native fresh."""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
import v4_native_decode as V

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
mode = sys.argv[1] if len(sys.argv) > 1 else "native"   # native | pb128 | v4
tasks = ["fda", "triviaqa"]
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to("cuda")
model.config.use_cache = True
model.eval()
if mode == "pb128":
    print("[gate] installing v4 pb=128 (all-hot gate)", V.install(model, pb=128, c=16), flush=True)
elif mode == "v4":
    print("[gate] installing v4 pb=32 c=16", V.install(model, pb=32, c=16, cold_bf16=1), flush=True)
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1)
res = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=50, bootstrap_iters=0,
                              log_samples=True)
for t in tasks:
    m = {k.split(",")[0]: round(v, 4) for k, v in res["results"][t].items()
         if isinstance(v, float) and "stderr" not in k}
    print(f"[{mode}] {t}: {m}", flush=True)
print(f"[{mode}] fda gen:", repr(res["samples"]["fda"][0]["resps"][0][0][:70]), flush=True)
