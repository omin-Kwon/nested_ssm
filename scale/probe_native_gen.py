"""Is the low recall-suite score an ENGINE artifact (naive fwd degenerates in
generation) or the model itself? Run the recall tasks through the NATIVE
transformers engine (cache-enabled, no patch) as the ground-truth reference."""
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to("cuda")
model.config.use_cache = True
model.eval()
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1)
res = lm_eval.simple_evaluate(
    model=lm, tasks=["fda", "swde", "squad_completion", "triviaqa", "nq_open", "drop"],
    limit=300, bootstrap_iters=0, log_samples=True)
out = {}
for t, m in res["results"].items():
    out[t] = {k.split(",")[0]: v for k, v in m.items()
              if isinstance(v, (int, float)) and "stderr" not in k}
    print(f"[native] {t}: " + " ".join(f"{k}={round(v,4)}" for k, v in out[t].items()),
          flush=True)
g = res["samples"]["fda"][0]["resps"][0][0][:80]
print("[native] fda gen sample:", repr(g), flush=True)
json.dump(out, open("nemo9b_recall_native.json", "w"), indent=1)
print("[native] DONE", flush=True)
