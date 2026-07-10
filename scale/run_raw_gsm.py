"""Raw (un-retrofitted) public Nemotron-9B on GSM8K — the missing first cell of
the 3-way comparison (raw / fresh=retrofit-tiering-off / v4=retrofit-tiering-on).
Isolates whether the R+decay retrofit preserves REASONING at full width."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SHARED = "/NHNHOME/ARC/arclab/shared/hub"
tok = AutoTokenizer.from_pretrained(MID, cache_dir=SHARED)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16,
                                             cache_dir=SHARED).to("cuda")
model.config.use_cache = True
model.eval()
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1)
res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"], limit=150, bootstrap_iters=0)
m = {k.split(",")[0]: v for k, v in res["results"]["gsm8k"].items()
     if isinstance(v, (int, float)) and "stderr" not in k}
print(f"[raw_gsm] gsm8k: {m}", flush=True)
json.dump(m, open("nemo9b_gsm_raw.json", "w"), indent=1)
print("[raw_gsm] DONE", flush=True)
