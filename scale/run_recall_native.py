"""Task suites through the NATIVE engine (coherent generation), on the
retrofitted 9B (R + tuned decay from ckpt).  Arms:
  fresh : retrofitted model via pb=128 dispatcher (R applies in decode too)
  v4    : v4_native_decode (prefill fresh, decode tiered pb/c, bf16 cold)
Usage: run_recall_native.py <fresh|v4> [--ckpt X] [--limit N] [--tasks ...]
       [--tag T] [--maxlen 4096  # RULER haystack length]"""
import sys, json, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from nemotron_retrofit import ActRotMask
import v4_native_decode as V

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SHARED_HUB = "/NHNHOME/ARC/arclab/shared/hub"   # read-only shared cache: model
# weights load from here explicitly; env HF_HUB_CACHE should point to a WRITABLE
# local hub so dataset downloads (gsm8k etc.) don't hit shared .lock PermissionError
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["raw", "fresh", "v4"])
ap.add_argument("--ckpt", default="nemo9b_rot_p4long.pt")
ap.add_argument("--limit", type=int, default=300)
ap.add_argument("--tasks", nargs="+",
                default=["fda", "swde", "squad_completion", "triviaqa",
                         "nq_open", "drop"])
ap.add_argument("--tag", default=None)
ap.add_argument("--maxlen", type=int, default=0,
                help="RULER max_seq_lengths metadata (0 = not a RULER run)")
ap.add_argument("--pb", type=int, default=32)
ap.add_argument("--c", type=int, default=16)
ap.add_argument("--model", default=MID,
                help="e.g. nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base (official base "
                     "numbers are for -Base; aligned ckpt underperforms base harness)")
args = ap.parse_args()
MID = args.model
mode, ckpt, limit, TASKS = args.mode, args.ckpt, args.limit, args.tasks

tok = AutoTokenizer.from_pretrained(MID, cache_dir=SHARED_HUB)
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, cache_dir=SHARED_HUB).to("cuda")
model.config.use_cache = True
tag = args.tag or f"nat_{mode}"
if mode == "raw":
    # stock public 9B: no ActRotMask, no ckpt, pure native forward
    model.eval()
    print(f"[{tag}] RAW un-retrofitted 9B (native forward, no R/decay)", flush=True)
else:
    mixers = [m for m in model.modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
    for m in mixers:
        m.act = ActRotMask(m.act, m.intermediate_size, m.n_groups,
                           m.ssm_state_size).to("cuda")
    saved = torch.load(ckpt)
    for i, m in enumerate(mixers):
        m.act.R.data.copy_(saved[i].to("cuda").float())
    if "decay" in saved:
        for i, m in enumerate(mixers):
            m.A_log.data.copy_(saved["decay"]["A_log"][i].to("cuda"))
            m.dt_bias.data.copy_(saved["decay"]["dt_bias"][i].to("cuda"))
    model.eval()
if mode == "v4":
    n = V.install(model, pb=args.pb, c=args.c, cold_bf16=1)
    print(f"[{tag}] v4 installed on {n} mixers (pb={args.pb} c={args.c} bf16cold)",
          flush=True)
elif mode == "fresh":
    # rotation caveat: native decode branch calls act on 2D -> ActRotMask would
    # skip R. Route ALL mixers through the v4 dispatcher with pb=128 (all-hot),
    # which mirrors native semantics but applies R in decode as well.
    n = V.install(model, pb=128, c=16, cold_bf16=0)
    print(f"[{tag}] fresh via pb=128 dispatcher on {n} mixers (R applies in decode)",
          flush=True)
lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1,
          max_length=args.maxlen + 1024 if args.maxlen else None)
kw = {}
if args.maxlen:
    kw["metadata"] = {"max_seq_lengths": [args.maxlen], "pretrained": MID,
                      "tokenizer": MID}
res = lm_eval.simple_evaluate(model=lm, tasks=TASKS, limit=limit,
                              bootstrap_iters=0, confirm_run_unsafe_code=True, **kw)
out = {}
for t, m in res["results"].items():
    out[t] = {k.split(",")[0]: v for k, v in m.items()
              if isinstance(v, (int, float)) and "stderr" not in k}
    print(f"[{tag}] {t}: " + " ".join(f"{k}={round(v,4)}" for k, v in out[t].items()),
          flush=True)
json.dump({"mode": mode, "ckpt": ckpt, "limit": limit, "results": out},
          open(f"nemo9b_recall_{tag}.json", "w"), indent=1)
print(f"[{tag}] DONE", flush=True)
