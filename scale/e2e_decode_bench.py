"""END-TO-END decode throughput at B=256 — FUSED KERNELS ONLY (paper-grade).

raw    : HF fast path (causal_conv1d_update + selective_state_update) — how the
         model is actually deployed via HF. NO eager anywhere.
fresh  : same dispatcher as v4 but pb=128 (all-hot SSU) — gate arm; must ~= raw
v4*    : v4_fused_decode (hot=SSU narrow slice, cold=quantized snapshot readout
         [bf16 cuBLAS bmm | fp8 Triton], flush=amortized cuBLAS)
Timing: model.generate over gen tokens after warmup; reports ms/step + agg tok/s.
Usage: e2e_decode_bench.py <raw|fresh|v4c16|v4c4|v4c16fp8|v4c4fp8> [--B 256]"""
import time, json, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SHARED = "/NHNHOME/ARC/arclab/shared/hub"
ARMS = {"raw": None, "fresh": (128, 16, "bf16"),
        "v4c16": (32, 16, "bf16"), "v4c4": (32, 4, "bf16"),
        "v4c16fp8": (32, 16, "fp8"), "v4c4fp8": (32, 4, "fp8"),
        "v4c16fp32": (32, 16, "fp32"), "v4c4fp32": (32, 4, "fp32")}
ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=list(ARMS))
ap.add_argument("--B", type=int, default=256)
ap.add_argument("--gen", type=int, default=128)
args = ap.parse_args()

from transformers.models.nemotron_h import modeling_nemotron_h as M

tok = AutoTokenizer.from_pretrained(MID, cache_dir=SHARED)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16,
                                             cache_dir=SHARED).to("cuda")
# fast-path resolution is LAZY (at mixer init) — assert only after model load
assert M.is_fast_path_available, "fused kernels (causal_conv1d, mamba_ssm) missing!"
model.config.use_cache = True
model.eval()
if ARMS[args.arm] is not None:
    import v4_fused_decode as VF
    pb, c, cold = ARMS[args.arm]
    n = VF.install(model, pb=pb, c=c, cold=cold)
    print(f"[e2e_{args.arm}] fused v4 on {n} mixers (pb={pb} c={c} cold={cold})",
          flush=True)

B = args.B
ids = torch.randint(1000, 30000, (B, 1), device="cuda")
gk = dict(do_sample=False, pad_token_id=0, eos_token_id=None)
with torch.no_grad():
    model.generate(ids, max_new_tokens=8, **gk)              # warmup
    torch.cuda.synchronize(); t0 = time.time()
    model.generate(ids, max_new_tokens=args.gen, **gk)
    torch.cuda.synchronize(); dt = time.time() - t0
ms = dt / args.gen * 1e3
toks = B * args.gen / dt
print(f"[e2e_{args.arm}] B={B} gen={args.gen}: {ms:.2f} ms/step, "
      f"{toks:,.0f} tok/s aggregate", flush=True)
json.dump({"arm": args.arm, "B": B, "ms_per_step": ms, "tok_s": toks},
          open(f"results/e2e_fused_{args.arm}_B{B}.json", "w"), indent=1)
