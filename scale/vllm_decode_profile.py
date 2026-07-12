"""Per-step decode breakdown of Nemotron-9B on vLLM (offline, in-process engine).

Buckets CUDA kernel time of a decode-heavy generate() into: SSU (state
update+readout), conv update, GEMMs (projections/MLP/lm_head), attention,
norms/elementwise, sampler, other. Also reports wall/step and GPU-busy/step.

Caveat: run on leftover memory of a shared GPU -> small B; small-B decode is
launch/weight-bound, so the STATE-OP SHARE here is a lower bound vs the B=256
doctrine point. The bandwidth arithmetic for B=256 is printed alongside.

Usage: CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
       ~/vllm_env/bin/python3 vllm_decode_profile.py [--mode raw|v4] [--batch 24]
"""
import argparse, os, torch

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="raw", choices=["raw", "v4"])
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--gen", type=int, default=64)
ap.add_argument("--util", type=float, default=0.11)
args = ap.parse_args()

if args.mode == "v4":
    os.environ["NESTED_SSM_CKPT"] = "/home/omin/nested_ssm/scale/nemo9b_rot_longcot.pt"
    os.environ["NESTED_SSM_MODE"] = "v4"
    os.environ.setdefault("NESTED_SSM_PB", "32")
    os.environ.setdefault("NESTED_SSM_C", "4")
    os.environ.setdefault("NESTED_SSM_COLD", "bf16")

from vllm import LLM, SamplingParams

llm = LLM(model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
          mamba_ssm_cache_dtype="float32",
          gpu_memory_utilization=args.util, max_num_seqs=args.batch,
          max_model_len=2048, enforce_eager=True,
          download_dir="/NHNHOME/ARC/arclab/shared/hub")

sp = SamplingParams(temperature=0, max_tokens=args.gen, ignore_eos=True)
prompts = [f"Request {i}: Summarize the theory of relativity." for i in range(args.batch)]
llm.generate(prompts, sp)                                   # warmup

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    llm.generate(prompts, sp)

BUCKETS = {
    "state-op (SSU)": ["selective_state", "selective_scan_update", "_state_update",
                       "ssu"],
    "conv update": ["causal_conv1d_update", "conv1d_update"],
    "prefill scan/conv": ["chunk_scan", "chunk_state", "bmm_chunk", "state_passing",
                          "causal_conv1d_fwd", "chunk_cumsum"],
    "GEMM (proj/MLP/lm_head)": ["gemm", "cutlass", "matmul", "mm_", "nvjet"],
    "attention": ["flash", "attn", "trtllm", "paged"],
    "v4 extra (rot/readout)": ["einsum", "bmm", "reduce_kernel"],
    "norm/elementwise": ["norm", "elementwise", "vectorized", "cat", "copy_",
                         "fill", "where", "softplus", "exp", "mul", "add"],
    "sampler/logits": ["softmax", "argmax", "topk", "sort", "gather"],
}

def bucket(name):
    n = name.lower()
    for b, keys in BUCKETS.items():
        if any(k in n for k in keys):
            return b
    return "other"

agg, total = {}, 0.0
for ev in prof.key_averages():
    t = ev.device_time_total
    if t <= 0:
        continue
    agg[bucket(ev.key)] = agg.get(bucket(ev.key), 0.0) + t
    total += t

steps = args.gen
print(f"\n=== decode-dominated generate: B={args.batch} gen={steps} mode={args.mode} "
      f"(eager, fp32 state) ===")
print(f"GPU busy total {total/1e3:.1f} ms -> {total/1e3/steps:.3f} ms/step-ish "
      f"(includes 1 prefill of ~12 tok/req)")
for b, t in sorted(agg.items(), key=lambda kv: -kv[1]):
    print(f"  {b:28s} {t/1e3:9.2f} ms  {100*t/total:5.1f}%")

# ---- B=256 bandwidth arithmetic (why cold-write reduction pays) ----
L, H, P, N, dt = 27, 128, 64, 128, 4
state_bytes = L * H * P * N * dt
print(f"\n[arithmetic @ B=256, fp32 state]")
print(f"  state R+W per decode step : 2 x 256 x {state_bytes/1e6:.1f}MB "
      f"= {2*256*state_bytes/1e9:.1f} GB")
print(f"  weights per step (shared) : ~17.4 GB (bf16 9B)")
print(f"  -> state traffic {2*256*state_bytes/17.4e9:.1f}x weights at B=256; "
      f"v4-c16 cuts state write ~(1-pb/N-1/c)= {(1-32/128-1/16)*100:.0f}% of writes + "
      f"cold reads stale (bf16 halves bytes again)")
