"""Batch sweep B in {1,16,32,64,128,256}: per-component decode breakdown on a
DEDICATED B200 via vLLM (in-process engine, eager, fp32 state) + wall/throughput.

One LLM instance (max slots for B=256), per B: warmup generate, then profiled
generate. Kernel time bucketed; wall measured around the profiled call.
Output: scale/results/vllm_sweep_breakdown.json  (plots: plot_roofline_sweep.py)

Usage: CUDA_VISIBLE_DEVICES=2 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
       ~/vllm_env/bin/python3 vllm_sweep_profile.py
"""
import argparse, json, os, time, torch

ap = argparse.ArgumentParser()
ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 32, 64, 128, 256])
ap.add_argument("--state_dtype", default="float32",
                help="fp32 state caps at B~1180 on 178GB (2048x113MB=232GB "
                     "does NOT fit -> bf16 for B=2048; capacity wall = PNM motivation)")
ap.add_argument("--util", type=float, default=0.85)
ap.add_argument("--out", default="results/vllm_sweep_breakdown.json")
args = ap.parse_args()

GEN = 64
BATCHES = args.batches

from vllm import LLM, SamplingParams
llm = LLM(model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
          mamba_ssm_cache_dtype=args.state_dtype,
          gpu_memory_utilization=args.util, max_num_seqs=max(BATCHES),
          max_model_len=2048,
          enforce_eager=True, download_dir="/NHNHOME/ARC/arclab/shared/hub")
sp = SamplingParams(temperature=0, max_tokens=GEN, ignore_eos=True)

BUCKETS = {
    "state-op (SSU)": ["selective_state", "selective_scan_update", "_state_update"],
    "conv update": ["causal_conv1d_update", "conv1d_update"],
    "prefill scan/conv": ["chunk_scan", "chunk_state", "bmm_chunk", "state_passing",
                          "causal_conv1d_fwd", "chunk_cumsum"],
    "GEMM (proj/MLP/lm_head)": ["gemm", "cutlass", "matmul", "mm_", "nvjet"],
    "attention": ["flash", "attn", "trtllm", "paged"],
    "norm/elementwise": ["norm", "elementwise", "vectorized", "cat", "copy_",
                         "fill", "where", "softplus", "exp", "mul", "add",
                         "einsum", "bmm", "reduce_kernel"],
    "sampler/logits": ["softmax", "argmax", "topk", "sort", "gather"],
}

def bucket(name):
    n = name.lower()
    for b, keys in BUCKETS.items():
        if any(k in n for k in keys):
            return b
    return "other"

from torch.profiler import profile, ProfilerActivity
results = {}
for B in BATCHES:
    prompts = [f"Request {i}: Summarize the theory of relativity." for i in range(B)]
    llm.generate(prompts, sp)                       # warmup (JIT, shapes)
    t0 = time.perf_counter()
    llm.generate(prompts, sp)
    wall_unprof = time.perf_counter() - t0
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        llm.generate(prompts, sp)
        wall = time.perf_counter() - t0
    agg = {}
    for ev in prof.key_averages():
        t = ev.device_time_total
        if t > 0:
            agg[bucket(ev.key)] = agg.get(bucket(ev.key), 0.0) + t / 1e3  # ms
    busy = sum(agg.values())
    results[B] = dict(
        buckets_ms=agg, busy_ms=busy, wall_ms=wall * 1e3,
        wall_unprofiled_ms=wall_unprof * 1e3, gen=GEN,
        ms_per_step_busy=busy / GEN, ms_per_step_wall=wall_unprof * 1e3 / GEN,
        tok_per_s=B * GEN / wall_unprof, state_dtype=args.state_dtype)
    print(f"B={B:4d}: wall {wall_unprof*1e3/GEN:7.2f} ms/step  "
          f"busy {busy/GEN:7.2f}  {B*GEN/wall_unprof:9.0f} tok/s", flush=True)

os.makedirs("results", exist_ok=True)
if os.path.exists(args.out):                       # merge into existing sweep
    old = json.load(open(args.out))
    old.update({str(k): v for k, v in results.items()})
    results = old
json.dump(results, open(args.out, "w"), indent=1)
print(f"SWEEP DONE -> {args.out}")
