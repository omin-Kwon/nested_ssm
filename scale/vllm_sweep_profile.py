"""Batch sweep B in {1,16,32,64,128,256}: per-component decode breakdown on a
DEDICATED B200 via vLLM (in-process engine, eager, fp32 state) + wall/throughput.

One LLM instance (max slots for B=256), per B: warmup generate, then profiled
generate. Kernel time bucketed; wall measured around the profiled call.
Output: scale/results/vllm_sweep_breakdown.json  (plots: plot_roofline_sweep.py)

Usage: CUDA_VISIBLE_DEVICES=2 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
       ~/vllm_env/bin/python3 vllm_sweep_profile.py
"""
import json, os, time, torch

GEN = 64
BATCHES = [1, 16, 32, 64, 128, 256]

from vllm import LLM, SamplingParams
llm = LLM(model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
          mamba_ssm_cache_dtype="float32",
          gpu_memory_utilization=0.85, max_num_seqs=256, max_model_len=2048,
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
        tok_per_s=B * GEN / wall_unprof)
    print(f"B={B:4d}: wall {wall_unprof*1e3/GEN:7.2f} ms/step  "
          f"busy {busy/GEN:7.2f}  {B*GEN/wall_unprof:9.0f} tok/s", flush=True)

os.makedirs("results", exist_ok=True)
json.dump(results, open("results/vllm_sweep_breakdown.json", "w"), indent=1)
print("SWEEP DONE -> results/vllm_sweep_breakdown.json")
