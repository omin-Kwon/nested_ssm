"""Kernel-level profile of one decode step: raw vs v4 — find where time goes.
Usage: profile_decode.py <raw|v4c4|v4c16> [--B 256]"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SHARED = "/NHNHOME/ARC/arclab/shared/hub"
ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=["raw", "v4c4", "v4c16"])
ap.add_argument("--B", type=int, default=256)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MID, cache_dir=SHARED)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16,
                                             cache_dir=SHARED).to("cuda")
model.config.use_cache = True
model.eval()
if args.arm != "raw":
    import v4_fused_decode as VF
    c = 4 if args.arm == "v4c4" else 16
    VF.install(model, pb=32, c=c, cold="bf16")

B = args.B
ids = torch.randint(1000, 30000, (B, 1), device="cuda")
gk = dict(do_sample=False, pad_token_id=0, eos_token_id=None)
with torch.no_grad():
    model.generate(ids, max_new_tokens=24, **gk)          # warmup + capture
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        model.generate(ids, max_new_tokens=32, **gk)
tbl = prof.key_averages().table(sort_by="cuda_time_total", row_limit=18)
print(f"===== {args.arm} B={B} (32 decode steps) =====")
print(tbl)
