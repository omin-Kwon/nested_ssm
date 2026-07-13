"""B=256 MEASURED self-speculative cycle: c hot-only draft steps + 1 chunked
verify pass vs c raw full steps. FUSED kernels only.

  t_raw    : HF fast-path decode ms/step
  t_draft  : hot-only (pb=32, no cold read/write/flush) decode ms/step
  t_verify : one (B, c) forward with primed cache (chunked-prefill branch =
             mamba_chunk_scan_combined with initial_states) — the verify pass
  mech(c)  = c*t_raw / (c*t_draft + t_verify(c))      [all drafts accepted]
  eff(c)   = mech(c) * acceptance-efficiency (from selfspec policy sim ~0.93)

Usage: CUDA_VISIBLE_DEVICES=<G> ~/nemo_env/bin/python3 bench_selfspec_e2e.py [--B 256]
"""
import argparse, json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SHARED = "/NHNHOME/ARC/arclab/shared/hub"
ap = argparse.ArgumentParser()
ap.add_argument("--B", type=int, default=256)
ap.add_argument("--gen", type=int, default=64)
ap.add_argument("--cs", type=int, nargs="+", default=[4, 6, 8, 16])
args = ap.parse_args()
B = args.B

from transformers.models.nemotron_h import modeling_nemotron_h as M
tok = AutoTokenizer.from_pretrained(MID, cache_dir=SHARED)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16,
                                             cache_dir=SHARED).to("cuda")
assert M.is_fast_path_available, "fused kernels missing!"
model.config.use_cache = True
model.eval()
gk = dict(do_sample=False, pad_token_id=0, eos_token_id=None)
ids = torch.randint(1000, 30000, (B, 1), device="cuda")

def time_decode(label):
    with torch.no_grad():
        model.generate(ids, max_new_tokens=8, **gk)
        torch.cuda.synchronize(); t0 = time.time()
        model.generate(ids, max_new_tokens=args.gen, **gk)
        torch.cuda.synchronize(); dt = time.time() - t0
    ms = dt / args.gen * 1e3
    print(f"[{label}] {ms:.2f} ms/step", flush=True)
    return ms

# 1) raw decode
t_raw = time_decode("raw")

# 2) verify chunk timing (raw path, primed cache, multi-token forward)
t_verify = {}
with torch.no_grad():
    prompt = torch.randint(1000, 30000, (B, 64), device="cuda")
    r = model(prompt, use_cache=True)
    cache = r.past_key_values
    for c in args.cs:
        chunk = torch.randint(1000, 30000, (B, c), device="cuda")
        for _ in range(3):                                    # warmup
            r = model(chunk, past_key_values=cache, use_cache=True)
            cache = r.past_key_values
        torch.cuda.synchronize(); t0 = time.time()
        REP = 10
        for _ in range(REP):
            r = model(chunk, past_key_values=cache, use_cache=True)
            cache = r.past_key_values
        torch.cuda.synchronize()
        t_verify[c] = (time.time() - t0) / REP * 1e3
        print(f"[verify c={c}] {t_verify[c]:.2f} ms/chunk", flush=True)

# 3) hot-only draft decode
import v4_fused_decode as VF
n = VF.install(model, pb=32, c=1 << 20, cold="off")
print(f"[draft] hot-only installed on {n} mixers", flush=True)
t_draft = time_decode("draft(hot-only)")

print(f"\nB={B}  t_raw={t_raw:.2f}  t_draft={t_draft:.2f} ms/step")
out = {"B": B, "t_raw": t_raw, "t_draft": t_draft, "t_verify": t_verify}
for c in args.cs:
    mech = c * t_raw / (c * t_draft + t_verify[c])
    eff = mech * 0.93                       # acceptance efficiency (policy sim)
    out[f"mech_c{c}"] = mech
    print(f"c={c:2d}: cycle {c*t_draft + t_verify[c]:7.2f} ms vs raw {c*t_raw:7.2f} "
          f"-> mech {mech:.2f}x  eff~{eff:.2f}x")
json.dump(out, open(f"results/selfspec_e2e_B{B}.json", "w"), indent=1)
print("SELFSPEC BENCH DONE")
