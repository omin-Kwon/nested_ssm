"""Gate for v4_native_decode._lean_prefill: on a short input (fits the torch
path), rotated-model logits and final recurrent states must match between
  A) torch_forward prefill (reference, ActRotMask applies R)
  B) _lean_prefill (causal_conv1d + Triton chunk scan, manual R)
Run: CUDA_VISIBLE_DEVICES=<G> ~/nemo_env/bin/python3 probe_lean_prefill.py"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemotron_retrofit import ActRotMask
import v4_native_decode as V

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SHARED_HUB = "/NHNHOME/ARC/arclab/shared/hub"
CKPT = "nemo9b_rot_longcot.pt"

import sys
ROTATE = "--raw" not in sys.argv          # --raw: no R, isolates the scan mirror
tok = AutoTokenizer.from_pretrained(MID, cache_dir=SHARED_HUB)
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, cache_dir=SHARED_HUB).to("cuda")
model.config.use_cache = True
V.M.is_fast_path_available = False
mixers = [m for m in model.modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
if ROTATE:
    for m in mixers:
        m.act = ActRotMask(m.act, m.intermediate_size, m.n_groups,
                           m.ssm_state_size).to("cuda")
    saved = torch.load(CKPT)
    for i, m in enumerate(mixers):
        m.act.R.data.copy_(saved[i].to("cuda").float())
    if "decay" in saved:
        for i, m in enumerate(mixers):
            m.A_log.data.copy_(saved["decay"]["A_log"][i].to("cuda"))
            m.dt_bias.data.copy_(saved["decay"]["dt_bias"][i].to("cuda"))
model.eval()
V.install(model, pb=128, c=16, cold_bf16=0)

text = open("/home/omin/nested_ssm/README.md").read()
ids = tok(text, return_tensors="pt").input_ids[:, :300].to("cuda")

outs, states = {}, {}
for name, lean in [("torch", 0), ("lean", 1)]:
    for m in mixers:
        m.v4cfg["lean_prefill"] = lean
        m._v4state = None
        # shrink torch-path chunked-SSD intermediates so the fp32 reference
        # fits next to the other tenant; exact math either way
        m.chunk_size = 64 if not lean else 256
    with torch.no_grad():
        r = model(ids, use_cache=True)
    outs[name] = r.logits.float().cpu()
    cache = r.past_key_values
    states[name] = torch.stack(
        [cache.layers[m.layer_idx].recurrent_states.float().cpu() for m in mixers])

dl = (outs["torch"] - outs["lean"]).abs()
scale = outs["torch"].abs().mean()
ds = (states["torch"] - states["lean"]).abs()
sscale = states["torch"].abs().mean()
top_t = outs["torch"][0, -50:].argmax(-1)
top_l = outs["lean"][0, -50:].argmax(-1)
agree = (top_t == top_l).float().mean().item()
print(f"logits: max|d|={dl.max():.4f} mean|d|={dl.mean():.5f} (scale {scale:.2f})")
print(f"states: max|d|={ds.max():.4f} mean|d|={ds.mean():.6f} (scale {sscale:.4f})")
print(f"greedy top-1 agreement (last 50 pos): {agree:.2f}")
print("GATE " + ("PASS" if agree >= 0.98 and dl.mean() / scale < 0.02 else "FAIL"))
