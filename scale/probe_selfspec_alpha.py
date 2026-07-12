"""Self-speculative viability: acceptance rate of the hot-only draft.

alpha = P(argmax under hot-only readout == full-model token | exact state).
Verification resets state exactly every chunk, so state carried = full-width;
the draft differs ONLY in readout width. Expected accepted run E[L]=a/(1-a).
Causal control: trained R vs identity R (nesting should be what buys alpha).

Run: CUDA_VISIBLE_DEVICES=<G> ~/nemo_env/bin/python3 probe_selfspec_alpha.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemotron_retrofit import ActRotMask
import v4_native_decode as V

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
HUB = "/NHNHOME/ARC/arclab/shared/hub"
CKPT = "nemo9b_rot_longcot.pt"
GEN = 180

tok = AutoTokenizer.from_pretrained(MID, cache_dir=HUB)
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, cache_dir=HUB).to("cuda")
model.config.use_cache = True
V.M.is_fast_path_available = False
mixers = [m for m in model.modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
for m in mixers:
    m.act = ActRotMask(m.act, m.intermediate_size, m.n_groups,
                       m.ssm_state_size).to("cuda")
saved = torch.load(CKPT)

PROMPTS = [
    "Explain why the sky is blue in three sentences.",
    "A train travels 60 km/h for 2.5 hours, then 80 km/h for 1.5 hours. "
    "Total distance? Show your work.",
    "Write a short Python function that reverses a linked list.",
]

def set_R(useR):
    for i, m in enumerate(mixers):
        m.act.R.data.copy_(saved[i].to("cuda").float() if useR
                           else torch.eye(m.ssm_state_size, device="cuda")
                           .repeat(m.n_groups, 1, 1))
        if "decay" in saved and useR:
            m.A_log.data.copy_(saved["decay"]["A_log"][i].to("cuda"))
            m.dt_bias.data.copy_(saved["decay"]["dt_bias"][i].to("cuda"))
    model.eval()

@torch.no_grad()
def gen_full(ids):
    V.install(model, pb=128, c=16, cold_bf16=0)          # full fresh
    out = model.generate(ids, max_new_tokens=GEN, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return out[0]

@torch.no_grad()
def alpha_hot_only(seq, prompt_len, pb):
    """Teacher-forced manual decode over full-model seq; hot-only readout;
    state advances full-width (verification semantics)."""
    V.install(model, pb=pb, c=1 << 30, cold_bf16=0, coldoff=1)
    r = model(seq[:prompt_len][None].to("cuda"), use_cache=True)
    cache = r.past_key_values
    hit = tot = 0
    logits = r.logits[0, -1]
    for t in range(prompt_len, len(seq) - 1):
        hit += int(logits.argmax().item() == seq[t].item())
        tot += 1
        r = model(seq[t][None, None].to("cuda"), past_key_values=cache,
                  use_cache=True)
        cache = r.past_key_values
        logits = r.logits[0, -1]
    return hit / max(tot, 1)

for useR, name in [(True, "rot"), (False, "id ")]:
    set_R(useR)
    als = []
    for p in PROMPTS:
        msgs = [{"role": "user", "content": p}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt",
                                      add_generation_prompt=True)
        if not torch.is_tensor(ids):            # BatchEncoding on this tokenizer
            ids = ids["input_ids"]
        ids = ids.to("cuda")
        seq = gen_full(ids)
        a = alpha_hot_only(seq.cpu(), ids.shape[1], pb=32)
        als.append(a)
        print(f"[{name}] alpha={a:.3f}  E[run]={a/(1-a+1e-9):.1f}", flush=True)
    m = sum(als) / len(als)
    print(f"[{name}] MEAN alpha={m:.3f}  E[run]={m/(1-m+1e-9):.1f}", flush=True)
