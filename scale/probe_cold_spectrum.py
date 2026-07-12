"""Is the COLD state low-rank under the trained R? (명제 1: nesting ~ PCA
ordering => spectrum concentration). Per head, SVD of S_cold (P x Nc) captured
energy at rank r — trained R vs identity, on real decode states.
If rank-8 captures >>90% under R (and much less under id), a rank-r sketch
computed at flush lets every token read r(P+Nc) instead of P*Nc: cold-read /4.8.
Run: CUDA_VISIBLE_DEVICES=<G> ~/nemo_env/bin/python3 probe_cold_spectrum.py"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemotron_retrofit import ActRotMask
import v4_native_decode as V

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
HUB = "/NHNHOME/ARC/arclab/shared/hub"
CKPT = "nemo9b_rot_longcot.pt"
PB = 32

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

prompt = ("The Treaty of Westphalia in 1648 ended the Thirty Years' War. "
          "Explain its three main consequences for European politics, "
          "then compute 37*89 step by step.")
ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")

for key, useR in [("rot", True), ("id", False)]:
    for i, m in enumerate(mixers):
        m.act.R.data.copy_(saved[i].to("cuda").float() if useR
                           else torch.eye(m.ssm_state_size, device="cuda")
                           .repeat(m.n_groups, 1, 1))
        if "decay" in saved and useR:
            m.A_log.data.copy_(saved["decay"]["A_log"][i].to("cuda"))
            m.dt_bias.data.copy_(saved["decay"]["dt_bias"][i].to("cuda"))
    model.eval()
    V.install(model, pb=128, c=16, cold_bf16=0)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=100, do_sample=False,
                             pad_token_id=tok.eos_token_id,
                             return_dict_in_generate=True)
    cache = out.past_key_values
    fracs = {r: [] for r in (4, 8, 16, 32)}
    for m in mixers:
        S = cache.layers[m.layer_idx].recurrent_states.float()   # (1,H,P,N)
        Sc = S[0, :, :, PB:]                                     # (H,P,Nc)
        sv = torch.linalg.svdvals(Sc)                            # (H,min(P,Nc))
        e = sv.pow(2)
        tot = e.sum(-1, keepdim=True) + 1e-12
        cum = e.cumsum(-1) / tot
        for r in fracs:
            fracs[r].append(cum[:, r - 1].mean().item())
    print(f"[{key:3s}] mean energy captured @rank " +
          "  ".join(f"r{r}={100*sum(v)/len(v):.1f}%" for r, v in fracs.items()),
          flush=True)
