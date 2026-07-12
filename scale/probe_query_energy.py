"""Does the learned R concentrate QUERY energy into hot dims?

At each decode step, per head: rho = ||C_cold||^2 / ||C||^2 (fraction of the
query's energy pointing at cold coordinates).  If nesting worked, rho should
be low for most (token, head) pairs under the trained R (=> those reads are
skippable), and ~cold_frac (96/128=0.75) under identity R.

Run: CUDA_VISIBLE_DEVICES=<G> ~/nemo_env/bin/python3 probe_query_energy.py
"""
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

STATS = {"rot": [], "id": []}
_orig_decode = V._v4_decode
def spy_decode(self, input_states, cache_params, attention_mask):
    out = _orig_decode(self, input_states, cache_params, attention_mask)
    return out
# hook rho collection inside the act wrapper instead: capture C after rotation
class Spy(torch.nn.Module):
    def __init__(self, act, key):
        super().__init__()
        self.act, self.key = act, key
    def forward(self, x):
        y = self.act(x)
        if y.dim() == 3 and y.shape[-1] == self.act.conv_dim:
            gn = self.act.ng * self.act.ds
            C = y[..., -gn:].float().view(y.shape[0], y.shape[1],
                                          self.act.ng, self.act.ds)
            num = C[..., PB:].pow(2).sum(-1)
            den = C.pow(2).sum(-1) + 1e-9
            STATS[self.key].append((num / den).flatten().cpu())
        return y
    @property
    def R(self): return self.act.R
    def rotmask(self, x): return self.act.rotmask(x)
    @property
    def conv_dim(self): return self.act.conv_dim
    @property
    def ng(self): return self.act.ng
    @property
    def ds(self): return self.act.ds

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
        for i, m in enumerate(mixers):
            m.A_log.data.copy_(saved["decay"]["A_log"][i].to("cuda"))
            m.dt_bias.data.copy_(saved["decay"]["dt_bias"][i].to("cuda"))
    for m in mixers:
        m.act = Spy(m.act, key) if not isinstance(m.act, Spy) else \
            Spy(m.act.act, key)
    model.eval()
    V.install(model, pb=128, c=16, cold_bf16=0)   # fresh semantics, R in decode
    with torch.no_grad():
        model.generate(ids, max_new_tokens=120, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    rho = torch.cat(STATS[key])
    q = torch.tensor([.1, .25, .5, .75, .9])
    print(f"[{key:3s}] rho quantiles {[round(v,3) for v in torch.quantile(rho, q).tolist()]}"
          f"  mean {rho.mean():.3f}  frac(rho<0.3)={100*(rho<0.3).float().mean():.1f}%"
          f"  frac(rho<0.1)={100*(rho<0.1).float().mean():.1f}%", flush=True)
    STATS[key] = []
