"""GHOST-lite baseline ckpt: calibration-scored PERMUTATION in place of R.

Scores each state dim n (per layer, per group) by an output-aware proxy:
  score_n = E_cal[C_n^2] * E_cal[S_n^2]   (observability x energy)
then builds permutation matrices sorting dims by score (desc) so that
"keep top-32" == our width-32 masking. Saved in the retrofit ckpt format
({i: P_i}), runnable via run_recall_native --coldoff 1 --pb 32.
This reproduces GHOST's select-and-truncate semantics inside our harness —
same code path, only the basis differs (calibration permutation vs trained R).

Run: CUDA_VISIBLE_DEVICES=<G> ~/nemo_env/bin/python3 make_ghost_perm.py
"""
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemotron_retrofit import ActRotMask
import v4_native_decode as V

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
HUB = "/NHNHOME/ARC/arclab/shared/hub"

tok = AutoTokenizer.from_pretrained(MID, cache_dir=HUB)
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, cache_dir=HUB).to("cuda")
model.config.use_cache = True
V.M.is_fast_path_available = False
mixers = [m for m in model.modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
for m in mixers:                       # identity R wrapper (raw basis)
    m.act = ActRotMask(m.act, m.intermediate_size, m.n_groups,
                       m.ssm_state_size).to("cuda")
    m.chunk_size = 64
model.eval()

# spy: accumulate E[C_n^2] per (layer, group, n) in the RAW basis
CSTAT = [torch.zeros(m.n_groups, m.ssm_state_size, device="cuda") for m in mixers]
CNT = [0]
class CSpy(torch.nn.Module):
    def __init__(self, act, li):
        super().__init__()
        self.act, self.li = act, li
    def forward(self, x):
        y = self.act(x)
        if y.dim() == 3 and y.shape[-1] == self.act.conv_dim:
            gn = self.act.ng * self.act.ds
            C = y[..., -gn:].float().view(-1, self.act.ng, self.act.ds)
            CSTAT[self.li] += C.pow(2).mean(0)
            if self.li == 0:
                CNT[0] += 1
        return y
    def rotmask(self, x): return self.act.rotmask(x)
    @property
    def R(self): return self.act.R
    @property
    def conv_dim(self): return self.act.conv_dim
    @property
    def ng(self): return self.act.ng
    @property
    def ds(self): return self.act.ds
    @property
    def inter(self): return self.act.inter

for i, m in enumerate(mixers):
    m.act = CSpy(m.act, i)

val = np.load("wt103_val_nemo.npy", mmap_mode="r")
with torch.no_grad():
    SSTAT = [torch.zeros(m.n_groups, m.ssm_state_size, device="cuda")
             for m in mixers]
    for b in range(16):                                   # 8 x 1024-token rows
        s = b * 512 * 4
        row = torch.from_numpy(val[s:s + 512].astype(np.int64))[None].to("cuda")
        out = model(row, use_cache=True)
        cache = out.past_key_values
        for i, m in enumerate(mixers):                   # E[S_n^2]: final states
            S = cache.layers[m.layer_idx].recurrent_states.float()  # (1,H,P,N)
            H, G = m.num_heads, m.n_groups
            SSTAT[i] += S.pow(2).mean((0, 2)).view(G, H // G, -1).mean(1)

perm = {}
for i, m in enumerate(mixers):
    score = (CSTAT[i] / max(CNT[0], 1)) * SSTAT[i]       # (G,N)
    order = torch.argsort(score, dim=-1, descending=True)
    G, N = score.shape
    P = torch.zeros(G, N, N)
    for g in range(G):
        # rotmask computes x' = einsum('...n,mn->...m', x, R): x'_m = R[m,:]·x
        # want x'_m = x_{order[m]}  ->  P[g, m, order[g, m]] = 1
        P[g, torch.arange(N), order[g].cpu()] = 1.0
    perm[i] = P
torch.save(perm, "nemo9b_ghost_perm.pt")
kept = [int((torch.argsort(CSTAT[i][0], descending=True)[:32]).min()) for i in (0, 13, 26)]
print(f"saved nemo9b_ghost_perm.pt (calibration {CNT[0]} forwards); "
      f"sanity min-kept-idx L0/L13/L26: {kept}")
