"""E-T1b: LAGGED traffic operator vs trained R.

E-T1 used same-token E[C_t B_t^T] (lag-0) and found ~random overlap with the
trained R. The true traffic operator is the decay-weighted cross-covariance
  M_lag = sum_tau a^tau E[C_t B_{t-tau}^T]
computed in one pass via a running decayed key bbar_t = a*bbar_{t-1} + B_t
and accumulating C_t bbar_t^T. Two decay constants probe short/long horizons.
If the trained-R subspace STILL mismatches -> "importance is loss-defined,
not covariance-defined" is fully confirmed.

Run: CUDA_VISIBLE_DEVICES=<G> ~/nemo_env/bin/python3 probe_M_lagged.py
"""
import json
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemotron_retrofit import ActRotMask
import v4_native_decode as V

MID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
HUB = "/NHNHOME/ARC/arclab/shared/hub"
CKPT = "nemo9b_rot_longcot2.pt"
PB = 32
DECAYS = [0.9, 0.99]

tok = AutoTokenizer.from_pretrained(MID, cache_dir=HUB)
model = AutoModelForCausalLM.from_pretrained(
    MID, dtype=torch.bfloat16, cache_dir=HUB).to("cuda")
model.config.use_cache = True
V.M.is_fast_path_available = False
mixers = [m for m in model.modules() if type(m).__name__ == "NemotronHMamba2Mixer"]
for m in mixers:
    m.act = ActRotMask(m.act, m.intermediate_size, m.n_groups,
                       m.ssm_state_size).to("cuda")
    m.chunk_size = 64
model.eval()

L, G, N = len(mixers), mixers[0].n_groups, mixers[0].ssm_state_size
Mlag = {a: torch.zeros(L, G, N, N, device="cuda") for a in DECAYS}
CNT = [0]

class LagSpy(torch.nn.Module):
    def __init__(self, act, li):
        super().__init__()
        self.act, self.li = act, li
    def forward(self, x):
        y = self.act.act(x)
        if y.dim() == 3 and y.shape[-1] == self.act.conv_dim:
            gn = self.act.ng * self.act.ds
            B = y[..., self.act.inter:self.act.inter + gn]
            C = y[..., self.act.inter + gn:]
            Bf = B.float().view(-1, self.act.ng, self.act.ds)   # (T,G,N)
            Cf = C.float().view(-1, self.act.ng, self.act.ds)
            T = Bf.shape[0]
            for a in DECAYS:
                bbar = torch.zeros(self.act.ng, self.act.ds, device=Bf.device)
                acc = torch.zeros(self.act.ng, self.act.ds, self.act.ds,
                                  device=Bf.device)
                for t in range(T):                              # sequential (decay)
                    bbar = a * bbar + Bf[t]
                    acc += torch.einsum('gm,gn->gmn', Cf[t], bbar)
                Mlag[a][self.li] += acc
            if self.li == 0:
                CNT[0] += T
            h = y[..., :self.act.inter]
            y = torch.cat([h, self.act.rotmask(B), self.act.rotmask(C)], dim=-1)
        return y
    def rotmask(self, x): return self.act.rotmask(x)
    @property
    def R(self): return self.act.R
    @property
    def conv_dim(self): return self.act.conv_dim
    @property
    def inter(self): return self.act.inter
    @property
    def ng(self): return self.act.ng
    @property
    def ds(self): return self.act.ds

saved = torch.load(CKPT)
for i, m in enumerate(mixers):
    m.act.R.data.copy_(saved[i].to("cuda").float())
    m.act = LagSpy(m.act, i)

val = np.load("wt103_val_nemo.npy", mmap_mode="r")
with torch.no_grad():
    for b in range(12):
        s = b * 512 * 4
        row = torch.from_numpy(val[s:s + 512].astype(np.int64))[None].to("cuda")
        model(row)
print(f"[M-lag] {CNT[0]} tokens", flush=True)

out = {}
for a in DECAYS:
    Ms = Mlag[a] / max(CNT[0], 1)
    tops, gaps, angs = [], [], []
    for i, m in enumerate(mixers):
        R = m.act.R.detach()
        for g in range(G):
            Msym = 0.5 * (Ms[i, g] + Ms[i, g].T)
            ev, evec = torch.linalg.eigh(Msym)
            ev = ev.flip(0); evec = evec.flip(1)
            mass = ev.abs()
            tops.append(float(mass[:PB].sum() / mass.sum().clamp(min=1e-9)))
            gaps.append(float((mass[PB-1] - mass[PB]) / mass[0].clamp(min=1e-9)))
            sv = torch.linalg.svdvals(R[g, :PB, :] @ evec[:, :PB])
            angs.append(float(sv.mean()))
    n = len(tops)
    out[str(a)] = dict(top32=sum(tops)/n, gap=sum(gaps)/n, cos=sum(angs)/n)
    print(f"decay={a}: top32 {sum(tops)/n:.3f}  gap@32 {sum(gaps)/n:.5f}  "
          f"R-vs-eig cos {sum(angs)/n:.3f}", flush=True)
json.dump(out, open("results/M_lagged.json", "w"), indent=1)
print("M LAGGED DONE")
