"""E-T1: explicit estimation of the traffic operator M = E[C B^T] (per layer,
per group) from calibration text — the theory-validation probe.

Outputs (per layer, averaged over groups):
  1. eigenvalue decay of sym(M) in the RAW basis  -> concentration exists?
     top-32 mass + spectral gap at n=32 (Davis-Kahan robustness quantity)
  2. principal angles between top-32 eigenspace of sym(M) and the TRAINED R's
     first-32 rows -> "SGD = implicit eigendecomposition" test
  3. empirical per-coordinate traffic in the trained basis
     T'_n = E[(RC)_n (RB)_n] -> is it actually sorted/decaying?

Run: CUDA_VISIBLE_DEVICES=<G> ~/nemo_env/bin/python3 probe_M_spectrum.py
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
Mstat = torch.zeros(L, G, N, N, device="cuda")   # E[C B^T] in RAW basis
CNT = [0]

class MSpy(torch.nn.Module):
    """Capture PRE-rotation B, C (raw basis) at every prefill forward."""
    def __init__(self, act, li):
        super().__init__()
        self.act, self.li = act, li
    def forward(self, x):
        y = self.act.act(x)                       # silu only (skip rotation here)
        if y.dim() == 3 and y.shape[-1] == self.act.conv_dim:
            gn = self.act.ng * self.act.ds
            B = y[..., self.act.inter:self.act.inter + gn]
            C = y[..., self.act.inter + gn:]
            Bf = B.float().reshape(-1, self.act.ng, self.act.ds)
            Cf = C.float().reshape(-1, self.act.ng, self.act.ds)
            Mstat[self.li] += torch.einsum('tgm,tgn->gmn', Cf, Bf)
            if self.li == 0:
                CNT[0] += Bf.shape[0]
            # continue the real path WITH rotation (trained-basis model runs)
            h = y[..., :self.act.inter]
            y = torch.cat([h, self.act.rotmask(B), self.act.rotmask(C)], dim=-1)
        return y
    # pass-throughs used elsewhere
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
    m.act = MSpy(m.act, i)

val = np.load("wt103_val_nemo.npy", mmap_mode="r")
with torch.no_grad():
    for b in range(24):
        s = b * 512 * 4
        row = torch.from_numpy(val[s:s + 512].astype(np.int64))[None].to("cuda")
        model(row)                                 # prefill only, no cache needed
print(f"[M-probe] {CNT[0]} tokens accumulated over {L}x{G} groups", flush=True)

Mstat /= max(CNT[0], 1)
out = {"layers": []}
for i, m in enumerate(mixers):
    R = m.act.R.detach()                           # (G,N,N) trained rotation
    tops, gaps, angs, sortedness = [], [], [], []
    for g in range(G):
        Msym = 0.5 * (Mstat[i, g] + Mstat[i, g].T)
        ev, evec = torch.linalg.eigh(Msym)         # ascending
        ev = ev.flip(0); evec = evec.flip(1)
        mass = ev.abs()
        tops.append(float(mass[:PB].sum() / mass.sum().clamp(min=1e-9)))
        gaps.append(float((mass[PB - 1] - mass[PB]) / mass[0].clamp(min=1e-9)))
        # principal angles: trained R rows 0..31  vs  top-32 eigenvectors
        A = R[g, :PB, :]                           # (32,N) rows
        Bv = evec[:, :PB]                          # (N,32) cols
        sv = torch.linalg.svdvals(A @ Bv)          # cos of principal angles
        angs.append(float(sv.mean()))              # 1.0 = same subspace
        # empirical traffic in trained basis, sortedness (Spearman-ish):
        Tn = torch.einsum('mn,am,an->a', Mstat[i, g], R[g], R[g]).abs()
        order = torch.argsort(Tn, descending=True).float()
        idx = torch.arange(N, device=order.device).float()
        sortedness.append(float(1 - (order - idx).abs().mean() / (N / 2)))
    out["layers"].append(dict(layer=i, top32_mass=sum(tops)/G,
                              gap_at_32=sum(gaps)/G, subspace_cos=sum(angs)/G,
                              sortedness=sum(sortedness)/G))
    print(f"L{i:02d}: top32 mass {sum(tops)/G:.3f}  gap@32 {sum(gaps)/G:.4f}  "
          f"R-vs-eig cos {sum(angs)/G:.3f}  sorted {sum(sortedness)/G:.3f}", flush=True)

agg = {k: sum(d[k] for d in out["layers"]) / L
       for k in ("top32_mass", "gap_at_32", "subspace_cos", "sortedness")}
out["mean"] = agg
print(f"\nMEAN: top32 mass {agg['top32_mass']:.3f} | gap@32 {agg['gap_at_32']:.4f} "
      f"| R-vs-eigenspace cos {agg['subspace_cos']:.3f} | sortedness {agg['sortedness']:.3f}")
json.dump(out, open("results/M_spectrum.json", "w"), indent=1)
print("M SPECTRUM DONE")
