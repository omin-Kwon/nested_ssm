"""MOTIVATION TABLE: what happens WITHOUT our method (H100 80GB, 7B GDN-hybrid).
 (M1) GPU-only ceiling vs state size dk  -- batch cap + BW -> throughput collapse
 (M2) naive offload, store-only: state lives in CXL as raw far-memory,
      GPU computes -> 2*S crosses the link EVERY token per request
 (M3) naive offload, PNM-computes-fresh: per-token rank-1 on PNM (no chunking)
      -> BW-generous upper bound (real PIM compute shape is worse; cf. Pimba SPUs)
 (M4) ours (nesting+v4, c=8) for contrast
"""
from serving_sim import cfg7b, HW, bmax
GB, TB = 1e9, 1e12

def tput_gpu_only(dk):
    m = cfg7b(dk=dk)
    B = bmax(m, HW, "B")
    if B <= 0: return 0, 0
    t = (m["W"] + B * m["kv"] + B * 2 * m["S_full"]) / HW["hbm_bw"]
    return B, B / t

print("== M1: GPU-only ceiling vs state size (accuracy knob dk) ==")
print(f"{'dk':>6s} {'state/req':>10s} {'B*':>6s} {'tok/s':>10s}")
for dk in (128, 256, 512, 1024, 2048):
    m = cfg7b(dk=dk)
    B, tp = tput_gpu_only(dk)
    print(f"{dk:>6d} {m['S_full']/1e6:>8.1f}MB {B:>6d} {tp:>10,.0f}")

m = cfg7b(dk=1024)
print("\n== M2: naive store-only offload (state in CXL, compute on GPU) ==")
B = bmax(m, HW, "D")                     # HBM freed -> big batch possible
t_link = B * 2 * m["S_full"] / (HW["n_pnm"] * HW["link_bw"])
t_gpu = (m["W"] + B * m["kv"]) / HW["hbm_bw"]
t = max(t_link, t_gpu)
print(f"B*={B}  link traffic {B*2*m['S_full']/1e9:,.0f} GB/token-step "
      f"-> {t*1e3:,.0f} ms/tok -> {B/t:,.0f} tok/s  (link-bound)")

print("\n== M3: naive full-offload, PNM computes FRESH per token (no chunking) ==")
t_pnm = B * 2 * m["S_full"] / (HW["n_pnm"] * HW["pnm_bw"])
t_gpu = (m["W"] + B * m["kv"]) / HW["hbm_bw"]
t = max(t_pnm, t_gpu)
need = B * 2 * m["S_full"] / t_gpu / HW["n_pnm"]
print(f"B*={B}  PNM traffic 2S/tok -> {t*1e3:.1f} ms/tok -> {B/t:,.0f} tok/s (PNM-BW-bound)")
print(f"  to reach GPU ceiling would need {need/TB:.2f} TB/s/device (CXL-PNM class: 1.6)")
print("  NOTE: BW-generous — assumes weak PNM compute sustains per-token rank-1 at")
print("  full DRAM BW (Pimba needed dedicated SPUs); real number lower.")

print("\n== M4: ours (nesting + v4, c=8) ==")
t_gpu = (m["W"] + B * m["kv"] + B * 2 * m["S_hot"]) / HW["hbm_bw"]
t_pnm = B * m["S_cold"] * (1 + 2 / 8) / (HW["n_pnm"] * HW["pnm_bw"])
t = max(t_gpu, t_pnm)
print(f"B*={B}  {t*1e3:.1f} ms/tok -> {B/t:,.0f} tok/s (GPU-BW-bound; PNM needs only "
      f"{B*m['S_cold']*(1+2/8)/t_gpu/HW['n_pnm']/TB:.2f} TB/s/dev)")
