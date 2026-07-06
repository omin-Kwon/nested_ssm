"""
S8 analytic roofline / serving model for Elastic Test-Time Memory (v4).

Decode-dominated serving of a 7B-class GDN-hybrid LM. Compares:
  A. GPU-only, small state (d_k=128), fresh              -- fast, low recall
  B. GPU-only, LARGE state, fresh                        -- accurate, slow
  C. GPU-only, LARGE state, v4 semantics (cold stale c)  -- BW relief only
  D. GPU+CXL-PNM v4 (hot on GPU fresh; cold resident on PNM, chunk replay c)

Key physics:
- gated-delta decay rescales the WHOLE state every token -> fresh (c=1)
  semantics costs 2*S bytes of state traffic per token per request.
  Chunk replay costs 2*S/c  ("staleness = PNM efficiency knob", traffic ∝ 1/c).
- v4 link traffic per token per request per GDN layer:
  down: k_cold,q_cold (2*dk_cold*2B) + v,r_hot (2*dv*2B) + gates(~8B)
  up:   y_cold (dv*2B) + residual-norm (4B)
All parameters are explicit; change and rerun.
"""
import argparse, math

def fmt(x):
    return f"{x:,.0f}" if x >= 100 else f"{x:.2f}"

def model_cfg(dk_total, dk_hot=128):
    return dict(
        L=32, gdn_frac=0.75,             # 24 GDN layers + 8 SWA-attn layers
        H=16, dv=128,
        dk=dk_total, dk_hot=min(dk_hot, dk_total),
        params=7e9, bytes_per=2,          # bf16
        swa_window=512,
    )

def hw_cfg():
    return dict(
        hbm_cap=80e9, hbm_bw=3.35e12, gpu_flops=990e12,   # H100-class
        gpu_reserve=10e9,                                  # activations/workspace
        n_pnm=4, pnm_cap=512e9, pnm_bw=1.6e12,             # per-device near-bank BW
        link_bw=32e9,                                      # CXL per device, per dir
    )

def state_bytes(m, dk):
    Lg = m["L"] * m["gdn_frac"]
    return Lg * m["H"] * m["dv"] * dk * m["bytes_per"]

def kv_bytes(m):
    La = m["L"] * (1 - m["gdn_frac"])
    return La * m["H"] * m["dv"] * 2 * m["swa_window"] * m["bytes_per"]

def step_time(cfg, hw, B, c, mode):
    """seconds per decode step at batch B; returns (t, bottleneck, feasible)"""
    m = cfg
    Lg = m["L"] * m["gdn_frac"]
    W = m["params"] * m["bytes_per"]
    S_full = state_bytes(m, m["dk"])
    S_hot = state_bytes(m, m["dk_hot"])
    S_cold = S_full - S_hot
    # ---- HBM capacity check
    if mode in ("A", "B", "C"):
        per_req_hbm = kv_bytes(m) + S_full
    else:                                            # D: only hot state on GPU
        per_req_hbm = kv_bytes(m) + S_hot
    hbm_need = W + hw["gpu_reserve"] + B * per_req_hbm
    if hbm_need > hw["hbm_cap"]:
        return None, "HBM-capacity", False
    if mode == "D" and B * S_cold > hw["n_pnm"] * hw["pnm_cap"]:
        return None, "PNM-capacity", False
    # ---- GPU time: weights + kv + state traffic on HBM; dense compute
    if mode == "A" or mode == "B":
        gpu_state_traffic = B * 2 * S_full           # fresh: whole state r+w each token
    elif mode == "C":                                # v4 semantics on GPU: hot fresh, cold chunked
        gpu_state_traffic = B * (2 * S_hot + 2 * S_cold / c)
    else:                                            # D: only hot on GPU
        gpu_state_traffic = B * 2 * S_hot
    t_gpu_bw = (W + B * kv_bytes(m) + gpu_state_traffic) / hw["hbm_bw"]
    t_gpu_fl = 2 * m["params"] * B / hw["gpu_flops"]
    # ---- link + PNM (mode D only)
    t_link = t_pnm = 0.0
    if mode == "D":
        # per token, per request, per GDN layer, per head (dk/dv are per-head dims):
        # down: k_cold+q_cold slices, v, r_hot, gates; up: y_cold + residual norm
        dn = (2 * (m["dk"] - m["dk_hot"]) + 2 * m["dv"]) * m["bytes_per"] + 8
        up = m["dv"] * m["bytes_per"] + 4
        link_bytes = B * Lg * m["H"] * (dn + up)
        t_link = link_bytes / (hw["n_pnm"] * hw["link_bw"])
        t_pnm = B * 2 * S_cold / c / (hw["n_pnm"] * hw["pnm_bw"])
    t = max(t_gpu_bw, t_gpu_fl, t_link, t_pnm)
    bn = ["GPU-BW", "GPU-FLOPs", "link", "PNM-BW"][
        [t_gpu_bw, t_gpu_fl, t_link, t_pnm].index(t)]
    return t, bn, True

def best_batch(cfg, hw, c, mode, bmax=4096):
    best = None
    for B in [2 ** i for i in range(4, 13)]:
        t, bn, ok = step_time(cfg, hw, B, c, mode)
        if not ok:
            break
        tput = B / t
        if best is None or tput > best[2]:
            best = (B, t, tput, bn)
    return best

def main():
    hw = hw_cfg()
    print(f"HW: HBM {hw['hbm_cap']/1e9:.0f}GB @ {hw['hbm_bw']/1e12:.2f}TB/s | "
          f"{hw['n_pnm']}x PNM {hw['pnm_cap']/1e9:.0f}GB @ {hw['pnm_bw']/1e12:.1f}TB/s, "
          f"link {hw['link_bw']/1e9:.0f}GB/s ea")
    m_small = model_cfg(128)
    m_large = model_cfg(1024)
    print(f"state/request: small(dk=128) {state_bytes(m_small,128)/1e6:.1f}MB | "
          f"large(dk=1024) {state_bytes(m_large,1024)/1e6:.1f}MB "
          f"(hot {state_bytes(m_large,128)/1e6:.1f} + cold {(state_bytes(m_large,1024)-state_bytes(m_large,128))/1e6:.1f})")
    print()
    rows = []
    rows.append(("A  GPU-only small (dk=128, fresh)", m_small, 1, "A"))
    rows.append(("B  GPU-only LARGE (dk=1024, fresh)", m_large, 1, "B"))
    for c in (4, 16, 64):
        rows.append((f"C  GPU-only LARGE, v4 sem. c={c}", m_large, c, "C"))
    for c in (1, 4, 16, 64):
        rows.append((f"D  GPU+PNM v4 (dk=1024) c={c}", m_large, c, "D"))
    print(f"{'config':38s} {'B*':>5s} {'ms/tok':>7s} {'tok/s':>9s}  bottleneck")
    base_B = None
    for name, m, c, mode in rows:
        r = best_batch(m, hw, c, mode)
        if r is None:
            print(f"{name:38s}  infeasible"); continue
        B, t, tput, bn = r
        print(f"{name:38s} {B:>5d} {t*1e3:>7.2f} {tput:>9,.0f}  {bn}")
        if mode == "B": base_B = tput
        if mode == "D" and base_B:
            print(f"{'':38s}  -> {tput/base_B:.1f}x vs B (iso-state-size)")

if __name__ == "__main__":
    main()
