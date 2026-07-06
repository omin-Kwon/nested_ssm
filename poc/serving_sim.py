"""S8 VERDICT LAYER: token-level serving simulation with queueing & latency.
Promotes the analytic ~3.8x (hypothesis) to a simulated verdict.

What the analytic roofline cannot see (and this sim adds):
 1. PNM chunk-REPLAY queueing: replays consume PNM bandwidth left over from
    per-token snapshot reads; if backlog exceeds the chunk window -> STALL.
 2. Link latency exposure per layer (request fired at projection, must return
    within the layer window) + link serialization across concurrent requests.
 3. Continuous batching with a prompt/decode mix: prefill jobs steal GPU time.
 4. Per-token jitter (lognormal service noise).

Configs: B  = GPU-only large state (fresh)
         D  = GPU+PNM v4 (hot fresh on GPU, cold resident on PNM, chunk c)
Deterministic-service discrete simulation over decode steps; workload = closed
system at max batch (capacity-limited, as in the roofline) plus a prefill duty
cycle given by the workload mix.
"""
import argparse, math
import numpy as np

GB, TB, US, MS = 1e9, 1e12, 1e-6, 1e-3

def cfg7b(dk=1024, dk_hot=128):
    L, gfrac, H, dv, By = 32, 0.75, 16, 128, 2
    Lg = L * gfrac
    S_full = Lg * H * dv * dk * By
    S_hot = Lg * H * dv * dk_hot * By
    return dict(L=L, Lg=Lg, H=H, dv=dv, dk=dk, dk_hot=dk_hot, By=By,
                params=7e9, W=14e9,
                S_full=S_full, S_hot=S_hot, S_cold=S_full - S_hot,
                kv=L * (1 - gfrac) * H * dv * 2 * 512 * By)

HW = dict(hbm=80 * GB, hbm_bw=3.35 * TB, reserve=10e9,
          n_pnm=4, pnm_bw=1.6 * TB, link_bw=32 * GB, rtt=2 * US)

def bmax(m, hw, mode):
    per = m["kv"] + (m["S_hot"] if mode == "D" else m["S_full"])
    return int((hw["hbm"] - m["W"] - hw["reserve"]) // per)

def simulate(m, hw, mode, c, steps=4000, prefill_frac=0.0, jitter=0.05, seed=0):
    """Returns tokens/s, mean/p99 step time, stall fraction."""
    rng = np.random.default_rng(seed)
    B = bmax(m, hw, mode)
    # --- static per-step service times
    if mode == "B":
        t_gpu = (m["W"] + B * m["kv"] + B * 2 * m["S_full"]) / hw["hbm_bw"]
        t_link = t_read = replay_per_boundary = 0.0
    else:
        t_gpu = (m["W"] + B * m["kv"] + B * 2 * m["S_hot"]) / hw["hbm_bw"]
        dn = (2 * (m["dk"] - m["dk_hot"]) + 2 * m["dv"]) * m["By"] + 8
        up = m["dv"] * m["By"] + 4
        t_link = B * m["Lg"] * m["H"] * (dn + up) / (hw["n_pnm"] * hw["link_bw"])
        t_read = B * m["S_cold"] / (hw["n_pnm"] * hw["pnm_bw"])        # per token
        replay_per_boundary = B * 2 * m["S_cold"] / (hw["n_pnm"] * hw["pnm_bw"])
    # prefill duty: fraction of GPU time consumed by prefill of other requests
    gpu_scale = 1.0 / max(1e-9, (1.0 - prefill_frac))
    backlog = 0.0                      # outstanding replay work (seconds of PNM time)
    times = np.empty(steps)
    stalls = 0
    for s in range(steps):
        jit = float(rng.lognormal(0.0, jitter))
        step_gpu = t_gpu * gpu_scale * jit
        # PNM this step must fit reads; leftover bandwidth services replay backlog
        step = max(step_gpu, t_link, t_read)
        # latency exposure per layer
        window = step / m["L"]
        step += m["Lg"] * max(0.0, hw["rtt"] - window)
        if mode == "D":
            if s % max(c, 1) == 0:
                backlog += replay_per_boundary
            pnm_free = max(0.0, step - t_read)     # PNM time left after reads
            backlog = max(0.0, backlog - pnm_free)
            # stall if backlog exceeds what the remaining chunk window can absorb:
            # snapshot for the NEXT boundary must be ready -> backlog must clear
            # within (c - s%c) steps of leftover capacity; enforce at boundaries.
            if s % max(c, 1) == max(c, 1) - 1 and backlog > 0:
                stalls += 1
                step += backlog                    # drain synchronously (stall)
                backlog = 0.0
        times[s] = step
    tput = B * steps / times.sum()
    return dict(B=B, tput=tput, mean_ms=times.mean() / MS,
                p99_ms=np.percentile(times, 99) / MS, stall_frac=stalls / steps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()
    m = cfg7b()
    print("== S8 VERDICT SIM (7B, dk=1024, hot=128) ==")
    for pf in (0.0, 0.3):
        b = simulate(m, HW, "B", 1, args.steps, prefill_frac=pf)
        print(f"\n-- prefill_frac={pf} --")
        print(f"B  GPU-only fresh      B*={b['B']:4d} {b['tput']:>9,.0f} tok/s "
              f"mean {b['mean_ms']:.2f}ms p99 {b['p99_ms']:.2f}ms")
        for n_pnm in (4, 8):
            hw = dict(HW); hw["n_pnm"] = n_pnm
            for c in (4, 8, 16, 64):
                d = simulate(m, hw, "D", c, args.steps, prefill_frac=pf)
                print(f"D  v4 c={c:<3d} n_pnm={n_pnm}  B*={d['B']:4d} {d['tput']:>9,.0f} tok/s "
                      f"mean {d['mean_ms']:.2f}ms p99 {d['p99_ms']:.2f}ms "
                      f"stalls {d['stall_frac']*100:4.1f}%  gain {d['tput']/b['tput']:.2f}x")
    # sensitivity: PNM bandwidth sweep at c=8 (the read-floor boundary)
    print("\n== read-floor sensitivity (c=8, n_pnm=4, prefill 0) ==")
    b = simulate(m, HW, "B", 1, args.steps)
    for bw in (0.8, 1.0, 1.2, 1.6, 2.0):
        hw = dict(HW); hw["pnm_bw"] = bw * TB
        d = simulate(m, hw, "D", 8, args.steps)
        print(f"pnm_bw {bw:.1f} TB/s/dev: {d['tput']:>9,.0f} tok/s "
              f"stalls {d['stall_frac']*100:4.1f}% gain {d['tput']/b['tput']:.2f}x")

if __name__ == "__main__":
    main()
