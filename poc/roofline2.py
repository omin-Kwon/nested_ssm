"""
S8 v2 — GAIN-BOUNDARY MAP (not "a more precise 4x").
Purpose 1: find where the gain COLLAPSES (sensitivity boundaries).
Purpose 2: certify the scale-up design point (c=4, hot=128, 8x expansion).
Stratum: still ANALYTIC (hypothesis layer). Verdict layer = queueing sim / measurement.
"""
BYTES = 2                     # bf16 activations/state
GB, TB, MS, US = 1e9, 1e12, 1e-3, 1e-6

def cfg(params_b=7e9, dk=1024, dk_hot=128):
    m = dict(L=32, gdn_frac=0.75, H=16, dv=128, dk=dk, dk_hot=dk_hot,
             params=params_b, swa=512)
    m["Lg"] = m["L"] * m["gdn_frac"]
    m["S_full"] = m["Lg"] * m["H"] * m["dv"] * m["dk"] * BYTES
    m["S_hot"] = m["Lg"] * m["H"] * m["dv"] * m["dk_hot"] * BYTES
    m["S_cold"] = m["S_full"] - m["S_hot"]
    m["kv"] = m["L"] * (1 - m["gdn_frac"]) * m["H"] * m["dv"] * 2 * m["swa"] * BYTES
    m["W"] = m["params"] * BYTES
    return m

HW = dict(hbm=80 * GB, hbm_bw=3.35 * TB, flops=990 * TB, reserve=10 * GB,
          n_pnm=4, pnm_cap=512 * GB, pnm_bw=1.6 * TB, link=32 * GB, rtt=2 * US)

def bmax(m, hw, mode):
    per = m["kv"] + (m["S_hot"] if mode == "D" else m["S_full"])
    free = hw["hbm"] - m["W"] - hw["reserve"]
    if free <= 0: return 0
    return int(free // per)

def t_step(m, hw, B, c, mode):
    if mode in "AB": st = B * 2 * m["S_full"]
    elif mode == "C": st = B * (2 * m["S_hot"] + 2 * m["S_cold"] / c)
    else: st = B * 2 * m["S_hot"]
    t_bw = (m["W"] + B * m["kv"] + st) / hw["hbm_bw"]
    t_fl = 2 * m["params"] * B / hw["flops"]
    t_link = t_pnm = 0
    if mode == "D":
        dn = (2 * (m["dk"] - m["dk_hot"]) + 2 * m["dv"]) * BYTES + 8
        up = m["dv"] * BYTES + 4
        t_link = B * m["Lg"] * m["H"] * (dn + up) / (hw["n_pnm"] * hw["link"])
        # CORRECTED: per-token cold READOUT scans the snapshot (NOT /c);
        # only the UPDATE amortizes by chunk replay.
        t_pnm = B * m["S_cold"] * (1 + 2 / c) / (hw["n_pnm"] * hw["pnm_bw"])
    base = max(t_bw, t_fl, t_link, t_pnm)
    lat = m["Lg"] * max(0, hw["rtt"] - base / m["L"]) if mode == "D" else 0
    names = ["GPU-BW", "GPU-FL", "link", "PNM-BW"]
    return base + lat, names[[t_bw, t_fl, t_link, t_pnm].index(base)]

def tput(m, hw, c, mode):
    B = bmax(m, hw, mode)
    if B <= 0: return 0, 0, "infeasible"
    t, bn = t_step(m, hw, B, c, mode)
    return B / t, B, bn

m7 = cfg()
hw = dict(HW)
g_D, B_D, _ = tput(m7, hw, 4, "D")
g_B, B_B, _ = tput(m7, hw, 4, "B")
print(f"reference (7B, dk=1024, c=4): D={g_D:,.0f} tok/s (B={B_D}) vs B={g_B:,.0f} (B={B_B}) -> {g_D/g_B:.1f}x\n")

print("== BOUNDARY 1: PNM internal BW (per device) needed to keep full gain ==")
for c in (1, 4, 16):
    B = bmax(m7, hw, "D")
    t_ceiling, _ = t_step(m7, hw, B, c, "D")  # includes pnm at current bw; recompute ceiling w/o pnm
    tb = max((m7["W"] + B * m7["kv"] + B * 2 * m7["S_hot"]) / hw["hbm_bw"],
             2 * m7["params"] * B / hw["flops"])
    need = B * m7["S_cold"] * (1 + 2 / c) / tb / hw["n_pnm"]
    print(f"  c={c:>2d}: PNM BW >= {need/TB:.2f} TB/s/device keeps GPU-BW ceiling "
          f"(have 1.6 -> {'OK' if need<=hw['pnm_bw'] else 'BOTTLENECK'})")

print("\n== BOUNDARY 2: CXL link ==")
B = bmax(m7, hw, "D")
dn = (2 * (m7["dk"] - m7["dk_hot"]) + 2 * m7["dv"]) * BYTES + 8
up = m7["dv"] * BYTES + 4
lb = B * m7["Lg"] * m7["H"] * (dn + up)
tb = (m7["W"] + B * m7["kv"] + B * 2 * m7["S_hot"]) / hw["hbm_bw"]
print(f"  link traffic {lb/GB:.2f} GB/token-step; needs >= {lb/tb/GB:,.0f} GB/s total "
      f"({lb/tb/GB/hw['n_pnm']:,.1f}/device; have {hw['link']/GB:.0f}) "
      f"-> utilization {lb/(hw['n_pnm']*hw['link'])/tb*100:.0f}% of window")
print(f"  fp8 activations would halve this. RTT exposure: layer window "
      f"{tb/m7['L']/US:,.0f}us vs RTT {hw['rtt']/US:.0f}us -> margin {tb/m7['L']/hw['rtt']:,.0f}x (latency non-issue)")

print("\n== BOUNDARY 3: accuracy-knob ceiling (dk expansion at c=4, link-capped) ==")
for dkx in (2, 4, 8, 16, 32):
    m = cfg(dk=128 * dkx)
    g, B, bn = tput(m, hw, 4, "D")
    gb, Bb, _ = tput(m, hw, 4, "B")
    r = g / gb if gb else float("inf")
    print(f"  {dkx:>2d}x (dk={128*dkx:>5d}): D={g:>9,.0f} tok/s (B={B}, {bn})  gain={'inf' if gb==0 else f'{r:.1f}x'}")

print("\n== BOUNDARY 4: model size (HBM pressure) ==")
for pb, name in ((3e9, "3B"), (7e9, "7B"), (34e9, "34B"), (70e9, "70B")):
    m = cfg(params_b=pb)
    gD, BD, bnD = tput(m, hw, 4, "D")
    gB, BB, bnB = tput(m, hw, 4, "B")
    tag = "ENABLEMENT (B infeasible)" if gB == 0 and gD > 0 else \
          (f"{gD/gB:.1f}x" if gB else "both infeasible")
    print(f"  {name:>4s}: D={gD:>9,.0f} (B={BD})  B-config={gB:>9,.0f} (B={BB})  -> {tag}")

print("\n== BOUNDARY 5: prefill dilution (E2E gain vs prefill time fraction f in config B) ==")
for f in (0.1, 0.3, 0.5, 0.7):
    print(f"  f={f:.1f}: E2E gain = {1/(f + (1-f)/ (g_D/g_B)):.2f}x  "
          f"(decode-only {g_D/g_B:.1f}x)")
print(f"  one-time cold-state ship: {m7['S_cold']/hw['link']/MS:.1f} ms/request "
      f"(amortized over 256 tok: {m7['S_cold']/hw['link']/256/US:.0f} us/tok — negligible)")

print("\n== DESIGN-POINT CERTIFICATE (REVISED after read-traffic correction) ==")
print("  Per-token cold READOUT sets a BW floor ~1.05TB/s/dev (B~1200) that c cannot amortize;")
print("  c only removes the update surcharge (x(1+2/c)).")
print("  (c=4, n_pnm=4x1.6TB/s): need 1.57 -> margin 1.02x = MARGINAL, not robust.")
print("  Robust options: c=8 (need 1.31, margin 1.2x) | c=16 (1.18, 1.36x) | n_pnm=8 (2x at c=4).")
print("  REVISED scale-up target: (c=8, hot=128, 8x, n_pnm=4)  [recall cost c8: D32 0.916 vs fresh 0.988]")
print("  or (c=4, n_pnm=8). Accuracy-knob ceiling now PNM-read-bound: 16x -> gain 3.3x (was link-bound).")
