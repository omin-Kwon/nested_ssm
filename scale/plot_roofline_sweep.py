"""Figures (PDF) from vllm_sweep_breakdown.json — B200, Nemotron-9B decode.

1) roofline.pdf        : B200 roofline + operational intensity of each decode
                         component; achieved points from measured times.
2) sweep_latency.pdf   : stacked per-component ms/step vs batch (busy) + wall.
3) sweep_throughput.pdf: tokens/s vs batch + SSU share annotation + v4 projection.

Run: ~/nemo_env/bin/python3 plot_roofline_sweep.py   (from scale/)
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = {int(k): v for k, v in
     json.load(open("results/vllm_sweep_breakdown.json")).items()}
BS = sorted(R)

# ---------------- B200 machine model + component models ----------------
BW = 8.0e12                    # HBM3e peak bytes/s
PEAK_BF16 = 2.25e15            # dense bf16 FLOP/s
L, H, P, N, DT = 27, 128, 64, 128, 4          # mamba stack, fp32 state
STATE = L * H * P * N * DT                     # bytes per request, all layers
PARAMS = 8.9e9

def flops_bytes(comp, B):
    """Analytic (FLOPs, bytes) per decode step."""
    if comp == "state-op (SSU)":
        e = B * L * H * P * N                  # state elements touched
        return 6 * e, 2 * e * DT               # ~6 flop/elem; R+W fp32
    if comp == "GEMM (proj/MLP/lm_head)":
        return 2 * B * PARAMS, PARAMS * 2      # weights bf16 read (act. negligible)
    if comp == "conv update":
        e = B * L * (H * P + 2 * 8 * N) * 4    # conv_dim window ops
        return 8 * e / 4, 2 * e
    if comp == "attention":
        kv = B * 4 * 2 * 128 * 8 * 80 * 2      # 4 layers, short ctx (~128) kv bf16
        return 2 * kv, kv
    if comp == "norm/elementwise":
        e = B * 56 * 4480 * 2                  # rough activations touched
        return e, 4 * e
    return None

COLORS = {"state-op (SSU)": "#d62728", "GEMM (proj/MLP/lm_head)": "#1f77b4",
          "norm/elementwise": "#7f7f7f", "conv update": "#2ca02c",
          "attention": "#9467bd", "prefill scan/conv": "#bcbd22",
          "sampler/logits": "#8c564b", "other": "#e377c2"}

# ----------------------------- 1. roofline -----------------------------
fig, ax = plt.subplots(figsize=(7, 5))
oi = np.logspace(-2, 4, 200)
ax.loglog(oi, np.minimum(oi * BW, PEAK_BF16), "k-", lw=2)
ax.axhline(PEAK_BF16, color="k", ls=":", lw=.8)
ax.annotate("HBM3e 8 TB/s", xy=(0.1, 0.1 * BW), rotation=33, fontsize=9,
            ha="center", va="bottom")
ax.annotate("bf16 dense 2.25 PFLOP/s", xy=(600, PEAK_BF16 * 1.15), fontsize=9)
ax.axvline(PEAK_BF16 / BW, color="gray", ls="--", lw=.6)

for comp in ["state-op (SSU)", "GEMM (proj/MLP/lm_head)", "conv update",
             "attention", "norm/elementwise"]:
    xs, ys = [], []
    for B in BS:
        fb = flops_bytes(comp, B)
        t = R[B]["buckets_ms"].get(comp, 0) / R[B]["gen"] / 1e3   # s/step
        if fb and t > 0:
            xs.append(fb[0] / fb[1])                # OI (flop/byte)
            ys.append(fb[0] / t)                    # achieved FLOP/s
    ax.plot(xs, ys, "o-", ms=5, lw=1, color=COLORS[comp], label=comp)
    if comp == "GEMM (proj/MLP/lm_head)":
        for B, x, y in zip(BS, xs, ys):
            if B in (1, 256):
                ax.annotate(f"B={B}", (x, y), fontsize=7,
                            xytext=(4, -10), textcoords="offset points")
ax.annotate("SSU: OI=0.75 — bandwidth-wall pinned\n(all B overlap)",
            xy=(0.75, 6 * 0.75e12), fontsize=8, color=COLORS["state-op (SSU)"],
            xytext=(2.2, 1.2e12),
            arrowprops=dict(arrowstyle="->", lw=.7))
ax.set_xlabel("operational intensity (FLOP/byte)")
ax.set_ylabel("achieved FLOP/s")
ax.set_title("B200 roofline — Nemotron-9B decode components (vLLM, fp32 state)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=.25, which="both")
fig.tight_layout(); fig.savefig("results/plots/roofline.pdf")

# ------------------------- 2. latency breakdown -------------------------
order = ["state-op (SSU)", "GEMM (proj/MLP/lm_head)", "norm/elementwise",
         "conv update", "attention", "sampler/logits", "prefill scan/conv",
         "other"]
fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(BS))
bot = np.zeros(len(BS))
for comp in order:
    v = np.array([R[B]["buckets_ms"].get(comp, 0) / R[B]["gen"] for B in BS])
    ax.bar(x, v, .6, bottom=bot, color=COLORS[comp], label=comp)
    bot += v
wall = [R[B]["ms_per_step_wall"] for B in BS]
ax.plot(x, wall, "k^--", ms=7, lw=1.2, label="wall / step (launch gaps incl.)")
for i, B in enumerate(BS):
    s = R[B]["buckets_ms"].get("state-op (SSU)", 0) / R[B]["gen"]
    ax.annotate(f"SSU {100*s/max(bot[i],1e-9):.0f}%", (x[i], bot[i]),
                ha="center", va="bottom", fontsize=8, color="#d62728")
ax.set_xticks(x, [str(b) for b in BS])
ax.set_xlabel("batch size"); ax.set_ylabel("ms per decode step")
ax.set_title("Decode-step latency breakdown vs batch (B200, eager, fp32 state)")
ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig("results/plots/sweep_latency.pdf")

# --------------------------- 3. throughput ---------------------------
fig, ax = plt.subplots(figsize=(7, 4.6))
tp = [R[B]["tok_per_s"] for B in BS]
ax.plot(BS, tp, "o-", lw=2, color="#1f77b4", label="raw (measured)")
# v4 projection: SSU busy cut by measured fused-bench factors, wall shifted
for cut, lab, c in [(2.14, "v4-c16-bf16 (SSU/2.14x, projected)", "#d62728"),
                    (1.62, "v4-c4-bf16 (SSU/1.62x, projected)", "#ff9896")]:
    tp2 = []
    for B in BS:
        ssu = R[B]["buckets_ms"].get("state-op (SSU)", 0) / R[B]["gen"]
        new_wall = R[B]["ms_per_step_wall"] - ssu * (1 - 1 / cut)
        tp2.append(B * 1e3 / new_wall)
    ax.plot(BS, tp2, "s--", lw=1.2, color=c, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks(BS, [str(b) for b in BS])
ax.set_xlabel("batch size"); ax.set_ylabel("decode throughput (tok/s)")
ax.set_title("Decode throughput vs batch — raw measured + v4 projection")
ax.legend(fontsize=8); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig("results/plots/sweep_throughput.pdf")
print("PLOTS DONE -> results/plots/{roofline,sweep_latency,sweep_throughput}.pdf")
