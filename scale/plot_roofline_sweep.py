"""Figures (PDF) from vllm_sweep_breakdown.json (+ bf16 extension) — B200,
Nemotron-9B decode.

1) roofline.pdf        : B200 roofline + OI of each decode component; B=1 and
                         B=2048 endpoints annotated; bf16-state point separate.
2) sweep_latency.pdf   : stacked per-component ms/step vs batch (busy) + wall.
3) sweep_throughput.pdf: tokens/s vs batch + v4 projection + fp32 capacity wall
                         (B>~1180: 113MB/req fp32 state no longer fits 178GB).

Run: ~/nemo_env/bin/python3 plot_roofline_sweep.py   (from scale/)
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = {int(k): v for k, v in
     json.load(open("results/vllm_sweep_breakdown.json")).items()}
BS = sorted(B for B in R if R[B].get("state_dtype", "float32") == "float32")
# bf16-state points (B=2048/2304) verified the ceiling story but batch cannot
# grow further there either — excluded from the main figures by default;
# pass --with-bf16 to re-include.
BF16 = ({B: v for B, v in R.items() if v.get("state_dtype") == "bfloat16"}
        if "--with-bf16" in sys.argv else {})

# ---------------- B200 machine model + component models ----------------
BW = 8.0e12                    # HBM3e peak bytes/s
PEAK_BF16 = 2.25e15            # dense bf16 FLOP/s
L, H, P, N = 27, 128, 64, 128
PARAMS = 8.9e9
FP32_CAP_B = int((178e9 * 0.88 - 17.4e9) / (L * H * P * N * 4))   # ~1180

def flops_bytes(comp, B, sdt=4):
    if comp == "state-op (SSU)":
        e = B * L * H * P * N
        return 6 * e, 2 * e * sdt
    if comp == "GEMM (proj/MLP/lm_head)":
        return 2 * B * PARAMS, PARAMS * 2
    if comp == "conv update":
        e = B * L * (H * P + 2 * 8 * N) * 4
        return 8 * e / 4, 2 * e
    if comp == "attention":
        kv = B * 4 * 2 * 128 * 8 * 80 * 2
        return 2 * kv, kv
    if comp == "norm/elementwise":
        e = B * 56 * 4480 * 2
        return e, 4 * e
    return None

COLORS = {"state-op (SSU)": "#d62728", "GEMM (proj/MLP/lm_head)": "#1f77b4",
          "norm/elementwise": "#7f7f7f", "conv update": "#2ca02c",
          "attention": "#9467bd", "prefill scan/conv": "#bcbd22",
          "sampler/logits": "#8c564b", "other": "#e377c2"}

os.makedirs("results/plots", exist_ok=True)

# ----------------------------- 1. roofline -----------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
oi = np.logspace(-2, 4, 200)
ax.loglog(oi, np.minimum(oi * BW, PEAK_BF16), "k-", lw=2)
ax.axhline(PEAK_BF16, color="k", ls=":", lw=.8)
ax.annotate("HBM3e 8 TB/s", xy=(0.1, 0.1 * BW), rotation=33, fontsize=9,
            ha="center", va="bottom")
ax.annotate("bf16 dense 2.25 PFLOP/s", xy=(500, PEAK_BF16 * 1.15), fontsize=9)
ax.axvline(PEAK_BF16 / BW, color="gray", ls="--", lw=.6)

def series(comp, data, sdt, marker, lab):
    xs, ys, bs = [], [], []
    for B in sorted(data):
        fb = flops_bytes(comp, B, sdt)
        t = data[B]["buckets_ms"].get(comp, 0) / data[B]["gen"] / 1e3
        if fb and t > 0:
            xs.append(fb[0] / fb[1]); ys.append(fb[0] / t); bs.append(B)
    ax.plot(xs, ys, marker, ms=5, lw=1, color=COLORS[comp], label=lab)
    return xs, ys, bs

for comp in ["state-op (SSU)", "GEMM (proj/MLP/lm_head)", "conv update",
             "attention", "norm/elementwise"]:
    xs, ys, bs = series(comp, {B: R[B] for B in BS}, 4, "o-", comp)
    # endpoint annotations: B=1 and largest fp32 B
    for tgt in (1, max(bs)):
        if tgt in bs:
            i = bs.index(tgt)
            ax.annotate(f"B={tgt}", (xs[i], ys[i]), fontsize=7, fontweight="bold",
                        xytext=(5, -11 if comp != "state-op (SSU)" else 7),
                        textcoords="offset points", color=COLORS[comp])
if BF16:
    xs, ys, bs = series("state-op (SSU)", BF16, 2, "s", "SSU bf16-state")
    xs2, ys2, bs2 = series("GEMM (proj/MLP/lm_head)", BF16, 2, "s", None)
    for x, y, B in list(zip(xs, ys, bs)) + list(zip(xs2, ys2, bs2)):
        ax.annotate(f"B={B}\n(bf16 state)", (x, y), fontsize=7, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points")
ax.annotate("SSU: OI 0.75 (fp32) / 1.5 (bf16)\n— pinned to the bandwidth wall,\n"
            "no kernel tuning can move it right;\nonly less traffic (v4) helps",
            xy=(0.75, 4.5e12), fontsize=8, color=COLORS["state-op (SSU)"],
            xytext=(3.5, 6e11), arrowprops=dict(arrowstyle="->", lw=.7))
ax.set_xlabel("operational intensity (FLOP/byte)")
ax.set_ylabel("achieved FLOP/s")
ax.set_title("B200 roofline — Nemotron-9B decode components (vLLM, measured)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=.25, which="both")
fig.tight_layout(); fig.savefig("results/plots/roofline.pdf")

# ------------------------- 2. latency breakdown -------------------------
ALL = BS + sorted(BF16)
order = ["state-op (SSU)", "GEMM (proj/MLP/lm_head)", "norm/elementwise",
         "conv update", "attention", "sampler/logits", "prefill scan/conv",
         "other"]
fig, ax = plt.subplots(figsize=(8.5, 5.2))
x = np.arange(len(ALL))
bot = np.zeros(len(ALL))
for comp in order:
    v = np.array([R[B]["buckets_ms"].get(comp, 0) / R[B]["gen"] for B in ALL])
    ax.bar(x, v, .62, bottom=bot, color=COLORS[comp], label=comp,
           hatch=["" if B not in BF16 else "//" for B in ALL][0])
    bot += v
for i, B in enumerate(ALL):                      # hatch bf16 bars post-hoc
    if B in BF16:
        ax.bar(x[i], bot[i], .62, fill=False, hatch="//", edgecolor="k", lw=.4)
wall = [R[B]["ms_per_step_wall"] for B in ALL]
ax.plot(x, wall, "k^--", ms=7, lw=1.2, label="wall / step (launch gaps incl.)")
for i, B in enumerate(ALL):
    s = R[B]["buckets_ms"].get("state-op (SSU)", 0) / R[B]["gen"]
    ax.annotate(f"SSU {100*s/max(bot[i],1e-9):.0f}%", (x[i], bot[i]),
                ha="center", va="bottom", fontsize=8, color="#d62728")
ax.set_xticks(x, [f"{b}" if b not in BF16 else f"{b}\n(bf16 state)" for b in ALL])
ax.set_xlabel("batch size"); ax.set_ylabel("ms per decode step")
ax.set_title("Decode-step latency breakdown vs batch (B200, eager; "
             f"fp32 state fits only to B≈{FP32_CAP_B})")
ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig("results/plots/sweep_latency.pdf")

# --------------------------- 3. throughput ---------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
tp = [R[B]["tok_per_s"] for B in BS]
ax.plot(BS, tp, "o-", lw=2, color="#1f77b4", label="raw fp32-state (measured)")
if BF16:
    bb = sorted(BF16)
    ax.plot(bb, [BF16[B]["tok_per_s"] for B in bb], "s", ms=8, color="#17becf",
            label="raw bf16-state (measured)")
# fp8-cold projection (deployment target): analytic byte ratio, fp32-hot
# normalized units where raw state R+W = 2.0 —
#   hot R+W 0.5 + cold fp8 read 0.1875 + flush (bf16 shadow R+W + fp8 snap)/c
#   c4 0.922 -> 2.17x | c16 0.746 -> 2.68x | c32 0.717 -> 2.79x (asymptote 2.91x)
for cut, lab, c in [(2.79, "v4-c32-fp8cold (SSU/2.79x, analytic proj.)", "#8b0000"),
                    (2.68, "v4-c16-fp8cold (SSU/2.68x, analytic proj.)", "#d62728"),
                    (2.17, "v4-c4-fp8cold (SSU/2.17x, analytic proj.)", "#ff9896")]:
    tp2 = []
    for B in BS:
        ssu = R[B]["buckets_ms"].get("state-op (SSU)", 0) / R[B]["gen"]
        new_wall = R[B]["ms_per_step_wall"] - ssu * (1 - 1 / cut)
        tp2.append(B * 1e3 / new_wall)
    ax.plot(BS, tp2, "s--", lw=1.2, color=c, label=lab)
ax.axvspan(FP32_CAP_B, 4096, alpha=.12, color="red")
ax.annotate(f"fp32 state exceeds HBM\n(113MB/req -> B>{FP32_CAP_B} needs "
            "state offload:\nthe CXL-PNM capacity motivation)",
            xy=(FP32_CAP_B * 1.15, max(tp) * .25), fontsize=8, color="darkred")
ax.set_xscale("log", base=2)
ticks = BS + sorted(BF16)
ax.set_xticks(ticks, [str(b) for b in ticks], fontsize=8)
ax.set_xlim(0.8, 3200)
ax.set_xlabel("batch size"); ax.set_ylabel("decode throughput (tok/s)")
ax.set_title("Decode throughput vs batch — raw measured + v4 projection")
ax.legend(fontsize=8); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig("results/plots/sweep_throughput.pdf")
print("PLOTS DONE -> results/plots/{roofline,sweep_latency,sweep_throughput}.pdf")
