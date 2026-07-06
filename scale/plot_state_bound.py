"""Motivation figure: decode tok/s vs per-sequence state size, measured."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
colors = {"gdn": "tab:blue", "gla": "tab:orange"}
names = {"gdn": "Gated DeltaNet (Qwen3.5/Kimi/Nemotron family)",
         "gla": "GLA (channel-gate family)"}
for fam in ["gdn", "gla"]:
    try:
        d = json.load(open(f"bench_state_bound_{fam}_heads.json"))
    except FileNotFoundError:
        continue
    budgets = sorted({p["budget"] for p in d["points"]})
    styles = [(":", "^"), ("--", "s"), ("-", "o"), ("-.", "D")]
    for budget, (ls, mk) in zip(budgets, styles[-len(budgets):]):
        pts = sorted([p for p in d["points"] if p["budget"] == budget],
                     key=lambda p: p["state_gb_seq"])
        if not pts:
            continue
        xs = [p["state_gb_seq"] for p in pts]
        ax.plot(xs, [p["toks"] for p in pts], ls, marker=mk, color=colors[fam],
                label=f"{names[fam]}, {budget:.0f}GB HBM budget")
        ax2.plot(xs, [p["B"] for p in pts], ls, marker=mk, color=colors[fam])
    # 1/x reference from the largest-budget curve
    ref = sorted([p for p in d["points"] if p["budget"] == budgets[-1]],
                 key=lambda p: p["state_gb_seq"])
    if ref and fam == "gdn":
        x0, y0 = ref[0]["state_gb_seq"], ref[0]["toks"]
        xs = [p["state_gb_seq"] for p in ref]
        ax.plot(xs, [y0 * x0 / x for x in xs], ":", color="gray",
                label=r"$\propto$ 1/state (capacity law)")
for a, yl in [(ax, "decode throughput (tok/s), measured"),
              (ax2, "max batch under HBM budget")]:
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("recurrent state per sequence (GB)")
    a.set_ylabel(yl); a.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Bigger state ⇒ GPU decode throughput collapses ~1/state")
ax2.set_title("Mechanism: HBM capacity ÷ state = batch")
plt.tight_layout()
plt.savefig("state_bound_motivation.png", dpi=150)
print("saved state_bound_motivation.png")
