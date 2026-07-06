"""Parse FINAL recall(width,D) grids from run logs and plot the money figure."""
import re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_final(path):
    txt = open(path).read()
    # take the block after the LAST 'FINAL'
    blk = txt.split("FINAL")[-1]
    rows = {}
    widths = None
    for line in blk.splitlines():
        m = re.match(r"D\\k\s+(.*)", line)
        if m:
            widths = [int(x) for x in m.group(1).split()]
        m2 = re.match(r"(\d+)\s+(.*)", line.strip())
        if m2 and widths:
            D = int(m2.group(1))
            vals = [float(x) for x in m2.group(2).split()]
            if len(vals) == len(widths):
                rows[D] = vals
    return widths, rows

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, (mode, path) in zip(axes, [("additive", "run_additive.log"),
                                   ("delta", "run_delta.log")]):
    widths, rows = parse_final(path)
    for D in sorted(rows):
        ax.plot(widths, rows[D], marker="o", label=f"D={D}")
    ax.axhline(0.95, ls="--", c="gray", lw=.8)
    ax.set_xscale("log", base=2); ax.set_xticks(widths)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("inference state width k (nested key-dim)")
    ax.set_title(f"{mode}  (one nested model)")
    ax.grid(alpha=.3)
axes[0].set_ylabel("MQAR recall")
axes[0].legend(title="#assoc.", fontsize=9)
fig.suptitle("Elastic Test-Time Memory PoC: one model, recall dialed by state width k\n"
             "H1: monotone recall vs k   |   H2: larger D needs larger k (capacity=recall)",
             fontsize=11)
fig.tight_layout()
fig.savefig("poc_grid.png", dpi=130)
print("saved poc_grid.png")
for mode, path in [("additive","run_additive.log"),("delta","run_delta.log")]:
    widths, rows = parse_final(path)
    print(f"\n{mode}: widths={widths}")
    for D in sorted(rows):
        # min k to reach 0.95
        kk = next((widths[i] for i,v in enumerate(rows[D]) if v>=0.95), None)
        print(f"  D={D:2d} -> k*(0.95)={kk}")
