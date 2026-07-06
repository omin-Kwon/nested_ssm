"""Compare nested runs vs dedicated fixed-width baselines (delta).
Focuses on the tax cells (intermediate width under high load)."""
import re, sys, glob

def parse_final(path):
    try: blk = open(path).read().split("FINAL")[-1]
    except FileNotFoundError: return None, None
    rows = {}; widths = None
    for line in blk.splitlines():
        m = re.match(r"D\\k\s+(.*)", line)
        if m: widths = [int(x) for x in m.group(1).split()]
        m2 = re.match(r"(\d+)\s+(.*)", line.strip())
        if m2 and widths:
            v = [float(x) for x in m2.group(2).split()]
            if len(v) == len(widths): rows[int(m2.group(1))] = dict(zip(widths, v))
    return widths, rows

# dedicated fixed-width delta baselines (seed 0)
W = [2, 4, 8, 16, 32]
ded = {}
for w in W:
    _, r = parse_final(f"tax_delta_{w}.log")
    if r: ded[w] = {D: r[D][w] for D in r}      # dedicated recall at its width

def show(tag, path):
    widths, rows = parse_final(path)
    if not rows:
        print(f"{tag}: (no data)"); return
    print(f"\n### {tag}  ({path})")
    print("recall grid:")
    print("D\\k  " + " ".join(f"{w:>6d}" for w in widths))
    for D in sorted(rows):
        print(f"{D:<4d} " + " ".join(f"{rows[D][w]:6.3f}" for w in widths))
    print("TAX vs dedicated (fixed_k - nested_k), + = nesting costs:")
    print("D\\k  " + " ".join(f"{w:>7d}" for w in widths))
    for D in sorted(rows):
        print(f"{D:<4d} " + " ".join(
            f"{ded[w][D]-rows[D][w]:+7.3f}" if w in ded else "   n/a " for w in widths))

runs = [("nested seed0 (orig)", "run_delta.log"),
        ("nested seed1", "exp_seed1.log"),
        ("nested seed2", "exp_seed2.log"),
        ("nested 6000-step", "exp_long.log"),
        ("mitig loss_pow=-0.5", "exp_lw05.log"),
        ("mitig loss_pow=-1.0", "exp_lw1.log")]
print("Dedicated (fixed-width) delta recall @ own width:")
print("D\\k  " + " ".join(f"{w:>6d}" for w in W))
for D in [4,8,16,32]:
    print(f"{D:<4d} " + " ".join(f"{ded[w][D]:6.3f}" if w in ded else "  n/a " for w in W))
for tag, path in runs:
    show(tag, path)

# focused summary: the sore cells (D in {16,32}, k=8)
print("\n===== FOCUS: nested recall at k=8 (dedicated: D16=%.3f, D32=%.3f) ====="
      % (ded[8][16], ded[8][32]))
print(f"{'run':22s} {'D16,k8':>8s} {'D32,k8':>8s}")
for tag, path in runs:
    _, rows = parse_final(path)
    if rows: print(f"{tag:22s} {rows[16][8]:8.3f} {rows[32][8]:8.3f}")
