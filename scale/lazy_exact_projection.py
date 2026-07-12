"""Analytic e2e projection of EXACT lazy state materialization across models.

Per-token state-op bytes (units: raw state R+W = 2.0):
  T(c) = r + b*c/2 + 2/c,  b = buffer/state ratio per token (~1/N)
  c* = sqrt(4/b) -> T* = r + 2*sqrt(b)
  exact tier: r=1.0 (fp32 read)   licensed tier: r=0.25 (fp8 read)
Delta families (GDN/KDA/GDN-2) need WY bookkeeping: buffer ~2x -> b2 = 2b.

e2e = 1 / (1 - s + s/cut), s = state-op share of decode step at the stated
batch (measured for Nemotron; computed from bytes for others).

Assumption-laden for non-Nemotron models — verify configs before publishing.
Run: python3 lazy_exact_projection.py
"""
import math

def tier(r, b):
    return 2.0 / (r + 2 * math.sqrt(b))

MODELS = [
    # name, family, s (state-op share at saturating B), b(buffer/state), note
    ("Nemotron-Nano-9B (dense hybrid 27M:4A)", "additive", 0.58, 1/114,
     "s MEASURED @B>=256 (57.6-60%)"),
    ("Falcon-H1/Granite-style dense mamba2 hybrid", "additive", 0.55, 1/114,
     "s assumed ~ Nemotron (same family/shape)"),
    ("Mamba-3 class pure SSM (~1-3B dense)", "additive", 0.95, 1/114,
     "weights tiny vs state at B=256 -> s~95%"),
    ("Qwen3-Next-80B-A3B (GDN 36:12, MoE)", "delta(WY)", 0.30, 2/114,
     "MoE expert streaming dominates weights at large B -> s~30% (rough)"),
    ("Kimi-Linear-48B-A3B (KDA 20:7, MoE)", "delta(WY)", 0.30, 2/114,
     "as above"),
    ("GDN-2 research scale (1.3B dense)", "delta(WY)", 0.90, 2/114,
     "dense small -> state dominates"),
]

print(f"{'model':47s} {'exact cut':>9} {'e2e exact':>9} {'fp8 cut':>8} {'e2e fp8':>8}")
for name, fam, s, b, note in MODELS:
    cut_x, cut_8 = tier(1.0, b), tier(0.25, b)
    e2e = lambda cut: 1 / (1 - s + s / cut)
    print(f"{name:47s} {cut_x:8.2f}x {e2e(cut_x):8.2f}x {cut_8:7.2f}x "
          f"{e2e(cut_8):7.2f}x   <- {note}")
print("\nc* = sqrt(4/b) ~", round(math.sqrt(4 / (1/114))), "(additive),",
      round(math.sqrt(4 / (2/114))), "(delta/WY)")
