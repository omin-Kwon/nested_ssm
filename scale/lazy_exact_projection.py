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
    ("GDN-2 research scale (1.3B dense)", "delta(WY)", 0.90, 2/114,
     "dense small -> state dominates"),
]

print(f"{'model':47s} {'exact cut':>9} {'e2e exact':>9}")
for name, fam, s, b, note in MODELS:
    cut_x = tier(1.0, b)
    print(f"{name:47s} {cut_x:8.2f}x {1/(1-s+s/cut_x):8.2f}x   <- {note}")
print("c* = sqrt(4/b) ~", round(math.sqrt(4 / (1/114))), "(additive),",
      round(math.sqrt(4 / (2/114))), "(delta/WY)")

# ---- MoE hybrids: s is a FUNCTION of batch (expert coverage saturates) ----
# s(B) = 2B*state / (2B*state + W_moe*f(B) + W_dense);  f = 1-(1-k/E)^B
# Kimi-Linear-48B-A3B assumed: 21 KDA layers x 2MB/req state, MoE 90GB bf16,
# 384 experts top-8, dense/attn ~6GB.  (VERIFY configs before publishing.)
print("\nKimi-Linear-48B-A3B — exact e2e vs batch (delta/WY cut "
      f"{tier(1.0, 2/114):.2f}x):")
st, Wm, Wd, k, E = 42e6, 90e9, 6e9, 8, 384
for B in (128, 256, 512, 1024, 2048, 4096):
    f = 1 - (1 - k / E) ** B
    s = 2 * B * st / (2 * B * st + Wm * f + Wd)
    cut = tier(1.0, 2/114)
    s8 = 2 * B * st / (2 * B * st + (Wm * f + Wd) / 2)   # fp8 WEIGHTS serving
    print(f"  B={B:5d}: s={s:4.0%} -> e2e {1/(1-s+s/cut):.2f}x   "
          f"(fp8-weight serving: s={s8:4.0%} -> {1/(1-s8+s8/cut):.2f}x)")
