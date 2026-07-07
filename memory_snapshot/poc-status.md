---
name: poc-status
description: "Nested-state MQAR PoC — setup, bugs fixed, and results confirming H1/H2 for Elastic Test-Time Memory"
metadata: 
  node_type: memory
  type: project
  originSessionId: 09706fad-d204-4de5-8592-38bd3d386694
---

PoC for [[research-direction-elastic-ttm]]. Code in /home/omin/nested_ssm/poc/ (nested_delta_mqar.py; plot_grid.py -> poc_grid.png). Env: 4× B200, torch 2.10, pure-PyTorch (no fla). gdn_out.txt = Gated DeltaNet paper.

**What it tests:** one gated-linear/delta recurrent LM trained with nested-dropout on the state KEY dim (Matryoshka), on Zoology-style MQAR, mixed #associations D per step. Eval = recall(width k, D) grid. State S ∈ R^{Dv×Dk}; nest = use first k key-dims at inference.

**Bugs found & fixed (important):** initial delta-only recurrence with NO positional emb / NO short conv got stuck at ~0.20 recall on easy D=8. Root cause: (1) MQAR queries are key tokens *reappearing* → delta rule OVERWRITES the stored value with junk at query positions; (2) no way to tell write-region from read-region. Fix: added absolute positional embedding + causal depthwise short conv (GDN uses it) + default recurrence = gated ADDITIVE linear attention (no overwrite). After fix, baseline solves MQAR D=8 at recall 1.000.

**RESULTS (3000 steps, head_dim=32, 1 head, 2 layers, D∈{4,8,16,32}, k∈{2,4,8,16,32}):**
- H1 CONFIRMED — one model, recall rises monotone & graceful with width k. e.g. additive D=8: k2..k32 = 0.31/0.69/0.95/1.00/1.00. The runtime dial exists.
- H2 CONFIRMED — min k for 0.95 recall scales ~linearly with D (additive D4→8, D8→16, D16→16, D32→~32; delta D32→32 hits 0.962). Reproduces Zoology "d≥N" law ON ONE MODEL by varying k at inference — a curve no published paper has shown.
- Delta (coupled dims via S@k) nests fine, even slightly BETTER at high load (D32,k32: delta 0.962 vs additive 0.936). No catastrophic recurrent-nesting failure at this scale.
- H3 (Matryoshka tax = fixed_k − nested_k) — KEY FINDING confirming the novelty-check prediction:
  - ADDITIVE: tax ≈ 0 or NEGATIVE (nesting free / even beneficial; columns independent → prefix is a valid standalone memory + multi-width loss regularizes).
  - DELTA: real POSITIVE tax concentrated at intermediate width k=8 under high load (D16,k8 +0.169; D32,k8 +0.217). Delta couples dims via S@k, so a truncated prefix is NOT a clean memory — a dedicated width-8 delta (D16=0.922) beats nested width-8 (0.753). Nesting destroys delta's "store more in fewer dims" advantage at middle widths. At k=2/4 and k=32 tax ≈ 0.
  - => the "recurrent-nesting tax" is REAL and measurable specifically for the expressive (delta/Gated-DeltaNet) rule we care about. Mitigating it (regularization / curriculum / dimension orthogonalization / stop-grad between prefix levels) is the algorithmic contribution. Single seed — needs seed replication.

**H3 follow-up (seed replication + mitigations, 2026-07):** Focus cell = nested recall at k=8 (dedicated delta: D16=0.922, D32=0.663).
- Tax ROBUST across seeds 0/1/2: D16,k8 = 0.75/0.79/0.77 (vs 0.922). Not noise.
- Tax is MOSTLY an OPTIMIZATION-BUDGET artifact, not fundamental: nested delta @ 6000 steps (2x) nearly closes it — D16,k8=0.910 (vs 0.922), D32,k8=0.666 (vs 0.663). Full 6000-step TAX grid ~0/negative everywhere. => coupled delta nests essentially FREE with enough training; the joint multi-width objective just converges slower than single-width.
- Simple loss-reweighting (weight ∝ width^p, p<0 to emphasize small widths) FAILS: p=-0.5 worse; p=-1.0 helps k8 slightly but destroys large widths (D32,k32: 0.96->0.67). Zero-sum, not a real fix.
- STRUCTURAL fix (hierarchical/residual nested delta, mode=nesteddelta) — NEGATIVE RESULT: it did NOT help; it made the tax WORSE. mean tax over {4,8,16}xD = +0.041 (nesteddelta) vs +0.020 (naive delta) at 3000 steps; k=8 D16=0.705/D32=0.397 (nesteddelta) vs 0.753/0.446 (naive). Hypothesis (residual hierarchy -> leading-block independence -> lower tax) was WRONG in practice: the residual-per-level structure lengthens gradient paths / starves upper levels, hurting optimization. This particular formulation backfired (could be the specific design, not the whole idea, but no win at this scale).
- BOTTOM LINE: the recurrent-nesting tax is an OPTIMIZATION-BUDGET artifact, not fundamental. Winning recipe is embarrassingly simple: naive nested delta + adequate training (6000 steps closes tax to ~0). No exotic recurrence needed. Simpler = stronger/more deployable claim: "nesting the recurrent state dim N is essentially free with standard training." The nesteddelta complexity was solving a non-problem.

**E6 3k PILOT findings (staleness, 2026-07-06):**
- (b) CONTROL = resounding causal win for the anchor: nested cold-stale c=4 D=16 recall 0.981 vs dedicated-32 same-config 0.040 (instant death at c>=2). Nested training concentrates function in the fresh leading dims → tail becomes stale-tolerable. Ordering CAUSES staleness tolerance.
- Staleness budget ∝ capacity HEADROOM: nested (a) D=8 gentle to c=16 (0.962), D=16 breaks at c≥8, D=32 (at capacity) breaks at c≥4 (0.589). Quantitative relation worth a figure.
- Mechanism discovery: in the SEPARATED MQAR layout (all writes before all queries), the (a)-degradation is from WRITE-path correction staleness (stale cold r_c corrupts delta's interference cancellation under load), NOT read recency — queries see everything in the snapshot by query time.
- **BENCHMARK FLAW CAUGHT: separated MQAR cannot test (c1)/read-recency** — honest-c1 (exact corr, stale readout) passes trivially there (layout artifact, must NOT be read as "Config B suffices"). Fix: make_imqar (interleaved writes/queries → young ages exist). Interleaved 6k verdict runs launched (exp_e6v2_imqar_*_6k.log) alongside separated 6k (exp_e6v2_*_6k.log; those remain valid for write-path staleness + (b) causality).
- Parked v4 idea: PNM-side exact chunk replay for cold updates (GPU ships r_hot per token, +d_v link) could fix write-path staleness at capacity — evaluate only if E6v2 shows it's needed.

**Caveats:** tiny scale (synthetic MQAR, ~2-layer, 1 head, head_dim 32); nesting is on ONE layer's per-head key-dim; recurrence run as O(L) python loop (fine at this size). Not yet: real LM/perplexity, TTT-MLP/Titans memory, GPU/PNM roofline, controller for choosing k.
