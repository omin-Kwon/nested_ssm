---
name: state-scaling-landscape-findings
description: "Verified literature landscape — state size vs accuracy in linear-attn/TTT, and the CXL-PNM/PIM prior-work gap"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 09706fad-d204-4de5-8592-38bd3d386694
---

Deep-research findings (105 agents, 23 primary sources, 25/25 claims passed 3-vote adversarial verification) for the [[research-direction-elastic-ttm]] project.

**State size = recall accuracy (high confidence, peer-reviewed):**
- Recall–throughput tradeoff is fundamental. Smaller recurrent state → worse associative recall. BASED (Arora/Ré, ICML 2024, arxiv 2402.18668); Zoology (ICLR 2024, arxiv 2312.04927): 82% of quality gap to attention = recall; gated-conv needs model dim ≥ seq len (linear) to solve MQAR; attention solves at d=64.
- BASED: state = f(feature dim d', window), 2nd-order Taylor → d'^2 state; state is an explicit tunable knob, not tied to seq length.

**Frontier: test-time-optimized memory (larger/expressive state → accuracy):**
- Titans (Google, arxiv 2501.00663): deep MLP memory L_M≥1, update S_t=η S_{t-1}−θ∇L, M_t=(1−α)M_{t-1}+S_t (momentum=surprise, α=forgetting). Deeper L_M → better perplexity all seq lengths but slower training. Explicit: "very long context cannot be compressed in small vector/matrix state."
- Gated DeltaNet (NVIDIA, ICLR 2025, arxiv 2412.06464): S∈R^{d×d} per head (~2MB @ 32 heads d=128); gate + delta rank-1. NIAH: Mamba2 84.5% / DeltaNet 92.1% / Gated DeltaNet 98.4%.
- Miras / "It's All Connected" (Google, arxiv 2504.13173): unifies models as associative memory (arch · attentional-bias · retention gate · learning algo). Medium confidence (small scale, OpenReview).
- TTT (Sun et al., arxiv 2407.04620): hidden state IS a model (TTT-Linear/TTT-MLP), updated by test-time gradient step; dual form for HW efficiency.

**Hardware landscape + THE GAP:**
- Pimba (MICRO-58 2025, arxiv 2507.10178): PIM for POST-transformer state update (Mamba2/GLA/RetNet/HGRN2). State-update AI = 4× attention but bandwidth-bound; up to 73.8% of latency at batch 128; 4.1×/2.1× throughput. BUT: fixed state, THROUGHPUT only — not accuracy-via-scaling. Also found "swamping effect": Float8 state → ppl 8114, needs MX8.
- FPGA persistent-state (Qwen3-Next, arxiv 2503.14376): Gated DeltaNet decode ~0.87 FLOP/B; state in on-chip mem → compute-bound, 60× energy. Explicitly fixes d=128, does NOT explore accuracy-vs-state.
- NeuPIMs (2403.00579): NPU GEMM + PIM GEMV/attention split template. CXL-PNM 1M-token (2511.00321): CXL holds overflow KV. Samsung/SK Hynix stack: GEMV/attention + DLRM embeddings.
- **GAP: everyone treats state as fixed-small and optimizes efficiency; NO ONE uses PNM capacity to SCALE the state for ACCURACY, and no one touches TTT/Titans (gradient/MLP) memory in HW.**

Roofline partition: OI~1 state carry → PNM; OI tens–hundreds chunk matmul → GPU; only chunk activations cross link.
