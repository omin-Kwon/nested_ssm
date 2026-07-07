---
name: related-work-and-novelty
description: "Novelty verdict + related work for Elastic Test-Time Memory (nested recurrent state) — what to cite, what to distinguish"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 09706fad-d204-4de5-8592-38bd3d386694
---

Related work / novelty for [[research-direction-elastic-ttm]] (from 4 parallel lit-review agents, 2026-07).

**NOVELTY VERDICT: novel, but pre-empt two neighbors.** Nesting the recurrent STATE dimension N itself (Matryoshka/ordered) so ONE continuously-updated state serves all budgets via prefixes, runtime-selectable per request, co-designed with near-memory capacity — is unpublished. No paper trains ONE model and reports a fine-grained accuracy-vs-recurrent-state curve by varying N at inference; producing that curve is itself a contribution.

**MUST distinguish (reviewers reach for these first):**
- **MatMamba** (arXiv:2410.06718, 2024) — THE threat. Matryoshka + Mamba2, but nests MODEL/CHANNEL width → each submodel is an independent narrower recurrence with its own smaller state. Does NOT nest d_state N, no recall framing, no multi-write stability analysis. Our delta: nest N itself; one shared state; confront error-compounding.
- **Nemotron Elastic** (arXiv:2511.16664, 2025) — elastic Mamba-Transformer but EXPLICITLY freezes SSM state dim N (resizes head count/channels). Confirms our knob is untouched.
- **StateX** (arXiv:2509.22630, 2025) — post-train state EXPANSION (one-time fixed larger model), opposite of runtime-elastic. Cite as feasibility evidence.
- **MatFormer** (2310.07707), **MatryoshkaKV** (2410.14731, nests stored KV cache rank — write-once, not recurrent), **MatMLA** (nests heads).

**Foundations to cite:** MRL (Kusupati, NeurIPS 2022, arXiv:2205.13147) — mechanism: sum CE loss over log-spaced prefixes, shared head uses first m cols (MRL-E); ~0% "Matryoshka tax" on WRITE-ONCE embeddings. Nested Dropout (Rippel, ICML 2014, arXiv:1402.0915) — stochastic prefix truncation, recovers PCA ordering.

**KEY RESEARCH INSIGHT:** MRL/MatFormer/MatMamba nest WRITE-ONCE objects (≈0% tax). A recurrent state is WRITTEN THOUSANDS of times → error compounding (TTT fast-weight magnitude explosion, arXiv:2505.23884). So a real "Matryoshka tax on recurrent state" likely EXISTS — measuring/fixing it is an unclaimed publishable result. This is the crux the PoC probes (additive = columns independent, easy/free nesting; delta = columns couple via S@k, hard case).

**Controller lit ("who picks width k per request"):** DS-Net learned gate (arXiv:2103.13258, per-input slimming ratio, easy/hard mining) = blueprint; CALM (arXiv:2207.07061) Learn-then-Test calibration for a provable global quality guarantee; PonderNet (2107.05407) probabilistic halting; Titans "surprise" = gradient norm as "needs more memory" signal; Mixture-of-Depths (2404.02258) = fix k options a priori for static tensor shapes (HW-friendly). Signals ranked: surprise/grad-norm > readout uncertainty (entropy/margin/hidden-saturation) > conformal set size > context length/density.

**Strongest motivation cites:** "Impossibility Triangle of Long-Context Modeling" (arXiv:2605.05066) — argues fixed state budget bounds worst-case recall; input-adaptive state width is the principled fix. Just Read Twice (2407.05483) — state 80→640 lowers ppl ~15%. BASED/Zoology recall-vs-state Pareto.

## Sparse linear attention 계열과의 구분 (2026-07-07 추가, 인용 검증됨)
- **SSE (Scaling Linear Attention with Sparse State Expansion, arXiv:2507.16577, 2025-07)**: state를 N 파티션으로 확장, write-read 게이트 + softmax top-k row-sparse 갱신. **MoM (arXiv:2502.13685, 2025-02, Shanghai AI Lab)**: 다중 독립 메모리 + 토큰 라우터.
- **구분 4축**: ① 근사 자리 — 그들은 WHAT을 버림(라우팅 탈락 슬롯은 영구 미수신, 내용의존·비가역), 우리는 WHEN만 미룸(dense-but-stale; 오차 = 최근 c토큰 cold 성분, age-국소·정확보상). ② HW — 그들 state는 전부 HBM 상주(capacity 불변, 오히려 악화; gather/scatter), 우리는 capacity 해방 + dense streaming(PNM 정합). ③ 결정 — 그들은 데이터패스에 토큰별 라우터, 우리는 controller-less(순서는 학습이 굽고 배치는 정적 절단). ④ 직교 — SSE/MoM의 확장 state도 K×V 직합이라 우리 nesting+티어링을 얹을 수 있음(경쟁 아닌 보완 포지셔닝).
- 한 줄: **"sparse-and-fresh vs dense-but-stale"** — 논문 related work의 핵심 대조 문장.
