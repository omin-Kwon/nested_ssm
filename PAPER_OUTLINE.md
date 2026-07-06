# Paper Skeleton — Elastic Test-Time Memory: Importance-Ordered State Placement with Differential Staleness for GPU+CXL-PNM Serving

*(working title; 모든 수치는 본 레포의 실험 로그에서 — 출처는 각 절에 표기)*

## Abstract (요지)
Linear-attention/GDN 계열의 recall은 recurrent state 크기에 지배되지만(Zoology d≥N), state는 학습 시 고정되고 GPU에서 3중 벌(HBM 용량 경쟁·OI≈1 대역폭 벽·실리콘 낭비)을 받는다. 우리는 (i) state 차원을 nested로 학습해 importance ordering을 만들고, (ii) 그 순서대로 hot(GPU, fresh)/cold(CXL-PNM 상주, chunk-exact 갱신 + stale readout)로 배치하는 **v4** 실행 의미론을 제안한다. 판정 실험으로 "staleness의 금기는 correction이지 readout이 아님"(age-지문으로 인과 분리)과 "nesting = hot 티어 유효성의 by-construction 보장"을 확립하고, 시뮬레이션으로 iso-state-size **~3.8×** throughput(또는 동일 throughput에 8× state)을 보인다. Qwen3.5가 채택한 GDN이 3/4 레이어를 차지하는 현 프론티어에 직접 적용 가능.

## 1. Introduction
- 문제: 고정 state = recall 상한 고정; 요청별 필요 memory 이질적; 선행 해법은 worst-case static provision.
- 기회: CXL-PNM = capacity-dense/저-compute — 큰 state의 "비싼 방향"과 정확히 반대 프로파일.
- 기여 4개: ① nested-N ordering(런타임 폭 dial + hot 티어 보장) ② v4 실행 의미론(correction-exact chunk replay + readout-stale + hot recency buffer) ③ age×D 지문 방법론(staleness 실패 모드의 인과 분리) ④ 시스템 정량(3.8×, 경계 지도, c=8 설계점).
- 시의성: Qwen3.5-397B(2026-02)·Kimi Linear가 GDN 계열 채택 — 우리가 티어링하는 state가 플래그십 안에 실재.

## 2. Background & Motivation
- §PROBLEM_SETTING 1–2 그대로: recall∝state 법칙, OI≈1 상수(GDN 디코드 0.87 FLOP/B 실측 인용), Pimba 73.8%, GPU 3자 충돌, 선행 4갈래(크게/hybrid/Titans/RAG)와 공통 비효율.
- Landscape 표: MatMamba(width nest)/Nemotron(N 고정)/StateX(1회 확장)/Pimba(고정 state throughput) — 전부 우리 축 비점유.

## 3. Method
### 3.1 Nested-N training
- 폭 메뉴 loss(또는 per-sample 폭 — A4에서 4× 빠른 등가 레시피 확립). in-kernel L2 norm과 zero-마스킹의 결합으로 절단 의미론이 깨끗함(구현 §gdn_a4).
- 결과 성질: 폭 dial(H1/H2), hot 자족성(보장 vs 우연: nested 0.98 vs dedicated 0.44 @ 실GDN k16).
### 3.2 v4 execution semantics
- 수식 + 데이터플로우(토큰당 교환 1회; staleness = window 연장; readout GEMV = PNM 바닥).
- 왜 이 형태인가의 판정 역사: blockdelta 반증(결합=용량 메커니즘) → "값싼 곳(타이밍)에서 굽힘".
### 3.3 Placement
- 2-티어(hot=GPU/cold=PNM), 티어=staleness 스케줄의 하드웨어 실현체. per-channel-gate 계열(KDA/GDN-2)에선 게이트가 배치를 "제안"(τ 정적, rank-corr 0.999) — nesting이 "보장".

## 4. Algorithm-Level Verdicts (전부 age×D 지문 방법론으로)
| 판정 | 결과 | 출처 |
|---|---|---|
| correction 금기 | (a) age-증가형 복리 붕괴(D-스케일) vs (c1) step-at-c | poc imqar 6k |
| v4 지배 | toy: D32c16 0.864 vs 0.416/0.634; **실GDN: c32 0.95(young 0.89, old 1.00), c8 비용 2%** | poc + scale A4 |
| hot 티어 인과 | nested-v4 young 0.88 vs dedic 0.60 (실GDN) | A4 |
| multi-head | 중복성(1-head stale 무해 0.99) + dedicated만 head 이질(0.893) — nested가 균질화 | A4 Q1 |
| 무중단 신축 (P4) | grow 16→64 0.98 / shrink 0.95, 재계산 0 | E4 |
| 실모델 게이트 지문 | τ 분포 72/15/6/7%, 정적(rank 0.999); "τ≫c" 규칙; slow-stale ppl +4% vs fast +194% | A2/A3 |
| **실LM 경고와 구조** | pretrained GLA(hot 티어 없음): 老recall 붕괴 / **A4(nested hot): 불발현(old 1.00)**; 언어-학습 최종 판정 = T3 (진행 중) | A3v2/A4/T3 |
| retrofit | toy: k8 90% of scratch, staleness 프로파일 보존(E8/E8v2). **실GDN(=Qwen3.5 케이스): 회전-only FT 50초로 from-scratch와 전 폭 동일**(k8 0.863 vs 0.871) — toy의 작은 폭 격차 소멸 | E8 + T4c |

## 5. System Evaluation
- Analytic(가설층): 4.0→3.8×(read-floor 정정 반영 — readout은 c로 amortize 불가, floor ~1.05TB/s/dev), 경계 지도(링크 62% fp16, prefill 희석 f=0.3→2.1×, 정확도 노브 상한 16×에서 PNM-read-bound).
- **판정층(시뮬):** 3.82× 생존(c≥8 stall 0%, p99 28.6ms); c=4/n4 = 6.6% stall → **설계점 (c=8, hot=128, 8×, n=4)**; read-floor 절벽 실증(≤1.2TB/s → 12.5% stall).
- 정직 캐비앗: 커널 실측 아님, 단일 노드, E2E 희석은 별도 지표.

## 6. Related Work
- lit-review 메모리의 인용 전부: Zoology/BASED, TTT/Titans/Miras, Gated DeltaNet(+GDN-2, KDA), MatMamba/Nemotron/StateX/MatryoshkaKV/MRL/nested dropout, Pimba/NeuPIMs/CXL-PNM 1M-token/삼성·하이닉스 스택, slimmable/OFA/DS-Net/CALM.

## 7. Limitations & Future
- toy→~55M 언어모델까지만(대규모 LM 미검증); 정확도 축의 스케일 외삽; controller(elastic dial) future work; per-channel-gate 계열의 positional 얽힘; 다중 노드.

## 8. 부록 후보
- 판정 사이클 로그(blockdelta 반증, 선등록 수정 이력 — 방법론 투명성), 경계 지도 전체, E4/E8 상세.

## 수치 구멍 — 전부 채워짐 ✅
- [x] **언어-학습 nested GDN (T3, 클로징 결과):** 폭 탄력성 17.56→16.26(nested) vs 절단 붕괴 83→16(dedicated); **arms ppl: nested-v4 +2~4%(≈공짜) vs nested-c1 ~10× vs dedicated-v4 +120%** → "v4 티어 실행은 nesting이 있어야만 공짜"가 실언어에서 성립. A3v2 경고 사슬 해소(GLA무장비→붕괴 / dedic hot만→2.2× / nested 둘다→+2%). needle은 35M 능력 바닥으로 측정 불가(한계 명시, ≥340M future).
- [x] A4 seed 재현: v4-c32 = 0.95/0.95/0.96 (3-seed)
- [x] **일반성(T7):** GLA 계열 — 탄력성 tax +4.9%, **v4 비용 0%**(additive엔 correction 금기 자체가 없음 — 이론 정합), c1 붕괴 9~15× 재현
- [x] **★ 4가족 일반성 완성(T8, Kimi/Nemotron 가족):** **KDA**(채널decay+delta) tax **+1.6%**(전 가족 최저), v4 +1.9~3.4%, c1 8.7~11.7×, dedic-v4 +92~115% / **M2**(Mamba2 SSD, 스칼라+additive) tax +7.1%, **v4 +0.05~0.4%(문자 그대로 공짜)**, **c1 49~64×(최대 붕괴)**, dedic-v4 +161~198%. 검증 게이트 4/4 정확 일치. → **클레임: "티어링 실행권은 nesting이 산다"는 gate 구조(스칼라/채널) × 갱신 규칙(delta/additive) 2×2 전부에서 성립** — 현존 프로덕션 선형 계열 전체 커버(GDN=Qwen, GLA, KDA=Kimi, M2=Nemotron/Falcon-H/Granite)
- [x] **tax-vs-예산(T7):** 언어 nesting tax는 ~4%로 **정체**(2.05B tok에도 불소멸 — toy와 달리 H3-소멸 없음). 클레임 교정: "~4% 안정 비용으로 탄력성+v4-실행권을 산다". **규모(120M, fineweb 1.5B tok): +5.0%로 유지** — 규모로도 안 줄지만 안 커짐(안정 비용 클레임 확정)
- [x] **120M 스케일 재현(T7 마감):** arms 패턴 재현·증폭 — nested-v4 +0.8~4.2% vs nested-c1 7-8.6× vs **dedic-v4 +276~369%**(35M의 +120%보다 악화; dedic-a-c16=nan). **needle 계측 생존(120M)**: nested fresh 0.28(young 1.00) / hot-alone nested 0.12 vs dedic 0.00 / **nested-v4 needle 0.25-0.31 ≈ fresh(老age 무손상)** / dedic 전 arms 0.00 — A3v2 경고 사슬을 살아있는 needle로 종결(32 probes 캐비앗)
- [x] **Motivation 실측 그림(§2용, `scale/state_bound_motivation.png`):** fla decode 커널 실측 — 고정 HBM 예산에서 tok/s ∝ 1/state (slope −1, GDN·GLA 곡선 중첩, 24GB: 22.4k→89 tok/s @ state 25MB→805MB, batch=1 capacity wall에서 법칙보다 더 추락), 커널 BW 2.4-2.6TB/s 고정(memory-bound 증거). 80GB(H100) 점은 analytic(M1 47.7k→6.8k)과 결합
- [x] 실GDN retrofit: 회전-only 50초 FT = from-scratch 동등 (k8 0.863 vs 0.871)

## 4.5. 결과 서사의 척추 (본문 Figure 순서 제안)
1. Fig1 개념도(hot/cold, v4 데이터플로우) → 2. 폭 탄력성(toy grid + LM ppl 곡선) → 3. age×D 지문(correction vs readout) → 4. v4 지배(toy+A4+LM ppl arms 3단) → 5. ordering 인과(hot-alone + v4-young + LM arms의 nested/dedic 대비) → 6. retrofit(before/after) → 7. §8 경계 지도 + 시뮬 → 8. E4 신축.
