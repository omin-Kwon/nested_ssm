# Elastic Test-Time Memory — Problem Setting

*알고리즘–HW Co-Design for a GPU + CXL-PNM hybrid platform*
(SNU ARClab · working draft)

---

## 1. Background

### 1.1 Linear-attention / SSM / TTT: 고정 크기 recurrent state
Softmax attention은 KV cache가 시퀀스 길이에 비례해 자라 O(n²) 비용이 든다. Linear attention·SSM·TTT 계열은 이를 **상수 크기 행렬 state**로 압축한 선형 RNN으로 재구성한다. 공통 갱신식:

$$S_t = \mathrm{diag}(\alpha_t)\,S_{t-1} + k_t v_t^\top,\qquad y_t = S_t^\top q_t$$

state $S\in\mathbb{R}^{d_k\times d_v}$는 시퀀스 길이와 무관하게 고정. 이 갱신은 **rank-1 outer-product + matvec** — 산술강도(OI)가 낮은 memory-bound 연산이다.

### 1.2 핵심 법칙: state 크기 = recall 용량
Zoology(Arora, Ré et al., ICLR 2024, arXiv:2312.04927)와 BASED(ICML 2024, arXiv:2402.18668)가 규명한 **근본적 tradeoff**:

- attention 대비 품질 격차의 **82%가 associative recall**에서 온다.
- gated-conv/recurrent 모델이 MQAR을 풀려면 **model dimension이 저장할 KV 쌍 수에 최소 선형(d ≥ N)**으로 커져야 한다. attention은 d=64 상수로 해결.
- recall 정확도는 **recurrent state 크기에 단조 비례**(BASED: state 1MB에서 ~100%, 65KB에서 ~50%).

즉 고정 state = **recall 상한이 고정** = linear attention의 근본 약점.

### 1.3 프론티어: test-time에 최적화되는 큰 memory
최근 계열은 정확히 "state를 더 크고 표현력 있게"로 간다:
- **TTT**(Sun et al., arXiv:2407.04620): state가 *모델 그 자체*, test-time gradient step으로 갱신.
- **Titans**(Behrouz et al., Google, arXiv:2501.00663): 깊은 MLP 장기 memory, momentum(surprise)+forgetting. **메모리 깊이↑ → 모든 시퀀스 길이에서 perplexity↑**. 명시: *"매우 긴 컨텍스트는 작은 state에 압축될 수 없다."*
- **Gated DeltaNet**(Yang/NVIDIA, ICLR 2025, arXiv:2412.06464): gating+delta rank-1. NIAH recall 98.4% (Mamba2 84.5%).

**결론: 이 계열은 본질적으로 큰 test-time state를 원하지만 HW(HBM 용량·compute)에 발목이 잡혀 있다.**

---

## 2. Motivation

### 2.1 하드웨어 비대칭과 roofline
- **GPU**: compute-dense, HBM 용량 부족. balance point ≈ 295 FLOP/B (H100).
- **CXL-PNM**: capacity-dense, 저-compute, 뱅크 근처 고대역. 저-OI memory-bound 연산에 이상적.

recurrent state 갱신의 **OI ≈ 1 FLOP/byte이고 state 크기에 무관하게 상수**(실측: Gated DeltaNet 디코드 0.87 FLOP/B). GPU에서 돌리면 텐서코어의 **~99.7%가 유휴**(Pimba: batch 128에서 state update가 latency의 73.8%). 그리고 state를 키워도 OI가 상수 → **"정확도 노브(큰 state) = PNM 노브"** 가 동일하다.

파티션: chunk 내부 matmul(OI 수십~수백) → GPU, inter-chunk state carry(OI≈1) → PNM. 큰 state는 PNM 상주, 링크는 chunk 활성값만 건넌다.

### 2.2 두 개의 뚫린 갭
1. **알고리즘 갭 — capacity가 학습 시 고정.** state 차원은 학습된 projection 가중치에 묶여 추론 때 못 바꾼다. 큰 state의 정확도 이득을 runtime에 dial할 방법이 없다.
2. **하드웨어 갭 — 아무도 정확도를 위해 state를 키우지 않는다.** Pimba(arXiv:2507.10178)·FPGA persistent-state(arXiv:2503.14376)는 *고정* state를 throughput용으로 가속할 뿐, 용량을 정확도로 전환하지 않는다. FPGA 논문은 "d=128 고정, 정확도-state trade는 미탐구"라 명시.

### 2.3 원리적 근거
"Impossibility Triangle of Long-Context Modeling"(arXiv:2605.05066): **고정 state 예산은 worst-case recall을 bound**하며, 그 탈출구는 **입력에 따라 state를 신축**시키는 것(dense 컨텍스트엔 크게, easy엔 작게)이라고 주장. 이것이 우리 아이디어의 원리적 정당화다.

### 2.4 우리 위치 — 공유된 문제, 미개척 해법
"입력적응적 state/memory가 필요하다"는 **motivation은 학계가 공유**한다(Impossibility Triangle, Titans/TTT의 test-time 적응, ACT/PonderNet/MoD/CALM의 adaptive computation). 이는 오히려 우리에게 유리하다 — *공인된 실재 문제*이기 때문. 그러나 남들이 적응시키는 대상은 다르다: Titans는 *고정 용량의 내용*, ACT/MoD는 *compute 깊이/토큰*, MatMamba/slimmable은 *모델 width*. **recurrent state 용량 N 자체를 요청별 runtime에 신축시키는 것은 아무도 하지 않는다.** 따라서 차별화는 "우리가 문제를 처음 봤다"가 아니라 **(a) 메커니즘(N을 nest) + (b) 하드웨어 co-design**에 실린다. 특히 "capacity-dense near-memory가 큰 state 끝단을 담게 하려고 state를 신축한다"는 프레이밍은 우리 고유다.

### 2.5 GPU-only의 3자 충돌 → CXL-PNM의 해소
우리 알고리즘은 "필요할 때 state를 크게"가 정확도의 원천이라 **runtime에 큰 state 수요를 스스로 생성**한다. GPU 단독에선 세 가지가 충돌한다:
1. **용량 벽:** state 총량 = B × layers × H × d_v × **d_k**. 정확도 노브(d_k↑)가 이를 곱으로 키워 HBM에서 모델 가중치와 경쟁 → 큰 state와 큰 배치를 동시에 못 함(정확도 vs throughput).
2. **낭비된 실리콘:** state update는 OI≈1(크기 무관 상수) → GPU(balance ~295 FLOP/B)에서 peak의 <1%. state를 키우면 더 bandwidth-bound → **정확도 노브가 워크로드를 GPU 최악 영역으로 밀어넣음**(Pimba: batch128 디코드 latency의 73.8%가 state update).
3. **탄력성이 독:** 요청별 가변 k = 가변 shape = ragged 배치·단편화 → GPU 배칭/커널의 천적.

**CXL-PNM이 각각 해소:** (1) capacity-dense 상주로 큰 state×큰 배치 동시 수용(cold 차원은 싼 용량, hot만 HBM); (2) 뱅크 근처에서 저-OI 갱신을 roofline 근처로 수행하고 **state는 상주해 토큰마다 링크를 안 건넘**(chunk 활성값만 이동 = 데이터 이동 최소화 충족); (3) ragged state가 PNM에선 그냥 주소 가능한 용량이라 배칭 문제 우회. **탄력 k가 곧 hot(GPU)/cold(PNM) 티어 경계**가 된다 — nested-N 알고리즘이 state를 두 하드웨어로 깨끗이 가르는 바로 그 축을 제공.

---

## 3. Core Idea — Importance-Ordered State Placement with Differential Staleness

*(v3 프레임 전환 — 상세·근거는 `CORE_ALGORITHM.md` §9. 이전 elastic-k 프레임은 controller 닭-달걀·배치 동기화 문제로 placement-first로 개정; elastic dial은 future work로 강등.)*

> **recurrent state의 차원을 nested(Matryoshka)로 학습해 importance ordering을 만들고, 그 ordering대로 state를 memory hierarchy에 배치한다 — hot 앞 차원은 GPU HBM에 fresh로, cold 뒤 차원은 CXL-PNM에 상주시켜 near-data로 갱신하되 c-step stale로 둔다. 모든 요청이 같은 경로를 탄다(controller 불필요).**

**Novelty anchor (E6 결과와 무관하게 성립):** ① ordering이 staleness의 **차등 적용**을 원리적으로 정당화한다(uniform chunking은 전 차원 균일 지연 — fresh hot 필요 + stale cold 충분의 구조가 없음). ② c-step staleness는 손해가 아니라 **PNM 효율 노브**다 — c개 rank-1 갱신을 chunked matmul 1회로 묶어 low-compute PNM의 OI를 높이고 링크 메시지를 1/c로 amortize. ③ KV offload와 다름 — importance-정렬 배치 + near-data **계산**(단순 저장 아님). 서사: hot = working memory(fresh) / cold = consolidated memory(lazy).

### 3.1 메커니즘
- **학습:** state의 key 차원에 nested-dropout(MRL, arXiv:2205.13147 방식) — 매 step 폭 k를 여러 granularity에서 취해 loss를 합산. 모델이 중요 정보를 **앞 차원에 몰아 담도록** 학습.
- **추론:** 요청/구간별로 폭 k 선택. k 작음 → 좁은 state, GPU 상주, 빠름, 낮은 recall. k 큼 → 넓은 state, PNM 확장, 높은 recall. **재학습 불필요.**
- **Controller("누가 k를 정하나"):** difficulty 신호로 k 결정 — Titans surprise(gradient norm), readout uncertainty(entropy/margin), 또는 컨텍스트 길이/밀도. 학습형 gate(DS-Net, arXiv:2103.13258) + 보정(CALM Learn-then-Test, arXiv:2207.07061)으로 전역 품질 보장.

### 3.2 Novelty (선점 방어)
| 인접 연구 | 무엇을 하나 | 우리와의 차이 |
|---|---|---|
| **MatMamba** (2410.06718) | 채널/블록 **width**를 nest → 독립적 좁은 recurrence들 | 우리는 **state 차원 N 자체**를 nest, 하나의 공유 state |
| **Nemotron Elastic** (2511.16664) | elastic Mamba지만 **N 고정** | 우리는 정확히 그 N을 신축 |
| **StateX** (2509.22630) | state를 1회 **확장**(고정) | 우리는 runtime 신축 |
| **Pimba** (2507.10178) | *고정* state throughput 가속 | 우리는 정확도 위해 state 신축 + PNM 상주 |

**핵심 미해결 = "recurrent-nesting tax".** MRL/MatMamba가 nest하는 것은 **write-once** 객체라 tax ≈ 0. 그러나 recurrent state는 **수천 번 덮어써지므로**(delta rule은 차원이 S@k로 결합) prefix가 유효한 memory로 남는지가 비자명하다. 이 tax를 측정·완화하는 것 자체가 미청구 기여다.

---

## 4. Hypotheses & PoC

작은 규모 MQAR(Zoology식)에서 nested gated-linear/delta state를 학습·검증:
- **H1 (runtime dial 존재):** 추론 폭 k↓ 시 recall이 **완만·단조** 감소.
- **H2 (capacity 노브):** 연관 수 D↑ 시 동일 recall에 **더 큰 k** 필요(state 크기 = recall 용량 재현).
- **H3 (Matryoshka tax):** nested 모델의 폭-k 성능이 폭-k 전용 학습 모델에 근접(또는 tax를 정량화 — recurrent nesting의 난이도 규명).

*(결과는 §6에 갱신)*

---

## 5. 기여 (예정)
1. recurrent state 차원 $N$을 nest해 **runtime-elastic한 test-time memory**를 만드는 최초의 알고리즘 — 한 모델에서 추론 시 $N$만 바꿔 정확도-메모리를 다이얼.
2. 단일 모델에서 **recall-vs-state-size 곡선**을 추론 시 $N$만 바꿔 측정(문헌 미존재)하고, **nesting이 표준 학습만으로 사실상 공짜**임을 규명(아래 §6).
3. GPU(dense compute)–CXL-PNM(대용량·저-OI state) **operator/용량 co-design**과 roofline·serving 이득 정량화(§7–8).

*(초기 가설이던 "recurrent-nesting tax 완화용 새 recurrence"는 §6에서 tax가 근본 문제가 아님이 밝혀져 기여에서 내림.)*

---

## 6. PoC Results

**설정:** 순수 PyTorch. Zoology식 MQAR, 혼합-D. gated-linear/delta recurrent LM(2-layer, 1-head, head_dim=32)을 state의 key 차원에 nested-dropout 걸어 학습(폭 k∈{2,4,8,16,32} loss 합산). 추론 시 폭 k별 `recall(k, D)` 측정. 3000 steps.

**초기 실패 → 원인 규명:** positional emb·short conv 없는 delta-only는 D=8에서 recall 0.20 정체. 원인 = MQAR 쿼리가 key 토큰의 *재등장*이라 delta rule이 쿼리 위치에서 value를 junk로 **overwrite**, 게다가 쓰기/읽기 구간 구분 불가. 수정(pos emb + causal short conv + overwrite 없는 additive) 후 baseline이 MQAR D=8 recall **1.000**.

**결과 (그림: `poc/poc_grid.png`):**

*ADDITIVE — recall(k, D):*
```
D\k    2      4      8     16     32
4    0.585  0.913  0.992  1.000  1.000
8    0.307  0.689  0.946  0.997  0.999
16   0.178  0.399  0.772  0.973  0.993
32   0.108  0.212  0.463  0.816  0.936
```
*DELTA — recall(k, D):*
```
D\k    2      4      8     16     32
4    0.594  0.908  0.991  1.000  1.000
8    0.347  0.710  0.938  0.999  1.000
16   0.191  0.432  0.753  0.985  0.997
32   0.114  0.224  0.446  0.853  0.962
```

- **H1 ✅** 한 모델에서 recall이 폭 k에 **단조·완만 증가** — runtime dial이 실재.
- **H2 ✅** 0.95 recall에 필요한 최소 k가 **D에 ~선형**(D4→8, D8→16, D16→16, D32→~32). Zoology "d≥N" 법칙을 **단일 모델·추론시 N 변경**으로 재현(문헌 미존재 곡선).
- **관찰:** 차원이 S@k로 결합되는 **delta도 깨지지 않고** nest되며 큰 D에서 오히려 약간 우세(D32,k32: 0.962 vs 0.936). 예상한 파국적 recurrent-nesting tax는 이 규모에선 없음.
- **H3 (Matryoshka tax) — 핵심 발견:** 폭별 *전용* 모델과 비교(`TAX = fixed_k − nested_k`).
  - **Additive: tax ≈ 0 또는 음수(이득).** 열(column)이 독립이라 prefix가 그대로 유효한 작은 memory → nesting이 공짜, 오히려 여러 폭 loss가 정칙화로 작용.
  - **Delta: 실제 양의 tax가 중간 폭(k=8)·고부하(D=16,32)에 집중**(D16,k8 **+0.169**; D32,k8 **+0.217**). delta는 `S@k`로 차원이 결합돼 prefix가 독립 memory가 아님 → 전용 width-8 delta(D16=0.922)가 nested width-8(0.753)를 크게 앞섬. **nesting이 delta의 "적은 차원으로 효율 저장"이라는 강점을 중간 폭에서 훼손.**
  - ➡️ delta에서 중간 폭 tax가 실재. **하지만 아래 후속 실험에서 근본 문제가 아님이 판명.**

**H3 후속 (seed 재현 · 학습량 · 완화 실험):** 기준 셀 = k=8 nested recall (전용 delta: D16=0.922, D32=0.663).
- **tax는 seed에 견고**(seed 0/1/2: D16,k8 = 0.75/0.79/0.77) — 노이즈 아님.
- **그러나 대부분 "학습량 부족" 아티팩트:** nested delta **6000스텝**이면 전용을 거의 따라잡음(D16,k8=0.910, D32,k8=0.666, 전체 tax 그리드 ≈0). → **결합된 delta도 충분히 학습하면 nesting이 사실상 공짜.** joint 다중폭 목적함수가 단일폭보다 느리게 수렴할 뿐.
- **단순 loss 가중은 해법 아님:** 작은 폭 강조(pow<0)는 큰 폭을 죽이는 zero-sum(D32,k32 0.96→0.67).
- **우리가 새로 설계한 hierarchical/residual nested delta(mode=nesteddelta) = NEGATIVE RESULT:** tax를 줄이긴커녕 키움(평균 tax +0.041 vs naive +0.020; k8 D16=0.705/D32=0.397 < naive 0.753/0.446). 가설(residual 계층→독립성→tax↓)이 실전에서 틀림 — residual 구조가 최적화를 방해. **이 형태·규모에선 이득 없음 → 폐기.**

**➡️ 핵심 결론:** recurrent-nesting tax는 **근본 문제가 아니라 최적화 예산 문제**. 이기는 레시피는 단순함 — **naive nested delta + 충분한 학습.** exotic recurrence 불필요 → *"recurrent 상태 차원 $N$의 nesting은 표준 학습만으로 사실상 공짜"* 라는 더 강하고 배포 친화적인 주장.

**한계:** 소규모 synthetic·소모델. 실제 LM perplexity·TTT-MLP/Titans memory·GPU/PNM roofline·k 선택 controller·**예상 이득 정량화(§8)**는 미착수(다음 단계).

---

## 7. Hardware Mapping (알고리즘 → GPU + CXL-PNM)

*(정밀도(int8 등) 같은 저수준 HW 디테일은 이 단계에서 불필요 — hot/cold 모두 fp16 가정. 지금은 operator/용량 배치 수준.)*

**연산을 산술강도(OI)로 가른다:**
- **GPU (compute-dense, 高-OI):** 입력 투영 $W_q x,W_k x,W_v x$, 게이트, short conv, FFN, output proj, LayerNorm — dense GEMM. + **hot(앞 $k_{hot}$) state의 recurrence**(빠른 경로).
- **CXL-PNM (capacity-dense, 低-OI):** **cold(뒤) state의 저장 + 갱신 + 읽기**. 갱신 $S{=}\alpha S+\beta(v-\alpha Sk)k^\top$ 와 읽기 $y{=}Sq$ 는 rank-1/matvec(OI≈1)이라 near-memory에 적합.

**데이터 흐름 (per token):**
```
GPU:  x_t → q,k,v (투영·conv)  ──(cold용 k/v/q 슬라이스, O(d) 벡터)──▶ PNM
GPU:  hot recurrence → y_hot                                        PNM: cold state 갱신 → y_cold
GPU:  y = y_hot + y_cold ◀──────(y_cold, O(d_v) 벡터)────────────── PNM
GPU:  y → FFN → 다음 토큰
```
- **큰 cold state는 PNM에 상주** — 토큰마다 링크를 안 건넘, $O(d)$ 벡터만 왕복(**데이터 이동 최소화** 충족).
- **다이얼 $k$ = hot/cold 경계 = GPU/PNM 경계.** 쉬운 요청($k\le k_{hot}$)은 전부 GPU; 어려운 요청의 확장분만 PNM.
- **오프로딩되는 용량:** $B \times L_{layer} \times H \times d_v \times d_k^{cold}$ — 배치·시퀀스가 커질수록 HBM을 넘치는 부분이 PNM으로.

## 8-a. Projected Gains — 첫 정량 결과 (analytic roofline v1, `poc/roofline.py`)

**⚠️ 층 표기 (감독 규율, 알고리즘 판정과 동일한 2층):** 아래 4.0×는 **analytic 상한(가설층)** — "구조적으로 4× 이득의 여지가 있다"이지 실측이 아님. 링크 latency·PNM 큐잉·chunk 동기화·커널 오버헤드 미포함, correction-exact replay가 c-토큰 내 완료된다는 가정. **판정층 = latency·큐잉 포함 serving 시뮬 (TODO).** 정확도 축(8× state의 이득)은 toy-H2 외삽 — §8과 규모 확대가 상호 인질: 속도 분자는 roofline이, 정확도 분모는 규모 실험이 검증해야 함.

7B-급 GDN-hybrid(24 GDN층+8 SWA층, H16, d_v128) decode 서빙. H100-급 GPU(80GB, 3.35TB/s) + 4× CXL-PNM(각 512GB, 뱅크근처 1.6TB/s, 링크 32GB/s). state/요청: dk=128 → 12.6MB, dk=1024 → 100.7MB(hot 12.6 + cold 88.1).

| config | B* | tok/s | 병목 |
|---|---|---|---|
| A. GPU-only 작은 state(dk128, fresh) | 1024 | 46,276 | GPU-BW |
| B. GPU-only 큰 state(dk1024, fresh) | 256 | 11,569 | GPU-BW(용량이 B 제한) |
| C. GPU-only 큰 state + v4 의미론 c=16 | 256 | 26,925 | GPU-BW |
| **D. GPU+PNM v4, c=4~64** | **1024** | **46,276** | GPU-BW |

**헤드라인:** ① **D = B 대비 4.0× (iso-state-size throughput)** — 그리고 **A와 동일 throughput에 8× state** = "작은-state 속도로 큰-state 정확도". ② **c 노브의 정량 확인:** c=1이면 PNM-BW 병목(3.1×), **c≥4부터 cold state가 사실상 공짜**(GPU-BW 천장 도달) — 알고리즘 판정에서 c=4의 recall 비용이 ~4.6%(D32: 0.942/0.988)였으므로 **c=4가 스윗스팟.** ③ C(staleness만, 용량 해방 없음) = 1.8~2.5× → **분해: staleness 기여 ~2×, 용량 해방 기여 나머지.** ④ 용량 headroom: B=1024에서 cold 총 90GB ≪ 2TB — 정확도 노브를 훨씬 더 돌릴 여지(궁극 한계는 링크).

**정직한 한계:** analytic(커널 오버헤드·prefill·다중노드 미포함), 파라미터는 class-대표값, **dk=1024의 정확도 이득은 toy H2 곡선의 외삽** — 실제 LM 검증이 규모 확대 항목. B* 는 2의 거듭제곱 탐색.

## 8-c. 이득 경계 지도 (roofline v2, `poc/roofline2.py` — 목적: 4× 정밀화가 아니라 "언제 무너지나")

*(세밀 배치 탐색으로 기준값 4.0×→3.8× — analytic 추정치가 "~4×"이지 정밀 숫자가 아님을 재확인.)*

**⚠️ 정정 (read-traffic 누락 수정):** 초판은 PNM 트래픽을 갱신($2S/c$)만 계상 — **cold readout $S^\top q_t$는 매 토큰 snapshot 전체를 읽으며 c로 amortize 불가**(stale state + fresh query는 가능하지만 그 역은 다른 질문에 답하는 것). PNM per-token 트래픽 = $S_{cold}(1 + 2/c)$. **c 노브가 amortize하는 것 = 갱신 + 연산 패턴(rank-1→chunked matmul) + 링크 메시지이며, readout 대역폭은 환원 불가한 바닥.**

| 경계 | 임계값 (정정 후) | 현재 margin | 판정 |
|---|---|---|---|
| **PNM 내부 BW** | **readout 바닥 ~1.05TB/s/dev**(B≈1200, c-무관) + 갱신: c=1 계 3.15(**병목**), **c=4 계 1.57**, c=8 1.31, c=16 1.18 | **c=4: 1.02× = 한계선**, c=8: 1.2×, c=16: 1.36×, n_pnm=8이면 2× | ⚠️ **readout이 진짜 PNM 사이징 제약** |
| **CXL 링크** | ≥20GB/s/dev (fp16) | 1.6× (이용률 62%) | ⚠️ fp8 활성값로 2× 완화 권장 |
| RTT 노출 | <0.5ms | 398× (레이어 창 795µs vs 2µs) | ✅ latency 비문제 |
| **정확도 노브 상한** | **16×부터 PNM-read가 병목**(gain 3.3×로 하락; 정정 전 link-bound 5.8×는 오류) | 8×는 GPU-BW flat 영역 내부 | ✅ 유효 범위 ≤8×(n=4) |
| 모델 크기 | 3B/7B/34B: 3.6/3.8/3.1× 유지 | — | ✅ |
| **Prefill 희석** | decode-only 3.8× → f=0.3: **2.1×**, f=0.5: 1.6× | — | ⚠️ decode-heavy 스코프 명시 |

**설계점 인증 (정정판):** (c=4, n_pnm=4)는 margin 1.02×로 **비-robust** → **개정 과녁: (c=8, hot=128, 8×, n_pnm=4)** [margin 1.2×, recall 비용 D32 0.916 vs fresh 0.988] **또는 (c=4, n_pnm=8)** [margin 2×]. 취약 지점 3개 = **PNM readout 바닥, 링크(fp16), prefill 비중** — 전부 선제 명시.

**규모 확대의 핵심 단일 질문 (사전 확정):** *toy에서 성립한 v4의 correction-exact/readout-stale 분리가 실제 GDN의 multi-head + output gate에서도 성립하는가* — head별 state가 독립이라 correction 금기가 head마다 다르게 나타날 수 있음(진짜 리스크 지점). perplexity/NIAH는 그 다음.

## 8-d. §8 판정층 결과 (serving 시뮬 v1, `poc/serving_sim.py`)

토큰-레벨 시뮬(replay 백로그 큐 + boundary stall 규칙, 레이어별 RTT 노출, lognormal jitter, prefill duty):
- **판정: ~3.8×가 큐잉·latency 하에서 생존.** c≥8: **3.82×, stall 0%**, p99 28.6ms (baseline B: 12.5k tok/s, 33ms).
- **c=4 + n_pnm=4: replay 백로그로 6.6% stall** (throughput은 3.81× 유지 — backlog drain이 쌈). analytic이 못 보던 큐잉 효과 확인 → **판정층 설계점도 c=8 지지.**
- **read-floor 절벽 실증:** PNM BW 0.8/1.0/1.2 TB/s/dev → **stall 12.5%**(매 boundary), gain 2.33/2.91/3.50×; 1.6부터 clean. analytic read-floor(~1.05TB/s)와 정합.
- 정직한 주석: (i) prefill은 양쪽 config에 동일 duty로 모델링 → **decode-공유 throughput 관점**에선 gain 3.82× 유지, **E2E per-request 관점**의 희석(f=0.3→~2.1×)은 §8-c의 analytic이 여전히 유효한 별도 지표. (ii) per-step 서비스 시간은 여전히 analytic 유도(커널 실측 아님); 다중 노드·비동기 per-request 스케줄링 미포함.

## 8-b. Projected Gains — 원래 평가 계획 (참고)

무엇을, 무엇 대비, 어떻게 보일지:

**정확도(accuracy):**
- 지표: recall(MQAR/NIAH), 장문 perplexity.
- 곡선: 단일 nested 모델의 **recall/ppl vs 다이얼 $k$** (PoC의 그림을 실제 LM 규모로).
- 비교: (a) 고정 작은 state 모델(빠르지만 부정확) (b) 고정 큰 state 모델(정확하지만 GPU에 안 맞음) — nested가 **한 모델로 두 지점을 다 커버**함을 보임.

**Throughput / Capacity (co-design 이득):**
- 모델: analytic roofline + serving 시뮬. 파라미터 — GPU HBM/BW, CXL-PNM 용량/BW/링크 BW, per-op OI.
- 핵심 비교 대상:
  1. **GPU-only, 고정 큰 state**: 큰 state가 HBM·배치와 경쟁 → 낮은 배치/throughput.
  2. **GPU-only, 고정 작은 state**: 높은 throughput, 낮은 정확도.
  3. **GPU+CXL-PNM + nested (우리)**: cold state를 PNM에 두어 **큰 유효 state를 유지하면서 배치↑** → 정확도 손실 없이 throughput↑.
- 예상 이득 출처(정성): (i) HBM 절약 → 배치 확대(state가 배치와 안 싸움), (ii) 저-OI state update를 PNM이 roofline 근처로(GPU는 <1%), (iii) 쉬운 요청 다수를 작은 $k$로 GPU 고속 처리.
- 참고 수치(문헌): Pimba가 고정 state PIM으로 **4.1×/2.1× throughput**(GPU/GPU+PIM 대비) — 우리는 여기에 **정확도-위한-용량 확장 + 요청별 elasticity**를 더함. 우리 목표 이득은 이 serving 모델로 **정량 추정**해야 함(TODO).

**요청 분포 민감도:** 이득은 "쉬운/어려운 요청 비율"에 의존 — easy 다수면 GPU 고속경로 이득 큼, hard 다수면 PNM 용량 이득 큼. 분포별 곡선을 그린다.
