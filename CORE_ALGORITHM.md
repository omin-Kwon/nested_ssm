# Core Algorithm — Nested Block-Delta Memory (설계 v1)

*Elastic Test-Time Memory의 core algorithm 설계. 담당: Fable 5 세션 (Opus 4.8 세션의 PoC/문헌 위에서).*
*읽기 전제: `PROBLEM_SETTING.md` §1–3, §7.*

---

## 0. 설계 원리 (한 문장)

> **알고리즘의 결합(coupling) 구조가 하드웨어의 통신 구조를 미러링한다** — 통신이 공짜인 티어 내부에는 dense delta 결합, 통신이 비싼 GPU↔PNM 경계에는 결합 zero.

## 1. 기존(naive nested delta)의 세 가지 구멍

1. **결합이 분할을 깬다:** delta 보정항 $S k$가 활성 폭 전체 합 → write 계산에 hot(GPU)·cold(PNM) 양쪽 readout이 모두 필요 → **토큰당 3회 직렬 링크 왕복**. §7의 "활성값만 건넌다"가 성립 안 함.
2. **전역 L2 norm이 폭 간 일관성을 깬다:** full-dim normalize 후 슬라이스 → 같은 앞 차원의 스케일이 선택 폭에 의존 → 폭마다 hot dynamics가 다른 모델이 되어 폭들이 학습에서 서로 싸움(중간-폭 tax의 구조적 원인 후보; 6000스텝 수렴 비용의 정체).
3. **GPU ragged 배칭:** 요청별 $k$가 GPU 커널 shape을 바꿈 → 배칭 파괴(§2.5(3) 미해결).

*(참고: 폐기된 hierarchical/residual nesteddelta는 residual을 블록 체인으로 전달 — 블록 간 "직렬" 결합이라 최적화·하드웨어 양쪽에 불리했음. 아래 설계는 병렬-합 구조로 정반대.)*

## 2. 알고리즘 정의

### 상태 배치
head마다 state를 nested 블록으로 분할:
$$S = [\underbrace{S_{hot}}_{\text{GPU, } w_0} \,|\, \underbrace{S_{c_1} | S_{c_2} | \cdots}_{\text{CXL-PNM, } w_1, w_2, \dots}], \qquad S_j \in \mathbb{R}^{d_v \times w_j}$$
폭 메뉴 = 블록 경계의 누적값 $\{w_0,\ w_0{+}w_1,\ \dots\}$ (MoD 교훈: 고정 메뉴 → static shape).

### 갱신 (per token, 활성 블록 $j$: 경계 $\le m$)
$$k^{(j)} = \mathrm{norm}(k[\,s_j{:}e_j\,]),\quad q^{(j)} = \mathrm{norm}(q[\,s_j{:}e_j\,]) \qquad \text{(블록별 L2 norm)}$$
$$S_j \leftarrow \alpha_t\, S_j + \beta_t^{(j)}\big(v_t - \alpha_t S_j k_t^{(j)}\big)\,k_t^{(j)\top} \qquad \text{(블록별 self-correcting delta)}$$
$$y_t = \sum_{j \le m} S_j\, q_t^{(j)}$$
- 모든 블록이 **같은 $v_t$** 를 받음(residual 체인 아님 — 병렬 합).
- $\beta^{(j)}$: 블록별 write 게이트(학습) — cold 블록을 "overflow"로 쓰는 법을 모델이 배움.
- $\alpha$: head 공유 decay (블록별 확장 가능).
- **티어 내부는 full delta 허용:** hot superblock 내부(GPU 로컬), 각 cold 블록 내부(PNM 로컬)는 완전 결합 delta. 결합을 끊는 곳은 **오직 티어/블록 경계**.

### 구조가 자동으로 주는 성질
| 성질 | 이유 | 시스템 효과 |
|---|---|---|
| **P1 통신 최소** | 블록 간 의존 zero | 링크 왕복 1회/token/layer, hot∥cold 완전 병렬, 직렬화 없음 |
| **P2 prefix-일관성 (exact)** | 블록 궤적이 폭 $m$·타 블록과 무관 | 폭들이 학습에서 안 싸움 → tax 구조적 제거 (레이어 단위; multi-layer는 하위 레이어 입력이 폭에 의존하므로 end-to-end는 근사) |
| **P3 GPU shape 불변** | hot 연산이 $k$와 무관 | 요청별 $k$가 달라도 GPU 배칭 완벽, raggedness는 PNM 주소 공간에만 |
| **P4 무중단 신축** | 새 블록 0-init 유효, truncate한 prefix 유효 | mid-stream grow/shrink에 재계산 zero |
| **P5 controller 신호 공짜** | write-residual $\|v - \alpha S k\|$ = memory-pressure (½‖v−Sk‖² gradient = Titans surprise와 동형) | hot residual 지속↑ → $k$ 키움. 추가 연산 0 |

### 정직한 tradeoff
- **표현력:** 경계 간 간섭-소거 불가 → additive ⊂ **block-delta** ⊂ full delta. PoC에서 additive ≈ delta였으므로 소규모 리스크 낮음; LM 규모에서 재검증 필요.
- **학습 비용:** 폭 메뉴 합산 loss는 ×|menu| forward. 완화: sandwich rule(최소+최대+임의 1) 또는 단일 레이어 한정 "1-forward 다폭 readout"(P2 덕분에 가능; multi-layer는 불가).

## 3. HW 매핑 (per layer)

**Decode (per token):**
```
GPU : x → q,k,v,α,β (proj/conv)         PNM : (요청별 활성 cold 블록만)
GPU : hot superblock full-delta → y_hot  PNM : 블록별 delta 갱신 → y_cold, residual-norm
      └─(k_cold,q_cold,v,α,β_cold ↓ ≈3·d_v 스칼라)┘   └─(y_cold ↑ d_v)─┘
GPU : y = y_hot + y_cold → out proj → FFN
```
- 왕복 1회·비직렬(overlap 가능). cold state는 PNM 상주, 링크는 O(d) 벡터만.
- **Prefill:** chunk 활성값(c×d)을 PNM에 보내고 PNM이 블록별 chunked(WY) 갱신을 로컬 수행 — 각 블록이 표준 delta memory이므로 기존 chunked form 그대로 상속.

**Controller:** $k$는 요청 시작 시(컨텍스트 길이/유형) + chunk 경계에서 hot-block residual-norm 이동평균으로 grow. 보정은 CALM-식 threshold calibration(추후).

## 4. 검증 계획 (PoC 하네스 그대로)

- **E1 (tax 붕괴):** `mode=blockdelta` nested @3000 vs 기존 naive delta @3000 vs 전용 baseline. **선등록 예측:** k8 셀 (D16, D32) = naive (0.753, 0.446) → 전용 (0.922, 0.663) 쪽으로 유의미 접근; k32는 naive(0.962) 대비 −0.02 이내.
- **E2 (prefix-일관성 unit test):** 무작위 가중치로 폭 8 vs 32 forward → 선행 블록 readout 동일성 수치 확인 (구조적 성질이므로 학습 불필요).
- **E3 (controller 신호):** 블록별 residual-norm이 D(기억 부하)와 상관하는지.
- **E4 (mid-stream 신축):** 시퀀스 중간에 k 8→32 성장 시 recall이 두 고정폭 사이에 놓이는지.

## 5. 상태
- v1(blockdelta): 구현·검증 완료 → **E1 반증, 폐기** (§6).
- **v2(pipedelta): 현행 설계** (§7). E5 검증 실행.

## 6. v1 결과 — E1 반증 (폐기)

- E2(prefix-일관성)는 **bit-exact 통과** — 구조는 의도대로 동작.
- **E1 실패:** blockdelta@3000 recall이 naive coupled@3000보다 전면 악화 — k8: D16 **0.321**(naive 0.753, 전용 0.922), D32 **0.167**(naive 0.446); k32,D32 **0.879**(naive 0.962); 평균 tax **+0.268** (naive +0.020의 13배). 선등록 예측 전부 미달.
- **교훈 (이 문서의 핵심 판단 근거):** delta의 **전 폭에 걸친 결합($Sk$)은 사치가 아니라 용량 메커니즘**이다 — 전체 활성 폭 간섭-소거가 있어야 $D>w$ 연관을 눌러 담는다. 블록 분할은 (i) 블록별 용량 상한 + (ii) 같은 $v$의 중복 저장(분업 부재)으로 이중 손해. **"경계에서 결합 끊기" = 하드웨어-완벽·알고리즘-재앙.** (주의: 실제 규모(블록 폭 64~128)에서는 손해가 줄 수 있으나 v2가 이 희망에 의존하지 않도록 설계.)

## 7. v2 — Pipelined Coupled Nested Delta (현행)

**전제(검증된 사실):** naive coupled nested delta는 충분한 학습으로 tax≈0 (§PROBLEM_SETTING 6). 즉 **학습 알고리즘은 이미 옳다.** 남은 것은 시스템 구멍 ①(직렬화)·③(ragged)이며, 결합을 끊지 않고 푼다:

- **구멍 ③ → 메뉴 설계:** 폭 메뉴 최소값 = $k_{hot}$. GPU는 모든 요청에서 항상 폭 $k_{hot}$ 연산 → shape 균일, 가변성은 PNM에만.
- **구멍 ① → readout 타이밍 하나만 변경:** delta write는 원래 pre-update state($S_{t-1}k_t$)를 쓰므로 PNM은 $k_t$ 수신 즉시 자기 몫 $r_c$를 **정확히** 계산 가능(staleness 없음). 직렬화의 유일한 원인은 cold readout이 post-update라는 것 → **cold readout만 pre-update로**:
$$y_t = \underbrace{S^{hot}_t q^{hot}_t}_{\text{post-update}} + \underbrace{\alpha_t\, S^{cold}_{t-1} q^{cold}_t}_{\text{pre-update}}$$
  의미 변화 = cold 차원에서만 현재 토큰의 write가 다음 토큰부터 가시(1-token visibility delay). write 자체·hot 경로·결합은 전부 exact.

**Decode 데이터플로우 (토큰당 교환 1회, 완전 파이프라인):**
```
GPU→PNM : (k_t, q_t 슬라이스, w_{t-1}, gates)   ← w는 한 토큰 늦게 도착, PNM이 즉시 적용
PNM     : r_c = S_cold@k_cold (exact),  y_c = α·S_cold@q_cold (pre-update)
PNM→GPU : (r_c, y_c)  [+ residual-norm(controller 신호)]
GPU     : w_t = β(v_t − α(S_hot k_hot + r_c)),  hot 갱신·readout,  y = y_hot + y_c
```

**원리 수정판:** ~~결합은 티어 안에서만~~ → **"알고리즘은 값싼 곳(readout 타이밍)에서 굽히고, 하중을 받는 곳(cross-dim 결합)에서는 굽히지 않는다."**

**유지되는 성질:** P3(메뉴로 확보), P5(residual = controller 신호), P4(grow는 0-init로 유효; shrink는 nested 학습 덕에 근사 유효). P2(exact prefix-일관성)는 포기 — 6k 결과가 "결합돼도 nested 학습으로 충분"을 이미 보임.

**E5 (선등록):** pipedelta@3000 그리드가 naive delta@3000 대비 전 셀 **±0.03 이내** (k≤8은 정의상 동일 경로; 실검증은 k16/k32 열).

## 8. 결과 (v2)
- E5(pipedelta@3k, cold 1-step stale readout): step2500 기준 k16/k32 열이 naive@3k와 동등~우세(D32,k32: 0.972 vs 0.962) → **c=1 staleness는 공짜.** 최종 수치는 §10.

## 9. v3 — Placement-first 프레임 전환 + Novelty Anchor (확정)

**프레임 전환:** elastic-k(controller가 요청별 폭 판정) → **placement-first**: 모든 요청이 항상 hot+cold 동일 경로. state를 memory hierarchy에 맞춰 분할 — hot(앞 차원)=GPU HBM·fresh, cold(뒤 차원)=CXL-PNM 상주·near-data 갱신·**c-step stale**. controller의 닭-달걀·배치 동기화 문제 소멸. elastic dial은 future work로 강등(퇴화 규칙: 짧은 컨텍스트는 cold 생략 — 닭-달걀 없음).

**NOVELTY ANCHOR (고정 — E6 결과와 무관하게 성립):**
> **Importance-ordered differential staleness.** nested 학습이 만든 importance ordering이 "어느 차원을 얼마나 stale하게 둬도 되는지"를 원리적으로 지정한다(단조 staleness 스케줄, 하드웨어 실현체는 티어: hot c=0, cold c=chunk). 그리고 c-step staleness는 손해가 아니라 **PNM 효율 노브**다 — c개 rank-1 갱신을 chunked(WY) matmul 1회로 묶어 low-compute PNM의 OI를 끌어올리고 링크 메시지를 1/c로 amortize.

- ❌ anchor로 쓰지 말 것: "stale-aware training으로 학습된 내성" — E6가 공짜로 나오면 무너짐(E6 결과에 인질). E7(stale-aware training)은 **조건부 강화책**: E6 절벽 시에만 투입.
- 절벽 분기에서도 placement는 생존(placement는 X=기여작음에 의존, staleness만 Y=완만변화에 의존) — 죽는 건 chunked amortization 마진뿐이고 E7이 되사올 수 있음.
- "그냥 chunking" 방어: uniform chunking은 전 차원 균일 지연; 우리는 ordering이 **차등** 적용을 정당화(fresh hot 필요 + stale cold 충분 — E6 (c)/(a)가 실증). "그냥 KV offload" 방어: importance-정렬 배치 + near-data recurrence 갱신(저장 아님 계산).
- 서사: hot=working memory(fresh), cold=consolidated memory(lazy) — 시간적 분업(최근은 hot+conv, 과거는 cold).

**E6 (staleness 내성, 대조군 포함 — anchor의 인과 입증):** chunk-refresh 의미론(c 경계마다 snapshot 공개, decay-보정), c ∈ {1,2,4,8,16}. **감독 지침 2개 반영:**
1. **판정은 수렴 스텝(6000)에서만** — E5 c=1 한 점·3k 중간치는 파일럿(H3 교훈: 학습예산이 결론을 뒤집은 전례). 3k vs 6k 곡선 비교 자체가 내성의 예산-민감도 검정.
2. **(c)의 절벽 원인 분리** — 원안 (c2: conv까지 stale)는 conv가 k/v에도 걸려 있어 **write 정렬 파괴라는 교란** 발생(read-recency 원인 분리 실패) → 대신 **age-resolved recall**(query-타깃 나이 구간별 정확도)로 원인을 직접 판독: conv 커버리지는 역학상 age ≤ 4(window)에 한정되므로, (c1)의 손실이 age ∈ (4, c]에 집중되면 "conv는 window만, 그 너머는 fresh hot state 필요"가 한 실험에서 입증됨.

실험군:
- (a) nested, cold(>8)만 stale, hot fresh → 본안 곡선 (전 age 평평 예측)
- (b) non-nested(전용 폭32), 같은 차원 stale → ordering 인과
- **(c1) nested, 전 state stale + conv fresh + exact-correction** = **honest Config B**(PNM이 chunk 경계에서 정확한 순차 replay, readout만 stale — 전 state를 PNM에 두는 실존 경쟁 배포안의 최강 형태. correction까지 stale하게 두면 과소평가라 corr_fresh로 공정화)

**판돈 명시:** honest-(c1)이 수렴 스텝·큰 c까지 살아남으면 anchor의 '필요성' 다리가 부러짐(uniform stale + fresh conv로 충분 → ordering 장식화, "conv-보정 chunked decode"로 표류). 선등록 예측: conv는 age ≤ 4만 커버 → (c1) 손실은 age ∈ (4,c] 질량에 비례해 c와 함께 증가, (a)는 평평 → **hot 티어의 가치 = "PNM 효율이 요구하는 큰 c를 recency 구멍 없이 살 권리"** (c-의존적 필요성으로 착지 — (c1)이 작은 c에서 살아도 anchor 생존).
- conv의 GPU 상주는 선택이 아니라 강제(활성값 4-tap FIR, capacity 없음) — 설계 질문은 "fresh hot state 티어의 존재"로 좁혀짐.

## 10. 결과 (v3 / E5 최종 / E6)

- **E5 최종: PASS 결정적** — pipedelta@3k vs naive@3k, stale 열(k16/k32) max|diff| **0.001** (기준 ±0.03). k8 열은 +0.07~0.08 개선(pre-update readout의 정칙화 추정).
- **E6 separated 6k 판정 — 예상 역전, 핵심 발견:**
  - **(a) cold-stale(correction도 stale): D=32에서 c≥4 붕괴**(0.479/0.064/0.253) — age 분해상 **전 age 광역 오염** = write-path 오염(stale cold correction이 delta 간섭-소거를 훼손, state 영구 오염).
  - **(c1) all-stale-honest(correction exact, readout만 stale): 전 c 생존**(D=32 c=16에서 0.949) — 손실은 **정확히 age ≤ c에 국소화**(c=16: age1-4 0.44 → age17+ 0.98).
  - ➡️ **"staleness의 금기는 readout이 아니라 correction."** cold 갱신은 PNM의 exact chunk replay로(자연스러운 chunked WY), readout만 stale 공개. hot 티어의 존재 이유 = **young age(≤c) 커버하는 fresh recency buffer.**
  - (b) 인과는 (a)-류 구성에서 확정 유지: nested cold-c4-D16 0.952 vs dedicated 0.050.
  - **ordering의 역할 재배치:** dedicated-(c1)도 separated에선 생존하지만 dedicated의 앞 8차원 readout은 무의미(k8 recall ~0.05) → **hot 티어를 유효한 memory로 만드는 것이 nesting** — 작은 fresh 티어의 자족성이 ordering의 시스템적 존재 이유.
- **⚠️ 지위 강등 (감독 지침): 위의 "correction 금기" 결론과 v4는 SEPARATED가 던진 가설이며 판정이 아님.** separated는 (c1)/recency를 못 재는 벤치라고 우리 스스로 판정했으므로(§레이아웃 결함), separated 기반 도출은 규율 위반. interleaved 3자 비교에서 재현될 때만 승격, 재현 실패 시 v4는 separated 아티팩트 산물로 폐기.
  - 인식론적 뉘앙스(선등록): 역전의 두 반쪽 중 **(a) 붕괴는 write-블록 내 현상이라 레이아웃 비판에 상대적으로 견고**(재현 예상, 정도는 완화 가능), **(c1) 생존이 바로 그 아티팩트 취약 셀**(진짜 미지수).
- **v4 (가설): tier-local writes** — cold 열은 PNM-exact write(GPU가 r_hot 동봉 → PNM이 완전한 correction으로 chunk replay), hot 열만 stale-corrected write. (a)의 fresh-hot readout + (c1)의 clean cold state 결합. 링크 비용 +d_v/token. interleaved 6k 3자 비교 실행 중(exp_e6v2_imqar_tierw_6k.log).
- **판정 기준 (확정, 감독 합의): age-프로파일 × 부하(D) 지문 대조.**
  - correction staleness → **age-flat** 열화(age≫c 포함), **D에 강하게 스케일** (메커니즘: stale $S$ 기준의 간섭-소거가 유령잔상/불완전 덮어쓰기를 state에 영구 기입, write 스트림 따라 복리 누적; 포화 시 key 비직교로 전역 분산 → flat. 저부하선 $Sk$ 항 자체가 작아 무해 = "staleness 예산 ∝ headroom"의 미시 설명)
  - readout staleness → **step at c** (age>c는 c=0 수준), D 의존 약함
  - v4 재현 시 → age>c는 c=0 수준 + **young-age 하한 ≈ nested의 fresh k=8 recall**(hot 티어 자족 용량). dedicated는 k=8 readout 무의미(~0.05)라 v4-류 구조 불가 → ordering 인과 재확인 지점.
- **v4 분리 판정 구조 (감독 못 + 정밀화 확정):** v4를 단일 승격/폐기가 아니라 다리별로 판정.
  | v4의 구성 요소 | 근거 | interleaved 판정 셀 |
  |---|---|---|
  | 다리1: correction-exact (chunk replay) | (a) 붕괴 — 견고 | (a)가 age-flat × D-스케일 열화를 재현하는가 |
  | 다리2: readout-stale (old만) | (c1) 생존 — 의심 | **(c1)/v4의 age > c recall ≈ c=0 수준인가** ← 다리2의 진짜 의존처는 (c1) 전체 생존이 아니라 이 step-프로파일. (c1)이 young age에서 무너지는 건 다리2의 실패가 아니라 **예상된 지문이자 hot 티어의 존재 이유** |
  | 다리3: hot 티어의 young 커버 | nested 자족성 | v4(age ≤ c) ≈ nested fresh k=8 수준 ≫ (c1)(age ≤ c) |
  다리2 FAIL 조건 = age > c에서도 열화. (a) 재현 + (c1) young-붕괴 조합이면 → v4는 폐기가 아니라 "다리1 확정 + 다리3 검증"으로 진행.
- interleaved(imqar) 판정 런 4종 실행 중: nested(a/c1) / dedicated(b) / **v4** / E8v2. 모든 결론은 separated(가설) / interleaved(판정) 2층으로 보고.
- **인시던트(수정 완료):** make_imqar의 per-item 루프 + GPU sync로 배치당 21.9s → 판정 런들이 기어감. 벡터화(argsort-셔플·cumsum-포지셔닝)로 **0.015s(1500×)**, 연관 정합성·young-age 존재 검증 후 4런 재실행.

## 12. INTERLEAVED 6K 최종 판정 (판정층 — 확정)

**다리 1 ✅ correction-exact 확정:** (a)가 imqar에서 D-스케일 붕괴 재현(c4: D8 −0.08 / D16 −0.30 / D32 −0.60). age 지문 정밀화: flat이 아니라 **age 증가에 따라 손상 증가**(D32,c4: age1-4 0.65 → age33+ 0.09) = **복리 누적 메커니즘의 직접 증거** (오래된 항목일수록 오염된 write에 더 오래 노출).
**다리 2 ✅ readout-stale(old) 확정:** (c1) age>c ≈ c=0 수준(c16에서 age17+ 0.97/0.95 vs 0.988), step-at-c 정확히 재현.
**다리 3 ✅ hot young 커버 확정:** v4 young(D32): c4 0.87 / c16 0.80 vs (c1) 0.48 / 0.08. 선등록 하한(fresh k8=0.763) 상회.
**➡️ v4 승격 (논문의 기술적 심장):** 전 c에서 (a),(c1) 동시 지배 — D32,k32: v4 0.981/0.965/0.942/0.916/**0.864** vs (a) …0.416, (c1) …0.634 (baseline 0.988).

**정직한 축소:** (b) ordering→staleness-내성 인과는 판정층에서 +0.05~0.06으로 축소(separated의 0.981-vs-0.040은 밀집-write 레짐 산물). imqar-학습 dedicated의 절단 readout은 쓰레기가 아님(k8,D32 0.484). **ordering의 인과는 "fresh 티어 품질"로 이전.**

**CLOSURE 판정 (선등록 미스 포함):** 예측 dedic-v4 young ≈0.48 → 실제 **0.70(c4)/0.64(c16)** vs nested-v4 0.87/0.80. ordering의 v4-young 기여 = **+0.16~0.17 (정량적 우위, 이진적 enable 아님)** — "young recall ≈ hot 티어 단독 품질 + conv"와 정합(0.763 vs 0.484 격차가 conv로 희석). old age에선 dedic이 근소 우위(작은 nesting tax). 총합은 nested-v4 승(+0.015~0.027).
**확정 anchor 문안:** ① **nesting = hot 티어 유효성의 by-construction 보장**(학습 목표가 보장) — un-nested의 prefix 품질은 우연적·벤치의존(separated-학습 0.05 vs imqar-학습 0.48). "보장 vs 우연" + multi-width elasticity 무료. ② staleness 내성 = 설계(correction-exact + readout-stale)의 산물. ③ "staleness = PNM 효율 노브"(트래픽 ∝ 1/c) = 순수 시스템 주장, §8이 정량화.

**v4 층위 표시 (감독 못 2):** 위 승격은 **알고리즘 층(recall 축)**. 시스템 층 이득은 §8에서 검증 — 첫 analytic 결과(PROBLEM_SETTING §8-a): **GPU+PNM v4가 iso-state-size 4.0×, c≥4에서 cold state 트래픽이 GPU-BW 천장 아래로(사실상 공짜), c=4 스윗스팟(recall 비용 ~4.6%).**
**논문 주의:** 총량 수치는 벤치 age 분포 의존 — 벤치 독립 진술은 age-분해 지문. (a)의 c8>c16 비단조 = chunk 경계 정렬 아티팩트 추정(각주).

**E8v2 (못 B) 판정:** X — rot-FT로 k8,D32 0.484→0.687(scratch 0.763의 90%), k≤4 격차 잔존, full-width 무손실. Y — rot-FT의 (a)/(c1) staleness 프로파일이 scratch와 사실상 동일((c1) c16 D32 0.632 vs 0.634). 단 (c1)-형 내성은 ordering-특이가 아니라 correction-exact의 산물(dedicated도 보임) — retrofit은 Y를 "보존"함. **배포 스토리: 기존 체크포인트 → 회전-only FT로 중간 폭 tiered-state 배포(최소 폭은 경량 해동), staleness 거동 보존.**

## 13. A4 판정 — 진짜 multi-head GDN에서의 최종 검증 (Phase A 완결)

fla GatedDeltaNet(4-head, short conv, output gate, in-kernel L2 norm), imqar 학습(nested {8,16,32,64} + dedicated-64), 검증 게이트(naive=fused=0.997) 통과. `scale/gdn_a4*.py/log/json`.

**Q1 ✅ v4 분리가 multi-head에서 성립.** nested D=64: fresh 1.00 / (a)c32 0.93 / (c1)c32 0.51(young 0.02) / **v4 c32 0.95(young 0.89, old 1.00)**, c8 비용 2%. 부수 발견: (i) correction-staleness가 toy보다 온화 — 용량 headroom 규칙 재확인, (ii) **multi-head 중복성** = head 1개 stale은 무해(0.99, 결함 내성), (iii) head 이질성은 dedicated에서만(head3 0.893), **nested는 균일(0.991–0.996)** — nested 학습이 head 역할을 균질화.
**Q2 ✅ hot 티어가 recall을 구조; A3v2 경고 불발현.** v4 young 0.88–0.95 vs (c1) 0.02–0.11; **v4 old 0.99–1.00**(pretrained GLA와 달리 老recall 붕괴 없음 — hot이 query-시점 상호작용 담당 + exact replay가 state 청정 유지). 한정: 태스크-학습 모델; 최종 실LM 판정 = 언어-학습 nested GDN(T3).
**Ordering 인과 최강 증거:** hot-단독 k16 nested 0.98 vs dedicated 0.44 → v4 young 0.88 vs 0.60.

## 11-c. E8 v1 결과 (separated 파일럿 — X만)
pretrain(fixed-32) → 동결 → rot-only Matryoshka FT 3000:
```
D\k      2      4      8     16     32     (from-scratch nested 6k 대비)
4     0.362  0.625  0.947  0.995  0.999    (0.761 0.969 0.997 1.000 1.000)
8     0.224  0.445  0.852  0.987  0.998    (0.533 0.895 0.985 1.000 1.000)
16    0.126  0.264  0.659  0.966  0.997    (0.320 0.688 0.910 0.994 0.998)
32    0.078  0.148  0.400  0.824  0.975    (0.175 0.401 0.666 0.931 0.974)
```
- **full-width 완전 보존**(k=32 열 동일) — 함수-보존 시작점 검증 ✓. 회전만으로 **실질적 ordering 이식**(k=8: 0.947/0.852 — pretrained 절단 시 ~0.05 대비) ✓.
- 단 **작은 폭(k≤4)에서 from-scratch와 뚜렷한 격차** — 회전만으론 tail까지 못 미침(backbone k-proj 동결 한계). ‖RᵀR−I‖=0.09~0.45.
- 판정: **X는 "부분 retrofit 가능"** — 중간 폭 elasticity는 회전만으로 확보, 최소 폭까지 원하면 경량 해동 필요. Y(staleness)는 E8v2에서.

## 11-b. E8 못 B 반영 (감독 지침)
E8 v1은 recall 그리드(성질 X=ordering retrofit)만 측정 — **anchor 검증에 불충분.** 회전 $R$은 정적 재정렬이라 X는 줘도 Y(state 변화율 억제)는 보장 없음($|[Rk]_{cold}|$가 작아질 이유 없음, backbone k-proj 동결). **E8v2**: imqar에서 pretrain→rot-FT 후 **staleness 내성 sweep까지 측정**(exp_e8v2_imqar.log). recall 그리드만 접근하고 staleness 내성이 안 따라오면 retrofit은 "elastic 모델"은 만들어도 "tiered-state 배포"는 못 만듦 — 그 경우 retrofit 스토리는 X까지로 한정하고 Y는 from-scratch(또는 backbone 일부 해동 FT)로.

## 11. E8 — retrofit 가능성 (pretrained → nesting FT)
delta recurrence는 key 공간의 직교 회전에 대해 full-width 함수 불변 → **backbone 동결 + identity-init 회전 R만 Matryoshka objective로 학습**하는 retrofit 레시피가 이론적으로 성립(학습 대상 = ordering뿐). E8 = fixed-32 "pretrained" 프록시 → rot-only FT → from-scratch nested와 그리드 비교. 실행 중(exp_e8_rotft.log). 성공 시 "기존 GDN/Qwen3-Next류 체크포인트를 값싸게 tiered-state로 개조" 배포 스토리.
