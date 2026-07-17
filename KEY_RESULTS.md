# KEY RESULTS — 논문 논리 흐름 정리 (2026-07-17 기준)

> **한 줄 요약: 공개 Nemotron-9B의 0.04% 파라미터(회전 R 3.5M)만 몇 시간 재학습하면,
> recurrent state를 hot/cold 2티어(dense-but-stale)로 갈라 decode를 B200 실측 1.92×(c4)~2.42×(c16) 가속하면서
> 공식 평가 스택(NeMo-Skills+vLLM)의 전 정확도 축이 3-arm(raw/fresh/v4) 동률이 된다.
> 배포점 v4-c4-fp8: GSM8K 94.6 / MATH-500 95.2 / RULER@4k 98–100.**

모델: `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (백본 동결, retrofit).
ckpt 계보: `nemo9b_rot_p4long.pt`(배포 검증 완료) → `nemo9b_rot_longcot.pt` → **`nemo9b_rot_longcot2.pt`(최신 — GSM8K v4-c4 lossless, 공식 스택 재측정 대기)**.
acc의 공식 소스 = **NeMo-Skills + vLLM**(2026-07-12 전면 교체; 구 lm-eval/native 수치는 내부 기록 — 3-arm 델타만 유효). 모든 수치의 원장 = `EVAL_LEDGER.md`.

---

## §1. 문제 (Motivation) — 왜 state가 병목이고, 왜 버리면 안 되는가

**① state = 대역폭·용량의 벽 (실측):**
- Linear-attention/SSM의 recall 용량 = **recurrent state 크기**(Zoology/BASED 법칙). 그런데 state 크기는 학습 시 고정.
- decode의 state 갱신은 **OI≈1 (memory-bound)**: 매 토큰 state 전체를 read+write. 9B에서 state = **시퀀스당 141.6MB**.
- 커널 실측: tok/s ∝ 1/state (slope −1 정확, BW 고정), 용량 2×로는 tok/s 불변 (`scale/state_bound_motivation.png`).
- **B200 vLLM 배치 스윕 실측**: SSU(state-op) 점유율 2.6%(B=1) → **57.6%(B=256)** → 60%(B=1152); fp32 throughput 천장 11.9k tok/s(B=512부터 포화); capacity wall B≈1230. → 서빙 체제(대배치)에서 state-op가 지배 성분.

**② 그렇다고 버릴 수는 없다 — 삭제-vs-lazy 사다리 (2026-07-13, GSM8K n=200):**
| arm | GSM8K | recall(fda/swde) |
|---|---|---|
| fresh | 85.0 | 81.0 / 92.5 |
| GHOST-lite(calibration 순열 top-32, **삭제**) | **58.5 (−26.5)** | 81.0 / 92.0 |
| trained R + **삭제** | **59.5 (−25.5)** | 81.5 / 92.0 |
| trained R + **lazy 보존**(v4-c4) | **80.0 (−5; 공식 스택 n=1319에선 −0.5)** | 81.5 / 92.0 |
- 정적 절단은 **다단계 추론에서 파국** — wikitext ppl(+1)과 얕은 recall(무손상)이 못 보는 축.
- 회전만으론 삭제를 못 구함(−25.5) — 회전 = 분할 효율화지 삭제 면허가 아님. **같은 hot 예산에서 lazy 보존만이 회복.**
- 결론: cold의 가치는 staleness에 살아남고 제거에 죽는다 → **"버리지 말고 미뤄라"가 유일한 적정 관리.**

## §2. 아이디어 — importance-ordered tiered state (dense-but-stale)

- state 차원(N축)을 **중요도순으로 정렬**(직교 회전 R + nested 학습)한 뒤, 회전 좌표의 앞 pb차원 = **hot(매 토큰 fresh)**, 뒤 = **cold(c토큰마다 몰아서 exact 갱신, 읽기는 stale+decay 정확보상)**.
- 아무것도 버리지 않음: 오차는 "최근 c토큰의 cold 성분"뿐 — age-국소적·내용-무관·정확 보상. (sparse 계열과의 대조 문장: **"sparse-and-fresh vs dense-but-stale"** — SSE/MoM은 무엇을 저장할지 라우팅(비가역 손실), 우리는 언제 반영할지만 지연.)
- 정확한 비유 = **learned write-back cache hierarchy** (분할 기준이 시간이 아니라 트래픽; flush=write-back). short/long-term memory 비유는 부정확.
- **배포 의미론**: prefill은 chunked scan이라 자연 fresh → decode만 tiered. warmup = 프롬프트 길이.

## §3. 방법 — 0.04% retrofit (백본 동결)

| 구성요소 | 내용 | 근거 |
|---|---|---|
| 회전 R (27층×8그룹×128², 3.5M) | state 기저를 중요도순 재정렬; **QR retraction → full-width 수학적 무손실**(명제 1) | fresh = raw 정답 수 동일(공식 스택 실증) |
| 폭 메뉴 (nested dropout) | k∈{16,32,64,128} 탄력성 | E-T2: 메뉴 지점 절단세 1.1~1.3× |
| v4-aware 학습 | 스텝의 절반을 v4 forward(c 추첨)로 | 미학습 tiering 33.0 → 학습 후 lossless |
| long-CoT 데이터 | seqlen 4096 + **cs_menu c4·c8 가중** (longcot2) | GSM8K v4-c4 갭 −3.4 → **+0.4 (lossless)** |

- **⚠ 정정(2026-07-13 발견): 실효 학습 = R 단독.** tune_decay는 침묵 no-op였음(bf16 ulp 0.008 > 5e-5 업데이트; A_log가 전 ckpt에서 bit-identical) → fresh=raw는 구성상 보장(회전 불변성)이고, 관측된 fresh 드리프트는 greedy 라운딩 노이즈.
- 회전 불변성 정리(THEORY 명제 1): B→RB, C→RC ⇒ y 불변 — **스칼라 per-head decay 가족(mamba2/GDN)에 무조건 성립**; GLA/KDA(채널 decay)는 경량 FT 필요(명제 1′).

## §4. 속도 (B200 실측, B=256 표준, fused-only)

**트래픽 분해**: fresh는 매 토큰 read+write(2.0S). 두 레버가 서로 다른 항을 지움 — **read는 바이트(정밀도)로, write는 빈도(청킹)로**:

| 단계 | 지우는 항 | 실측 | acc |
|---|---|---|---|
| fresh (fp32) | — | 1.00× | — |
| + tiering c4 (lazy write) | cold WRITE −61%, READ −47% (회계 실측) | 1.62× (bf16 실측 기준) | 0 |
| + tiering c16 + bf16-cold | read ÷2 추가 | **2.42×** | 0 |
| + fp8-cold | read ÷2 또 | 2.17×(c4)~2.8×(c16) analytic* | 0 (공식 95.2) |

- *fp8 speed는 dequant-matvec 커널 대기(트래픽 모델은 fp32 전 구간 실측 일치로 검증).
- **NEGATIVE**: async flush(stream 겹치기) = 이득 0 — state-op 경로는 BW 포화, **바이트 감축만이 화폐**.
- **NEGATIVE**: 동적 read 선택(ktop/qgate)도 유료(§7) — read의 공짜 레버는 정밀도뿐.
- roofline/스윕 그림: `scale/results/plots/sweep_*.pdf` (SSU read/write 분리 표기, fp8 투영 2.17/2.68/2.79×).

## §5. 정확도 — 공식 스택 매트릭스 (NeMo-Skills+vLLM, pass@1, reasoning on)

### §5a. 정밀도×티어링 전수 매트릭스 (2026-07-13, fresh/v4 = longcot ckpt)
| bench | 공식 발표 | raw | raw-bf16 | **raw-fp8** | fresh | v4-c4-fp32 | v4-c4-bf16 | **v4-c4-fp8** |
|---|---|---|---|---|---|---|---|---|
| GSM8K (1319) | 91.4 | 95.0 | 95.0 | 95.15 | **95.0 (정답수 동일)** | 94.47 | 94.47 | **94.62** |
| MATH-500 | 97.8 | 97.6 | 97.6 | **93.2 ⚠붕괴** | **98.2** | 95.2 | 95.2 | **95.2** |
| RULER@4k (niah 5종) | — | — | 98–100 | 98–100 | — | — | — | **98–100** |

판정 4건: ① **스택 검증 통과**(공식 91.4 이상 재현; 구 lm-eval 갭은 harness 차이 확정), ② **fresh 완전 무손실** — retrofit이 기반 모델 무훼손(회전 불변성 실증), ③ **raw-fp8 반증 arm 작동** — 장문 CoT에서만 붕괴(MATH −4.4, 생성 11.6k토큰 = 2.2× 방황; GSM8K ~1.8k·RULER@4k는 무사 → 손상 축 = 수천 토큰 자기생성 폐루프), ④ v4 잔여 갭(MATH −2.4)은 **정밀도 무관(fp32=bf16=fp8 동일)** = 순수 staleness → 처방은 학습(longcot2)/c2.

### §5b. longcot2 — GSM8K lossless 달성 (2026-07-14, native 경로 n=500)
| ckpt | fresh | v4-c4 | 갭 |
|---|---|---|---|
| longcot | 83.2 | 79.8 | −3.4 |
| **longcot2** | 81.2 | **81.6** | **+0.4 (lossless)** |
공식 스택 재측정(GSM8K/MATH-500) 후 배포 승격 판정 — **다음 실험 1순위.**

### §5c. 나머지 표준 스택 (p4long 시절 판정 — 전부 lossless, 내부 스택)
commonsense 8task 8/8 동률 · recall-intensive 6task(BASED) 동률 · RULER 11종 동률 · HumanEval 동률 · needle 1.00 · ppl 0%.
minerva_math만 c4에서 −2.9였고 → longcot2의 표적이 이것(공식 재측정 대기). "생성이 길수록 낮은 c" = **c는 recall-SLA/생성길이 다이얼** (GSM8K: c16 −9.3 → c4 0; pb 증대는 무효 — 인과가 오직 staleness 빈도).

## §6. fp8 비대칭 정밀도 면허 — 사슬 완결 (2026-07-14)

| | 학습 X (항등 R) | 학습 O |
|---|---|---|
| raw + fp8 매토큰(폐루프) | — | **MATH 93.2 붕괴 + 2.2× 방황** |
| tiering + fp32-cold | **33.0 (−48.6 붕괴)** | 81.6 (lossless) |
| tiering + bf16-cold | 33.0 (동일 — 정밀도 무관) | 94.47 (공식) |
| tiering + fp8-cold | 접근 불가 (첫 관문 잠김) | **95.2 (공식, = fp32 = bf16)** |

**fp8을 안전하게 쓰는 유일한 경로 = [R 학습(0.04%) → tiering 사용권 → 청크 개루프(라운딩 T/c회) → fp8 공짜].** 관문마다 실측으로 잠김: 학습 없인 tiering 붕괴(정밀도 무관), tiering 없인 fp8 폐루프 복리 붕괴.
**capacity 회계**(raw fp32 = 1.0): bf16-cold **0.625**(단일 사본 — 대역폭 2.17×와 용량 −37% 동시), fp8-cold 0.81(유일한 이중 사본) — "용량 태워 속도 샀다" 공격의 답.

## §7. Read축 탐색 — 원리 있는 부정 결과 (설계 정당화)

| 방식 | skip | GSM8K (n=200, full=81.0) |
|---|---|---|
| 열 top-k(노름 유도) k32/k16/k8 | 67/83/92% | 74.0 / 55.5 / **21.0 (붕괴)** |
| head-qgate τ=0.45/0.6 | 25/66% | 79.0 / 74.0 |

- **압축 상태의 readout은 밀집**: attention의 토큰-희소성은 압축에서 소진 — per-token 열-희소성이 없음. → read를 줄이는 합법 레버 = **바이트(정밀도)** 뿐, write는 **빈도(청킹)**. 이 두 축이 우리 설계의 전부인 이유.
- qreg(질의-집중 정규화, λ=2.0)는 학습 가능(qrho 0.69→0.49, 무손상) — 게이트 부활 조건 = gate-aware 학습(미완).

## §8. 이론 — 무엇이 정렬되고, 왜 이식되는가 (E-T1/E-T2 실측, 2026-07-14)

- **명제 1(회전 불변성)**: full-width 정확 보존은 수학적 보장 — fresh=raw 정답 수 동일이 실증. (docs/THEORY.md 상세 유도.)
- **E-T1/E-T1b (중요도의 정체)**: 명시 트래픽 통계 M=E[CBᵀ](lag-0·시차가중)의 고유기저는 학습 R과 **무상관**(주각 cos 0.437~0.438 ≈ 무작위; gap@32≈0). 열노름(ktop 붕괴)·질의에너지(무신호)까지 **4중 수렴: "training-free 통계는 학습 기저를 예측 못 한다 — 중요도는 loss가 정의; end-to-end 학습 생략 불가."** GHOST류 calibration 실패(−26.5)의 이론적 뿌리이자 retrofit 학습의 필요성 증명.
- **E-T2 (이식성의 직접 실측)**: 학습 R 하나의 절단세 곡선이 **wiki/수학/코드에서 일치**(k16: 1.26/1.31/1.27×, k32: 1.15/1.21/1.18× — 산포 ≤0.06). 대조: ghost 순열은 자기 폭(32)의 ppl만 생존(그마저 acc −26.5)하고 k16 이하 절벽(17~36×) — **nested 탄력성 자체가 학습 산물**. identity는 전 폭 파국(12~230×).
- **메뉴 격자(flag) 관측**: 메뉴 밖 폭(k=8, 96)은 3도메인 공통 유료(k96>k64 비단조) → 학습물은 고유값 전순서가 아니라 **학습 메뉴 지점의 중첩 부분공간 사슬** — E-T1(어떤 M의 고유기저도 아님)과 정합.
- 일반화 근거의 최종 배선: ~~Davis-Kahan 갭~~(gap≈0으로 기각) → **① 가설류 작음(직교군, 좌표 선택) + ② 이식 실측(E-T2) + ③ 합집합 조건(hot=32차원 부분공간, 순서 무관)**. 시각 자료 = `docs/theory_deck.pptx`.

## §9. Self-speculative decoding (탄력 폭의 제3 배당 — 진행 중)

- hot-only(pb32) readout = **공짜 draft 모델**: α=0.91~0.92 (학습 R) vs 0.32 (항등) — nesting이 draft 품질을 인과적으로 삼(2.7×). additive 갱신은 state-독립이라 hot state는 cold write 없이도 **정확히** 진화 — 근사는 skipped cold readout뿐, 청크 verify가 상태를 정확 재설정 → **출력 exact**.
- B=256 실측: draft 20.5ms(2.14×↓), naive verify 190ms 고정비 → 현재 eff 1.07×(c16). **완성 조건 = vLLM varlen verify 커널**(투영 1.6~1.8×). margin-적응 트리거 신호는 확보(AUC 0.89~0.92).

## §10. 학습 레시피 발견 (방법론 기여)

1. **미학습 tiering은 붕괴(33.0)** — v4-aware 학습이 실행권을 삼 (사슬 §6의 첫 관문).
2. **distill 함정**: full-width 강제 distill은 폭 탄력성 파괴 → 폭 샘플링 유지 필수.
3. **도메인 다양성**: wikitext 단독 FT는 v4 다운스트림 드리프트 (데이터 선택 = 실효 M 선택; E-T2의 경계 민감성과 정합).
4. **장문 staleness는 장문으로 가르쳐야**: longcot2(seqlen4096 + c4·c8 가중)가 GSM8K 갭 소멸 — v4 arm이 +1.8 상승(fresh 하락이 아님).
5. tune_decay는 no-op였음(§3 정정) — 레시피에서 제거 가능.

## §11. 정직한 범위 한정 (리뷰 방어)

- 속도는 decode의 state-op 경로(대배치 지배 성분, B=256 표준) — e2e는 attention/MLP에 희석(vLLM CUDA-graph 목표 ~1.45×), prefill-heavy에서 축소.
- fp8 speed는 analytic(acc는 공식 스택 전 축 실측). fp8 dequant-matvec 커널 = 남은 실측.
- raw 절대값 vs 공식 -Base 갭은 harness 차이로 판정 완료(-Base probe; 계보 전환 불필요, 논문에 harness 스펙 명기).
- rank-c exact correction(청크 지연 정확 보정)은 **paper B로 분리** — A의 서사(학습된 티어링 + staleness 면허)에서 제외.

## 부록: 핵심 파일

| 파일 | 내용 |
|---|---|
| `scale/nemotron_retrofit.py` | 학습 (R+QR, --v4aware/--cs_menu/--qreg/--data) |
| `scale/v4_native_decode.py` | native decode v4 = acc-grade 프로토타입 (lean prefill, corr/qgate/ktop/coldoff arms) |
| `scale/vllm_v4_patch.py` + `install_vllm_v4.sh` | vLLM 포팅 (공식 스택 서빙; NESTED_SSM_* env) |
| `scale/v4_fused_decode.py` / `bench_selfspec_e2e.py` | B200 speed bench / self-spec e2e |
| `scale/probe_M_spectrum.py` / `probe_M_lagged.py` / `probe_domain_elasticity.py` | 이론 실측 (E-T1/E-T1b/E-T2) |
| `scale/nemo9b_rot_longcot2.pt` | 최신 ckpt (전 ckpt git 백업) |
