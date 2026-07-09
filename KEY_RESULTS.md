# KEY RESULTS — 논문 논리 흐름 정리 (2026-07-09 기준)

> **한 줄 요약: 공개 Nemotron-9B의 0.04% 파라미터(회전 R 3.5M + decay ~7k)만 몇 시간 재학습하면,
> recurrent state를 hot/cold 2티어로 갈라 decode를 B200 실측 2.42× 가속하면서
> 모든 정확도 축(ppl·needle·commonsense 8태스크·recall 6태스크·다운스트림)이 fresh와 동률이 된다.**

모델: `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (백본 동결, retrofit). 최종 ckpt: `scale/nemo9b_rot_p4long.pt`.
최종 배포 구성: **v4-c16 + bf16-cold + warmup(=prefill)**. 모든 수치는 재현 로그/json이 `scale/`에 있음.

---

## §1. 문제 (Motivation) — 왜 state가 병목인가

- Linear-attention/SSM의 recall 용량 = **recurrent state 크기**(Zoology/BASED 법칙). 그런데 state 크기는 학습 시 고정.
- decode의 state 갱신은 **OI≈1 (memory-bound)**: 매 토큰 state 전체를 read+write. 9B에서 state = **시퀀스당 141.6MB** → 대배치에서 가중치가 아니라 **state 트래픽이 지배**.
- 실측: tok/s ∝ 1/state (slope −1 정확, 커널 BW 고정 2.4~2.6TB/s), 용량 2×(80→160GB)로는 tok/s 불변 — **대역폭·용량 둘 다 state가 벽** (`scale/state_bound_motivation.png`).

## §2. 아이디어 — importance-ordered tiered state (dense-but-stale)

- state 차원을 **중요도순으로 정렬**(nested/Matryoshka 학습)한 뒤, 앞 pb차원 = **hot(매 토큰 fresh)**, 뒤 = **cold(c토큰마다 몰아서 exact 갱신, 읽기는 stale+decay 보정)**.
- 아무것도 버리지 않음(sparse가 아니라 **dense-but-stale**): 오차는 "최근 c토큰의 cold 성분"뿐, age-국소적·정확 보상.
- **배포 의미론**: prefill(프롬프트)은 원래 chunked scan이라 자연 fresh → decode(생성)만 tiered. 즉 **warmup = 프롬프트 길이** — 짧은 요청은 애초에 tiering 안 함.

## §3. 방법 — 3.5M 파라미터 retrofit (백본 동결)

| 구성요소 | 내용 | 근거 실험 |
|---|---|---|
| 회전 R (27층×8그룹×128², 3.5M) | state 기저를 중요도순으로 재정렬 | T10: **QR retraction → full-width 정확히 무손실 보장**(soft penalty는 실패 — 다양체 사영이 정답) |
| 폭 메뉴 (nested dropout) | k∈{16,32,64,128} 탄력성 | truncation 붕괴(k64 2810) → 회복(+1%) |
| **v4-aware 학습** | 50% 스텝을 v4 forward(c 추첨 {4..64})로 | v4 비용 +5.9%→0%로 수렴 |
| **tune_decay** (A_log/dt, 0.05×lr, ~7k개) | head별 decay가 staleness에 적응 | **배포축 수렴 ~3× 가속**(1.2k스텝=plain 4k스텝) |
| **다양한 데이터** (wikitext+fineweb 50/50) | 도메인 과적합 방지 | §7 함정 참조 |

총 학습: ~5.6k 스텝 ≈ 수 시간/1 GPU. **fresh(full-width) 경로는 회전 불변성으로 수학적으로 보존.**

## §4. 속도 결과 (B200 실측, 9B 실차원, B=448, baseline = fla fused 커널 + fp32 state = 오늘의 표준 구현)

**트래픽 분해**: fresh는 매 토큰 state를 2번 옮김(read+write=2.0S). 두 레버가 서로 다른 항을 지움:

| 단계 | 지우는 항 | **실측 속도** | acc 비용 |
|---|---|---|---|
| fresh | — | 1.00× (8,730 tok/s) | — |
| + tiering (lazy write, c=16) | cold **write** ÷16 | **1.81×** | **0** (70.23) |
| + cold **bf16** | cold **read** ÷2 | **2.42×** (21,111 tok/s) | **0** (70.25) |
| + cold **fp8**(scaled) | cold read ÷2 또 | ~2.8× (analytic*) | **0** (70.07) |

- *fp8 speed는 dequant-matvec 커널 필요(트래픽 모델은 fp32 커브 전 구간에서 실측 일치로 검증됨).
- **tiering 단독 상한 ~2×** (c64 실측 2.05×): readout 때문에 cold **read**는 매 토큰 남음 → quant가 정확히 그 항을 공격 (상보 관계, 곱 아님).
- **NEGATIVE (중요)**: async flush(별도 stream 겹치기) = 이득 정확히 0 — state-op 경로는 BW 포화라 **바이트 감축만이 화폐**. (§8 memory-bound 전제의 실측 확인. 단 PNM은 메모리 시스템이 분리라 겹침이 진짜 공짜 — lag acc 실측 = PNM-deferred-flush 비용.)
- 커널 함정: fla fused는 B×H≤65535 (CUDA grid).

## §5. 정확도 결과 — 전 축 lossless (retrofitted p4long, v4-c16-bf16cold-warm)

| 축 | fresh | v4 tiered | 판정 |
|---|---|---|---|
| wikitext ppl | 6.44 | **6.45 (0%)** | 문자 그대로 공짜 |
| needle recall (나이 400+) | 1.00 | **1.00 / 1.00** (c4/c16) | 만점 |
| commonsense 8태스크 (L1k) | 70.42 | **70.25** | 8/8 노이즈 이내 (boolq 85.5/85.1 등) |
| **recall-intensive 6태스크** (BASED: fda/swde/squad/tqa/nq/drop, native decode 실측) | 41.3 | **41.0** | 6/6 노이즈 이내 |
| retrofit 자체 (fresh vs 원본) | 71.43 | 71.33 / 생성도 raw와 동일 | retrofit 무손실 |

- **학습 전 대비**: v4 ppl +5.9~10.9% → 0%, needle 0.67(c64) → 1.00 — "비용이 학습 예산의 함수"임을 확정.
- 다운스트림의 마지막 −1pt는 **짧은-시퀀스 아티팩트**였고(§2 배포 의미론의 warmup이 해소), 긴-시퀀스 증거(ppl/needle/recall 스위트)는 독립 성립.

## §6. 비대칭 정밀도 면허 (실측) — "tiering이 quant를 공짜로 만든다"

같은 scaled-fp8을 어디에 넣느냐:

| 위치 | 라운딩 주입 | ppl | needle |
|---|---|---|---|
| per-token 재귀 (baseline이 필요한 것) | T번 | **+5.1%** | **0.92** |
| cold snapshot (우리 구조) | **T/c번** | **0%** | **1.00** |

- bf16은 양쪽 다 무사(9B, ~1k 지평) → 정직한 baseline 재앵커 = bf16-fresh. 그 대비 우리 fp8-cold ≈ **2.2×** (트래픽 0.47S vs 1.0S).
- **클레임**: baseline이 어떤 정밀도로 돌든, cold 티어는 한 단계 아래를 견딤 — 그 아래는 baseline이 깨지는 지점. (Quamba/MambaQuant 등 state 양자화와 경쟁이 아닌 **합성** 관계.)

## §7. 학습 레시피 발견 3건 (방법론 기여)

1. **tune_decay 가속**: decay 7k개 해동(0.05×lr)이 배포축 수렴 ~3× 가속 + needle 만점 + 게이트 무손상.
2. **distill 함정**: full-width 강제 distill은 폭 탄력성 파괴(k16 8.10→9.24) → **폭 샘플링 유지** 필수.
3. **도메인 과적합 다이버전스**: wikitext 단일 FT는 in-domain 지표(ppl/needle)를 올리면서 **v4 다운스트림을 −1pt 드리프트**(fresh는 회전불변으로 무사) → R-학습 데이터는 다양해야(fineweb 혼합으로 회복). fp32 컨트롤로 bf16 무관 확정.
   - 방법론 각주: lm-eval limit=500은 ±1pt 샘플 편차 — 후보 간 비교는 limit≥1000.

## §8. 배포 서사 (2단)

1. **GPU-only (새 HW 0개)**: 위 실측 2.42×(bf16)/~2.8×(fp8). capacity(batch)는 불변.
2. **+CXL-PNM 계층**: cold를 PNM 상주 — analytic 3.8× + state capacity 8×. cold 갱신(chunk matmul, OI↑)이 near-memory에 정합; **PNM-deferred-flush의 acc 비용도 실측 확보**(lag semantics). 설계점 (c, hot 1/4~1/8) 전 구간 robust (§8 boundary map).

## §9. 정직한 범위 한정 (리뷰 방어)

- 속도는 **decode의 state-op 경로**(대배치 지배 성분, Pimba 방법론과 동일) — 모델 전체는 attention/MLP에 희석, prefill-heavy에서 축소 (boundary map에 정량).
- fp8 speed는 analytic (acc는 전 축 실측).
- 남은 eval: RULER 표준화, full-set(no-limit), MMLU/GSM8K.
- 엔진 앵커: naive 엔진은 생성에서 degenerate(발견·우회 완료 — recall 수치는 native decode 실측), arms 절대값은 자기일관 엔진 내 비율.

## 부록: 핵심 파일

| 파일 | 내용 |
|---|---|
| `scale/nemotron_retrofit.py` | 학습 (R+QR, --v4aware/--tune_decay/--distill/--cs_menu/--data) |
| `scale/nemo9b_eval.py` | ppl/needle 판정 (v4/c1/lag/cold_bf16/fp8/warm 의미론) |
| `scale/nemo9b_lmeval.py` | lm-eval harness (다운스트림) |
| `scale/v4_native_decode.py` | **native decode의 v4 구현 = 배포 프로토타입** (pb128 게이트 검증) |
| `scale/bench_v4_decode.py` | B200 speed bench (fresh/v4/async/bf16cold/hot-only/eager) |
| `scale/nemo9b_rot_p4long.pt` | 최종 ckpt (R + decay) |
