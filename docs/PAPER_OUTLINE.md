# Paper A Skeleton — Elastic Test-Time Memory: Learned Hot/Cold Tiering of Recurrent State (2026-07-17 재작성)

*(working title 후보: "Tiered-State SSMs: Learned Write-Back Caching of Recurrent State for Fast Decoding" / "Dense-but-Stale: ...")*
*모든 수치 = KEY_RESULTS.md(헤드라인)·EVAL_LEDGER.md(원장) 실측. 구판 outline(소규모+PNM 서사)은 git 역사에.*
*스코프 확정(유저): GPU-only (PNM은 후속), rank-c exact correction은 paper B로 분리·본문 제외.*

## Abstract (요지)
Linear-attention/SSM의 decode는 매 토큰 recurrent state 전체를 read+write하는 memory-bound 연산이며, 대배치 서빙에서 지배 성분이다(B200 실측: state-op 점유 57.6% @B=256). state를 줄이면 recall이 죽고(Zoology 법칙), 잘라내면 다단계 추론이 붕괴한다(GSM8K −26.5 실측 — wikitext ppl이 못 보는 축). 우리는 제3의 길을 제시한다: **아무것도 버리지 않되, 덜 중요한 부분은 낡은 채로 읽고(stale read) 몰아서 정확히 갱신한다(lazy exact write)**. 공개 Nemotron-9B의 0.04%(직교 회전 R)만 재학습해 state 기저를 중요도순으로 정렬하면, 회전 좌표의 정적 prefix 분할(hot/cold)만으로 이 실행이 성립한다: full-width 정확도는 회전 불변성으로 **수학적으로 보존**되고(fresh=raw 정답 수 동일), tiered 실행은 B200에서 **1.92×(c4)~2.42×(c16)** 를 공식 평가 스택(NeMo-Skills+vLLM) 전 벤치 lossless로 산다. 부산물로 cold 티어는 정밀도 한 단계를 공짜로 얻는다(fp8-cold = fp32, 반면 raw-fp8은 장문 CoT에서 붕괴 — 라운딩 주입 빈도 T→T/c의 비대칭 면허). 이론 분석으로 "중요도는 어떤 training-free 통계로도 대체 불가(loss가 정의)"와 "학습된 기저의 도메인 이식성"을 실측으로 확립한다.

## 1. Introduction
- 문제 사슬: ① state=recall(법칙) → 크게 유지해야 함 ② decode state-op = OI≈1, 대배치 지배(57.6%) ③ 삭제/희소화는 비가역 손상(사다리 실측).
- 아이디어 한 줄: **learned write-back cache hierarchy** — 시간이 아니라 트래픽으로 나뉘는 hot/cold, 분할은 학습이 R에 구움(런타임 controller 0, 메타데이터 0, 연속 블록).
- 기여 5: ① 0.04% 회전 retrofit + 회전 불변성 정리(full-width 무손실 보장) ② v4 tiered 실행 의미론(hot fresh + cold lazy-exact-write/stale-read + decay 정확보상) ③ 공식 스택 3-arm 전수 lossless @ B200 1.92~2.42× ④ 비대칭 정밀도 면허(사슬 4-cell 실측) ⑤ 이론: loss-정의 중요도(4중 수렴) + 이식성 직접 실측.
- 시의성: Qwen3.5/Kimi 등 프론티어가 linear 계열 채택; 가족 일반성은 소규모 4가족(GDN/GLA/KDA/M2) 실측으로 뒷받침.

## 2. Motivation (실측 2단)
- **Fig-M1 state wall**: tok/s ∝ 1/state (slope −1), 용량 2× 무효; B200 배치 스윕(SSU 점유 2.6→57.6→60%, 천장 11.9k tok/s, capacity wall B≈1230).
- **Fig-M2 삭제 사다리**: fresh 85.0 / GHOST-lite 58.5 / trained-R 삭제 59.5 / lazy 80.0(공식 −0.5) — ① 삭제 희생축=다단계 추론(ppl·얕은 recall 무손상 뒤에 숨음) ② 회전≠삭제 면허 ③ lazy 보존만이 회복. → 설계 공간에서 "버리기"를 원리적으로 제거.

## 3. Method
### 3.1 Rotation retrofit (0.04%)
- 명제 1: B→RB, C→RC(그룹별 직교) ⇒ S→SRᵀ, y 불변 — 스칼라 per-head decay(mamba2/GDN)에 무조건; QR retraction으로 다양체 위 학습. 실효 학습 = R 단독(tune_decay no-op 판명 — 정직 기술).
- nested 폭 메뉴 {16,32,64,128} + v4-aware(c 추첨) + long-CoT 혼합(cs_menu c4·c8 가중, seqlen 4096) = longcot2 레시피. 총 수 GPU-시간.
### 3.2 v4 execution semantics
- hot(pb=32/128): 매 토큰 fresh R/W. cold: c토큰마다 chunk-exact flush(개루프), 읽기는 snapshot + exp(glog) decay 정확보상 — 오차 = 최근 c토큰의 cold 성분(age-국소, 내용 무관).
- 배포 의미론: prefill 자연 fresh(warmup=프롬프트), decode만 tiered. c = recall-SLA/생성길이 다이얼(§5).
### 3.3 왜 이 형태인가 (부정 결과들이 조각한 설계)
- read를 "골라 읽기"는 불가: 압축 state에 per-token 열-희소성 없음(ktop 81→21 붕괴; head-gate도 유료) → **read는 바이트(정밀도), write는 빈도(청킹)** 만이 합법 레버.
- async flush 이득 0(BW 포화) → 바이트 감축만이 화폐.
- 미학습 tiering 33.0 → 실행권은 학습이 산다.

## 4. Theory & Analysis (docs/THEORY.md + theory_deck)
- 4.1 회전 불변성(명제 1) — full-width 보존의 수학적 보장.
- 4.2 **중요도는 loss가 정의한다 (E-T1/E-T1b)**: 명시 M=E[CBᵀ](lag-0·시차)·열노름·질의에너지 전부 학습 R과 무상관(cos≈0.44, gap@32≈0) — calibration류(GHOST)가 실패하는 이유이자 end-to-end retrofit의 필요성.
- 4.3 **이식성의 직접 실측 (E-T2)**: 한 R의 절단세 곡선이 wiki/수학/코드 일치(k16 1.26/1.31/1.27×); ghost는 자기 폭 밖 절벽 — nested 탄력성 = 학습 산물. 메뉴 격자(flag) 관측(off-menu 폭 유료).
- 4.4 fp8 면허 메커니즘: 라운딩 주입 T→T/c + 개루프 flush — §6 4-cell이 예측 그대로.

## 5. Main Results (공식 스택 = NeMo-Skills + vLLM, 3-arm 원칙)
- **Table-1 정밀도×티어링 매트릭스**(§5a of KEY_RESULTS): raw 95.0/97.6 = fresh 95.0/98.2(정답수 동일) / v4-c4-fp8 94.6/95.2 / RULER 98–100. (longcot2 공식 재측정으로 최종 갱신 예정.)
- **Table-2 B200 속도**: 1.62×(c4-bf16) / 2.42×(c16-bf16) / fp8 analytic 2.17~2.8×; READ −47%/WRITE −61% 회계; roofline·배치 스윕 그림.
- **Fig c-다이얼**: GSM8K c16 −9.3 → c4 0 (pb 무효 — 인과=staleness 빈도); MATH만 c4 잔존 → longcot2 학습으로 회복(GSM8K 실증 +0.4, MATH 공식 재측정 대기). "생성 길이에 c를 다이얼."
- 표준 스택 커버리지: commonsense 8 / recall 6(BASED) / RULER 11 / HumanEval — 전부 3-arm 동률.

## 6. Asymmetric Precision License (§6 of KEY_RESULTS)
- 4-cell 사슬 표 + raw-fp8 붕괴의 지문(장문에서만: MATH −4.4·2.2× 방황, GSM8K/RULER@4k 무사 — 손상 축=자기생성 길이).
- capacity 회계: bf16-cold 0.625×(속도 2.17×와 용량 −37% 동시), fp8-cold 0.81×.
- 포지셔닝: Quamba/MambaQuant와 경쟁 아닌 합성 — 기여 = "staleness 구조가 만드는 티어별 정밀도 예산(÷c)".

## 7. Ablations & Negative Results (원리 있는 부정 결과 절 — 리뷰 방어의 척추)
| 항목 | 결과 | 메시지 |
|---|---|---|
| 미학습 tiering | 33.0 (정밀도 무관) | 실행권=학습 |
| 삭제(GHOST/trained-R) | −26.5/−25.5 | 버리기 원리적 배제 |
| async flush | 0 | BW 포화 — 바이트만 화폐 |
| ktop/qgate 동적 read | 붕괴/유료 | readout 밀집 — read 레버=정밀도 |
| distill 강제 | 탄력성 파괴 | 폭 샘플링 필수 |
| wikitext 단독 FT | 다운스트림 드리프트 | 데이터=실효 M 선택 |
| 명시 통계 기저(E-T1) | cos≈random | 중요도=loss 정의 |

## 8. Discussion / Future
- **Self-spec(제3 배당, 짧은 절)**: hot-only draft α=0.91(항등 0.32 — nesting 인과 2.7×), 출력 exact; varlen verify 커널 = 완성 조건(1.6–1.8× 투영). 완성되면 결과 절 승격.
- 가족 일반성: 소규모 4가족(GDN/GLA/KDA/M2) 2×2 실측 + GDN-2/Mamba-3 적용 조건(명제 1′).
- 후속: PNM 계층(cold 상주 — analytic 3.8×+capacity 8×), gate-aware 학습, pb 메뉴 재학습(~2.7×), fp8 dequant 커널.

## 9. Limitations
- 속도는 decode state-op 경로(e2e는 희석 — vLLM CUDA-graph 목표 ~1.45×); fp8 speed는 analytic; 단일 모델 9B 중심(가족 증거는 소규모); harness 스펙 명기(-Base 갭 판정 포함).

## Related Work (배선 완료 — memory/related-work 참조)
- 대조축: **sparse-and-fresh vs dense-but-stale** (SSE/MoM/Quest) · 정적 삭제(GHOST/Moore) · 고정 state HW(Pimba) · nesting 선행(MatMamba/MRL/MatryoshkaKV — 전부 write-once) · state 양자화(Quamba — 합성).

## Figure plan (척추)
1. 개념도(write-back cache, hot/cold 데이터플로우) → 2. Motivation 2단(state wall + 삭제 사다리) → 3. Table-1 매트릭스 → 4. B200 속도(스윕+roofline+회계) → 5. c-다이얼 → 6. fp8 사슬 4-cell → 7. 이론 2패널(E-T1 cos + E-T2 이식 곡선 = F5_measured) → 8. 부정 결과 표.
