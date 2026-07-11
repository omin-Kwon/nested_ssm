# HISTORY — 판정 연대기 (HANDOFF에서 이관된 역사 지층)

> 시간순 상세 판정 로그의 원본은 메모리 `research-direction-elastic-ttm.md`(= 레포 `memory_snapshot/`).
> 이 파일은 HANDOFF에 쌓였던 세션별 완결 요약을 보존 이관한 것 (2026-07-12 문서 정리).

## 2026-07-11~12: 3-arm 원칙 확립 + 표준 eval 스택 완전 커버
- run_recall_native.py에 raw 모드 추가 — 모든 정확도 비교는 raw/fresh/v4 3-arm.
- RULER 11개 lossless, HumanEval lossless, **minerva_math만 v4-c4 −2.9**(최장 CoT) → long-CoT 재학습(seqlen 4096) 착수.
- GSM8K c-sweep: c16 66.0 → c4 78.0(=fresh) 완전 회복; pb 무효 → 원인=staleness 빈도(c)만.
- raw 타당성 검증: aligned ckpt를 base harness로 평가한 것이 공식 -Base 수치와 갭(hellaswag/MATH/RULER) → -Base probe/계보 전환 검토.

## 2026-07-09: 다운스트림 gap 봉합 + 정밀도 면허 + eval 커버리지
- **warmup window**(첫 W토큰 fresh; 배포 의미론=prefill): 짧은-시퀀스 아티팩트 해소 — fresh 70.42 vs v4+warm 70.35 동률. c축 전체 무손실화(c16+warm 70.25) → **배포점 v4-c16+bf16cold+warm = 실측 2.42×**.
- **비대칭 정밀도 면허 실측**: bf16-fresh 깨끗(재앵커) / scaled-fp8 per-token +5.1%·needle 0.92 ↔ cold snapshot 0%·1.00 — tiering이 quant를 면허(주입빈도 ÷c). fp8-cold 다운스트림 70.07 통과. 최종 스택 ≈2.8×(vs fp32-fresh)/2.2×(vs bf16-fresh).
- **Eval 커버리지**: commonsense 8/8 lossless; recall-intensive 6/6 lossless — naive 엔진 생성 degenerate 함정 발견 → native decode에 v4 구현(v4_native_decode.py, pb128 게이트). 함정: native decode는 act 2D 호출이라 ActRotMask 스킵.

## 2026-07-08 밤: B200 실측 + 밤샘 학습 탐색
- **B200 speed 실측**(bench_v4_decode.py): sync v4 c4/8/16/64 = 1.31/1.53/1.81/2.05×. async flush 기각(BW 포화 — 바이트만 화폐). **bf16-cold: c4 1.92×/c16 2.42×, acc 0**. fla 커널 B×H≤65535.
- **밤샘 레시피 탐색**: tune_decay 발견(수렴 3×), distill은 폭샘플 유지 필수, **wikitext 단일도메인 함정**(in-domain↑·다운스트림 −1pt) → mixed 처방.
- **P4-long**(`nemo9b_rot_p4long.pt`) = 최종 배포 ckpt: v4-c4 ppl 0%, needle 1.00/1.00, 폭 최강, 다운스트림 중립. 교훈: 후보 비교는 limit≥1000.

## 2026-07-08 오전 (T12): 9B 예산부족 확정 + tiering-aware 재현
1. 9B 플레인 3k 연장(`nemo9b_rot_qr3k.pt`): needle v4-c64 0.67→0.96 — c64 약점=학습량 부족 확정.
2. v4-aware R-학습(`nemo9b_rot_v4aware.pt`): k16 9.44→8.10.
3. 판정 스위트: v4-c16 +2.3%/v4-c64 +5.0%; hot-alone 0.08→0.46(5.75×) = T11 tiering-aware 9B 재현.
4. lm-eval 3구성(limit500): orig 71.43/retro_fresh 71.33/retro_v4-c16 67.63 → retrofit 대가=0.

## 2026-07-07 오후 (T9/T10 + motivation 마감)
- **T10 (Nemotron-9B)**: 회전 3.5M + QR 사영 1000스텝 → full-width 손실 정확히 0(8.43→8.44), k64 +1%/k32 +7%/k16 +18%. soft 패널티 실패(다양체 사영이 정답). remote-code 경로 깨짐 → native transformers 5.13 격리 venv(~/nemo_env) 필수.
- **T9 (GLA-340M)**: 4.6분 nested-FT로 탄력성 완전 복원(k16 200→13.4), tax +2.9%; arms c1 12-18× → v4 +9~13%.
- **Motivation 풀예산 실측**: 80GB에서 50.4k→1.6k tok/s(slope −1), BW 4.97TB/s 고정, 160GB(용량 2×)에서 tok/s 동일. `scale/state_bound_motivation.png`.
- 곡선 3축 완비: 폭 k / pb(무릎 hot≥12%) / c(로그형).
- **9B 풀 판정표**: needle fresh 1.00 / v4-c16 0.96 / c64 0.67 / hot-alone 0.25 / c1 0.08.
- **T11 tiering-aware 학습(M2-35M 파일럿)**: v4 비용 ≤0, 무릎 평탄(hot 12% +0.2%), c 외삽 메뉴 4배 밖(c256 +0.5%). 함정: 상삼각 decay exp backward 0×inf=NaN → 마스크를 exp 이전에.
- 2단 배포 서사 확정: GPU-only ~2× → +PNM 3.8×+state 8×.

## 2026-07-07 (T8): 4가족 일반성 마감
KDA: tax +1.6%, v4 +1.9~3.4%, c1 8.7~11.7× / M2: tax +7.1%, v4 +0.05~0.4% 공짜, c1 49~64×. **2×2(게이트 스칼라/채널 × 갱신 delta/additive) 전부 성립.** 코드 `scale/{kda,m2}_lm{,_eval}.py`.

## 2026-07-06 밤 (T7, 120M)
tax +5.0%(안정 비용 확정); arms 재현·증폭; needle 계측 생존(120M). 데이터 `scale/gdn_lm_eval_lm120-*.json`.
- GLA 일반성: tax +4.9% / v4 0% / c1 9~15×. 언어 tax ~4% 정체(H3-언어판 기각).
- GDN-35M 100k: nested-100k k8 > dedic-30k k64.

## 2026-07-06 (무인 세션 T1–T5)
- T3: 실언어 "v4는 nesting이 있어야만 공짜" — nested-v4 +2~4% vs dedicated-v4 +120% vs nested-c1 ~10×.
- T4c: 실GDN retrofit — 회전-only 50초 FT = from-scratch 동등(seed 3 재현).
- T2 §8: 3.82× 생존(c≥8 stall 0%), read-floor 절벽. T5: PAPER_OUTLINE 수치 완결.

## Phase A (toy, ~2026-07-05)
- H1 runtime dial 실재 / H2 capacity 노브(k ∝ D 선형) / H3 tax=학습량 부족.
- NEGATIVE: hierarchical/residual nested delta(v1 Nested Block-Delta) 폐기 — delta의 full-width 결합이 곧 용량 메커니즘.
- v3 placement-first 전환(hot fresh/cold stale), E5/E6 staleness 판정(정확-보정 필수, readout stale은 양성), v4 tier-local writes 도출.
- A2 게이트지문, A3 실모델 staleness, A4 multi-head GDN Q1·Q2 YES. §8 analytic ~3.8×.
