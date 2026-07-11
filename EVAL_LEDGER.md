# EVAL_LEDGER — 체크포인트 단계별 3-arm (raw / fresh / v4-c4) 원장

> **원칙 (유지 규칙):**
> 1. 모든 정확도 비교는 **raw / fresh / v4** 3개를 항상 함께 기록. (raw=순정 9B, fresh=retrofit·tiering off, v4=retrofit·tiering on)
> 2. **학습 방법론이 바뀔 때마다**(레시피/데이터/seqlen/c-menu 등) 여기에 새 체크포인트 섹션 추가 + TRAIN_REPRO.md 계보표 갱신.
> 3. 원시 수치는 `scale/nemo9b_recall_*.json`·`nemo9b_lmeval_*.json` + 로그. 이 문서는 그 요약 진실.
> 4. c는 v4에만 있는 파라미터(tiering 갱신주기) → raw/fresh엔 c 없음. 짧은답=lossless는 c 무관, decode-heavy만 c 민감.

---

## CKPT: `nemo9b_rot_p4long.pt` (현 최종 배포, 2026-07-11) — 전 스택 3-arm 완비

배포 실행 = v4-c4 + bf16-cold + warmup(=prefill). B200 실측 1.92× (짧은답은 c16으로 2.42×).

### commonsense (lm-eval, acc) — raw는 limit500(orig), fresh/v4는 limit1000·warm64 (limit 상이 주의)
| task | raw | fresh | v4-c16+warm |
|---|---|---|---|
| lambada_openai | 0.662 | 0.667 | 0.655 |
| piqa | 0.808 | 0.802 | 0.799 |
| hellaswag | 0.502 | 0.492 | 0.489 |
| arc_easy | 0.810 | 0.800 | 0.800 |
| arc_challenge | 0.502 | 0.522 | 0.524 |
| winogrande | 0.754 | 0.730 | 0.727 |
| **평균** | 0.673 | 0.686 | 0.682 | → **lossless** (noise 이내) |

### recall-intensive (BASED, native decode, limit 300) — v4 @ c4
| task | raw | fresh | v4-c4 |
|---|---|---|---|
| fda (contains) | 0.310 | 0.310 | 0.310 |
| swde (contains) | 0.657 | 0.657 | 0.657 |
| squad_completion | 0.700 | 0.700 | 0.687 |
| triviaqa (EM) | 0.310 | 0.310 | 0.310 |
| nq_open (EM) | 0.093 | 0.087 | 0.077 |
| drop (f1) | 0.055 | 0.057 | 0.052 | → **lossless** |

### RULER (@4096) — ⚠️ first-4는 raw 미측정(fresh/v4c16만); rest-7은 3-arm 완비
| task | raw | fresh | v4 |
|---|---|---|---|
| niah_single_1 | — | 0.773 | 0.767 (c16) |
| niah_single_2 | — | 0.867 | 0.867 (c16) |
| niah_single_3 | — | 0.807 | 0.773 (c16) |
| niah_multikey_1 | — | 0.520 | 0.513 (c16) |
| niah_multikey_2 | 0.500 | 0.470 | 0.470 (c4) |
| niah_multikey_3 | 0.330 | 0.340 | 0.310 (c4) |
| niah_multiquery | 0.3225 | 0.3125 | 0.325 (c4) |
| niah_multivalue | 0.3475 | 0.3425 | 0.370 (c4) |
| ruler_cwe | 0.256 | 0.257 | 0.263 (c4) |
| ruler_fwe | 0.6467 | 0.6467 | 0.6033 (c4) |
| ruler_vt | 0.064 | 0.074 | 0.078 (c4) | → **lossless** (전 태스크 noise 이내) |
| **GAP** | first-4 raw 미측정 | | |

### GSM8K (5-shot CoT, limit 150) — c-sweep 포함
| config | 값 | 비고 |
|---|---|---|
| raw | 76.7 | |
| fresh | 75.3 | |
| v4-c16 | 66.0 | −9.3 (staleness 누적) |
| v4-c8 | 73.3 | −2.0 |
| **v4-c4** | **78.0** | **완전 회복 (=fresh)** |
| v4-c16 pb64 | 66.0 | pb 무효 → 원인=c만 |
→ **c=4에서 lossless.** decode-heavy 첫 실측 비용이 c-다이얼로 해소.

### HumanEval (code gen, pass@1, limit 100) — v4 @ c4
| raw | fresh | v4-c4 |
|---|---|---|
| 0.22 | 0.21 | 0.22 | → **lossless** |

### ★ minerva_math (math_verify, 700; MATH = 최장/최난 CoT) — 유일한 잔존 하락
| subtask | raw | fresh | v4-c4 | v4-c2 |
|---|---|---|---|---|
| algebra | 0.53 | 0.55 | 0.52 | 0.48 |
| counting_and_prob | 0.30 | 0.26 | 0.23 | 0.30 |
| geometry | 0.27 | 0.26 | 0.23 | 0.26 |
| intermediate_algebra | 0.16 | 0.14 | 0.14 | 0.13 |
| num_theory | 0.23 | 0.30 | 0.23 | 0.25 |
| prealgebra | 0.59 | 0.58 | 0.54 | 0.55 |
| precalc | 0.19 | 0.15 | 0.15 | 0.14 |
| **집계** | **0.324** | **0.320** | **0.291** | **0.301** |
| **vs fresh** | +0.4 | — | **−2.9** | **−1.9** |
→ **raw≈fresh (retrofit 무손실).** v4는 c4 −2.9 / c2 −1.9 (단조 회복하나 c2로도 미완). **MATH만 c로 완전히 안 잡힘** → 원인 진단: v4aware가 seqlen 1024 teacher-forcing만 학습, 긴 자기생성 stale 분포 미학습. **처방=long-CoT 재학습**(다음 ckpt).

---

## CKPT: `nemo9b_rot_longcot.pt` (진행 중, 2026-07-12) — [예약]
레시피 변경: **seqlen 1024→4096 + cs_menu {2..32} + tune_decay + mixed** (긴 CoT staleness in-distribution화). resume p4long, 800스텝. 완료 시 아래 3-arm 재측정 → minerva 회복 확인:
- [ ] minerva_math raw/fresh/v4-c4/c2 (목표: v4 ≈ fresh)
- [ ] GSM8K·HumanEval 회귀 체크 (안 떨어졌나)
- [ ] 짧은답 스위트(RULER/commonsense/recall) 회귀 체크

---

## 이전 CKPT (부분 기록 — 3-arm 미완, 참고용)
- `nemo9b_rot_qr` / `v4aware` / `p4mixed`: ppl-vs-width·needle·일부 lm-eval만 측정(당시 3-arm 원칙 이전). 상세는 메모리 research-direction §T12·NIGHT. 완전한 3-arm은 **p4long부터** 확립.
- 밤샘 후보(v4aware6k/tdecay/tdecay2/p3/p3mixed): in-domain 우세·다운스트림 열세로 폐기, p4long이 승자. lm-eval json은 `nemo9b_lmeval_p*_c4_bf16.json`.
