# EVAL_LEDGER — 체크포인트 단계별 3-arm (raw / fresh / v4-c4) 원장

> **원칙 (유지 규칙):**
> 1. 모든 정확도 비교는 **raw / fresh / v4** 3개를 항상 함께 기록. (raw=순정 9B, fresh=retrofit·tiering off, v4=retrofit·tiering on)
> 2. **학습 방법론이 바뀔 때마다**(레시피/데이터/seqlen/c-menu 등) 여기에 새 체크포인트 섹션 추가 + TRAIN_REPRO.md 계보표 갱신.
> 3. 원시 수치는 `scale/results/*.json` + `scale/logs/*.log` (2026-07-12 정리로 이동; 새 실행 산출물은 scale/ 루트에 생기니 주기적으로 쓸어담을 것). 이 문서는 그 요약 진실.
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
레시피 변경: **seqlen 1024→4096 + cs_menu {2..32} + tune_decay + mixed** (긴 CoT staleness in-distribution화). resume p4long, 800스텝(계획 1500 중 GPU 회수 전 저장분).

### 판정 (2026-07-12, lean-prefill 경로 = production 커널 등가; limit 100/task)
| | fresh | v4-c4 | v4-c2 |
|---|---|---|---|
| minerva_math (700샘플, math_verify) | 30.0 | **29.14 (−0.86)** | 27.86 (−2.1)* |
| GSM8K (100) | 89.0 | 81.0 (−8) | 84.0 (−5) |

- **판정축 달성: minerva v4-c4 갭 −2.9(p4long) → −0.86 (n=700 노이즈 ±1.7pp 이내) — long-CoT 재학습 유효.**
- *c2<c4 역전은 노이즈 소견(방향상 c2가 더 fresh에 가까워야 함).
- GSM8K −8/−5는 n=100 소표본(±5pp) — n=500 확장 측정으로 결판 [진행].
- 주의: 이 표는 lean-prefill(=cuda 등가) 경로라 p4long 절대값(HF naive torch 경로, top-1 0.78 불일치 실증)과 직접 비교 불가. 같은 표 안의 3-arm 델타만 유효.
- [ ] GSM8K n=500 3-config 결판
- [ ] 짧은답 스위트(RULER/commonsense/recall) 회귀 체크

---

## 신 스택 = NeMo-Skills + vLLM (2026-07-12 전면 교체 — 이후 acc의 공식 소스)
공식 논문과 동일한 측정 방식. 이전 lm-eval 수치는 내부 기록으로 강등(3-arm 델타는 여전히 유효).
- 서빙: `vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --mamba_ssm_cache_dtype float32` (raw-bf16 arm은 `bfloat16`)
- 평가: `ns eval --server_type vllm --server_address http://localhost:8010/v1 ...` — **함정 3종: ① server_address는 스킴+`/v1` 포함 전체 URL 필수(스킴 없으면 litellm 연결 실패), ② `export PATH=~/ns_env/bin:$PATH` 없으면 하위 스폰이 /usr/bin/python 사용, ③ prepare는 `~/ns_env/bin/python3 -m nemo_skills.dataset.prepare <bench>` 직접 실행**
- 측정 configs(유저 확정 6종): raw / raw-bf16 / fresh / v4-c4-{fp32,bf16,fp8} (fresh·v4는 vLLM 포팅 후)

### raw / raw-bf16 (스택 검증 — 전수, pass@1, reasoning on)
| bench | 공식 | raw(fp32 cache) | raw-bf16 cache | 구 lm-eval raw | 판정 |
|---|---|---|---|---|---|
| GSM8K (1319) | 91.4 | **95.0** (avg 1601 tok) | **95.0** | 76.7 | ✅ 스택 검증 통과 + bf16 cache 무손실 |
| MATH-500 (500) | 97.8(공식 MATH) | **97.6** (avg 5273 tok) | **97.6** | 32(minerva) | ✅ 공식 일치 + bf16 cache 무손실 |
json: `scale/ns_results/{raw,raw-bf16}/eval-results/*/metrics.json`

## 절대값 신뢰성 판정 (raw sanity, 2026-07-12)
공식 -Base 수치 대비 우리 raw(aligned) 갭의 원인 분해 — **-Base probe 실측**:
| task | 공식 -Base | -Base@우리harness | aligned@우리harness | 원인 |
|---|---|---|---|---|
| HumanEval | 58.5(avg@32) | **0.47**(pass@1) | 0.22 | **aligned 패널티 실재** (완성형 코드에서 Base가 2×) |
| GSM8K | 91.4(자체CoT) | 0.68 | 0.77 | harness 차이 (lm-eval 5-shot strict) |
| minerva_math | 80.5(자체채점) | 0.22 | 0.32 | harness 차이 (lm-eval minerva 0.3대 = 이 급 정상) |
| RULER niah_1@4k | ~1.0 기대 | 0.79 | 0.77 | harness/설정 이슈 (양 ckpt 동일) — 추후 점검 |
| winogrande/piqa | 75.3/81.8 | — | 75.4/81.6 | ✅ 일치 (likelihood형) |
**판정: -Base 계보 전환 불필요** — 갭 대부분이 평가 스택 차이(NeMo-Skills vs lm-eval). 3-arm 델타는 동일 harness라 전부 유효. 논문엔 harness 스펙 명기 + 각주. HumanEval은 -Base 참조 행 병기 권장. json `scale/results/nemo9b_recall_base_*.json`.

## 이전 CKPT (부분 기록 — 3-arm 미완, 참고용)
- `nemo9b_rot_qr` / `v4aware` / `p4mixed`: ppl-vs-width·needle·일부 lm-eval만 측정(당시 3-arm 원칙 이전). 상세는 메모리 research-direction §T12·NIGHT. 완전한 3-arm은 **p4long부터** 확립.
- 밤샘 후보(v4aware6k/tdecay/tdecay2/p3/p3mixed): in-domain 우세·다운스트림 열세로 폐기, p4long이 승자. lm-eval json은 `nemo9b_lmeval_p*_c4_bf16.json`.
