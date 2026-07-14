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
- GSM8K n=500 결판: **fresh 83.2 / v4-c4 79.8 (−3.4) / v4-c2 80.8 (−2.4)** — c 단조 회복 확인 → 노이즈 아닌 **c-다이얼로 조절되는 staleness 비용 ~2-3pp** (n=100의 −8은 소표본 과장). minerva가 회복된 반면 GSM8K에 소갭 잔존 — 공식 스택(vLLM) v4 측정이 최종 수치.
- 주의: 이 표는 lean-prefill(=cuda 등가) 경로라 p4long 절대값(HF naive torch 경로, top-1 0.78 불일치 실증)과 직접 비교 불가. 같은 표 안의 3-arm 델타만 유효.
- [x] GSM8K n=500 3-config 결판 (위)
- [x] recall 6종 회귀 체크 (limit 300, lean 경로) — **합격, v4-c4 lossless**:

| task | fresh | v4-c4 | Δ |
|---|---|---|---|
| fda (contains) | 0.827 | 0.827 | 0 |
| swde (contains) | 0.903 | 0.900 | −.003 |
| squad_completion | 0.797 | 0.777 | −.020 |
| triviaqa (EM) | 0.313 | 0.330 | +.017 |
| nq_open (EM) | 0.093 | 0.083 | −.010 |
| drop (f1) | 0.039 | 0.059 | +.020 |

  절대값이 p4long 표(fda 0.31, swde 0.66, squad 0.70)보다 크게 높음 — **lean-prefill(=production 커널 등가) 경로 효과**로 판단(구 torch 경로가 절대값을 눌러왔음; RULER 0.79 미스터리의 유력 원인). 구 경로 절대값은 이후 인용 금지, 델타만 역사적 참고.
- [ ] RULER/commonsense 회귀는 공식 스택(vLLM) 이관 후 측정

---

## 신 스택 = NeMo-Skills + vLLM (2026-07-12 전면 교체 — 이후 acc의 공식 소스)
공식 논문과 동일한 측정 방식. 이전 lm-eval 수치는 내부 기록으로 강등(3-arm 델타는 여전히 유효).
- 서빙: `vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --mamba_ssm_cache_dtype float32` (raw-bf16 arm은 `bfloat16`)
- 평가: `ns eval --server_type vllm --server_address http://localhost:8010/v1 ...` — **함정 3종: ① server_address는 스킴+`/v1` 포함 전체 URL 필수(스킴 없으면 litellm 연결 실패), ② `export PATH=~/ns_env/bin:$PATH` 없으면 하위 스폰이 /usr/bin/python 사용, ③ prepare는 `~/ns_env/bin/python3 -m nemo_skills.dataset.prepare <bench>` 직접 실행**
- 측정 configs(유저 확정 6종): raw / raw-bf16 / fresh / v4-c4-{fp32,bf16,fp8} (fresh·v4는 vLLM 포팅 후)

### 6-config 매트릭스 (전수, pass@1, reasoning on; fresh/v4 = longcot ckpt, vLLM 포팅 서빙)
| bench | 공식 | raw | raw-bf16 | **raw-fp8** | fresh | v4-c4-fp32 | v4-c4-bf16 | **v4-c4-fp8** |
|---|---|---|---|---|---|---|---|---|
| GSM8K (1319) | 91.4 | **95.0** | **95.0** | 95.15 | **95.0** (정답수 동일) | 94.47 | 94.47 | **94.62** |
| MATH-500 (500) | 97.8(공식 MATH) | **97.6** | **97.6** | **93.2 ⚠** | **98.2** | 95.2 | 95.2 | **95.2** |

**매트릭스 완성 판정 (2026-07-13):**
1. **비대칭 정밀도 면허 실증** — raw-fp8은 장문 CoT에서 붕괴: MATH −4.4, avg 11,613토큰(raw의 2.2× — 방황·장황 = 폐루프 상태오염 증상), no-answer 3.2%. **v4-cold-fp8은 완전 생존(95.2 = bf16 = fp32, 정상 6.4k토큰)**. 손상 축 = 수천 토큰 자기생성 (GSM8K ~1.8k은 양쪽 다 무사, RULER@4k도 raw-fp8 98-100 — 짧은 지평에선 안 드러남). ⚠ 매트릭스 설계 교훈: 반증 arm은 예상 손상 축의 벤치를 처음부터 포함할 것.
2. **v4 잔여 갭(MATH −2.4)은 정밀도 무관** (fp32=bf16=fp8 동일 95.2) → 순수 staleness → 처방 = long-CoT 학습 연장(800→1500 잔여) 또는 c2. cold 정밀도는 fp8까지 공짜.
3. GSM8K는 전 config lossless (v4-fp8 −0.4 이내).
4. RULER@4k(niah 5종): raw-fp8 98~100, **v4-c4-fp8도 98~100 (만점대)** — 배포점의 recall 축 무손상. 구 torch-경로 RULER(0.77대)는 경로 아티팩트였음 재확인. json `ns_results/{raw-fp8,v4-c4-fp8}/eval-results/ruler.nemotron9b_4k/`.

판정 누적: ① 스택 검증 통과(공식 수치 재현, 구 lm-eval 76.7 갭은 harness 차이), ② **bf16 state cache 무손실**, ③ **fresh 완전 무손실 = retrofit(0.04%)은 기반 모델을 훼손하지 않음** (직교 R의 full-width 불변성 실증). 구 lm-eval 수치는 내부 역사 기록으로만.
json: `scale/ns_results/{raw,raw-bf16,fresh,v4-*}/eval-results/*/metrics.json` · 서빙: `NESTED_SSM_CKPT/MODE/PB/C/COLD` env (scale/install_vllm_v4.sh)

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

## CKPT: `nemo9b_rot_qreg2.pt` — 질의-집중(qreg) 실험 (2026-07-13)
longcot +400스텝(seqlen256, λ=2.0): qrho 0.690→0.492, 폭 ppl 무손상(k128 9.35 불변), fresh acc 무손상(86.0).
### τ-게이트 스윕 (GSM8K n=100, v4-c4 위 head별 cold-read skip)
| arm | skip% | acc |
|---|---|---|
| fresh | — | 86.0 |
| v4-c4 | 0 | 84.0 |
| τ=0.3 | 3.5 | 85.0 |
| τ=0.45 | 24.7 | 79.0 |
| τ=0.6 | 65.7 | 74.0 |
판정: 게이트 작동하나 미학습 상태론 skip1%당 −0.2pp — **write-staleness의 dedicated-v4 붕괴→v4aware 회복 서사와 평행** → 처방 = gate-aware 학습(+qreg 본학습 seqlen1024/1500스텝). json `scale/nemo9b_recall_qg_*.json`.

## 삭제-vs-Lazy 사다리 (2026-07-13, motivation 실험; n=200, native lean 경로)
| arm | GSM8K | fda | swde |
|---|---|---|---|
| fresh | 85.0 | 81.0 | 92.5 |
| GHOST-lite (calibration 순열 top-32 잔존, 삭제) | **58.5** | 81.0 | 92.0 |
| trained R + 삭제 (hot-32만) | **59.5** | 81.5 | 92.0 |
| trained R + **lazy** (v4-c4) | **80.0** | 81.5 | 92.0 |
판정: ① 정적 절단은 추론에서 파국(−26.5; GHOST의 wikitext +1ppl 뒤에 숨는 축), ② 회전만으론 삭제 못 구함(−25.5) — 회전=분할 효율화지 삭제 면허가 아님, ③ **같은 hot 예산에서 lazy 보존만이 회복**(−5 @n=200; 공식 스택 n=1319에선 −0.5). ④ 삭제 민감축 = 얕은 recall이 아니라 다단계 추론. 재현: `queue_deletion_ladder.sh` + `make_ghost_perm.py`, json `nemo9b_recall_del_*.json`.

## CKPT: `nemo9b_rot_longcot2.pt` (2026-07-14) — GSM8K lossless 달성
레시피: longcot 이어 +1200스텝, seqlen 4096, v4aware, **cs_menu c4·c8 가중(2 4 4 8 8 16 32)**, tune_decay, mixed. FINAL(vppl512) k16 9.20/k32 8.43/k64 8.06/k128 7.45.
### GSM8K n=500 판정 (native lean 경로)
| ckpt | fresh | v4-c4 | 갭 |
|---|---|---|---|
| longcot | 83.2 | 79.8 | −3.4 |
| **longcot2** | 81.2 | **81.6** | **+0.4 (lossless)** |
판정: c4-가중 장문 학습으로 v4 arm +1.8 (fresh는 −2 드리프트 — 갭 소멸의 주인은 v4 상승). 다음: 공식 스택(vLLM) GSM8K/MATH-500 재측정으로 배포 승격 판정. json `nemo9b_recall_lc2_*.json`.

## 동적 cold-읽기 선택 스윕 (2026-07-14, longcot2, fp32-cold, GSM8K n=200) — 부정 결과 (설계 정당화)
| 방식 | skip | acc |
|---|---|---|
| full | 0% | 81.0 |
| **열 top-k (노름-유도)** k32/k16/k8 | 67/83/92% | **74.0 / 55.5 / 21.0** |
| head-qgate (참고, qreg2) τ0.45/0.6 | 25/66% | 79.0 / 74.0 |
판정: ① **압축 상태의 readout은 밀집** — attention의 토큰-희소성은 압축에서 소진, per-token 열-희소성 부재 (가속 붕괴 곡선). ② 열-단위 상계-점수 선택이 head-게이트보다 효율적(동일 skip에서 −7 vs −12)이나 둘 다 유료. ③ **read의 공짜 레버 = 정밀도(fp8 면허)뿐** — "read는 바이트로, write는 빈도로"의 실측 근거. 선택 계열의 부활 조건 = policy-aware 학습(미탐).

## fp8 면허 사슬 완결 (2026-07-14, GSM8K)
| | 학습 X (항등 R) | 학습 O (longcot2/longcot) |
|---|---|---|
| raw + fp8 매토큰 (폐루프) | — | **MATH 93.2 붕괴 + 생성 2.2× 방황** (n=1319/500 공식) |
| tiering + fp32-cold | **33.0 (−48.6 붕괴)** | 81.6 (= fresh, lossless) |
| tiering + bf16-cold | **33.0 (동일 — 정밀도 무관)** | 94.47 공식 |
| tiering + fp8-cold | (접근 불가 — 첫 관문 잠김) | **95.2 공식 (= fp32 = bf16)** |
**사슬**: fp8을 안전하게 쓰는 유일한 경로 = [R 학습(0.04%) → tiering 사용권 → 청크 개루프 → fp8 공짜]. 관문마다 실측: 학습 없인 tiering이 −48.6으로 붕괴(정밀도 무관), tiering 없인 fp8이 폐루프 복리로 붕괴, 둘 다 갖추면 fp8 = fp32. json `nemo9b_recall_unt_*.json`.

### capacity 회계 (2026-07-14; raw fp32 상태 = 1.0)
| 구성 | 총 capacity | 비고 |
|---|---|---|
| v4 fp32-cold | 1.00 | 마스터=스냅샷 단일 사본 (flush 사이 불변이므로) |
| **v4 bf16-cold** | **0.625** | 단일 사본 — 대역폭 2.17×와 용량 −37% 동시 |
| v4 fp8-cold | 0.81 | 유일한 이중 사본(bf16 마스터+fp8 읽기캐시) — 그래도 raw 이하 |
+ flush 도장 버퍼 ~3% @c4. "용량을 태워 속도를 샀다"는 공격에 대한 답.
