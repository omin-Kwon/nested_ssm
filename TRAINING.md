# TRAINING — nested 학습 방법론 + 전체 런 결과 총괄 (2026-07-07)

논문 §3.1(방법)과 §4(판정)의 원장(ledger). 개념적 "왜"는 THEORY.md, 판정 역사는 CORE_ALGORITHM.md.

## 1. 학습 방법론 (모든 LM 쌍 공통 레시피)

**Nested 학습 = 기존 학습 루프에 딱 한 줄**: 배치의 샘플마다 폭 w를 {8,16,32,64}에서 균등 추첨 →
state key-dim의 앞 w채널만 남기고 q,k를 zero-mask → 그대로 표준 CE 학습.
(폭별 loss 합산 방식 대비 4× 저렴함이 A4에서 확립된 등가 레시피. dedicated 대조군은 `--fixed_only 64`.)

```
optimizer  AdamW(lr 6e-4, wd 0.1, betas (0.9, 0.95)), grad-clip 1.0, warmup 200 (linear)
정밀도     fp32 master weights + bf16 autocast  ← full-bf16은 update rounding으로 정체 (인시던트 기록 §3)
batch/seq  16 × 1024 (120M은 fineweb에서 동일)
데이터     wt103 (136M tok, tokenizer fla-hub/gla-340M-15B, vocab 32k) / 120M은 fineweb-edu 1.5B tok
모델 shape 35M: d512 L6 H8 dk64 dv128 · 120M: d768 L12 H12 (모든 가족 동일 — apples-to-apples)
스텝       35M쌍 30k (tax-vs-budget은 100k) · 120M쌍 120k
```

**가족별 유일한 차이 (마스킹 지점의 norm 처리):**
| 가족 | 커널 | 절단 의미론 처리 |
|---|---|---|
| GDN (Qwen3.5) | chunk_gated_delta_rule | in-kernel qk-l2norm이 활성 prefix를 자동 정규화 — 무수정 |
| KDA (Kimi) | chunk_kda | 동일 (in-kernel l2norm) — GDN 레시피 그대로 |
| GLA | chunk_gla | in-kernel norm 없음 → **명시적 F.normalize(q,k) 필수** (없으면 초기화 폭발) |
| M2 (Mamba2/SSD) | chunk_simple_gla | 동일 — 명시적 qk-norm |

코드: `scale/gdn_lm.py`(gdn_a4.py의 masked_gdn_forward 사용) · `gla_lm.py` · `kda_lm.py` · `m2_lm.py`. 각 파일의 masked_*_forward가 마스킹 한 줄의 실체.

**Retrofit(기존 체크포인트 → nested)**: per-head 회전 R만 학습(identity 초기화, 직교 패널티, backbone 동결), 3000스텝 ≈ **50초**. 스칼라 decay 가족(GDN/M2)에서 이론상 정확(회전 불변성), 실GDN에서 from-scratch와 동등 확인. 채널 decay 가족(GLA/KDA)은 불변성이 깨져 일반 FT 필요. 코드: `scale/gdn_a4_retrofit.py`.

## 2. 전체 학습 결과 원장

### 2-a. 폭 탄력성 + nesting tax (학습 FINAL val ppl; 같은 파일 내 비교만 유효)
| 쌍 (스텝) | nested k8/k16/k32/k64 | dedic k64 | **tax** | dedic 절단 붕괴 |
|---|---|---|---|---|
| GDN-35M (30k) | 20.03/19.33/18.97/18.74 | 18.01 | +4.1% | 83→16 (별도 측정) |
| GDN-35M (100k, 2.05B tok) | k8 16.19 ··· k64 15.30 | 14.74 | +3.8% | — (tax 정체 판정용) |
| GDN-120M (120k, fineweb) | 44.84/41.05/38.35/37.07 | 35.30 | **+5.0%** | k16 128.7, k8 522 |
| GLA-35M (30k) | 22.86/···/21.96 | 20.93 | +4.9% | — |
| KDA-35M (30k) | 21.42/20.69/20.04/19.82 | 19.51 | **+1.6% (최저)** | — |
| M2-35M (30k) | 23.28/22.73/22.44/22.50 | 21.01 | +7.1% | — |

판정: ① 탄력성 단조 — 전 가족 성립. ② tax는 학습량(100k)으로도 규모(120M)로도 소멸하지 않고 **~2~7% 정체**(H3-언어판 기각) → 클레임 "안정적 소액 비용으로 탄력성+v4-실행권 구매". ③ nested-100k의 k8(16.19)이 dedic-30k의 k64(18.01)보다 우수.

### 2-b. Staleness arms 판정 (eval 윈도우 ppl; 검증 게이트 naive=fused 전 모델 통과)
| 가족 | fresh | nested-v4 (c16/c64) | nested-c1 | dedic-v4 | dedic-c1 |
|---|---|---|---|---|---|
| GDN-35M | 16.26 | 16.58/16.84 **(+2~4%)** | 152/198 (~10×) | 35.2 (+120%) | — |
| GDN-120M | 37.07 | 37.37/38.64 **(+0.8~4.2%)** | 261/319 (7-8.6×) | 132.7/165.5 (+276~369%) | 533/733 |
| GLA-35M | 19.61 | 19.60/19.67 **(0%)** | 176/299 (9-15×) | — | — |
| KDA-35M | 17.52 | 17.85/18.12 **(+1.9~3.4%)** | 152/205 (8.7-11.7×) | 33.2/37.3 (+92~115%) | 101/133 |
| M2-35M | 19.82 | 19.83/19.90 **(+0.05~0.4%)** | **979/1263 (49-64×)** | 49.3/56.2 (+161~198%) | 1259/2311 |

- delta 가족(GDN/KDA)엔 (a)-arm도 측정: LM 부하에선 v4와 유사 (correction 금기의 발현은 toy/A4 recall 지문에서 확립 — c≪age와 hot 커버가 ppl에선 가림).
- **2×2 불변 클레임**: 게이트(스칼라/채널) × 갱신(delta/additive) 전 조합에서 nested-v4 ≈ free / nested-c1 붕괴 / dedic-v4 파탄.
- needle(120M만 계측 생존): nested fresh 0.28(young 1.00) / hot-alone nested 0.12 vs dedic 0.00 / nested-v4 0.25-0.31 ≈ fresh / dedic 전 arms 0.00. (32 probes 캐비앗)

### 2-c. Retrofit
실GDN(=Qwen3.5 케이스): 회전-only 50초 FT, k8 recall 0.863 vs from-scratch 0.871 — 전 폭 동등. 3-seed 재현.

## 3. 안정화 인시던트 체크리스트 (재현 시 필독)
1. **full-bf16 AdamW 정체** — 1e-5급 업데이트가 bf16 가중치에서 반올림 소멸 → fp32 master 필수.
2. **GLA/M2 from-scratch 폭발** — in-kernel norm 부재 → 명시적 qk L2-norm.
3. **autocast dtype 불일치** — logsigmoid/normalize가 fp32로 승격 → 커널 진입 전 q,k,gk를 v.dtype으로 cast.
4. **초기 CE O(100+)는 정상** (기본 emb 초기화의 logits std ~23) — 폭발로 오판 금지.
5. fp32-logits upcast OOM → bf16 CE. / triton "Cannot find ptxas" → TRITON_PTXAS_PATH. / transformers==4.57.1 고정.
6. eval 윈도우 ≠ 학습 FINAL 윈도우 — **ppl은 같은 표 안에서만 비교**.
