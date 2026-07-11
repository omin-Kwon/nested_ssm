# 실제 모델별 고정 state 크기와 80GB 최대 batch (config.json 실측 기반, 2026-07-06)

전제: **state 크기는 pre-train 시점에 (레이어수 × 헤드수 × d_k × d_v)로 완전 고정** — request가 1토큰이든 100만 토큰이든 동일. state dtype은 fp32(fla 커널·HF cache 관례; bf16이면 ÷2 → batch ×2, 법칙 불변). weights bf16, workspace ~2GB 가정. 하이브리드는 어텐션 레이어 KV가 컨텍스트에 비례해 **추가**됨(별도 표기).

| 모델 (가족) | 파라미터/weights | 선형 레이어 구성 | **state/req (fp32)** | **80GB 최대 batch** | 이때 state 점유 |
|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B (GDN 3:1) | 80B MoE / bf16 160GB | 36 GDN: 32vh×128×128 | **75.5MB** (+KV 24.6KB/tok×12층) | 80GB 1장 **불가**(fp8+TP2 필요) | — |
| Kimi-Linear-48B-A3B (KDA≈GDN) | 48B MoE / bf16 96GB, **fp8 48GB** | 20 KDA: 32h×128×128 | **41.9MB** (+MLA KV 1.15KB/tok×7층) | fp8 기준 **≈716** (32k ctx KV 포함 시 ≈100) | 30GB=37% |
| GLA-1.3B (fla-hub, 우리 실험 계열) | 1.3B / 2.6GB | 24 GLA: 4h×256×512 | **50.3MB** | **≈1,499** | 75GB=94% |
| Mamba2-2.7B (state-spaces) | 2.7B / 5.4GB | 64 SSD: d_inner5120×N128 | **167.8MB** | **≈432** | 73GB=91% |
| Nemotron-Nano-9B-v2 (Mamba2 하이브리드) | 8.9B / 17.8GB | 27 Mamba2: 128h×80×128 (+4 attn) | **141.6MB** (+KV 8.2KB/tok×4층) | **≈424** (32k ctx 시 ≈150) | 60GB=75% |
| RWKV7-World3-2.9B | 2.9B / 5.8GB | 32 WKV: 40h×64×64 | **21.0MB** | **≈3,438** | 72GB=90% |

계산식: batch = (80GB − weights − 2GB) ÷ state/req. conv state(<1%)는 무시. 출처: 각 모델 HF config.json (fetch 스크립트는 이 파일과 같은 커밋의 로그 참조).

## 읽는 법 (motivation 배선)
1. **작은 모델일수록 GPU의 75–94%가 weights가 아니라 state** — 선형 모델 serving의 용량 소비자는 state다.
2. **실모델 state는 21MB(RWKV7)–168MB(Mamba2)** — 우리 실측 그림(`state_bound_motivation.png`)의 x축(25–805MB)이 정확히 이 실구간을 덮음.
3. state는 pre-train에 고정되므로 **recall을 올리려면(N↑, Zoology d≥N) state를 m배 키워야 하고 → batch가 m분의 1로 붕괴**(실측 slope −1). 예: GLA-1.3B 1,499 → 8×에서 187 → 16×에서 94; Mamba2-2.7B 432 → 54 → 27.
4. RWKV7의 3,438 batch는 "state를 작게 고정한 대가"의 반대면 — recall 상한을 낮춰 throughput을 산 것(BASED tradeoff의 실물).
