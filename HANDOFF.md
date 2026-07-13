# HANDOFF — 새 세션 이어받기 (현재 상태만; 역사는 docs/HISTORY.md)

## ★★ RESUME (2026-07-12, compact 직전 고정) — 이 순서로 즉시 재개
**대전제(유저 확정)**: ① acc는 **NeMo-Skills+vLLM 공식 스택으로 전면 교체**(lm-eval 수치는 내부 기록으로 강등), 측정 configs = raw / raw-bf16 / fresh / v4-c4-{fp32,bf16,fp8}. ② speed는 **state-op/e2e 병기, B=256 표준**, eager 금지(fused만). ③ 모든 acc 비교는 3-arm(raw/fresh/v4) 필수.

**재개 액션 (우선순위):**
1. **vLLM raw 서빙 재기동** (죽음 — GPU3를 타 유저가 회수): GPU 빈 것 확인 후
   `HF_HUB_CACHE=/NHNHOME/ARC/arclab/shared/hub CUDA_VISIBLE_DEVICES=<G> ~/vllm_env/bin/vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --mamba_ssm_cache_dtype float32 --port 8010 --gpu-memory-utilization 0.85`
   (이전 기동 성공·응답 검증됨. raw-bf16 arm = `--mamba_ssm_cache_dtype bfloat16` 플래그만.)
2. **[✅ 완료 07-12] ns GSM8K raw = 95.0 — 스택 검증 통과** (공식 91.4 이상, 구 lm-eval 76.7 갭은 harness 차이 확정). 커맨드 함정 확정판:
   `export PATH=~/ns_env/bin:$PATH; ns eval --server_type vllm --server_address http://localhost:8010/v1 --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 --benchmarks gsm8k --output_dir ~/nested_ssm/scale/ns_results/raw`
   (**server_address는 스킴+`/v1` 전체 URL 필수**; PATH 미설정 시 하위 스폰이 /usr/bin/python; prepare는 `~/ns_env/bin/python3 -m nemo_skills.dataset.prepare <bench>` 직접.)
3. **longcot 판정** (큐 대기 중, pid 2236918 / `queue_longcot_verdict.sh`): GPU 창 열리면 자동 — longcot ckpt(재학습 완료됨)의 minerva/gsm8k를 fresh/v4c4/v4c2로. **판정축: minerva v4-c4 −2.9가 회복되나** → EVAL_LEDGER longcot 슬롯 채우기.
4. **vLLM 포팅 (본체)**: `~/vllm_env/lib/python3.12/site-packages/vllm/model_executor/models/nemotron_h.py` + `layers/mamba/`에 R 회전 + v4 tiered decode 이식 → fresh/v4 arm 서빙 + **CUDA-graph e2e**(목표 ~1.45×, busy-share 64% 기준). 참고 구현: `scale/v4_fused_decode.py`(HF관 동일 로직, acc-grade, 함정 주석 포함).
5. **fp8 state-op v2**: bench_v4_decode의 fp8 arm에 e2e-v2식 lean flush(bf16 shadow master) 이식 + Triton readout 튜닝 — 현재 fp8 speed는 미완(v1 flush 낭비로 bf16보다 느림).

**살아있는 프로세스/산출물:**
- ns eval(pid 2236915) — 죽은 서버 참조 중, 죽이고 2번으로 재실행
- longcot verdict 워처(pid 2236918) — 유지(GPU 감시 중)
- `nemo9b_rot_longcot.pt` — long-CoT 재학습 완료본(800스텝 seqlen4096, FINAL k16 8.62 — wikitext 폭은 열세지만 판정축은 minerva 회복)
- vllm_env/ns_env 설치 완료(ns는 legacy-resolver 산물 — 재설치 시 동일 방법)

**최종 확정 수치(변동 없음)**: state-op(SSU앵커, B=256) v4-c4-bf16 **1.62×**/c16 **2.36→2.14×**; e2e(HF python) c16 1.04×(wall)/1.10×(busy)/상한 1.45×(vLLM 목표); acc(구 스택) = EVAL_LEDGER. 디스크: omin 208→78G 정리 완료(컨테이너 하위레이어 이슈는 관리자 몫), 대형 아티팩트는 `/NHNHOME/ARC/arclab/omin/nested_ssm_artifacts/`.

**읽는 순서: `README.md` → `KEY_RESULTS.md`(결과) → 이 파일(현황) → `TRAIN_REPRO.md`(재현) → `EVAL_LEDGER.md`(수치 원장).**
프로젝트: **Elastic Test-Time Memory** — recurrent state를 hot/cold 2티어(dense-but-stale)로 갈라 decode를 가속. 공개 Nemotron-9B의 0.04%(회전 R+decay)만 재학습 → **B200 실측 2.42×, 전 정확도 축 fresh 동률.**

## 지금 실행 중 (2026-07-12)
1. **long-CoT 재학습** (GPU2, `scale/retrain_longcot.log` → `nemo9b_rot_longcot.pt`): seqlen 4096 + cs_menu {2..32} + tune_decay + mixed, 800스텝(200마다 저장). 목적 = minerva_math v4-c4 −2.9 회복(긴 자기생성 stale 분포 미학습이 원인). **완료 시**: EVAL_LEDGER의 longcot 슬롯 채우기 — minerva/GSM8K 3-arm + 짧은답 회귀 체크.
2. **-Base ckpt sanity probe** (GPU3, `scale/probe_base_sanity.log`): 우리 raw(aligned ckpt) 점수가 공식 -Base 수치와 갭(hellaswag 65.7 vs 79.9, GSM8K 76.7 vs 91.4, MATH 32 vs 80.5, RULER 낮음) → -Base가 공식과 맞으면 harness 무죄 = **retrofit 계보를 -Base로 전환 검토**(TRAIN_REPRO 체인 재실행 ~4-7 GPU-h). likelihood형(wino 75.4=75.3, piqa 81.6=81.8)은 이미 일치.

## 다음 후보 (우선순위)
1. longcot 판정 → EVAL_LEDGER 갱신 (+회귀 체크)
2. -Base probe 판정 → 계보 전환 여부 결정
3. PAPER_OUTLINE 재작성(docs/PAPER_OUTLINE.md — 07-07 이후 미갱신, KEY_RESULTS 수치로 배선)
4. fp8 dequant-matvec Triton 커널(analytic 2.8×를 실측으로; 우리쪽만 유리한 커널)
5. Arora 컨택(docs/CONTACT_ARORA.md; 발송 전 Stuart Sul 선연락 → 지도교수 승인)
6. 서빙 시뮬 prefill-겹침 flush / pb 메뉴 재학습(~2.7×)

## 환경 필수 (전체 상세는 TRAIN_REPRO §0)
- **Nemotron = 반드시 `~/nemo_env/bin/python3`** (transformers 5.13 native; remote-code 경로는 깨짐 ppl 3700). 소규모 가족(GDN/GLA/KDA/M2) = 시스템 python3 + fla 4.57.1 핀 + `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.
- `HF_DATASETS_CACHE`/`HF_HUB_CACHE`는 쓰기가능 로컬로(공유캐시 lock PermissionError); 모델은 cache_dir=공유hub로 로드.
- nohup은 `bash -c "cd <절대경로> && ..."` + 로그 절대경로. pkill 자기-매치 주의. 발사 전 `nvidia-smi`(타 유저 GPU 이동/OOM).
- 평가: 생성형은 `run_recall_native.py`(naive 엔진은 생성 degenerate); **정확도 비교는 항상 raw/fresh/v4 3-arm**; 후보 비교 limit≥1000.
- **⚠ fused-커널 빌드 후 함정(07-12 eval, 07-13 학습에서도 실증)**: nemo_env에 커널이 import되면 `is_fast_path_available=True`가 되어 mixer.forward가 `cuda_kernels_forward`를 탐 → torch_forward를 패치하는 v4/fresh 설치와 ActRotMask(R)가 **침묵 무효화**. 증상: (eval) fresh==v4 소수점까지 동일 / (학습) orth=0.0000·k폭 ppl 전부 동일·R 무그래디언트. run_recall_native와 **nemotron_retrofit(학습)** 둘 다 모델 로드 후 `is_fast_path_available=False` 강제로 봉합됨. torch 경로 prefill은 chunked-SSD 중간텐서가 T×chunk_size 비례 — 공유 GPU 자투리에선 `m.chunk_size=64` 축소(exact라 수치 불변).

## 파일 지도
| 위치 | 내용 |
|---|---|
| `README.md` | 입구: 요약·읽기순서·빠른재현·디렉토리 지도 |
| `KEY_RESULTS.md` | ★ 논문 논리 흐름 + 헤드라인 수치 단일 소스 |
| `EVAL_LEDGER.md` | ★ 체크포인트별 3-arm 정확도 원장(태스크별 전체 값) |
| `TRAIN_REPRO.md` | ★ 9B retrofit 재현 레시피(ckpt 4단계 체인 정확 커맨드) |
| `docs/NARRATIVE.md` | 전체 논리(구 PROBLEM_SETTING) — 서사·novelty·HW매핑·§8 |
| `docs/BACKGROUND.md` | 초심자 서사(논문 intro 모체) |
| `docs/ALGORITHM.md` | 알고리즘 설계 역사 v1~v4 + 성질/검증(구 CORE_ALGORITHM) |
| `docs/THEORY.md` | 명제 1-3 + 1′(4가족 불변성) |
| `docs/TRAINING.md` | 소규모(35M~340M) 가족 학습 방법론+결과 원장 |
| `docs/HISTORY.md` | 판정 연대기(세션별 완결 요약 이관본) |
| `docs/PAPER_OUTLINE.md` | 논문 뼈대(⚠ 07-07 이후 미갱신 — 재작성 대기) |
| `docs/MODEL_STATE_TABLE.md` / `docs/CONTACT_ARORA.md` / `docs/archive/` | 참조표 / 컨택 초안 / 폐기·원문 |
| `scale/` | 9B·가족 실험 코드(.py/.sh) + ckpt(.pt) + **results/(json)·logs/(log)** |
| `poc/` | toy PoC(MQAR) 코드·결과 |
| `memory_snapshot/` | 세션 메모리 사본(시간순 판정 로그 = research-direction-elastic-ttm.md) |
