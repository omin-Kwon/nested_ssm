# HANDOFF — 새 세션 이어받기 (현재 상태만; 역사는 docs/HISTORY.md)

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
