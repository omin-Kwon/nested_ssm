# TRAIN_REPRO — 9B retrofit 처음부터 재현 (single source)

> 목표: 공개 `nvidia/NVIDIA-Nemotron-Nano-9B-v2` → 최종 배포 ckpt `nemo9b_rot_p4long.pt`
> 를 **처음부터 재현**. `.pt`는 git 미포함(ARC 서버 `scale/`에만) → 이 레시피가 유일한 재현 경로.
> 코드: `scale/nemotron_retrofit.py` (학습), `scale/nemo9b_eval.py`·`nemo9b_lmeval.py`·`run_recall_native.py` (평가).

## 0. 환경 (필수)
- **반드시 `~/nemo_env/bin/python3`** — transformers 5.13 native NemotronH. remote-code(trust_remote_code) 경로는 forward 깨짐(ppl 3700). system fla 핀(4.57.1)과 격리된 venv(`--system-site-packages`).
- `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`
- `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (OOM 완화)
- 모델 가중치는 공유 hub 읽기전용에서 로드(`cache_dir=/NHNHOME/ARC/arclab/shared/hub`); **데이터셋 캐시는 쓰기가능 로컬로** — `export HF_DATASETS_CACHE=/home/omin/.cache/huggingface/datasets HF_HUB_CACHE=/home/omin/.cache/huggingface/hub` (공유 캐시 .lock PermissionError 회피).
- 평가용 추가 pip: `wonderwords nltk`(RULER), `antlr4-python3-runtime==4.11 sympy math_verify`(minerva_math), lm-eval 실행 시 humaneval은 `confirm_run_unsafe_code=True`.
- 결정성: `--seed 0`(torch.manual_seed) + 고정 npy. GPU 커널 비결정성으로 소수점 미세차 가능.

## 1. 데이터 준비 (1회)
```bash
cd scale
~/nemo_env/bin/python3 prep_wt103_nemo.py       # -> wt103_train_nemo.npy (136M tok), wt103_val_nemo.npy
                                                # Nemotron tokenizer로 wikitext-103 토큰화
~/nemo_env/bin/python3 prep_fineweb_nemo.py     # -> fineweb_train_nemo.npy (24M) + mixed_train_nemo.npy
                                                # mixed = wt103 24M + fineweb 24M (50/50, 48M tok) — 도메인 다양성 필수
```
- **왜 mixed**: wikitext 단일 도메인 FT는 in-domain(ppl/needle) 올리며 v4 다운스트림 −1pt 드리프트(회전이 도메인 과적합). fineweb 혼합으로 회복.

## 2. ckpt 계보 = 4단계 resume 체인 (raw → p4long)
각 단계 `CUDA_VISIBLE_DEVICES=<idle>` + §0 env. 백본 전체 동결, **회전 R(27L×8G×128²=3.5M) + (2단계부터) decay(A_log/dt_bias ~7k)만** 학습. QR retraction으로 full-width는 수학적으로 보존.

**단계 1 — 회전+QR (raw에서)** → `nemo9b_rot_qr.pt` (~1000스텝)
```bash
~/nemo_env/bin/python3 nemotron_retrofit.py --steps 1000 --batch 4 --seqlen 1024 \
  --lr 3e-3 --cosine --data wt103_train_nemo.npy --log_every 200 --save nemo9b_rot_qr.pt
```
- 폭 메뉴 nested-dropout {16,32,64,128}, 직교 패널티(--orth 1e-2 기본) + 매 스텝 QR 사영. full-width 손실 정확히 0(8.43→8.44). torch 폴백 ~2.1s/step.

**단계 2 — v4-aware (chunked SSD 엔진, 50% 스텝 v4)** → `nemo9b_rot_v4aware.pt` (+2000)
```bash
~/nemo_env/bin/python3 nemotron_retrofit.py --steps 2000 --batch 2 --seqlen 1024 \
  --lr 2e-3 --cosine --v4aware --resume nemo9b_rot_qr.pt --log_every 400 --save nemo9b_rot_v4aware.pt
```
- --v4aware: 자체 `chunked_mixer_forward`(chunked SSD). 50% 스텝을 v4(c 추첨, pb=32)로 → v4 실행 비용을 0으로 학습.

**단계 3 — tune_decay + mixed 데이터** → `nemo9b_rot_p4mixed.pt` (+1200)
```bash
~/nemo_env/bin/python3 nemotron_retrofit.py --steps 1200 --batch 2 --seqlen 1024 \
  --lr 2e-3 --cosine --v4aware --cs_menu 4 8 16 32 64 --tune_decay --data mixed_train_nemo.npy \
  --resume nemo9b_rot_v4aware.pt --log_every 400 --save nemo9b_rot_p4mixed.pt
```
- --tune_decay: A_log/dt_bias를 0.05×lr로 해동(head별 decay가 staleness에 적응) → 배포축 수렴 ~3× 가속.
- --cs_menu에 4 추가(배포점 c=4 학습). --data mixed(도메인 다양성).

**단계 4 — 연장 (동일 레시피)** → `nemo9b_rot_p4long.pt` = 최종 배포 ckpt (+2400)
```bash
~/nemo_env/bin/python3 nemotron_retrofit.py --steps 2400 --batch 2 --seqlen 1024 \
  --lr 2e-3 --cosine --v4aware --cs_menu 4 8 16 32 64 --tune_decay --data mixed_train_nemo.npy \
  --resume nemo9b_rot_p4mixed.pt --log_every 600 --save nemo9b_rot_p4long.pt
```

**단계 5 — [진행 예정] long-CoT 재학습** → `nemo9b_rot_longcot.pt`
minerva_math가 v4-c4에서 −2.9(가장 긴 CoT의 staleness 누적) → **seqlen 4096 + 작은 c 메뉴**로 긴 자기생성 stale 분포를 in-distribution화:
```bash
~/nemo_env/bin/python3 nemotron_retrofit.py --steps 1500 --batch 1 --seqlen 4096 \
  --lr 2e-3 --cosine --v4aware --cs_menu 2 4 8 16 32 --tune_decay --data mixed_train_nemo.npy \
  --resume nemo9b_rot_p4long.pt --log_every 300 --save nemo9b_rot_longcot.pt
```
(스크립트 `scale/retrain_longcot.sh` = smoke 4스텝 후 full 자동.)

### ckpt 계보 표
| ckpt | resume from | 스텝 | 추가 플래그 | 데이터 | 역할 |
|---|---|---|---|---|---|
| nemo9b_rot_qr | (raw) | 1000 | — | wt103 | 회전+QR (탄력성) |
| nemo9b_rot_v4aware | qr | +2000 | --v4aware | wt103 | v4 실행 비용 0 |
| nemo9b_rot_p4mixed | v4aware | +1200 | --v4aware --tune_decay --cs_menu 4.. | mixed | decay적응+도메인 |
| **nemo9b_rot_p4long** | p4mixed | +2400 | 동일 | mixed | **최종 배포** |
| nemo9b_rot_longcot | p4long | +1500 | + seqlen4096 --cs_menu 2.. | mixed | [진행] long-CoT |

## 3. 평가 재현 (최종 ckpt 검증)
```bash
# ppl/needle 판정
~/nemo_env/bin/python3 nemo9b_eval.py --ckpt nemo9b_rot_p4long.pt --tag p4long --cs 4 16
# lm-eval 다운스트림 (3구성: orig/retro_fresh/retro_v4)
~/nemo_env/bin/python3 nemo9b_lmeval.py --config retro_v4 --ckpt nemo9b_rot_p4long.pt --c 4 --cold_bf16 1 --warm 64 --bs 8 --limit 1000
# 생성형 스위트 3-arm (recall/RULER/math/code) — 항상 raw/fresh/v4 세 개
~/nemo_env/bin/python3 run_recall_native.py {raw|fresh|v4} [--c 4 --pb 32] --tasks <...> [--maxlen 4096]
# B200 speed
python3 bench_v4_decode.py --Bs 448 --cs 4 8 16 64   # (system python + fla)
```
- 헤드라인 수치·판정은 `KEY_RESULTS.md`. 시간순 판정 로그는 메모리 `research-direction-elastic-ttm.md`.

## 4. 핵심 함정 (재현 시)
- native decode 브랜치는 self.act를 2D로 호출 → ActRotMask 회전이 조용히 스킵됨. fresh arm도 pb=128 디스패처(`v4_native_decode.install`) 경유해야 R 적용.
- naive_mixer_forward(nemo9b_eval)는 teacher-forcing ppl은 맞지만 **생성에서 degenerate** → 생성형 평가는 native decode(`run_recall_native.py`) 사용.
- v4aware 학습 상삼각 decay exp backward 0×inf=NaN → 마스크를 exp 이전에 -inf로.
- 엔진 자기모순: transformers 순정 torch_forward(7.83) vs naive(6.44) — arms는 자기일관 엔진 내 비율만 유효.
