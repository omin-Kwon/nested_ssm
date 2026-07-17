# nested_ssm — Elastic Test-Time Memory (tiered recurrent state)

**한 문단**: Linear-attention/SSM의 recall 용량은 recurrent state 크기가 결정하는데, decode에서 그 state를 매 토큰 통째로 read+write하는 것이 memory-bound 병목이다(OI≈1, 9B에서 시퀀스당 141.6MB; B200 실측 state-op 점유 57.6% @B=256). 우리는 state 차원을 중요도순으로 정렬(nested)한 뒤 **앞쪽 hot 차원은 매 토큰 fresh, 뒤쪽 cold 차원은 c토큰마다 몰아서 갱신(읽기는 stale+decay 보정)** 하는 *dense-but-stale* 티어링을 제안한다. 공개 **Nemotron-Nano-9B-v2의 0.04% 파라미터**(회전 R 3.5M)만 몇 시간 재학습하면:

> **B200 실측 decode 1.92×(c4)~2.42×(c16)**, **공식 평가 스택(NeMo-Skills+vLLM) 3-arm 전 벤치 lossless** — 배포점 v4-c4-fp8: GSM8K 94.6 / MATH-500 95.2 / RULER@4k 98–100 (raw-fp8은 장문 CoT에서 붕괴 = 비대칭 정밀도 면허 실측). GSM8K는 longcot2 ckpt로 v4-c4 = fresh 달성(공식 재측정 대기). 이론 실측: 중요도는 loss가 정의(E-T1 4중 수렴) + 학습 기저의 도메인 이식성(E-T2). 장기 확장: cold를 CXL-PNM에 상주(+3.8×, state capacity 8×).

## 처음 읽는 순서
1. **`KEY_RESULTS.md`** — 논문 논리 흐름 그대로의 결과 정리 (문제→방법→속도→정확도→한정)
2. **`HANDOFF.md`** — 지금 실행 중/다음 액션/환경 필수
3. `docs/NARRATIVE.md` — 전체 논리 (동기·novelty·HW 매핑·§8 이득 분석)
4. **`TRAIN_REPRO.md`** — 처음부터 재현 (데이터 prep → ckpt 4단계 체인 정확 커맨드)
5. **`EVAL_LEDGER.md`** — 체크포인트별 raw/fresh/v4 3-arm 정확도 원장 (태스크별 전체 값)
6. 깊이: `docs/ALGORITHM.md`(설계 역사) · `docs/THEORY.md`(명제) · `docs/HISTORY.md`(판정 연대기) · `memory_snapshot/research-direction-elastic-ttm.md`(시간순 상세 로그)

## 빠른 재현
```bash
# 환경·데이터·학습 체인: TRAIN_REPRO.md 참조 (요약: ~/nemo_env/bin/python3 필수)
cd scale
# 최종 ckpt 평가 (ppl+needle):
~/nemo_env/bin/python3 nemo9b_eval.py --ckpt nemo9b_rot_p4long.pt --tag check --cs 4 16
# 3-arm 생성형 평가 (raw/fresh/v4):
~/nemo_env/bin/python3 run_recall_native.py v4 --c 4 --pb 32 --tasks gsm8k --limit 150
# B200 speed bench (system python + fla):
python3 bench_v4_decode.py --Bs 448 --cs 4 8 16 64
```

## 디렉토리 지도
```
KEY_RESULTS.md  EVAL_LEDGER.md  TRAIN_REPRO.md  HANDOFF.md   ← 4대 현재 문서 (루트)
docs/           서사·이론·역사·논문뼈대·참조 (NARRATIVE/ALGORITHM/THEORY/TRAINING/HISTORY/…)
docs/archive/   폐기·원문 (H100_BENCH, GDN 논문 전문)
scale/          9B retrofit + 4가족(GDN/GLA/KDA/M2) 실험 코드·ckpt
scale/results/  평가 결과 json     scale/logs/  실행 로그
poc/            toy PoC (MQAR nested-state)
memory_snapshot/ 세션 메모리 사본 (판정 로그·문헌·novelty)
```

## 핵심 코드 (scale/)
| 파일 | 역할 |
|---|---|
| `nemotron_retrofit.py` | 9B retrofit 학습 (R+QR; --v4aware/--tune_decay/--cs_menu/--data/--save_every) |
| `nemo9b_eval.py` | ppl/needle 판정 (v4/c1/lag/cold_bf16/fp8/warm 의미론) |
| `nemo9b_lmeval.py` | lm-eval 다운스트림 harness |
| `run_recall_native.py` | **3-arm(raw/fresh/v4) 생성형 평가** — native decode 경로 |
| `v4_native_decode.py` | native decode의 v4 구현 = 배포 프로토타입 (pb128 게이트 검증) |
| `bench_v4_decode.py` | B200 decode speed bench (fresh/v4/async/bf16cold/…) |
| `nemo9b_rot_p4long.pt` | 현 최종 배포 ckpt (in-repo 백업, 14MB) |

모델: `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (백본 동결 retrofit). 하드웨어: 4× B200 공유 서버.
