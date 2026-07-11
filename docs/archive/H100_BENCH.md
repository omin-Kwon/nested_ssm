# H100 실측 지침 — Motivation 그림 (state-capacity-bound throughput)

**이 문서는 H100 서버의 새 세션이 단독으로 실행 가능하도록 쓴 지침이다.**
프로젝트 전체 맥락은 `HANDOFF.md` → `PAPER_OUTLINE.md`, 판정 로그는 `memory_snapshot/research-direction-elastic-ttm.md` (이 레포에 스냅샷됨 — 새 서버에 `~/.claude` 메모리가 없어도 여기 다 있음).

## 1. 목적
논문 §2 Motivation의 실측 그림: **"recurrent state가 커지면 고정 HBM에서 decode tok/s가 ∝1/state로 붕괴한다"** 를 실제 H100 1장(80GB가 물리 전부)에서 측정. 지금 그림(`scale/state_bound_motivation.png`)은 B200에서 8/16/24GB 예산 에뮬레이션으로 잰 것 — 기울기 −1 법칙은 실측됐지만, **"진짜 H100 한 장의 전량"** 점이 있으면 반박 불가가 된다.

## 2. 필요한 파일 (체크포인트·데이터 불필요 — 랜덤 가중치 벤치)
- `scale/bench_state_bound.py` (자기완결; fla decode 커널 실측)
- `scale/plot_state_bound.py` (예산 자동 인식)

## 3. 환경 (필수 순서대로)
```bash
pip install torch  # CUDA 지원 빌드
pip install flash-linear-attention==0.5.1 transformers==4.57.1  # transformers 버전 고정 필수(5.x는 fla와 비호환)
export TRITON_PTXAS_PATH=$(command -v ptxas)   # triton이 "Cannot find ptxas" 내면 필수 (예: /usr/local/cuda/bin/ptxas)
```
확인: `python3 -c "from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule; print('ok')"`

## 4. 실행 (유휴 H100 1장에서, 총 ~15분)
```bash
cd scale
# 본선: heads축(순수 capacity 효과; dk축은 fla 커널 열화가 섞임 — 부록용)
CUDA_VISIBLE_DEVICES=<idle> python3 bench_state_bound.py --fam gdn --axis heads --budgets 80 76 72 --ms 1 2 4 8 16 32
CUDA_VISIBLE_DEVICES=<idle> python3 bench_state_bound.py --fam gla --axis heads --budgets 80 76 72 --ms 1 2 4 8 16 32
# 부록: dk축(실전은 법칙보다 더 나쁨을 보이는 각주)
CUDA_VISIBLE_DEVICES=<idle> python3 bench_state_bound.py --fam gdn --axis dk --budgets 72 --ms 1 2 4 8 16
python3 plot_state_bound.py   # -> state_bound_motivation.png
```
- **--budgets 80 76 72인 이유:** 스크립트는 예산 전량을 state로 채우고 커널의 일시 할당(+1~3GB)이 얹힌다. H100 실가용 ~79GB라 80은 OOM으로 자동 skip될 수 있음(잡아서 skip함, 죽지 않음). **성공한 최대 예산 곡선을 본선으로 쓰면 된다.** 다른 프로세스가 GPU에 있으면 그만큼 더 내려야 함(`nvidia-smi`로 확인).
- 결과 json: `bench_state_bound_{gdn,gla}_heads.json`, `..._gdn_dk.json` (실행마다 덮어씀 — B200 결과 보존하려면 실행 전 `mkdir b200_ref && cp bench_state_bound_*.json b200_ref/`).

## 5. 기대 결과 (B200 24GB-예산 레퍼런스; H100은 예산 ~3배·BW ~0.42배)
| state/seq | B200-24GB tok/s | 비고 |
|---|---|---|
| 25MB (m=1) | 22,408 | |
| 50MB | 11,505 | ×0.51 |
| 101MB | 5,627 | ×0.25 |
| 201MB | 2,730 | slope −1 유지 |
| 403MB | 1,217 | |
| 805MB (m=32) | 89 | **batch=1 capacity wall — 법칙보다 더 추락** |
- 검증 포인트: ① log-log 기울기 ≈ −1, ② 커널 달성 BW가 큰 batch에서 평평(H100 HBM3 피크 3.35TB/s의 상당 비율; B200에선 2.4–2.6TB/s), ③ GDN·GLA 곡선 중첩, ④ 최대 m에서 batch가 한 자릿수로 떨어지며 법칙 아래로 추락(용량 벽).
- H100 대략 예상: 76GB 예산이면 m=1에서 B≈2,900; tok/s는 GEMM/커널 시간이 B200보다 길어져 절대값은 낮지만 **기울기 −1과 wall 위치(state ≈ 예산)가 핵심**.

## 6. 끝나면
1. json + png를 이 레포 `scale/`에 커밋(또는 B200 결과와 함께 `plot`을 두 하드웨어 오버레이로 확장 — 선택).
2. `PAPER_OUTLINE.md` §수치 구멍의 "Motivation 실측" 항목에 H100 수치 갱신, `memory_snapshot/research-direction-elastic-ttm.md`(및 새 서버 메모리)에 한 줄 판정 추가.
3. 벤치 설계 배경·주의(예산-에뮬레이션 한계, dk축 커널 열화, kern_frac 필드 의미)는 `bench_state_bound.py` 도크스트링과 HANDOFF §4-b 참조.
