# HANDOFF — 새 세션 첫 읽기용

프로젝트: **Elastic Test-Time Memory** — GPU + CXL-PNM hybrid를 위한 알고리즘–HW co-design.
읽는 순서: 이 파일 → `PROBLEM_SETTING.md`(전체 논리) → `poc/`(코드·결과) → 메모리(`~/.claude/projects/-home-omin-TTT-PNM/memory/`).

---

## 1. 한 문단 요약
Linear-attention/SSM/TTT의 정확도는 **recurrent state 크기 = recall 용량**에 지배되는데(Zoology/BASED), 지금은 그 크기를 **학습 시 하나로 고정**한다. 우리는 state의 차원 $N$(Gated DeltaNet의 key-dim $d_k$)을 **nested(Matryoshka)로 학습**해, 한 모델에서 **추론 시 폭 $k$를 요청별로 다이얼**한다(정확도↔메모리 Pareto, 재학습 0). **hot(앞) 차원은 GPU, cold(뒤) 확장은 capacity-dense CXL-PNM에 상주**시켜, state update(OI≈1, memory-bound)를 near-memory로 처리하고 큰 state를 배치와 안 싸우게 한다. **다이얼 $k$ = GPU/PNM 경계.**

## 2. 확정된 것 (decisions)
- **Motivation:** 고정 state = recall 상한 고정(작으면 잊음/크면 비쌈), 요청마다 필요 memory는 다른데 하나로 못박음. (동기는 학계 공유 — Impossibility Triangle 등 — 이나 "state $N$을 runtime 신축"은 미개척.)
- **알고리즘 수정:** Gated DeltaNet state의 $d_k$를 nested-dropout으로 학습(폭별 loss 합산). 추론 시 폭 $k$ 선택.
- **베이스 모델:** Gated DeltaNet(진입점). 장기 확장: TTT-MLP/Titans memory.
- **HW 매핑:** dense compute→GPU, 대용량·저-OI state→PNM. 정밀도(int8 등) 저수준 디테일은 지금 불필요(hot/cold 모두 fp16 가정).
- **Novelty:** MatMamba(width nest, N 아님)·Nemotron(N 고정)·Pimba(고정 state throughput)와 명시적 구분. state $N$ 자체를 runtime 신축하는 유일 연구.

## 3. PoC로 밝혀진 것 (facts)
- **H1 ✅ runtime dial 실재:** 한 모델에서 recall이 폭 $k$에 단조·완만 증가. (`poc/poc_grid.png`)
- **H2 ✅ capacity 노브:** 0.95 recall에 필요한 $k$가 연관수 $D$에 ~선형(Zoology d≥N을 단일 모델·추론시 재현 — 문헌 미존재 곡선).
- **H3 결론:** naive nesting의 tax(delta 중간 폭)는 **근본 문제 아님 = 학습량 부족.** 6000스텝이면 전용 모델 거의 따라잡음 → **"nesting은 표준 학습만으로 사실상 공짜."**
- **NEGATIVE:** 우리가 새로 설계한 hierarchical/residual nested delta(mode=nesteddelta)는 tax를 **오히려 키움 → 폐기.** exotic recurrence 불필요.

## 4-b. ★ 새 세션 이어받기 (최신 상태, 2026-07-06 밤)

**★ 지금이 H100 서버의 새 세션이라면: 첫 임무 = `H100_BENCH.md` 그대로 실행** (motivation 그림의 H100 실측판; 체크포인트 불필요, ~15분). 이 서버에 `~/.claude` 메모리가 없으면 **`memory_snapshot/`(레포 내)이 메모리 전체의 사본**이다 — research-direction-elastic-ttm.md가 시간순 판정 로그.

**전부 완료, 실행 중인 것 없음 (2026-07-07 오후). T9/T10 배포 판정 + 풀예산 motivation까지 마감:**
- **T10 (Nemotron-9B 공개 하이브리드):** 회전 3.5M + **QR 사영** 1000스텝 → full-width 손실 **정확히 0**(8.43→8.44), k64 +1%/k32 +7%/k16 +18% 회복. soft 패널티는 실패(교훈: 다양체 사영이 정답). remote-code 경로 깨짐 → **native transformers 5.13 격리 venv(~/nemo_env)** 필수. torch 폴백 ~2.1s/step.
- **T9 (GLA-340M 공개 pretrained):** 4.6분 nested-FT로 탄력성 완전 복원(k16 200→13.4), tax +2.9%, 대조군 인과 증명; arms: c1 12-18× → **v4 +9~13%**(from-scratch의 0%까진 FT 예산 더 필요 — 정직 캐비앗).
- **Motivation 풀예산 실측:** 80GB에서 50.4k→1.6k tok/s(slope −1 정확), 커널 BW 4.97TB/s 고정, **160GB(용량 2×)에서 tok/s 동일** — "용량만으론 불변" 완결. `state_bound_motivation.png` 교체됨.
- 곡선 3축 완비: 폭 k(탄력성) / pb(무릎, hot≥12% 안전) / c(로그형 ~+1%p/배).

**T8 완료(2026-07-07): 4가족 일반성 판정 마감** — KDA(Kimi): tax +1.6%, v4 +1.9~3.4%, c1 8.7~11.7×, dedic-v4 +92~115% / M2(Mamba2=Nemotron): tax +7.1%, **v4 +0.05~0.4% 문자 그대로 공짜**, c1 49~64×, dedic-v4 +161~198%. 검증 게이트 4/4 정확. **2×2(게이트 스칼라/채널 × 갱신 delta/additive) 전부 성립 = 프로덕션 선형 계열 전체.** 코드 `scale/{kda,m2}_lm{,_eval}.py`, 데이터 json/log 동명. 상세는 메모리/PAPER_OUTLINE.

**T7 마감 + Motivation 실측 그림 (2026-07-06 밤 완료):**
- **T7 마감(120M):** tax +5.0%(안 줄지만 안 커짐 — "안정 비용" 클레임 확정); arms 재현·증폭(nested-v4 +0.8~4.2% / nested-c1 7-8.6× / dedic-v4 +276~369%, dedic-a-c16=nan); **needle 계측 생존** — nested fresh 0.28(young 1.00), hot-alone nested 0.12 vs dedic 0.00, nested-v4 needle ≈ fresh(老age 무손상), dedic 전 arms 0.00. 데이터: `scale/gdn_lm_eval_lm120-*.json`, `eval_lm120_*.log`.
- **Motivation 실측 그림:** `scale/state_bound_motivation.png` (`bench_state_bound.py`/`plot_state_bound.py`, json은 `bench_state_bound_{fam}_{axis}.json`). tok/s ∝ 1/state slope −1 실측(GDN·GLA 중첩, 예산 8/16/24GB; 24GB에서 22.4k→89 tok/s, batch=1 capacity wall), 커널 BW 2.4-2.6TB/s 고정. 80/160GB 실측은 타 유저 sglang이 GPU 1/3 점유 + GPU 2 사용 금지로 불가 → H100 점은 analytic(M1)과 결합. **한가한 GPU 창이 생기면 `--budgets 80 160`으로 재실측할 것.**
- 다음 후보: 논문 본문 집필(뼈대는 PAPER_OUTLINE, 수치 전부 채워짐), ≥340M nested LM(needle 강화), 커널 실측 §8.
- **완료 시 해야 할 일 (T7 마감):**
  1. `cd scale && CUDA_VISIBLE_DEVICES=? python3 gdn_lm_eval.py --ckpt gdn_lm120_nested.pt --tag lm120-nested --d 768 --layers 12 --heads 12` (+dedic도; 크기 인자 필수 — 이미 CLI 추가됨)
  2. 판정 축: ① 규모에서 tax 거동(35M은 ~4% 정체 — 120M에서 줄어드나), ② arms(v4 비용), ③ **needle 재시도**(120M이면 induction 살아날 가능성 — 35M에선 능력 바닥으로 0.00)
  3. 결과를 메모리 + PAPER_OUTLINE 구멍에 기록, T7 완료 처리

**T7까지의 확정 판정 (메모리 research-direction-elastic-ttm에 상세):**
- GLA 계열 일반성: 탄력성 tax +4.9% / **v4 비용 0%**(additive엔 correction 금기 없음 — 이론 정합) / c1 붕괴 9~15×
- 언어 nesting tax: **~4%로 정체**(2.05B tok에도 불소멸, H3-언어판 기각) — 클레임: "~4% 안정 비용으로 탄력성+v4-실행권"
- GDN-35M 100k: nested k64 15.30 vs dedic 14.74; nested-100k의 k8이 dedic-30k의 k64보다 우수

**환경 필수 (재현 시):** `export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`, transformers==4.57.1 고정, fp32-master+bf16-autocast로 LM 학습(full-bf16은 정체), pkill은 자기-매치 주의, GPU 1/3 사용(0/2는 타 유저).

## 4-a. 무인 세션 완결 요약 (2026-07-06, T1–T5 전부 완료)
- **T3 클로징 결과: 실언어에서 "v4는 nesting이 있어야만 공짜"** — nested-v4 ppl +2~4% vs dedicated-v4 +120% vs nested-c1 ~10×. 폭 탄력성 17.56→16.26. (needle은 35M 능력바닥 — 한계 명시)
- **T4c 헤드라인: 실GDN retrofit** — 회전-only 50초 FT = from-scratch 동등(Qwen3.5 케이스). seed 3개 재현, E4 신축 확인.
- **T2 §8 판정층:** 3.82× 생존(c≥8 stall 0%), read-floor 절벽 실증. **T5:** PAPER_OUTLINE.md 수치 완결.
- 인시던트 3건 처리 기록(OOM/bf16 정체/pkill 자기매치) — scale/ 코드에 교훈 반영.

## 4. 상태 / 다음 스텝
*(경과: elastic-k → placement-first + v4 → toy 판정 → **Phase A 완결**: A2 게이트지문(τ정적·72/15/6/7), A3 실모델 staleness(ppl "τ≫c" 확정 + recall 경고), **A4 진짜 multi-head GDN에서 Q1·Q2 모두 YES**(v4 성립 + hot 티어가 recall 구조, CORE_ALGORITHM §13). §8은 analytic(~3.8×, read-floor 정정 포함)까지. 상세: CORE_ALGORITHM §9–13, PROBLEM_SETTING §8-a/c, 메모리 research-direction-elastic-ttm.)*
1. **T2 §8 판정층:** 큐잉·latency 포함 serving 시뮬 (진행 중)
2. **T3 실언어 nested GDN:** wikitext 학습 → ppl vs 폭 + staleness 최종 판정 (진행 중)
3. **T4:** A4 seed 재현 / E4 mid-stream 신축 / 실GDN retrofit
4. **T5:** 논문 스켈레톤 (PAPER_OUTLINE.md)
5. (이후) controller/elastic dial, TTT-MLP/Titans 확장, 다중노드

## 5. PoC 재현법
```bash
cd /home/omin/TTT-PNM/poc
# H1/H2 핵심 그리드 (additive & delta):
python3 nested_delta_mqar.py --mode delta --steps 3000 --Ds 4 8 16 32 \
    --widths 2 4 8 16 32 --head_dim 32 --n_heads 1        # ~10min on 1 B200
python3 plot_grid.py                                       # -> poc_grid.png
# tax 분석: fixed-width baselines(--fixed_only w) 후 analyze_tax.py
```
- 핵심 코드: `poc/nested_delta_mqar.py` (mode: additive/delta/nesteddelta; nested-dropout on state key-dim; MQAR 데이터젠; train_mixed).
- 주의(중요 버그·교훈): delta-only + pos emb/short conv 없으면 MQAR에서 query-overwrite로 학습 실패 → pos emb + causal short conv + (기본)overwrite 없는 additive 필수.
- 환경: 4× B200, torch 2.10, 순수 PyTorch(no fla). GPU 0/2는 타 유저와 공유될 수 있음(1/3 사용).

## 6. 파일 지도
- `TRAINING.md` — **학습 방법론(공통 레시피+가족별 norm 처리) + 전체 런 결과 원장(탄력성/tax 표, arms 판정 표, retrofit) + 안정화 인시던트 체크리스트.** 논문 §3.1/§4의 원장.
- `BACKGROUND.md` — **초심자용 상세판 (합의된 서사 아크):** 공책→요약판→recall법칙→HW성격→통째-오프로딩 딜레마(A/B)→nesting+v4+계층배치→"10×/+120%/+2%" 인과 3숫자→payoff 두 얼굴→FAQ. 논문 intro/motivation의 모체.
- `PROBLEM_SETTING.md` — Background/Motivation/Core Idea/Novelty/PoC결과/HW매핑/이득평가계획 (전체 논리).
- `CORE_ALGORITHM.md` — **core algorithm 설계 v1 (Nested Block-Delta Memory):** "결합은 티어 안에서만" 원리, naive delta의 3가지 구멍(링크 직렬화·전역 norm 폭 간 불일치·GPU ragged), 블록별 norm/게이트 + block-diagonal delta, 성질 P1–P5, 검증 E1–E4.
- `poc/nested_delta_mqar.py` — 메인 실험 코드. `plot_grid.py`, `analyze_tax.py` — 분석. `poc_grid.png` — money figure. `poc/*.log` — 실험 로그.
- `gdn_out.txt` — Gated DeltaNet 논문 원문(참고).
- 메모리(`~/.claude/.../memory/`): research-direction-elastic-ttm, state-scaling-landscape-findings, related-work-and-novelty, poc-status.
