# Memory Index

**새 세션 첫 행동: `/home/omin/nested_ssm/KEY_RESULTS.md`(★ 헤드라인 수치 단일 소스) → `HANDOFF.md` 순으로 읽을 것.** 현 상태(2026-07-09): **9B retrofit 아크 사실상 완결** — 최종 ckpt `scale/nemo9b_rot_p4long.pt`, 배포점 v4-c16+bf16cold+warm = **B200 실측 2.42×, 전 정확도 축 lossless**(ppl 0%/needle 1.00/commonsense 8/8/recall 6/6). 남은 것: RULER·full-set·MMLU/GSM8K, fp8 dequant 커널, PAPER_OUTLINE 배선, Arora 컨택. 레포 = github.com/omin-Kwon/nested_ssm (모든 문서·코드·결과 push됨; 메모리 사본 = `memory_snapshot/`, 양쪽 갱신 시 동기화). 큰 아티팩트(*.pt, *.npy)는 git 미포함 — ARC 서버(10.55.3.200)에만 있음.

- [Research direction: Elastic Test-Time Memory](research-direction-elastic-ttm.md) — **프로젝트 전체 판정 로그(시간순, 최신이 아래)**: 방향 확정 → v1 반증 → v2/v4 → placement-first 전환 → toy/A4/실언어 판정 → §8 시뮬 → retrofit → GLA 일반성 → T7 마감(120M) → motivation 실측(slope −1) → T8 4가족 → T9/T10 공개 pretrained retrofit → 9B 풀판정 → T11 tiering-aware → T12(예산부족 확정·v4aware) → **B200 실측 Pareto(2.42×)·async 기각·bf16/fp8-cold·warmup gap 봉합·비대칭 정밀도 면허·분리 ablation·NIGHT 레시피 탐색(tdecay/도메인함정)·recall 스위트 완결(native decode v4)**. 모든 헤드라인 수치 포함
- [State-scaling landscape findings](state-scaling-landscape-findings.md) — 검증된 문헌 지형(state=recall 법칙, Pimba 갭), 인용 목록
- [Related work and novelty](related-work-and-novelty.md) — novelty 판정(MatMamba/Nemotron/StateX/SSE/MoM 구분 + **GDN-2/Mamba-3 신규**), controller 문헌, 인용 지침
- [PoC status and findings](poc-status.md) — toy PoC(H1/H2/H3) 상세 + 버그 교훈(pos-emb/short-conv 필수 등)

프로젝트 문서(레포): **KEY_RESULTS(★ 논문 논리 흐름·헤드라인 수치 단일 소스)** / **TRAIN_REPRO(★ 9B retrofit 처음부터 재현 레시피: 데이터prep+ckpt 4단계 resume 체인 raw→p4long 정확 커맨드+계보표, .pt는 git미포함이라 유일 재현경로)** / HANDOFF(이어받기) / TRAINING(학습 방법론+결과 원장) / THEORY(명제 1-3 + 1′) / PAPER_OUTLINE(뼈대) / BACKGROUND(초심자 서사) / CORE_ALGORITHM(설계 역사) / PROBLEM_SETTING(§8 전체 논리) / H100_BENCH(선택) / CONTACT_ARORA(컨택 메일+체크리스트) / scale/MODEL_STATE_TABLE(실모델 state표)
