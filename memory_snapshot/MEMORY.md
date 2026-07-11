# Memory Index

**새 세션 첫 행동: `README.md` → `KEY_RESULTS.md` → `HANDOFF.md` 순으로 읽을 것 (2026-07-12 레포 정리: 서사/이론/역사 문서는 `docs/`로, 로그/json은 `scale/logs/`·`scale/results/`로 이동; HANDOFF는 현황만·역사는 docs/HISTORY.md).** 현 상태(2026-07-12): 최종 ckpt `scale/nemo9b_rot_p4long.pt`, 배포점 v4-c16+bf16cold+warm = **B200 실측 2.42×, 표준 스택 3-arm lossless**(commonsense 8·recall 6·RULER 11·GSM8K·HumanEval; **minerva_math만 −2.9 → long-CoT 재학습(seqlen4096) 진행 중**). 병행: **-Base ckpt sanity probe**(우리 raw=aligned가 공식 -Base 수치와 갭 → 계보 전환 검토). 남은 것: PAPER_OUTLINE 재작성, fp8 dequant 커널, Arora 컨택. 레포 = github.com/omin-Kwon/nested_ssm. **9B retrofit ckpt들(각 14MB)은 git에 백업됨**; 큰 아티팩트(gla/m2 .pt, *.npy)만 ARC 서버(10.55.3.200) 전용.

- [Research direction: Elastic Test-Time Memory](research-direction-elastic-ttm.md) — **프로젝트 전체 판정 로그(시간순, 최신이 아래)**: 방향 확정 → v1 반증 → v2/v4 → placement-first 전환 → toy/A4/실언어 판정 → §8 시뮬 → retrofit → GLA 일반성 → T7 마감(120M) → motivation 실측(slope −1) → T8 4가족 → T9/T10 공개 pretrained retrofit → 9B 풀판정 → T11 tiering-aware → T12(예산부족 확정·v4aware) → **B200 실측 Pareto(2.42×)·async 기각·bf16/fp8-cold·warmup gap 봉합·비대칭 정밀도 면허·분리 ablation·NIGHT 레시피 탐색(tdecay/도메인함정)·recall 스위트 완결(native decode v4)**. 모든 헤드라인 수치 포함
- [State-scaling landscape findings](state-scaling-landscape-findings.md) — 검증된 문헌 지형(state=recall 법칙, Pimba 갭), 인용 목록
- [Related work and novelty](related-work-and-novelty.md) — novelty 판정(MatMamba/Nemotron/StateX/SSE/MoM 구분 + **GDN-2/Mamba-3 신규**), controller 문헌, 인용 지침
- [PoC status and findings](poc-status.md) — toy PoC(H1/H2/H3) 상세 + 버그 교훈(pos-emb/short-conv 필수 등)

프로젝트 문서(레포): **README(입구)** / **KEY_RESULTS(★ 헤드라인 수치 단일 소스)** / **TRAIN_REPRO(★ 재현 레시피: ckpt 4단계 체인)** / **EVAL_LEDGER(★ 체크포인트별 3-arm 원장)** / HANDOFF(현황만) / docs/: NARRATIVE(구 PROBLEM_SETTING)·BACKGROUND·ALGORITHM(구 CORE_ALGORITHM+최종형 1페이지)·THEORY·TRAINING(소규모 원장)·HISTORY(판정 연대기)·PAPER_OUTLINE(⚠미갱신)·MODEL_STATE_TABLE·CONTACT_ARORA·archive/(H100_BENCH, GDN 원문)
