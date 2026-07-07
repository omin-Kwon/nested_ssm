# Memory Index

**새 세션 첫 행동: `/home/omin/nested_ssm/HANDOFF.md` §4-b를 먼저 읽을 것.** 실행 중 런·재발사 커맨드·T12 프로토콜(9B v4-aware 재학습 → nemo9b_eval → lm-eval 3구성)·환경 필수사항 전부 거기 있음. 레포 = github.com/omin-Kwon/nested_ssm (모든 문서·코드·결과 push됨; 메모리 사본 = `memory_snapshot/`, 양쪽 갱신 시 동기화). 큰 아티팩트(*.pt, *.npy)는 git 미포함 — ARC 서버(10.55.3.200)에만 있음. H100 실측(H100_BENCH.md)은 B200 idle 80/160GB 실측 완료로 이제 선택사항.

- [Research direction: Elastic Test-Time Memory](research-direction-elastic-ttm.md) — **프로젝트 전체 판정 로그(시간순, 최신이 우선)**: 방향 확정 → v1 반증 → v2/v4 → placement-first 전환 → toy/A4/실언어 판정 → §8 시뮬 → retrofit → GLA 일반성 → tax 정체 → T7 마감(120M) → motivation 풀예산 실측(slope −1, 용량2×→불변) → T8 4가족 → pb무릎/c로그 곡선 → T9/T10 공개 pretrained retrofit(GLA-340M, Nemotron-9B QR) → **9B 풀판정(needle v4-c16 0.96) → T11 tiering-aware 학습 성공(v4비용≤0, c외삽 4배)** → 2단 배포 서사·SSE/MoM 구분. 모든 헤드라인 수치 포함
- [State-scaling landscape findings](state-scaling-landscape-findings.md) — 검증된 문헌 지형(state=recall 법칙, Pimba 갭), 인용 목록
- [Related work and novelty](related-work-and-novelty.md) — novelty 판정(MatMamba/Nemotron/StateX 구분), controller 문헌, 인용 지침
- [PoC status and findings](poc-status.md) — toy PoC(H1/H2/H3) 상세 + 버그 교훈(pos-emb/short-conv 필수 등)

프로젝트 문서(레포): HANDOFF(이어받기·T12 프로토콜) / TRAINING(학습 방법론+전체 결과 원장) / THEORY(명제 1-3 + 1′ 가족불변성) / PAPER_OUTLINE(뼈대, 수치 배선 완료) / BACKGROUND(초심자 서사) / CORE_ALGORITHM(설계 역사) / PROBLEM_SETTING(§8 전체 논리) / H100_BENCH(선택) / CONTACT_ARORA(컨택 메일 최종본+체크리스트) / scale/MODEL_STATE_TABLE(실모델 state표)
