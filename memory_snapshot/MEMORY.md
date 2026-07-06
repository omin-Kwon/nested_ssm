# Memory Index

**새 세션 첫 행동: `/home/omin/TTT-PNM/HANDOFF.md` §4-b를 먼저 읽을 것.** 다음 행동·환경 필수사항이 거기 있음. **프로젝트는 H100 서버로 이관 중(2026-07-06)**: 레포는 github.com/SNU-ARC/ttt_pnm, 새 서버 첫 임무는 레포의 `H100_BENCH.md`(motivation 그림 H100 실측). 메모리 전체가 레포 `memory_snapshot/`에 스냅샷되어 있음(양쪽에서 갱신 시 동기화할 것). 큰 아티팩트(*.pt, *.npy)는 git 미포함 — 이 서버(ARC, 10.55.3.200)에만 있음.

- [Research direction: Elastic Test-Time Memory](research-direction-elastic-ttm.md) — **프로젝트 전체 판정 로그(시간순, 최신이 우선)**: 방향 확정 → v1 반증 → v2/v4 → placement-first 전환 → toy/A4/실언어 판정 → §8 시뮬 → retrofit → GLA 일반성 → tax 정체 → **T7 마감(120M: tax +5%, dedic-v4 +276~369%, needle 생존) → motivation 실측 그림(∝1/state slope −1)**. 모든 헤드라인 수치 포함
- [State-scaling landscape findings](state-scaling-landscape-findings.md) — 검증된 문헌 지형(state=recall 법칙, Pimba 갭), 인용 목록
- [Related work and novelty](related-work-and-novelty.md) — novelty 판정(MatMamba/Nemotron/StateX 구분), controller 문헌, 인용 지침
- [PoC status and findings](poc-status.md) — toy PoC(H1/H2/H3) 상세 + 버그 교훈(pos-emb/short-conv 필수 등)

프로젝트 문서(레포): HANDOFF(이어받기) / H100_BENCH(H100 실측 지침) / BACKGROUND(초심자용 서사) / THEORY(명제 3개+motivation 수치) / PAPER_OUTLINE(논문 뼈대, 수치 배선 완료) / CORE_ALGORITHM(설계 역사·판정) / PROBLEM_SETTING(§8 포함 전체 논리)
