# Simran Arora 컨택 메일 초안 (2026-07-07)

**보내기 전 체크리스트**: ① 지도교수(Jae W. Lee 교수님) 승인 + cc 여부 결정, ② 첨부물 준비(1-pager PDF, Fig 2장: `state_bound_motivation.png` + 9B retrofit 표), ③ arXiv 프리프린트가 있으면 링크 교체(현재는 "in preparation"), ④ 그녀의 최신 논문(Cartridges ICLR'26) 한 번 훑고 문장 정확성 확인.

---

Subject: **Extending the recall–throughput tradeoff to a runtime dial: importance-ordered recurrent state across memory tiers**

Dear Prof. Arora,

I'm Omin Kwon, a PhD student in Jae W. Lee's group at Seoul National University. My research is on accuracy-preserving memory systems for efficient LLM serving: MAGE (arXiv:2602.14209) showed that in block-diffusion LLMs the sparse KV subset is identifiable before decoding and can be reused near-losslessly, and HERALD (arXiv:2606.21633, w/ Ion Stoica) turned that into a CPU–GPU tiered KV serving system (2.47× throughput at 5–10% KV budget).

I'm writing because our current project builds directly on your Zoology/Based results — and I believe extends them along an axis you haven't published on. Your work established that recall is bought with state size, fixed at training time. We make that state **elastic at inference**: nested (Matryoshka-style) training orders the state's key dimensions by importance in a single model, so a runtime "width dial" traces the recall–capacity curve (a single-model, inference-time version of your d≥N law). The ordering then licenses **placement**: a hot prefix stays fresh on GPU while the cold tail lives in far memory, updated exactly-but-late and read stale — *dense-but-stale*, in contrast to sparse-and-fresh selection (and unlike KV, recurrent state can't tolerate dropped writes — we verified stale *corrections* compound catastrophically while stale *reads* are age-local).

Highlights: on a public 9B hybrid (Nemotron-Nano-9B-v2), a rotation-only retrofit (3.5M params, backbone frozen) preserves full-width quality exactly while enabling the dial; under tiered-stale execution, needle recall is 0.96 vs 1.00 fresh (vs 0.25 if the cold tail is dropped). We also measured the serving-side law your tradeoff implies: decode throughput ∝ 1/state at fixed HBM, flat in capacity — which placement, not sparsity, relieves. This holds across GDN/GLA/KDA/Mamba2.

A preprint is in preparation; I'd greatly value 20 minutes of your feedback — especially on the accuracy-evaluation side (we're following the Based/JRT protocol) and possible connections to your test-time memory line (Cartridges).

Best regards,
Omin Kwon
Seoul National University

---

## 톤·전략 노트
- 첫 문단 = 신뢰 구축(그녀 언어의 실적: near-lossless + tiered KV + Stoica 공저), 둘째 문단 = "당신 법칙의 확장"이라는 훅(경쟁이 아니라 계승), 셋째 = 숫자 3개만(9B retrofit / needle 0.96 / slope −1), 넷째 = 작고 구체적인 부탁(20분, 평가 프로토콜 피드백)과 Cartridges 연결.
- "dense-but-stale vs sparse-and-fresh"는 그녀가 한 줄로 이해할 수 있는 프레이밍이라 본문에 유지.
- 후속 옵션: ES-FoMo(ICML, 그녀가 공동 조직) 제출을 언급하는 변형본도 가능 — 단 데드라인 확인 후.
