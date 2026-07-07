# Simran Arora 컨택 메일 (v2 — 협업 타진용 컴팩트 버전, 2026-07-07 확정)

**전략**: 방법 비공개(목표 수준까지만), 협업 의사 + Zoom 요청에 집중, ~180단어.
**보내기 전**: ① 지도교수(Jae W. Lee 교수님) 승인/cc, ② **Stuart Sul(ssul@cs.stanford.edu, Arora와 ParallelKittens/TK2.0 공저자)에게 먼저 연락** — 이름 언급 허락 또는 warm intro 요청(응답률 최선).

---

Subject: Prospective collaboration on memory-efficient linear attention serving

Dear Prof. Arora,

I am Omin Kwon, a Master's student in Prof. Jae W. Lee's group at Seoul National University - you may know our group through Stuart Sul, who did his research internship with us before joining Stanford. My research is on system-efficient LLM memory and serving systems. Most recently, I developed a sparse attention technique for block diffusion LLMs (Oral, AdaptFM @ ICML '26) and designed a CPU-GPU offloading technique for serving them efficiently, which is currently under review.

Working on KV-cache systems led me to linear attention, whose fixed-size state is far more memory-efficient than traditional KV-based architectures - and your Zoology/Based line of work convinced me that state size is the real currency of recall. Reading your papers sparked the idea I am now pursuing: what if the recurrent state - the memory that buys recall - were trained as a nested structure that separates a small hot core from a large cold remainder? The two could then be managed differently at serving time: the hot core kept fresh every token, while the cold majority is updated and refreshed far more lazily. Even in a GPU-only setting this eliminates most of the per-token state traffic, and it extends naturally to deeper memory hierarchies. I have promising early results on production-grade architectures.

I am based in Korea, but I am a diligent and highly motivated student, and I would be honored to explore a remote research collaboration. I am also planning to apply for PhD programs in Fall 2027 - another reason I would deeply value the chance to work with you. Would you be open to a short Zoom chat?

Best regards,
Omin Kwon
Seoul National University

---

# (참고) v1 — 상세 공개 버전 (미사용, Zoom 성사 후 팔로업 자료로 재활용 가능)



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
