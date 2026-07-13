
## ⚠ 계보 정정 (2026-07-14): tune_decay는 침묵 무효였음
A_log/dt_bias가 p4mixed→longcot2 네 ckpt에서 **비트 동일** — bf16 ulp(~0.008) < decay 업데이트(5e-5~1e-4)라 매 스텝 반올림 소멸. 함의: ① **전 계보의 유효 학습 파라미터 = R only (3.5M, 0.04%)**, ② fresh=raw는 회전 불변성으로 **구성상 보장** (n=500 fresh 요동 ±2pp는 greedy 라운딩 노이즈), ③ 문서의 "R+decay" 서술 정정 필요. decay를 실제로 켜려면 fp32 마스터 필요 — 현재로선 불필요 판단(v4 lossless가 R만으로 달성됨).
