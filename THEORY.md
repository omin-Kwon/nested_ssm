# Why It Works — 관찰과 수학적 근거
*(BACKGROUND.md §6의 "왜 되는가"를 명제 수준으로. 각 명제는 [직관] → [수학] → [우리 관찰] 순.)*

표기: state $S_t \in \mathbb{R}^{d_v \times N}$, 갱신 $S_t = \alpha_t\,S_{t-1} + w_t k_t^\top$, delta에서 $w_t = \beta_t (v_t - \alpha_t S_{t-1} k_t)$, 조회 $y_t = S_t q_t$. hot = 앞 $p$개 열, cold = 나머지. boundary $b$, chunk 길이 $c$.

---

## 명제 1 — Nesting 학습은 key 공간을 "중요도 순 기저"로 회전시킨다

**[직관]** 폭 $k$를 무작위로 잘라 loss를 물리면, 앞 차원들은 *모든* 폭의 loss에 등장하고 뒤 차원들은 큰 폭에서만 등장한다. 앞 차원은 "혼자서도 답해야 하는" 부담을 지고, 뒤 차원은 "앞이 이미 답한 것의 잔차"만 맡게 된다.

**[수학]** 목적함수 $\mathbb{E}_{k \sim \text{menu}}[\mathcal{L}_k]$에서 차원 $j$의 파라미터($W_k, W_q$의 $j$번째 행)는 **$k \ge j$인 loss 항에서만 gradient를 받는다** — gradient 수의 단조 비대칭이 정렬 압력의 원천이다. 선형 케이스의 정확한 결과가 존재한다: 선형 오토인코더에 nested dropout을 걸면 최적해가 **정확히 PCA(고유값 내림차순 기저)** 임이 증명되어 있다(Rippel et al., ICML'14). 우리의 readout $y=\sum_j S[:,j]\,q_j$도 차원별 기여의 합이라, prefix-loss는 "prefix 부분합의 자족성"을 같은 방식으로 강제한다. 비선형 전체 모델에 대한 정리는 아니므로 우리는 이를 **PCA-유사 직관 + 실측 확인**으로 위치시킨다.

**[결정적 보조 증거 — 정렬은 '새 능력'이 아니라 '기저 선택'이다]** delta recurrence는 key 공간의 직교 회전 $R$에 대해 함수 불변이다: $k \to Rk,\ q \to Rq$이면 $S \to SR^\top$이고 $y = SR^\top R q = Sq$ (스칼라 $\alpha$가 회전과 교환하기 때문 — **per-head 스칼라 decay 계열, 즉 GDN/Qwen3.5의 성질**). 따라서 "정렬된 모델"은 기존 모델과 **같은 함수 공간의 다른 기저**일 뿐이다. 이를 실험이 확인했다: backbone을 얼리고 회전 $R$만 50초 학습하자 from-scratch nested와 전 폭 동등(k8: 0.863 vs 0.871). **정렬에 필요한 것은 새 용량이 아니라 기저의 재선택뿐이라는 명제의 실증이다.** (채널별 대각 decay 계열(GLA/KDA)은 $\text{diag}(\alpha)R \neq R\,\text{diag}(\alpha)$라 불변성이 깨진다 — 그 계열은 경량 해동 FT가 필요; 실측 90% 수준.)

**[관찰]** ① recall(k)의 단조 곡선(0.87→1.00), ② hot-단독 품질: nested **0.98** vs 비-nested 우연 0.13~0.48, ③ 회전-retrofit 성공.

---

## 명제 2 — Cold 갱신의 "늦지만 정확함(lazy-but-exact)"은 recurrence의 결정론적 fold 구조와 affine 합성에서 나온다

**[직관]** state 갱신은 "입력이 정해지면 결과가 정해지는" 순차 계산이다. 재료(k, v, 게이트)를 버퍼에 모아뒀다가 나중에 같은 순서로 돌리면 똑같은 state가 나온다 — 늦는 것과 틀리는 것은 다르다.

**[수학]** 두 사실:
**(a) 지연의 정확성.** 갱신은 fold다: $S_t = F(S_{t-1};\,u_t)$, $u_t = (k_t, v_t, \alpha_t, \beta_t, r^{hot}_t)$. $F$는 벽시계가 아니라 $(S, u)$에만 의존하므로, 같은 $S_b$에서 같은 $u_{b+1..b+c}$를 재생하면 $S_{b+c}$가 **비트 단위로 동일**하다. delta correction이 $S_{\tau-1}$(pre-update state)을 읽는 구조라 재생 순서에서 항상 가용하다 — **correction의 지연 계산이 근사가 아닌 이유.**
**(b) chunk 묶음의 가능성 (affine 합성).** 각 스텝은 $S$에 대해 affine이다: $F(S;u) = M_u S + N_u$ (delta: $M_u = \alpha(I - \beta k k^\top)$, $N_u = \beta v k^\top$). affine의 합성은 affine이므로 c스텝이 $S_{b+c} = \big(\prod M\big) S_b + (\text{저}랭크 항들의 합)$ 한 방으로 접힌다(WY 표현). 결과: **state를 chunk당 1회만 읽고 쓰며(트래픽 $2S/c$), 연산이 rank-1 낱개가 아닌 작은 행렬곱**이 된다 — 약한 PNM 연산기가 감당하는 모양. *(a)가 "정확함"을, (b)가 "효율"을 준다 — staleness가 PNM의 window를 벌어주는 수학적 면허.*

**[관찰]** naive-vs-fused 검증 게이트 통과(replay 의미론의 수치 동일성), §8 판정 시뮬에서 c≥8일 때 replay backlog stall 0%.

---

## 명제 3 — Stale-read의 오차는 "최근 c토큰의 기여"로 정확히 분해되고, correction-stale의 오차는 state에 재귀한다 (금기의 수학)

**[직관]** 읽기만 낡으면 "방금 쓴 걸 아직 못 볼" 뿐이고, 쓰기의 참조가 낡으면 "잘못 지운 것이 장부에 영구히 남아 다음 계산을 또 오염"시킨다.

**[수학]** snapshot을 감쇠-보정($G = \prod \alpha$)해 읽으면, 감쇠항은 정확히 상쇄되고:
$$y^{fresh}_t - y^{stale}_t \;=\; \sum_{\tau = b+1}^{t} \Big(\prod_{s>\tau}\alpha_s\Big)\, w_\tau \,\big(k_{\tau}^{cold} \!\cdot\! q_t^{cold}\big)$$
— **오차 = 최근 $(t-b) \le c$개 write의 cold 기여, 그것뿐.** 세 귀결:
**(i) age-국소성:** $\tau \le b$인 모든 연관은 두 readout에서 동일 → 손상은 age ≤ c에 국한된다.
**(ii) 회복 가능성:** 빠진 write $w_\tau k_\tau^\top$의 **hot 성분은 fresh state에 이미 있다**($w_\tau k_\tau^{hot\top}$). 명제 1이 "hot prefix 읽기 ≈ 전체 읽기의 거친 근사"를 보장하므로, 빠진 항이 hot 품질로 대체된다. **stale의 비용이 정확도가 아닌 '일시적 해상도'로 전가되는 메커니즘.**
**(iii) 금기 — correction이 낡으면:** write가 $\tilde w_\tau = \beta(v - \alpha \tilde r_\tau)$로 계산되면 오차 $e_\tau = \alpha(r_\tau - \tilde r_\tau)$가 $\Delta S_\tau = -\beta\, e_\tau k_\tau^\top$로 **state에 기입**된다. 이후 모든 correction이 오염된 $S$를 참조하므로 $e_{\tau'}$가 $\Delta S$ 항들을 상속 — **오차의 재귀(복리)**. 1차 근사로 어떤 항목의 손상은 "그 항목이 겪은 이후의 stale-창 write 수"에 비례해 누적되고(→ **age가 오랠수록 손상 큼**), key들이 비직교가 되는 용량 포화에서 간섭이 전역화된다(→ **부하 D에 스케일, headroom 규칙**).
요약 이분법: **readout-stale = 출력에 더해지는 일시적 오차(transient, age-국소) / correction-stale = state에 곱해지는 영구 오차(persistent, 복리).**

**[관찰]** (i) step-at-c 지문: age>c recall 0.97~1.00, age≤c만 하락. (ii) v4 young 0.88~0.95 vs hot 없을 때 0.02~0.11; 실언어 ppl: hot 없으면 152, 있으면 16.6. (iii) correction-stale의 **age-증가형** 붕괴(0.65→0.09)와 D-스케일(D8 −0.08 / D32 −0.60) — (i)과 정반대 프로파일이라 두 실패 모드가 실험적으로 분리됨.

---

## 명제 → 관찰 매핑 요약

| 명제 | 예측 | 실측 |
|---|---|---|
| 1 정렬 | prefix 자족성, 회전만으로 이식 가능 | hot-단독 0.98 vs 0.13~0.48; 50초 retrofit 동등 |
| 2 lazy-exact | replay = 수치 동일, c로 트래픽·연산모양 개선 | 검증 게이트 통과; c≥8 stall 0% |
| 3(i,ii) read-stale 허용 | step-at-c + hot 커버 | old 0.97+, young 0.88 (vs 0.02) |
| 3(iii) correction 금기 | age-증가·D-스케일 복리 붕괴 | 0.65→0.09, D-스케일 확인 |

## Motivation 확정 수치 (H100 80GB, 7B GDN-hybrid, 4× CXL-PNM — `poc/roofline_motivation.py`)

| 구성 | tok/s | 병목 / 비고 |
|---|---|---|
| GPU-단독, dk=128 (작은 기억) | 47,679 | 빠르지만 recall 하한 낮음 |
| GPU-단독, dk=1024 (8× 기억) | **12,479** | HBM 용량이 배치를 417로 캡 (dk=2048이면 6.8k — 정확도 노브가 throughput을 7× 학살) |
| **naive ① 저장-오프로딩** (CXL=원격메모리, 계산 GPU) | **636** | 토큰마다 2S가 링크 왕복 — *안 옮긴 것보다 20× 악화* |
| **naive ② 전체-오프로딩, fresh** (매 토큰 PNM 처리) | ≤31,789 | **3.74TB/s/dev 필요(비존재)**; rank-1 낱개 연산은 약한 PNM이 실제론 더 못 함(관대한 상한) |
| naive ③ 전체-오프로딩, chunked | ~4×대 가능 | **정확도 파괴: 언어 ppl 16→152 (10×)** |
| **우리 (nesting + v4, c=8)** | **47,679** | GPU-BW 천장 = 작은-기억과 동일 속도; PNM 요구 1.31TB/s/dev, ppl +2~4% |

*읽는 법: 큰 기억은 GPU에서 4~7× 벌금(M1), 순진한 회피는 20× 참사(①)거나 존재하지 않는 하드웨어를 요구(②)하거나 정확도를 부순다(③). 재정렬+두 규칙이 이 4중 딜레마를 동시에 피하는 유일한 지점.*

### naive ①②③의 동작 방식 (오해 방지용 정밀 정의)

| | state 위치 | state 계산: 누가·언제 | 매 토큰 링크 통과물 | 죽는 지점 |
|---|---|---|---|---|
| **① 저장-오프로딩** | CXL (원격 메모리/tiering) | **GPU**, 매 토큰 → state가 GPU로 와야 함 | **state 전체 ×2 (200MB/req)** | 링크 (636 tok/s — 26× 느린 파이프에 같은 트래픽) |
| **② 전체-오프로딩 fresh** | PNM 상주 | **PNM**, 매 토큰 즉시 (GPU와 같은 스케줄) | 작은 벡터만 ✓ | PNM 내부: 토큰당 2S 스캔 = 3.74TB/s/dev 요구 + rank-1 낱개는 약한 연산기의 최악 모양 |
| **③ 전체-오프로딩 chunked** | PNM 상주 | PNM, **c토큰마다 몰아서** (읽기는 snapshot) | 작은 벡터만 ✓ | 정확도: fresh 경로 전무 → 최근 c토큰 기억 공백 (ppl 10×, young recall 0.02 실측) |
| **우리** | hot=GPU / cold=PNM | hot 매 토큰 fresh / cold c마다 **exact replay** | 작은 벡터만 ✓ | — (47.7k, ppl +2%; nesting이 hot의 커버 품질을 보장) |

*비유: ① 장부를 창고에 두고 한 글자마다 장부째 왕복. ② 창고 사서에게 한 글자마다 장부 전체를 넘기라 시킴. ③ 사서가 하루 한 번만 정리 — "방금 말한 것"을 아무도 모름. 우리 = 손에 든 수첩(fresh) + 창고 장부(몰아서 정확히 정리, 읽기만 낡게).*
