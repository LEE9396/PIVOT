# PIVoT v2 집필 브리프 (모든 집필 에이전트 공통)

## 0. 무엇을 하는 작업인가
`~/Downloads/ICRA-27-ROBIN` (v1 초안) 을 **`paper guideline v2.txt` 방향으로 개정**한다.
작업 트리는 `~/Desktop/PIVOT/paper_v2/` (v1 을 복사해 둠). v1 파일을 **고쳐 쓴다**.
가이드라인 원문: `~/Desktop/PIVOT/paper guideline v2.txt` — **반드시 먼저 읽을 것.**

## 1. 양식 규칙 (v1 과 동일하게 유지)
- IEEEtran conference, `main.tex` 가 `1_intro/2_relwork/3_method/4_exp/5_conclusion` 을 `\input`.
- **본문은 영어.** 각 문단 뒤에 `% [번역] ...` 로 한국어 번역 주석을 단다 (v1 관행. 반드시 유지).
- 미확정/실물대기 값은 `\rev{...}` (빨강) 또는 `\Pending` 매크로.
- 정리 환경: `lemma/proposition/corollary/remark` 는 main.tex 에 이미 선언됨.
- 저자·소속은 v1 그대로.

## 2. 제목·용어 (확정)
- 제목: `PIVoT: Part-Wise Identification of Volume-Aware Density via Multi-Topology Reorientation`
- 표기는 **PIVoT** 로 통일 (v1 본문의 `PIVOT` 을 전부 교체).
- **"excitation", "orbit" 금지어.** 대체: measurement pose / reorientation / measurement path.

## 3. 구조 (가이드라인 v2 준수)
Method 5절: A Problem Formulation and Overview / B Part-Wise Volume Estimation /
C Information-Theoretic Joint Configuration Selection / D Multi-Topology Measurement Path Design /
E Active Sensing and Uncertainty-Aware Density Update
Experiments 5절: A Setup / B Part-Wise Volume Estimation / C Part-Wise Density Identification (main) /
D Measurement Pose and Path Design / E Sim-Ready Asset Validation
Related Work 3분류: 1) Sim-ready reconstruction of articulated objects
2) Vision-based physical property estimation  3) Interaction-based physical property estimation

## 4. 절대 틀리면 안 되는 과학적 사실 (v1 정리 + 신규 실측)

### 4.1 준정적 렌치가 결정하는 것
정지 렌치는 **0차·1차 질량 모멘트(총질량과 1차 모멘트)만** 구속한다.
- Prop.(rank ceiling): `rank M(theta) <= 1 + min(3, P)`
- Cor.(rigid capped at four): 형상을 고정하면 중력 방향을 몇 개 쓰든, 몇 번 반복하든 rank 4 를 못 넘는다.
  따라서 P>=5 인 강체 조립품은 준정적 렌치만으로 식별 불가.
- **관성 텐서는 준정적으로 식별 불가능하다.** 정지 자세에서 omega=0, alpha=0 이므로
  tau = I*alpha + omega x (I omega) 가 항등적으로 0 이 되어 I 항이 사라진다.
  → 본 논문의 관성 텐서는 **메시 기하 x 식별된 밀도에서 유도**되는 것이지 측정되는 것이 아니다.
  이 점을 Method E 와 Limitation 에 명시할 것. (리뷰어 예상 질문 대응)

### 4.2 측정 자세 3개의 근거 — **v2 가이드라인 문구를 그대로 쓰지 말 것**
가이드라인은 "중력 3방향이 선형독립이어야 관측된다" 고 썼으나 이는 부정확하다.
**정확한 서술 (사용자 승인됨):**
- 정규직교 삼면체이면 `sum_k g_k g_k^T = I_3` 이라 정보행렬 M 이 **삼면체 전체 회전에 불변**(Cor. triad invariance).
- 즉 3방향은 rank 를 올리려는 것이 아니라 (직교 2방향이면 이미 rank 4 도달),
  **rank 상한을 달성하면서 정보행렬을 등방(isotropic)으로 만들어 조건수를 최소화**하고,
  그 결과 **삼면체의 SO(3) 방향 전체가 정보 면에서 공짜 자유도**가 된다.
- 이 공짜 자유도가 바로 파지·도달·충돌회피·관절보호에 쓸 수 있는 여유이며,
  Multi-Topology Measurement Path Design 은 이 자유도를 topology 별로 소진하는 문제다.

**실측 근거 (`study_tilt.py`, 이미 실행됨 — 본문에 인용할 것):**
| 방향 집합 | 정보이득 [nat] | 방향당 | 최대 관절토크 |
|---|---|---|---|
| 1 (아래로만) | 12.8811 | 12.8811 | 0.124 N·m |
| 2 (직교) | 25.4052 | 12.7026 | 1.031 N·m |
| **3 (직교, 채택)** | **26.3535** | 8.7845 | 1.031 N·m |
| 3 (비직교 60도) | 26.2069 | 8.7356 | 0.893 N·m |
| 4 (정사면체) | 26.9242 | 6.7311 | 0.765 N·m |
| 6 (±축 전부) | 27.7305 | 4.6218 | 1.031 N·m |
→ 3방향 직교가 **수확체감의 꺾이는 지점**. 그 위는 이미 채워진 부분공간.

**삼면체 회전의 자유도가 실제로 쓸모 있다는 증거 (매우 중요, contribution 2 의 핵심):**
- 정준 삼면체: 정보 26.3535, 최대 관절토크 **1.031 N·m**
- 토크 최소 회전: 정보 26.3535 (**+0.0%**, 정보 손실 없음), 최대 관절토크 **0.620 N·m (-39.9%)**
→ **정보를 전혀 잃지 않고 관절에 걸리는 최대 토크를 39.9% 줄일 수 있다.**
  "관절을 손상시키지 않는다" 는 주장의 정량적 근거가 바로 이것이다.
- 한 방향만 쓰면: 최고 22.3645 vs 최저 8.2290 nat, **2.72배** 차이.
  관절 축과 중력이 나란하면 그 관절 아래 질량의 토크가 0 이라 안 보인다.
  정규직교 3방향은 이 편차를 평균내 없앤다.

### 4.3 이론적 하한 (`study_theory.py`, 실행됨. 상대오차 ~1e-16 으로 수치 검증)
라운드 하한 `R_min = ceil((P-1)/3)`, 누적 rank 상한 `1 + min(3R, P)`.
| P | 라운드별 누적 rank | 상한 | R_min | 달성 |
|---|---|---|---|---|
| 2 | [2,2,2,2] | [3,3,3,3] | 1 | 1 |
| 3 | [3,3,3,3] | [4,4,4,4] | 1 | 1 |
| 4 | [4,4,4,4] | [4,5,5,5] | 1 | 1 |
| 5 | [4,5,5,5] | [4,6,6,6] | 2 | 2 |
| 6 | [4,6,6,6] | [4,7,7,7] | 2 | 2 |
실물 물체: 2link 미지수 3 / 3link 미지수 5, 단일형상 rank 4, R_min 2 / desklamp 미지수 3, rank 3, R_min 1.
**중요**: rank 하한은 generic 하게 tight 하지만 **실제 폐루프는 더 걸린다**.
그 차이를 만드는 것은 rank 가 아니라 **조건수**이고, 조건수는 **각도 추정 정밀도**가 만든다.
이것이 논문의 실질적 한계이며 정직하게 써야 한다.

### 4.4 부피 오차의 전파 — "Volume-Aware" 의 근거
회귀행렬이 rho_i 의 계수로 부피 V_i 를 그대로 쓰므로:
- **질량 m_i = rho_i * V_i 는 부피 오차에 불변**이다. 잡음이 평균화되면 참질량에 수렴한다.
- 그러나 **밀도 rho = m/V 는 부피 오차를 1:1 로 그대로 이고 간다.**
- **관성 텐서는 s^2 로 틀린다** (메시가 배율 s 만큼 틀리면).
- **도심(centroid) 오차는 다르다.** 도심은 C(theta) 에 직접 들어가 **질량으로 전파된다** (약 2.4%/mm).
→ 그래서 이 논문은 부피 정제를 한다: 질량만 맞으면 되는 것이 아니라
  **밀도와 관성까지 맞아야 sim-ready** 이기 때문이다. 이 논리를 Method B 와 Exp B 에서 명확히 세울 것.
→ 표에는 질량 오차와 밀도 오차를 **둘 다** 싣되, 위 불변성 때문에 고정 V 에서는 두 값이
  수치적으로 같다는 점을 각주로 정직하게 밝힐 것 (독립적인 두 열인 척하지 말 것).

### 4.5 힌지
custom object 2개는 힌지를 **독립 미지수**로 둔다 (플랜트의 별도 link 가 아니라 추정 벡터의 별도 성분).
힌지는 부피 7.45 cm^3 에 질량 41 g → 밀도 5.5 g/cm^3 로 **부피는 작고 질량은 집중**되어 있으며
다른 part 에 가려져 관측이 어렵다. 실제 관절체에서 흔한, 더 어려운 일반적 경우다.
**실측 근거 (`study_hinge.py`, 실행됨)**: 힌지를 미지수에서 빼면(0 g 로 두면)
부위 최대 오차 2.63%, 힌지 오차 79.10%, 그런데 알고리즘이 주장하는 반폭은 0.83% 에 불과하다.
→ **틀렸는데 맞았다고 확신하는 실패**. 힌지를 부위로 세어야 하는 이유의 정량적 근거.

## 5. 실험 물체 (사용자 확정)
GT 는 `~/Downloads/gt_list.md` 가 **권위 있는 최신본**. v1 표의 부피와 다르므로 v1 값을 쓰지 말 것.

| 물체 | part | 부피 cm^3 | 질량 g | 밀도 g/cm^3 |
|---|---|---|---|---|
| Stand Lamp | Base | 273.00 | 399 | 1.375 |
| | Support | 84.50 | 85 | 1.166 |
| | Head | 64.10 | 87 | 1.647 |
| | Assembly | 421.60 | 571 | — |
| Laptop | Display | 354.10 | 379.25 | 1.071 |
| | Base | 798.60 | 1137.75 | 1.455 |
| | Assembly | 1152.70 | 1517 | — |
| Custom 2-link | Long link | 135.72 | 49 | 0.361 |
| | Short link | 108.12 | 42 | 0.388 |
| | Hinge | 7.44708 | 41 | 5.506 |
| | Assembly | 251.28708 | 132 | — |
| Custom 3-link | Link 0 (long) | 288.12 | 100 | 0.347 |
| | Link 1 (middle) | 208.40 | 92 | 0.441 |
| | Link 2 (short) | 162.28 | 69 | 0.425 |
| | Hinges x2 | 14.89416 | 82 | 5.506 |
| | Assembly | 673.69416 | 343 | — |

**Laptop 은 mesh/URDF 자산을 팀원이 제작 중이다.** 따라서:
- GT 열과 vision baseline 열은 채운다.
- **Ours 열은 `\Pending` (TBD)** 로 두고 각주로 "asset in preparation" 이라 밝힌다.
- 본문에서 "four objects" 라고 쓰되, 밀도 식별 결과는 세 물체에서 보고한다고 정확히 서술.

## 6. 하드웨어 (v1 그대로)
6축 Rainbow Robotics RB5-850E, AIDIN AFT200-D80-C 손목 F/T (sigma_f=0.10 N, sigma_tau=0.003 N·m/축),
Robotiq 2F-85 그리퍼, 단일 Intel RealSense D456. 계획·기구학·시뮬레이션은 모두 Drake.
관절각은 RGB-D 스트림의 pose tracker 가 읽는다 (**명령한 값이 아니라 관측한 값** — 그래서 설계변수 자체가 오차를 갖고, TLS 로 흡수한다).

## 7. Related Work — 확인된 사실 (지어내지 말 것)

### 7.1 Sim-ready reconstruction (물성 없음)
- **RORA** arXiv:2608.04842, Hyesung Lee, Youngseon Lee, Kyutae Lee, Dongjun Lee, Yongseok Lee (2026).
  3DGS+mesh 하이브리드, convex decomposition, human-in-the-loop joint 제안. **물성 없음.** 본 연구의 출발점.
- **ArtVIP** arXiv:2506.04941, ICLR 2026. "physical fidelity" 를 표방하나 **전문 모델러의 수작업 자산 라이브러리**이고
  자동 추정기가 아니다. → "물성을 다루지 않는다" 고 단정하지 말고 **"수작업 저작이며 실측 기반 식별이 아니다"** 로 쓸 것.
- **JODA** arXiv:2605.09954 (2026). 관절의 마찰·댐핑·detent 등 **관절 동역학**을 VLM 으로 추론. 질량/밀도/관성은 없음.
- 물성 없는 기하·관절 전용: Ditto(CVPR22), PARIS(ICCV23), Real2Code(2024), ScrewSplat(CoRL25),
  ArtSplat(arXiv:2605.24304), ArtGS(arXiv:2502.19459).

### 7.2 Vision-based physical property estimation
- **PhysX 계열**: PhysX-3D(NeurIPS25 Spotlight, arXiv:2507.12465), PhysX-Omni(arXiv:2605.21572),
  PhysX-Anything(CVPR26, arXiv:2511.13648). **재질 DB 에서 밀도를 조회**해 메시 부피와 곱해 질량·관성을 계산.
  실측 식별이 아니다 — 우리와 정확히 대비되는 지점.
- **PUGS** ICRA 2025, arXiv:2502.12231 (Shuai 외). 3DGS + zero-shot 질량 예측, 물체 단위.
- **SiPhy** ECCV 2026, arXiv:2607.22355 (Hoang Le 외). 표기는 **SiPhy** (SIPHY 아님). 단일 이미지, part 단위.
- NeRF2Physics(CVPR24, arXiv:2404.04242), GaussianProperty(arXiv:2412.11258), PhysGS(arXiv:2511.18570).
- **정확히 쓸 것**: 이들은 **실측을 평가용 GT 로만 쓰고 추정 입력으로는 쓰지 않는다.**
  "실측과 접점이 전혀 없다"고 쓰면 부정확 (PUGS·NeRF2Physics·PhysGS 는 ABO-500 실측 질량으로 평가함).

### 7.3 Interaction-based physical property estimation
- 단일 강체: "The Sum of Its Parts"(Nadeau, Giamou, Kelly, arXiv:2302.06685) — **하나의 강체 안의 재질 영역**별
  관성 파라미터. 여기서 "part" 는 관절로 연결된 링크가 아니다. **우리 "per-part" 와 혼동될 수 있으니 반드시 구분해 쓸 것.**
  "Fast Object Inertial Parameter Identification"(ICRA22, arXiv:2203.00830),
  "Active Mass Distribution Estimation from Tactile Feedback"(arXiv:2303.01010, 단일 강체 블록).
- 다중 자세 기반 rigid 관성 식별의 표준 결과 (본문에 유도로 제시할 것, 특정 논문에 귀속시키지 말 것):
  질량만이면 1자세, **질량+CoM 이면 비평행 2자세**로 충분 (torque 식 tau=[F]x r 이 자세마다 rank 2 구속이고
  null space 가 F 방향이라, 비평행 두 자세의 null space 교집합이 자명해짐).
  센서 바이어스까지 함께 풀면 비공면 3방향이 필요. **관성 텐서는 정적으로 불가능**(4.1 참조).
  동적 여기가 필요한 고전: Atkeson-An-Hollerbach(IJRR 1986), Kubus-Kröger-Wahl(IROS 2008, recursive TLS).
- **★ 반드시 인용하고 구분할 것 — 가장 가까운 선행연구**:
  **Kumar, Essa, Ha, Liu, "Estimating Mass Distribution of Articulated Objects using Non-prehensile Manipulation",
  arXiv:1907.03964 (2019).** 실물 UR10 으로 2·3링크 관절체의 **link별 질량**을 추정한다.
  차이점: (i) **비전(RealSense + 링크별 QR)** 으로 추정하고 F/T 를 추정에 쓰지 않는다,
  (ii) **밀어내기(non-prehensile push)** 이지 파지·재배치가 아니다,
  (iii) 연속 회귀가 아니라 **5개 이산 질량분포 클래스 분류**(정확도 81.4%)다.
  → 이 논문을 언급하지 않으면 리뷰어가 놓친 선행연구로 지적한다. 한 문장으로 명시적으로 구분할 것.
- **RigPI** (arXiv:2606.25212, IROS 2027, bib key `he2027rigpi`): 실물 xArm6 + ATI Axia80 F/T.
  그러나 실물 검증 대상(서랍, 캐비닛 문, 오븐 문)은 모두 **움직이는 링크가 1개뿐**이다.
  다중 링크 주장은 시뮬레이션/이론에 그친다. → "실물에서 독립 미지질량 링크 2개 이상을 F/T 로 식별한 사례는 없다" 로 좁힐 것.
- Martín-Martín & Brock (IJRR 2019 등) 의 interactive perception 계열은 **관절의 마찰/저항 프로파일**을
  추정하며 링크의 질량·밀도·관성이 아니다.

### 7.4 주장 문구 (3축을 동시에 유지해야 살아남는다)
> 손목 F/T 실측에 근거하여, 하나의 기구 안에서 **독립적으로 미지인 질량을 가진 링크가 2개 이상**인
> 관절체의 **링크 자체의 밀도**를 체계적으로 식별한 연구는 없다.
(i) 센싱 = F/T (비전 아님) → Kumar 2019 와 구분
(ii) 독립 미지질량 링크 2개 이상 → RigPI 실물 검증과 구분
(iii) 링크의 질량/밀도 (관절 마찰 아님) → Martín-Martín/Sturm 계열과 구분

### 7.5 OED / active exploration
- Gautier & Khalil, "Exciting Trajectories for the Identification of Base Inertial Parameters of Robots",
  IJRR 11(4):362-375, 1992 (CDC 1991 버전도 있음). 조건수 최소화.
- Swevers, Ganseman, Tükel, De Schutter, Van Brussel, "Optimal Robot Excitation and Identification",
  IEEE T-RA 13(5):730-740, 1997. **저자 5명 전부 표기할 것.**
- Wensing, Niemeyer, Slotine, "A Geometric Characterization of Observability in Inertial Parameter
  Identification", IJRR 43(14), 2024, arXiv:1711.03896. RPNA = Recursive Parameter Nullspace Algorithm.
- **★ v1 인용 오류**: `hausman2015` 와 `memmel2024asid` 는 **서로 무관한 별개 논문**이다.
  Hausman 외, "Active Articulation Model Estimation through Interactive Perception", ICRA 2015 (파티클 필터).
  Memmel 외, "ASID: Active Exploration for System Identification in Robotic Manipulation", ICLR 2024 Oral,
  arXiv:2404.12308 (Fisher information 기반). v1 이 둘을 붙여 쓴 곳이 있으면 반드시 분리할 것.

## 8. 사용자 개입 서술 (표현 주의)
- reconstruction 단계 개입은 **RORA 와 동일**하게 상속된다. 숨기지 말 것 (Fig.2 (a) 패널에 사람 아이콘).
- physics 추정을 위해 **추가되는** 개입은 제시된 joint configuration 으로 물체를 조정하는 것뿐.
- **"개입 지점이 한 곳"이라고 쓰지 말 것. "추가되는 개입은 한 곳"으로 정확히.**

## 9. 정직성 규칙 (엄수)
- 실물 로봇 데이터는 아직 없다. 실물 열은 `\Pending`.
- Sim-ready asset 검증(Exp E)은 **시뮬-시뮬 대리 검증**이다. 실세계 검증인 것처럼 쓰지 말 것.
- 물체가 적어 통계적 유의성을 주장할 수 없다. 물체별 결과를 표에 모두 드러내고 평균에만 의존하지 말 것.
- 없는 수치를 지어내지 말 것. 없으면 `\Pending` 또는 `\rev{[pending: ...]}`.
