# CUTS (8/30 압축, intro/relwork/conclusion 담당 에이전트)

## 1_intro.tex (883 -> 700)
- 5문단의 파이프라인 단계 열거(부피 보정 -> 형상 제시 -> 경로 계획 -> 렌치 측정 -> 밀도 갱신)를
  삭제 — 기여 bullet 3개와 중복. 원문: "Starting from the part-level reconstruction of
  [lee2026rora], PIVoT corrects each part's volume by multi-view-consistency-based mesh
  refinement and per-part scaling, suggests the joint configuration of highest expected
  information gain to the user, plans collision-free measurement paths through three
  measurement poses under different gravity directions, measures the wrench at each pose,
  and updates the per-part densities with their uncertainty until convergence."

## 2_relwork.tex (763 -> 598)
- Kumar 문단의 마무리 다리 문장 삭제: "Doing so requires jointly designing the joint
  configuration and the measurement poses, which is precisely what PIVoT does."
- 8/27 판본·8/20 판본 주석 블록(약 180줄) 삭제 — git 과 v1 트리에 보존.
- "exciting trajectories---their term for the dynamic identification runs" ->
  "identification trajectories" (금칙어 인접 표현 정리).

## 5_conclusion.tex (345 -> 280)
- 표현만 축약. 모든 한계 항목(부피 오차의 라운드 비용, 관성 텐서 유도량,
  조건수가 실질 한계, 재구성 오류·개입 상속, "추가되는 개입은 한 곳") 유지.

## 4_exp.tex 압축 (8/30, 2,914 -> 1,498 단어)에서 잘라낸 산문

구조 변경: Table I -> Fig. 3 캡션 흡수(가이드라인 v2 절약 수순), 구 Fig. 8
(sim-vs-real 자리표시자) -> Fig. 7 캡션의 \rev 로 흡수. tab:objects /
fig:assetreal 라벨은 해당 그림 환경 안에 보존.

표 캡션·각주가 동일 내용을 이미 싣고 있어 산문에서 지운 문장 (내용은 논문에 잔존):
- single 의 수렴 플래그 무의미(정지 규칙 미참조), single 커버리지는 폭 때문 — 표 III 각주.
- 관절한계 볼록 상자 상세("한계 안 두 IK 해를 잇는 직선이 못 벗어난다") — 표 IV 캡션.
- 3링크 조건수 1e16~1e18 이 구조적(rank 4 대 미지수 5)이라는 상세 — 표 IV 각주.
- vision 변환 상세(PUGS Gaussian 적분·질량범위 중점, GT 부피 분율 분배, PhysX-Omni 조립체 채점) — 표 III 각주.
- asset 검증의 공유 상수 목록(접촉 강성·소산·마찰·관절 댐핑), 초기조건 5개 사전 고정, 27/27 1위 — 표 V 캡션.
- PhysX-Omni 두 커스텀 물체 N/A 제외, 커스텀 변환은 상대 질량만 이전 — 표 V 각주.

완전히 삭제한 내용 (논문 어디에도 남지 않음):
- 하드웨어 상세: "Rainbow Robotics" 상호, AIDIN AFT200-D80-C 형번, sigma_f=0.10 N /
  sigma_tau=0.003 N·m, hand-eye 캘리브레이션, 그리퍼 타어링(대상보다 무거움), 커스텀 물체
  링크 수(2·3링크 — Fig.3 캡션의 links 열에는 잔존).
- 이론 수치 검증 문단: 무작위 형상 300개에서 식 (3) 상대오차 7.9e-16, 따름정리 1 동일
  정밀도, 명제 2 영공간 7.5e-14, 명제 1 rank 천장 위반 0.
- worst-part 밀도오차 중앙값 나열: 0.18 % [0.10~0.27], 0.11 % [0.08~0.14], 0.23 %
  [0.12~0.46]; uniform 49.1 / 321.0 / 34.6 % (표 III 에 부위별 값 잔존).
- 합성 사슬 P=2,3 수치(uniform 319 / 263 %, 나머지 <= 1.5 %).
- 중력 방향 수치 서술: 1방향 12.88 nat 대 2방향 25.41, 삼면체 26.35, 정사면체 +0.57,
  6축 +1.38, 방향당 12.9 -> 4.6 nat, 단일 방향 8.23~22.36 nat (표 IV(a) 에 잔존).
- 마찰 힌지 유지 토크 사양 0.9 N·m, Drake 전수 스윕 최대 부하 0.158 N·m, 여유 5.7배.
- 경로 수치 서술: 충돌률 25.0/20.9/21.1 %, 경로 길이 +1.5~7 %, 계획시간 0.11~0.43 s
  (표 IV(b) 에 잔존); 최악 구간 평균 최소여유 음수(2링크 transit -13.8 mm; 표 IV 의
  Clear. 열과 각주에 개념 잔존).
- "정보가 많은 형상 = 까다로운 형상" 문단 전체: 램프에서 직선 충돌 형상 8개가 전부 정보
  상위 9개 안(Delta I >= 8.86 nat / 전범위 6.9~9.3), 8.85 nat 이하는 전부 무충돌; 3링크는
  상관 약함(도달가능 23 중 19 충돌) -> 램프 하나로 일반화 안 함; IK 실패 2링크 1/5,
  3링크 2/25, 램프 6/25.
- "지렛대는 F/T 센서가 아니라 각도 추적기" 문장 (Remark 3 참조는 잔존).
- release 궤적이 관절 한계에 내려앉는다는 문장의 부연("정보는 과도응답에 있고 RMSE 가
  그것을 적분한다") — 앞 절반은 잔존.
- asset push 절대 오차 26~44 mm (표 V 에 잔존).

## 3_method.tex — 8/30 압축(3,556 → 2,145 단어)에서 옮겨 둔 내용

### Experiments(Table IV, tab:path)로 이관 — 본문에서 삭제된 수치
- 방향 수 수확체감 수치열: 1방향 12.88 nat / 직교 2방향 25.41 / 직교정규
  삼면체 26.35 / 정사면체 4방향 26.92 / 여섯 축 방향 27.73, 방향당 수확
  12.9 → 4.6 nat. (4_exp.tex L374~ 가 동일 수치를 이미 서술; Method 에는
  "삼면체 = 수확체감 곡선의 무릎" 형태만 유지.)
- 단일 방향 비등방성 수치: 배향에 따라 8.23~22.36 nat (2.72배). (4_exp 유지.)
- 토크최소 회전 전후 절대 정보량 26.3535 nat. (Method 에는 0.0 % 손실 /
  1.031 → 0.620 N·m, −39.9 % 만 유지; 절대값은 4_exp 에 있음.)
- 단일 방향 비등방성의 기전 문장("중력과 나란한 관절 축은 자기 축에 대한
  모멘트를 만들지 못해 그 관절 아래(원위) 질량을 숨긴다")은 4_exp L380-382
  에 남아 있어 Method 에서 삭제.

### 증명 처리 (진술·라벨은 전부 보존)
- 보조정리 1 (lem:closed): 증명을 한 문장으로 축약 (전개 + [g]x^T[g]x =
  I - gg^T + k 합산).
- 명제 1 (prop:rank): IEEEproof 삭제 → "PSD 순서 + rank 준가법성, 표준"
  한 구절로 강등. 원 증명: g_k g_k^T ⪯ I_3 에서 두 항이 준양정, rank
  준가법성으로 rank M ≤ rank(vv^T) + rank(C^T(·)C) ≤ 1 + rank C, C ∈ R^{3×P}.
- 명제 3 (prop:rounds): IEEEproof 삭제 → "C(theta_r) 쌓아 그람 행렬 +
  준가법성" 한 구절로 강등. 원 증명: C = [C(θ1)^T … C(θR)^T]^T ∈ R^{3R×P},
  Σ_r C^T C = C^T C 의 rank ≤ min(3R, P).
- 명제 2 (prop:null): 증명 짧아 유지.

### 기타 통합/삭제
- Wensing/base-parameter 대비 문단 → 한 문장으로 축약. 삭제분: "밀도
  매개화는 물체당 10개 관성 파라미터 공간에 거는 선형 제약으로 물체당
  미지수 하나만 남긴다"; "식별 가능 부분공간의 밀도 좌표 투영 =
  span{v} ⊕ row(C(θ))"; "base parameter set~\cite{gautier1990direct,
  mayeda1990base} 도 형상 개수는 묻지 않는다(제약 없는 동적 운동 가정)".
  (gautier1990direct/mayeda1990base 인용은 2_relwork L136 에 존재.)
- "강체 방법은 사전지식으로 메울 수밖에 없다" 문단 → cor:rigid 진술의
  꼬리절("---the closure rigid-object methods~\cite{nadeau2023sum,
  pfaff2025scalable,shuai2025pugs} must buy")로 이동.
- A 소절의 영공간 예고 문단 절반(힘 행/토크 행 상세, "손목 재배치는 같은
  행을 재저울질" 전개)은 명제 1·2 가 정식으로 담당하므로 요약형으로 축소.
- 라운드당 "숫자 18개(eighteen scalars per round)" 색채 서술 삭제.
- 정지 규칙의 "복잡도가 다른 물체를 같은 부위당 기준으로" 부연 삭제
  (Remark 3 인용이 근거를 대신함).
- GT 서술의 "(스탠드 램프, 노트북)" 예시 괄호 삭제 (4_exp 에 있음).

## 8/30 8-페이지 맞춤 (fit-to-8 에이전트)

### tables/table4_path_v2.tex — panel (a) 전체 삭제 (중력 방향 수확체감 표)
본문 IV-D 산문이 무릎(knee)·2.72배 비등방성·0.0 % / −39.9 % 를 이미 싣고 있어
표 (a) 를 지우고 삼면체 26.35 nat 대 6방향 27.73 nat 수치를 산문 한 구로 흡수.
복원용 원문:
```
\textit{(a) Gravity directions per round}\\[2pt]
\begin{tabular}{lccc}
\toprule
Direction set & Info [nat] & per dir.\ & \shortstack{Peak torque\\{[N$\cdot$m]}} \\
\midrule
1 (downward)            & 12.88 & 12.88 & 0.124 \\
2 (orthogonal)          & 25.41 & 12.70 & 1.031 \\
\textbf{3 (orthonormal, ours)} & \textbf{26.35} & 8.78 & 1.031 \\
3 (non-orthogonal, $60^\circ$) & 26.21 & 8.74 & 0.893 \\
4 (tetrahedral)         & 26.92 & 6.73 & 0.765 \\
6 ($\pm$ axes)          & 27.73 & 4.62 & 1.031 \\
\midrule
Triad, canonical orient.\ & 26.3535 & -- & 1.031 \\
\textbf{Triad, torque-min rot.} & \textbf{26.3535} {\tiny($+0.0\%$)} & -- & \textbf{0.620} {\tiny($-39.9\%$)} \\
\bottomrule
\end{tabular}\\[6pt]
```
캡션의 "Top: ..." 문장도 함께 삭제.

### 참고문헌 감축 (38 -> 27 항목; reference.bib 은 그대로, \cite 만 제거)
제거: jin2025ugraph, eppner2018physics, lee2020geometric, lee2021optimal,
mayeda1990base, janot2014instrumental, zhai2024nerf2physics,
xu2025gaussianproperty, kim2025screwsplat, lee2026artsplat, li2026uniphysgen.
유지(후보였으나 남김): atkeson1986estimation (Lemma 1 "classical fact" 귀속),
hausman2015active (interactive perception 구분 문장의 유일 인용).
함께 지운 산문:
- 2_relwork: "Active variants choose the next reorientation---still of one
  rigid body \cite{jin2025ugraph}." (문장 전체 — jin 인용을 싣기 위한 문장)
- 2_relwork: 밀도 문장에서 "NeRF2Physics, GaussianProperty," 명칭 제거
  (문장은 PhysGS·SiPhy 로 유지), 부피 문장에서 "and UniPhysGen" 제거.
- 1_intro: "...can hide entirely different mass distributions" 의
  \cite{xu2025gaussianproperty} 제거 (주장 자체는 유지).
- 3_method: \cite{lee2020geometric}(양수 상자 제약), \cite{janot2014instrumental}
  (TLS remedy), \cite{lee2021optimal}(D-최적 목록 + restricted-trajectory 구)
  의 인용만 제거, 개념 서술은 유지.

### tables/table5_asset_v2.tex — 9행 -> 3행 (push 만; Fig. 7 의 시나리오와 일치)
release·drop 의 순서는 본문 한 문장으로 이전 ("orders of magnitude on release
and drop"), 27/27 1위 주장은 본문·캡션 모두 유지. 2link release 구조적 둔감
각주는 삭제 (본문 산문에 동일 서술 잔존 — 보호 항목). 복원용 원행:
```
 & release$^{\sharp}$ & \textbf{0.00} & 0.81 & 1.77 & 2.00 \\   (2-link)
 & drop    & \textbf{0.17}  & 12.80  & 8.88   & 5.87 \\
 & release & \textbf{0.01}  & 5.62   & 5.43   & 6.89 \\        (3-link)
 & drop    & \textbf{0.55}  & 10.89  & 10.65  & 11.29 \\
 & release & \textbf{0.00}  & 0.59   & 3.76   & 6.00 \\        (Stand lamp)
 & drop    & \textbf{0.03}  & 10.35  & 19.75  & 41.48 \\
각주: $^{\sharp}$ structurally insensitive to density: with the root link welded
the 2-link release is a single pendulum, whose motion is invariant to scaling
that link's density---its sub-millimetre entries reflect that invariance, not
estimation skill.
```

### tables/table3_main_v2.tex — laptop 2행 -> 각주 1문장 (수치 전부 각주로 이전)
"4 물체, 밀도 결과는 3" 서술은 IV-A 산문에 그대로. 복원용 원행:
```
\multirow{2}{*}{Laptop$^{\dagger}$}
 & Display & 354.10 &  379.25 & 1.071 & 76.7 & 182.5 & 417.3 & \Pending & \Pending & \Pending & \Pending \\
 & Base    & 798.60 & 1137.75 & 1.425 & 30.1 & 108.0 & 611.8 & \Pending & \Pending & \Pending & \Pending \\
```

### 산문 중복 제거 (8-페이지 압축; 보호 수치는 모두 다른 자리에 잔존)
- 3_method III-D: "Measurement agrees ...: the triad is the knee of the
  diminishing-returns curve, later directions landing in an already-filled
  subspace, and single-direction anisotropy is averaged away." -> "Measurement
  agrees (Sec. IV-D)." (수치·무릎 서술은 IV-D 산문이 보유)
- 3_method III-D: "Measurably so: the torque-minimizing rotation preserved
  information exactly ($0.0\%$ loss) while cutting the peak torque about the
  object's own joints from $1.031$ to $0.620$ N·m ($-39.9\%$)---the
  quantitative basis for claiming measurement does not endanger the joints."
  -> 한 구("quantified in Sec. IV-D"); 0.0 % / 1.031→0.620 / −39.9 % 와
  "관절을 상하게 하지 않는다" 주장 근거 문구는 IV-D 로 일원화.
- 4_exp IV-B: "Density error follows volume error near-exactly one-to-one.
  Part is far less sensitive---but not exactly invariant, drifting to 3--6%..."
  문단을 캡션(그림 4)과 겹치지 않게 축약 — 3~6 %, 1:1, s^2(19.1 %) 수치는
  산문에 유지 (보호 항목).
- 5_conclusion 2문단: 표 III 이 싣는 커버리지 수치 나열(16/16, 24/24, 22/24 /
  14/16, 15/24, 4/24)을 합산형(62/64 vs 33/64)으로, single 반폭 나열
  (2.6/28.7/42.5)은 최악값(42.5 %)만 — 전체 수치는 IV-C 산문·표 III 에 잔존.
  힌지 사양 괄호(7.4 cm^3, 5.51 g/cm^3)는 Fig.3 캡션·III-A 잔존으로 삭제.
- tables/table3 각주: "On the 2-link object Ours and Single are byte-identical
  ... (Sec. IV-C)" 문장 삭제 (IV-C 산문에 동일 서술 — 보호 항목 잔존).
- tables/table4 캡션: "Joint-limit violations are zero ... not by limit
  violation." 문장 삭제 (IV-D 산문에 동일 서술 — 보호 항목 잔존).

### 2차 산문 압축 (8-페이지)
- 3_method III-A: 예고 문단 삭제 — "A static wrench constrains only the total
  weight $v^{\!\top}\!\rho$ and first moment $C(\theta)\rho$; for $P>4$ some
  density redistributions are invisible, wrist reorientation cannot reveal
  them, and only articulation changes what is visible." (명제 1·2와 따름정리
  2가 같은 내용을 엄밀히 진술 — 중복 예고)
- 3_method III-E: "---where the estimate re-enters the design, the sole reason
  the loop must close (Cor. 1)" 구 삭제 (Cor. 1 뒤 문단의 "루프는 오직 각도
  오차를 흡수하기 위해 존재" 와 중복)
- 4_exp IV-E 개입 문단 축약 (개입 서술 자체는 intro·method·결론·표 V 에 잔존)
- 4_exp IV-C: "(0.25, 0.51, 0.59 %)" 나열 삭제 ("below 0.6 %" 유지)
- 참고문헌 4건 추가 제거 (27 -> 23): cao2026physxanything (PhysX 계열은
  physx3d + omni 로 대표), liu2023paris·mandi2024real2code (재구성 나열 4건
  -> 2건: ditto + artgs), activetactile2023 (bib 자체가 UNVERIFIED 저자 미상
  — 제출 위생상 제거; host 절 "tactile feedback actively estimates a rigid
  block's mass distribution" 도 삭제)

## 8/30 최종 상태 (fit-to-8 에이전트 종료 시점)
- 페이지: 11 -> 9 (build/main.log "Output written on main.xdv (9 pages)").
- 참고문헌: 38 -> 23 항목 (venue 명칭은 IEEE 표준 약어로 축약; 항목 자체는
  reference.bib 에 전부 보존, \cite 만 제거).
- 형식 수단: \raggedbottom 제거(flushbottom), float 간격 8/6pt, 데이터 그림
  4개 0.74\textwidth, 초록 ~55단어 축약, Ack 를 참고문헌 앞으로.
- 9 -> 8 을 막는 것 (전부 금지 항목):
  1. placeholder 그림 3개 (fig1/fig2_pipeline/fig3_setup) 합계 ~0.6쪽 —
     "footprint 유지가 정직한 페이지 수" 지시로 축소·삭제 불가.
  2. 실데이터 그림 4개 (~0.75쪽) — 결과물, 삭제 금지; 더 줄이면 판독 불가.
  3. 남은 산문은 표·그림 재서술이 아닌 방법론 본문 (Remark 2, 79x 잔차 함정,
     관절 가시성 설계, 합성 사슬 수치 등) — "산문은 재서술 문단만" 규칙 밖.
  4. 보호 목록의 정직성 단서·정리 진술·\rev/\Pending 전부 유지 필요.
  5. 참고문헌 23개는 전부 보호(12) 또는 위치 서술 담지 항목.
