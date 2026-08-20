# 논문 수정 계획 — abstract / intro / related work

대상: `~/Downloads/ICRA-27-ROBIN/` (08/18 버전).
`3_method.tex`, `4_exp.tex`, `5_conclusion.tex` 는 **초안을 이미 써 넣었습니다.**
이 문서는 나머지 세 절의 **수정 방향과 교체 문안**입니다.

> 이론 근거는 [CONTRIBUTION.md](CONTRIBUTION.md), 증명은 [theory.tex](theory.tex),
> 수치 검증은 [study_theory.py](study_theory.py).

---

## 0. 먼저 — 기계적으로 고칠 것 (5분)

| 위치 | 현재 | 고칠 것 |
| --- | --- | --- |
| `1_intro.tex` 3문단 첫 줄 | "we propose **GS-Physics**" | "we propose **PIVOT**" |
| `1_intro.tex` 3문단 중간 | "—**GS-Physics** runs a closed-loop" | "—**PIVOT** runs a closed-loop" |
| `main.tex` abstract 끝 | "**five types** of articulated objects" | "**four** articulated objects" |
| `1_intro.tex` 3문단 끝 | "**five types** of articulated objects" | "**four** articulated objects" |
| `1_intro.tex` contribution 3 | "on **five** articulated objects" | 아래 3장에서 통째 교체 |

`main.tex` 서문에는 정리 환경 선언(`\newtheorem{lemma}` 등)을 **이미
추가했습니다.** `3_method.tex` 가 이걸 씁니다.

---

## 1. Abstract — 무엇이 문제인가

현재 abstract 는 **"관절체는 아직 아무도 안 했다"** 만 말하고,
**"우리가 무엇을 알아냈다"** 를 말하지 않습니다. 지금 문장으로는 리뷰어가
기여를 *"파이프라인을 만들었다"* 로만 읽습니다. 이론 기여를 주장하려면
abstract 에 **정리의 결론 한 줄**이 반드시 들어가야 합니다.

### 무엇을 바꾸나

| 현재 | 바꿀 방향 |
| --- | --- |
| "iterative closed-loop procedure that explores and estimates at positions with the highest information gain" | **왜** 그래야 하는지가 없음 → "단일 형상은 rank 4 를 못 넘으므로 관절 운동이 **필요조건**" 을 넣는다 |
| "five types of articulated objects" | 4개. 그리고 **`p=2..8` 시뮬레이션 스윕**을 같이 적어 일반화 근거를 보인다 |
| 기여가 전부 시스템 | **식별성 분석**을 첫 기여로 올린다 |

### 교체 문안 (그대로 붙여도 됩니다)

```
Beyond rigid bodies, the real world contains a wide variety of articulated
objects, making their faithful treatment in simulation essential. Simulating
them requires each part's density in addition to its geometry, yet density is
precisely what a camera cannot observe, and existing measurement-grounded
methods keep the object rigid. We show that this is not a matter of engineering
convenience: under quasi-static wrench sensing the information matrix admits a
closed form, from which it follows that a single articulation configuration
determines at most four independent density combinations---regardless of how
often the wrist is reoriented---and that the invisible directions are exactly
those redistributions preserving total mass and first moment. Changing the
object's own configuration is therefore necessary, not merely helpful, beyond
four bodies, and the analysis yields a lower bound on the number of
configurations that is computed from geometry alone. Building on this, we
present PIVOT, which actively selects articulation configurations to collapse
the null space of the density estimate while accounting for the fact that the
configuration itself is measured with error, and compiles the result into a
sim-ready asset. We validate on four real articulated objects---two custom
objects with exact per-part ground truth, a scanned desk lamp, and an unopened
laptop---where the predicted configuration counts are met with equality, and on
a simulation sweep over chains of two to eight links that locates the practical
limit of the method and shows it is set by angular sensing precision rather
than by the estimator.
```

> 분량은 현재와 비슷합니다(약 210단어). 길면 마지막 문장의 물체 나열을 줄이세요.

---

## 2. Introduction — 무엇이 문제인가

지금 intro 는 **흐름이 좋습니다.** 1문단(왜 real2sim), 2문단(왜 밀도가
문제인가), 3문단(우리 방법) 구조를 유지하세요. 바꿀 것은 **3문단과 기여
목록**입니다.

### 2.1 3문단 — "null space" 를 말하다 말았습니다

현재 이렇게 쓰여 있습니다.

> "Since a single grasp and joint configuration constrains only a few density
> combinations—leaving the rest in a null space that repeated measurement
> cannot resolve—..."

**"a few" 와 "a null space" 가 정확히 무엇인지 말하지 않습니다.**
이제 말할 수 있으니 말해야 합니다. 이 한 곳만 고쳐도 논문의 성격이 바뀝니다.

**교체 문안:**

```
In this work, we propose PIVOT, which estimates the per-part density and
geometry of portable articulated objects through real-world robot--object
interaction. Starting from a part-level articulated 3DGS reconstruction, we
formulate static wrist force/torque measurements as linear constraints on the
unknown density vector. Writing the information matrix in closed form makes the
structure of the problem explicit: a single articulation configuration
determines at most four independent density combinations no matter how many
gravity directions are used, and the unresolved directions are exactly the mass
redistributions that preserve total mass and first moment about the sensor.
Articulation is the only mechanism that moves them into view. PIVOT therefore
runs a closed-loop procedure with minimal human involvement: it selects the
joint configuration whose expected information gain is highest---searching over
configurations alone, since the gravity triad provably carries no
information---verifies that the configuration is graspable, collision-free, and
reachable under the object's shape at that configuration, and iterates until
the estimate converges, writing the result back into the reconstructed asset.
Because the configuration is measured by a camera rather than commanded, the
design variable itself carries error; we account for this with a
total-least-squares formulation, and show that it, not the rank condition, sets
the practical limit of the method.
```

### 2.2 기여 목록 — 통째 교체  ★ 먼저 검토하실 부분

원칙: **원래 쓰신 문체 그대로** — 명사구로 시작하는 한 문장, 22~30 단어.
바뀌는 것은 (a) 식별성 분석이 1번으로 들어오고, (b) five → four,
(c) 원래 2·3번은 거의 그대로 둡니다.

```latex
\begin{itemize}
    \item Identifiability analysis showing that a single joint configuration
    determines at most four independent density combinations---so that
    articulation becomes necessary beyond four parts---and bounding the number
    of configurations required.
    \item Active estimation strategy that selects the most informative joint
    configurations to collapse the null space of the density estimate,
    minimizing overall estimation time.
    \item Pipeline with minimal user intervention that turns real-world
    observations and interactions into sim-ready articulated assets faithful in
    both appearance and dynamics, without damaging joints.
    \item Real-world experiments on four articulated objects, two with
    controllable ground-truth densities, where the predicted number of
    configurations is met with equality, and a simulation sweep to eight links.
\end{itemize}
```

**순서를 왜 이렇게 두나.** 분석 → 그 분석에서 나온 전략 → 그걸 담은 파이프라인
→ 검증. 지금처럼 파이프라인이 1번이면 리뷰어가 끝까지 시스템 논문으로 읽습니다.
2·3번은 원문 그대로이고 위치만 바뀝니다.

**1번이 이렇게 쓰인 이유.**
- "at most four **independent** density combinations" — 정확히는 밀도의 독립
  선형결합 개수입니다. `independent` 를 빼면 부정확해집니다.
- "**beyond four parts**" — 부위가 4개 이하면 단일 형상으로도 될 수 있으므로,
  이 한정어가 없으면 과장이 됩니다.
- "bounding the number of configurations **required**" — 하한임을 명시.

**4번이 이렇게 쓰인 이유.** "met with equality" 가 이론과 실험을 잇는 가장 짧은
표현입니다. 2-link 는 하한 1에 관측 1, 3-link 는 하한 2에 관측 2입니다.

### 대안 — 3개로 더 줄이고 싶으시면

```latex
\begin{itemize}
    \item Identifiability analysis showing that a single joint configuration
    determines at most four independent density combinations, making
    articulation necessary beyond four parts, and bounding the rounds required.
    \item Pipeline with minimal user intervention that actively selects the most
    informative joint configurations to collapse the null space of the density
    estimate, turning real-world interactions into sim-ready articulated assets.
    \item Real-world experiments on four articulated objects and a
    two-to-eight-link simulation sweep, where the predicted round counts are met
    with equality and the method's practical limit is located.
\end{itemize}
```

2번에서 원래의 "능동 전략"과 "파이프라인"을 한 항목으로 합친 것입니다.
다만 **원문 두 문장의 선명함이 줄어들어 4개 안을 권합니다.**

### 2.3 2문단에 한 문장 추가 (선택)

2문단 끝의 *"Physical property estimation of articulated objects grounded in
real measurements remains absent."* 뒤에 아래를 붙이면, 3문단의 이론이
갑자기 튀어나오지 않고 자연스럽게 이어집니다.

```
This absence is not incidental: as we show in Sec.~III, keeping the object
rigid caps what any number of quasi-static measurements can determine, which is
why prior work must supply the remaining degrees of freedom with priors.
```

---

## 3. Related Work — 무엇이 문제인가

현재 related work 는 **잘 쓰였습니다.** 세 소절 구분(asset 생성 / vision-only /
measurement-grounded)이 적절하고, baseline 을 각 소절에서 지정하는 방식도 좋습니다.

문제는 **두 가지**입니다.

### 3.1 문제 A — 고전 여기설계 문헌이 통째로 빠져 있습니다 ★ 가장 위험

지금 related work 에는 **optimal excitation trajectory design** 계열이
한 편도 없습니다. 그런데 우리는 D-최적 설계를 씁니다.

> **리뷰어가 "이건 Swevers 1997 이후 30년 된 표준 아닌가"라고 물으면
> 지금 원고로는 답이 없습니다.** 이건 desk-reject 급 위험입니다.

**반드시 새 소절을 추가하세요.** 그리고 *"우리가 새로 만든 재료는 없다"* 를
먼저 인정한 뒤 차이를 말하는 것이 훨씬 강합니다.

**추가 문안** (`2_relwork.tex` 의 C 소절 **앞**에 넣으세요):

```latex
\subsection{Optimal Excitation Design}

Designing maximally informative motions for inertial parameter identification
is a mature subject: classical criteria based on the parameter-estimation
covariance date to Gautier and Khalil and to Swevers et al., and remain the
standard formulation \cite{gautier1992exciting,swevers1997optimal}. More
recently, active exploration has been learned rather than optimized offline,
maximizing Fisher information over interaction policies \cite{memmel2024asid}.
In all of these the design variable is the \emph{robot's} trajectory and the
manipulated object is a fixed rigid payload, so the regressor depends on robot
motion alone and the joint configuration is read from encoders. Our setting
inverts this: the robot moves quasi-statically and the information comes from
how the \emph{object} folds, which makes the articulation configuration the
design variable. Three consequences follow that do not arise classically---the
configuration is not actuated by the robot but held by friction, it is observed
by a camera and therefore enters as an errors-in-variables term in the very
quantity being designed, and it changes the object's shape and hence which
grasps and motions remain feasible. Interactive-perception methods likewise
select informative actions on articulated objects, but to recover kinematic
structure rather than mass distribution \cite{hausman2015active}.
```

`reference.bib` 에 `gautier1992exciting`, `swevers1997optimal`,
`memmel2024asid`, `hausman2015active`, `janot2014instrumental` **다섯 항목을
이미 추가했습니다.**

### 3.2 문제 B — C 소절이 "왜 다른가"를 나열로만 말합니다

현재 마지막 문단은 *"In contrast, our VLM-free formulation infers a Bayesian
posterior over part-wise density and post-tare sensor bias..."* 로 **차이를
열거**합니다. 열거는 약합니다. 이제 **원리로** 말할 수 있습니다.

**C 소절 마지막 문단 끝에 추가:**

```
We make this distinction precise in Sec.~III: quasi-static wrench sensing
constrains only the zeroth and first mass moments, so a rigid assembly admits
at most four identifiable density combinations irrespective of the number of
measurements or wrist orientations. Methods that keep the object rigid must
therefore close the remaining directions with priors---homogeneous-part
geometry \cite{nadeau2023sum}, a single-payload model \cite{pfaff2025scalable},
or language-derived material guesses. We close them with measurements, by
changing the articulation configuration itself.
```

### 3.3 EIV/TLS 는 우리 것이라고 주장하지 마세요

`3_method.tex` 초안에서도 그렇게 썼지만, **총최소제곱으로 회귀행렬 오차의
편향을 없앤다는 것은 로봇 식별 문헌의 기존 결과입니다** (IDIM-TLS 계열,
`janot2014instrumental`). 우리 기여는 도구가 아니라 **"오차원이 다름 아닌
설계변수 자신"** 이라는 결합 구조입니다. C 소절에 한 문장으로 명시해 두면
리뷰어가 과장으로 읽지 않습니다.

---

## 4. 반영 순서 (권장)

1. 기계적 수정 5곳 (0장) — 5분
2. Related Work 새 소절 (3.1) — **가장 위험한 구멍부터 막기**
3. 기여 목록 교체 (2.2) — 논문의 성격을 바꾸는 지점
4. Intro 3문단 교체 (2.1)
5. Abstract 교체 (1장)
6. Related Work C 소절 보강 (3.2, 3.3)

---

## 5. 아직 못 한 것

- **컴파일 검증**: 이 PC 에 `pdflatex` 이 없습니다. Overleaf 에 올려
  한 번 돌려 보세요. `3_method.tex` 가 `lemma`/`proposition`/`corollary`/
  `remark` 환경과 `IEEEproof` 를 씁니다 (전자는 `main.tex` 에 추가해 뒀고,
  후자는 IEEEtran.cls 가 제공).
- **`4_exp.tex` 의 실물 수치**: F/T 센서가 없어 전부 `\rev{}` 로 비워 뒀습니다.
  빨간 글씨로 표시되므로 채울 곳을 눈으로 찾을 수 있습니다.
- **`figures/study_scaling.png`**: 스윕이 끝나면 논문 폴더로 복사하고
  `4_exp.tex` 의 `\includegraphics` 주석을 푸세요.
