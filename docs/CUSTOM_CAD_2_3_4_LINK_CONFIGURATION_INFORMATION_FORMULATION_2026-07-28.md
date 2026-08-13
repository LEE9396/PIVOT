# Custom CAD 2/3/4-Link Configuration Information 검증과 수식

작성일: 2026-07-28
상태: CAD-derived robust hidden-GT simulation 완료 / real calibration 대기

## 1. 이번 검증의 범위

`custom_object_cad`를 기준으로 다음 modular object family를 만들었다.

| Object | 구성 | Nominal envelope volume |
| --- | --- | --- |
| 2-link | Parent + Child | 1,200 + 960 cm³ |
| 3-link | Parent + Child + Middle | 1,200 + 960 + 960 cm³ |
| 4-link | Parent + Child + Middle + End | 1,200 + 960 × 3 cm³ |

- Parent: 150 × 100 × 80 mm
- Child/Middle/End: 120 × 100 × 80 mm
- hinge gap: 8 mm
- 각 연결부: torque hinge 2개
- Parent는 PGC-140-50의 약 100 mm jaw opening으로 clamp
- RB5-850E–AFT200–PGC–Object 순서

2-link는 실제 Parent/Child CAD를 그대로 사용한다. 3/4-link는 Child envelope를
반복한 modular digital prototype이다. 실제 출력 전에는 양쪽 hinge recess를
가진 Middle CAD를 추가 export해야 한다.

## 2. 현실성을 위해 넣은 오차

Simulator/evaluator만 다음 hidden 값을 안다.

- PLA 출력 재료량과 내부 centered insert mass
- 실제 envelope volume과 제조 scale 편차
- part COM 편차
- joint axis 위치 편차
- grasp transform 편차
- 실제 hinge holding torque
- F/T run-constant bias
- 실제 관절각과 hold drift

Robot은 다음만 사용한다.

- 복원된 exterior-envelope volume
- VLM density mean/covariance
- empty-gripper tare
- 6축 F/T hold average
- D435i 2대와 D456 1대의 관절각/covariance
- nominal CAD joint/grasp geometry

현재 분포:

| 항목 | Simulation 설정 |
| --- | --- |
| volume estimate error | 2.5% σ |
| print scale | 0.3% σ |
| part COM model error | 1.5 mm σ |
| joint-axis error | 1.0 mm σ |
| grasp transform error | 2.0 mm σ |
| fused camera angle error | D435i 0.8°/0.8°, D456 0.5° |
| torque-hinge pair | 2.40 ± 0.25 N·m |
| F/T | 100-sample hold mean + 500-sample tare |

이 값 중 CAD 치수와 STL solid volume은 workspace asset에서 얻었다.
F/T·camera·hinge 분포는 아직 engineering assumption이므로 실측 후 교체한다.

## 3. Hidden part mass

STL에서 얻은 printed-material volume과 PLA 1,240 kg/m³를 사용한다.

- Parent printed mass: 약 0.373 kg
- Child printed mass: 약 0.322 kg

centered hidden insert를 각각 0.10, 0.30, 0.05, 0.40 kg 넣는 controlled
configuration을 사용했다. nominal part mass는 약

\[
(0.473,\ 0.622,\ 0.372,\ 0.722)\ {\rm kg}
\]

이며 4-link total은 약 2.19 kg이다. PGC 권장 payload 3 kg 안에 있지만,
4-link lever arm과 grasp torque는 별도 gate를 통과해야 한다.

## 4. 관절 kinematics

각 관절의 opening angle을 \(\alpha_j\)로 둔다. \(\alpha_j=180^\circ\)는
assembly preview처럼 평평하게 펼친 상태다. 상대 bend는

\[
\theta_j=-(\pi-\alpha_j)
\]

이고 누적 회전은

\[
R_j=R_{j-1}R_y(\theta_j)
\]

이다. CAD length, 8 mm gap과 hinge-top 높이를 사용해 각 part의 F/T-frame
CoM 위치

\[
r_i(q,\gamma)
\]

를 재귀적으로 구한다. \(\gamma\)는 grasp, joint axis, COM, scale 오차를
포함하는 geometry nuisance다.

## 5. 6축 F/T 측정식

외부 envelope volume \(V_i\), effective density \(\rho_i\), sensor-frame
gravity \(g_S\)에 대해 part \(i\)의 wrench column은

\[
y_i(q,\gamma)=
\begin{bmatrix}
V_i g_S\\
V_i r_i(q,\gamma)\times g_S
\end{bmatrix}.
\]

따라서

\[
\boxed{
z_k=Y(q_k,\gamma)\rho+b+\epsilon_k
}
\]

이다. \(b\)는 F/T bias이며 tare observation으로 함께 추정한다.

Robot state를

\[
x=[\rho_1,\ldots,\rho_P,b_{F_x},\ldots,b_{\tau_z}]^\top
\]

로 두면 한 hold의 observation matrix는

\[
H_k=[Y(q_k)\ \ I_6]
\]

이다.

## 6. 현실 오차를 포함한 covariance

단순 F/T noise만 쓰지 않고 camera와 geometry uncertainty를 wrench 공간으로
전파한다.

\[
\boxed{
R_{\mathrm{eff},k}
=R_{\mathrm{FT}}
+J_q\Sigma_qJ_q^\top
+\alpha J_\gamma\Sigma_\gamma J_\gamma^\top
}
\]

- \(J_q=\partial(Y(q)\mu_\rho)/\partial q\)
- \(J_\gamma=\partial(Y(q,\gamma)\mu_\rho)/\partial\gamma\)
- \(\alpha=2.25\): finite-sample model covariance inflation

\(J_\gamma\Sigma_\gamma J_\gamma^\top\)는 80개의 scale·volume·COM·joint-axis·
grasp perturbation으로 근사했다. \(\alpha\)는 현재 residual correlation을
보수적으로 덮기 위한 값이며 실제 calibration residual로 다시 맞춰야 한다.

## 7. Posterior update

\[
K_k=\Sigma_{k-1}H_k^\top
(H_k\Sigma_{k-1}H_k^\top+R_{\mathrm{eff},k})^{-1}
\]

\[
\mu_k=\mu_{k-1}+K_k(z_k-H_k\mu_{k-1})
\]

\[
\Sigma_k=(I-K_kH_k)\Sigma_{k-1}(I-K_kH_k)^\top
+K_kR_{\mathrm{eff},k}K_k^\top.
\]

마지막 식은 numerical positive-semidefinite 성질을 보존하는 Joseph form이다.

## 8. 정보량·rank·null space

GT를 사용하지 않고 Robot posterior density block만으로 configuration의
추가 정보량을 계산한다.

\[
\boxed{
\Delta I_k=
\frac12\log
\frac{\det\Sigma_{\rho,k-1}}
{\det\Sigma_{\rho,k}}
}
\]

관측된 configuration을 쌓은 행렬

\[
\bar Y_K=[Y(q_1)^\top,\ldots,Y(q_K)^\top]^\top
\]

에 대해

\[
\operatorname{rank}(\bar Y_K)=P
\]

이면 모든 density direction이 measurement에 나타난다.

\[
\operatorname{nullity}=P-\operatorname{rank}(\bar Y_K)
\]

는 아직 구분할 수 없는 part-density 조합 수다.

part별 uncertainty 감소는

\[
u_{i,k}^{std}=1-\frac{\sigma_{i,k}}{\sigma_{i,0}}
\]

로 보고한다. 분산 감소율을 쓸 때는

\[
u_{i,k}^{var}=1-\frac{\Sigma_{k,ii}}{\Sigma_{0,ii}}
\]

라고 명시해야 한다.

## 9. Hinge safety filter

joint \(j\)의 downstream gravity torque는

\[
\tau_j(q,\rho)=
\left|
\sum_{i>j}
a_j^\top
\left[
(r_i-o_j)\times(V_i\rho_i g)
\right]
\right|
\]

이다.

prior density, hinge capacity와 geometry uncertainty를 sampling하여

\[
\boxed{
\Pr\left[
\tau_j(q)\le0.8\,\tau^{hold}_j,\ \forall j
\right]\ge0.95
}
\]

인 configuration만 정보량 비교에 넣었다. 실제 hidden execution에서
\(\tau_j>\tau_j^{hold}\)가 발생하면 F/T measurement를 폐기한다.

## 10. Configuration 선택식

안전 후보 집합을 \(\mathcal Q_{\rm safe}\)라 하면 다음 lexicographic
objective를 사용했다.

\[
\boxed{
q_k^\star=
\arg\max_{q\in\mathcal Q_{\rm safe}}
\left(
\operatorname{rank}(\bar Y_{k-1}\oplus Y(q)),
\Delta I(q),
-U_{0.95}(q)
\right)
}
\]

\(U_{0.95}\)는 hinge-capacity usage의 95 percentile이다.

즉,

1. null-space를 먼저 줄이고
2. 같은 rank에서는 정보량을 최대화하고
3. 그래도 같으면 hinge torque margin이 큰 상태를 고른다.

## 11. 선택된 configuration과 nominal trace

### 2-link

| 순서 | Opening angle | rank/null | ΔI | 최종 density error |
| --- | --- | --- | ---: | --- |
| 1 | `(165°)` | `2/0` | 4.543 nat | 4.2%, 1.0% |

최종 posterior/prior std 비:

\[
(0.145,\ 0.102)
\]

### 3-link

| 순서 | Opening angles | rank/null | ΔI |
| --- | --- | --- | ---: |
| 1 | `(165°,105°)` | `2/1` | 4.532 nat |
| 2 | `(75°,75°)` | `3/0` | 1.676 nat |

최종 posterior/prior std 비:

\[
(0.176,\ 0.180,\ 0.162)
\]

### 4-link

| 순서 | Opening angles | rank/null | ΔI |
| --- | --- | --- | ---: |
| 1 | `(75°,165°,75°)` | `2/2` | 4.585 nat |
| 2 | `(165°,105°,75°)` | `3/1` | 1.683 nat |
| 3 | `(75°,75°,75°)` | `4/0` | 0.969 nat |

최종 posterior/prior std 비:

\[
(0.371,\ 0.272,\ 0.314,\ 0.229).
\]

첫 configuration이 가장 많은 총 정보를 주지만 rank는 2뿐이다. 세 번째
configuration의 정보량은 0.969 nat로 작아도 마지막 null direction을 제거해
full rank를 만든다는 점이 중요하다.

## 12. 300회 Monte Carlo

| Object | Density relative RMSE | p95 density error | 95% interval coverage |
| --- | --- | --- | --- |
| 2-link | 4.6%, 4.4% | 8.7%, 8.0% | 98.3%, 94.3% |
| 3-link | 5.8%, 5.4%, 3.7% | 11.3%, 9.7%, 6.9% | 97.3%, 99.3%, 100% |
| 4-link | 9.8%, 9.0%, 11.1%, 6.0% | 18.3%, 17.7%, 17.4%, 10.7% | 100%, 97.7%, 100%, 100% |

해석:

- link 수가 늘면 같은 F/T에서 density column이 더 강하게 결합되어 오차가
  증가한다.
- 4-link 평균 RMSE는 약 6–11%지만 p95에서는 일부 part가 17–18%에 도달한다.
- 4-link coverage 100%는 좋은 정확도라기보다 현재 covariance inflation이
  보수적이라는 뜻이다.
- 2-link Part 2 coverage 94.3%는 목표 95%와 가깝다.

## 13. Hinge torque 확인

4-link nominal trace의 actual hidden torque/capacity:

| 상태 | joint torque [N·m] | capacity [N·m] |
| --- | --- | --- |
| 1 | 1.18, 0.96, 0.52 | 2.17, 2.56, 2.16 |
| 2 | 1.40, 0.37, 0.51 | 동일 |
| 3 | 1.17, 0.94, 0.12 | 동일 |

모든 상태가 actual holding capacity 아래이며, Robot prior 기준 95% safety
screen도 통과했다.

## 14. 시각화

Drake scene에는 다음을 넣었다.

- 실제 `parent_body/lid`, `child_body/lid` CAD를 OBJ로 변환해 표시
- RB5-850E
- AFT200 wrist sensor
- PGC-140-50, joint position 20 mm/jaw → 약 100 mm opening
- 검은 2.3 × 1.1 m table과 배경
- D435i×2와 D456 관측선
- configuration 설정 → 10 cm lift → hold → lower
- 상태별 rank/nullity, information gain, Robot density/std, evaluator GT error

현재 PGC는 configuration 중 slip이 없다고 가정한 fixed-jaw rigid-grasp다.
접근·finger closing·contact pressure·slip은 아직 별도 physics gate다.

## 15. 남은 현실성 한계

- 3/4-link Middle/End의 양면 hinge recess CAD 미출력
- torque hinge의 실제 측정 curve 없음
- F/T·camera noise가 실측 calibration이 아님
- insert 고정 후 실제 part CoM 측정 전
- exterior-envelope mesh를 아직 nominal box volume으로 대체
- movable jaw/contact/slip dynamics 미검증
- covariance inflation 2.25가 real residual로 calibration되지 않음

따라서 현재 결과는 기존 임의 link보다 훨씬 현실적인 CAD-derived validation이고,
수식·상태 선택·난이도 scaling을 검증한다. 하지만 논문의 real-system 최종
오차로 인용하려면 출력물과 센서 calibration으로 동일 실험을 다시 실행해야 한다.

## 16. 재현

```bash
cd robot_learning

./scripts/run_drake_env.sh python \
  scripts/run_custom_cad_configuration_information_experiment.py \
  --trials 300

./scripts/run_drake_env.sh python \
  tests/test_custom_cad_configuration_information.py

./scripts/run_drake_env.sh python \
  scripts/visualize_drake_custom_cad_configuration_information.py \
  --part-count 4
```

산출물:

- `progress/artifacts/2026-07-28/custom_cad_configuration_information/custom_cad_results.json`
- `custom_cad_2_link_trace.png`
- `custom_cad_3_link_trace.png`
- `custom_cad_4_link_trace.png`
