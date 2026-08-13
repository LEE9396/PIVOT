# Drake RB5 contact/F/T custom-object validation

Date: 2026-07-28

## 1. 검증 질문

기존 configuration-information 실험은 물체 wrench를 수식으로 생성했기
때문에, “실제 로봇과 그리퍼 접촉 없이 이 정보가 어떻게 측정되는가?”라는
한계가 있었다. 이번 검증은 다음 질문을 분리해 확인한다.

1. RB5-850E가 실제 관절 동역학으로 목표 grasp 자세에 도달하는가?
2. PGC-140-50의 두 손가락이 자유 물체를 접촉으로 잡고 들어 올리는가?
3. AFT200 장착 위치의 joint reaction으로 접촉 유래 6축 wrench를 읽을 수
   있는가?
4. torque-hinge joint가 들어 올리는 동안 허용 각도 안에서 유지되는가?
5. 같은 grasp 방식이 2/3/4-link 물체로 확장될 때 어디서 실패하는가?

이 검증은 density estimator의 정확도 실험이 아니다. estimator 앞에 실제로
필요한 `robot → gripper contact → articulated object → wrist reaction`
측정 사슬의 feasibility test다.

## 2. 시뮬레이션 구성

- simulator: Drake discrete `MultibodyPlant`, 1 ms
- robot: RB5-850E URDF, 6개 actuated revolute joint, native PD
- robot mount: 실험 설정의 `[0, -0.74, 0.72] m`, yaw 90°
- F/T 위치: RB5 link6와 AFT200 bracket 사이 weld reaction
- gripper: PGC-140-50, 두 prismatic finger를 독립 actuation
- pad: compliant hydroelastic rubber proxy
- object: `custom_object_cad` Parent와 반복 Child module
- object root: world에 고정하지 않은 free rigid body
- internal joints: revolute joint + torque-hinge friction + holding spring
- table: 2.3 m × 1.1 m, top 0.75 m
- pickup:
  1. 물체가 테이블에 놓인 상태에서 gripper open
  2. 0.6–1.6 s 동안 finger close
  3. 2–4 s 동안 TCP를 준정적으로 100 mm 상승
  4. 5 s까지 hold 후 측정

기본 point contact에서는 패드의 면 접촉 토크를 표현하지 못해 물체가
비현실적으로 회전했다. 최종 장면은 compliant pad와 rigid object 사이의
hydroelastic contact를 사용한다.

## 3. 2-link 검증 결과

기본 grasp는 Parent 중심에서 hinge 방향으로 22.5 mm 이동시켰다. 이는
GT density를 estimator에 제공한 것이 아니라, 긴 articulated assembly를
Parent 중앙만 잡을 때 생기는 중력 모멘트를 줄이는 물리적 grasp-frame
설정이다.

| 지표 | 결과 | gate |
| --- | ---: | ---: |
| TCP 상승 | 89.10 mm | 참고 |
| Parent 상승 | 89.64 mm | ≥ 70 mm |
| Parent/TCP 상승비 | 1.006 | 참고 |
| grasp 상대이동 | 0.80 mm | ≤ 5 mm |
| grasp 상대회전 | 1.08° | ≤ 3° |
| 최대 internal-joint drift | 0.67° | ≤ 1.5° |
| final contact | 4 hydroelastic contacts | ≥ 2 |
| 최종 wrist reaction force | `[2.08, -7.03, -56.40] N` | raw |
| 최종 wrist reaction torque | `[-0.01, -8.49, -2.04] N·m` | raw |

모든 contact/lift/stability gate를 통과했다. 따라서 실제 RB5 model을 사용해
contact-derived wrist signal을 생성하는 경로는 성립한다.

raw wrist reaction에는 AFT bracket, sensor, gripper 자체 중량과 동역학이
포함된다. 실제 density update에는 각 pose에서 tare 또는 known-tool
inverse-dynamics subtraction이 반드시 필요하다.

## 4. 3/4-link stress test

3/4-link는 2-link와 같은 Parent-only parallel-jaw grasp를 그대로 사용했다.
이는 성공을 강제하기 위한 test가 아니라 확장 한계를 찾기 위한 stress
test다.

| Links | Parent lift | grasp 이동 | grasp 회전 | joint drift | 결과 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 89.64 mm | 0.80 mm | 1.08° | 0.67° | PASS |
| 3 | 약 89.16 mm | 3.07 mm | 15.74° | 0.67° | grasp rotation FAIL |
| 4 | 약 52 mm | 44–48 mm | 5.8–6.9° | 0.79–0.84° | lift/slip FAIL |

핵심 해석은 다음과 같다.

- torque-hinge stabilization 자체는 2/3/4-link 모두 약 1° 이내로 가능하다.
- link 수가 늘수록 assembly CoM이 Parent grasp 영역 밖으로 이동한다.
- 같은 PGC Parent-only grasp를 무조건 재사용하면 3-link는 회전하고,
  4-link는 미끄러진다.
- 따라서 4-link 실패를 estimator 실패로 해석하면 안 된다. 먼저
  grasp feasibility gate가 orbit/configuration selection 앞에 있어야 한다.
- 실험에서는 part 수별 grasp frame, 보조 fixture, 또는 더 짧은 module을
  설계한 뒤 같은 contact test를 다시 통과시켜야 한다.

## 5. 연구 formulation과의 연결

accepted configuration \(q_k\)에서 측정되는 raw wrist wrench는

\[
w_{\mathrm{raw},k}
= w_{\mathrm{tool},k}
+Y(q_k,\dot q\approx0,\ddot q\approx0)\rho
+b_{\mathrm{FT}}+\epsilon_k
\]

로 본다. 여기서 estimator 입력은

\[
z_k=w_{\mathrm{raw},k}
-\hat w_{\mathrm{tool},k}
-\hat b_{\mathrm{FT}}
\]

이다. 다음 조건을 만족하지 못한 measurement는 density posterior update에
넣지 않는다.

\[
\Delta q_{\mathrm{joint}}\le1.5^\circ,\quad
\Delta x_{\mathrm{grasp}}\le5\text{ mm},\quad
\Delta R_{\mathrm{grasp}}\le3^\circ.
\]

이번 결과에서 2-link만 세 조건을 모두 만족한다. 3/4-link 신호를 그대로
density 식에 넣으면 contact slip을 물성치 오차로 잘못 흡수한다.

## 6. 재현

```bash
cd robot_learning
./scripts/run_drake_env.sh \
  python scripts/simulate_drake_rb5_contact_ft_custom_object.py \
  --part-count 2
```

실시간 Meshcat:

```bash
./scripts/run_drake_env.sh \
  python scripts/simulate_drake_rb5_contact_ft_custom_object.py \
  --part-count 2 --live
```

회귀 테스트:

```bash
./scripts/run_drake_env.sh \
  python -m unittest tests.test_drake_rb5_contact_ft_custom_object
```

## 7. 남은 실험

1. wrist reaction에서 known tool/gripper wrench를 빼는 tare pipeline 추가
2. grasp close/lift 동안 200 Hz wrench와 contact/slip trace 저장
3. 3/4-link용 grasp 후보를 collision/contact simulation으로 screening
4. 실제 CAD의 측정 mass·CoM·hinge torque로 proxy 값을 교체
5. accepted grasp에서 configuration별 regressor rank와 posterior 감소 연결
6. 실제 RB5+AFT200에서 동일 lift protocol로 sim-to-real wrench 비교
