# Drake RB5 articulated-object validation

Drake에서 RB5-850E, AFT200 wrist F/T mount, DH Robotics PGC-140-50,
2/3/4-link custom articulated object를 실제 동역학·접촉으로 검증하는
재현용 저장소입니다.

핵심 질문은 “로봇이 물체를 실제로 잡지 않은 synthetic wrench”가 아니라,
다음 측정 사슬이 성립하는지입니다.

```text
RB5 joint dynamics
  → movable PGC fingers
  → hydroelastic grasp contact
  → free articulated object lift
  → AFT200 mount reaction wrench
  → grasp / hinge acceptance gate
```

## 현재 검증 결과

| Links | Parent lift | Grasp drift | Grasp rotation | Joint drift | Verdict |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 89.64 mm | 0.80 mm | 1.08° | 0.67° | PASS |
| 3 | 89.16 mm | 3.07 mm | 15.74° | 0.67° | grasp rotation FAIL |
| 4 | 54.52 mm | 43.88 mm | 6.02° | 0.84° | lift/slip FAIL |

3/4-link 실패는 숨기지 않습니다. 동일한 Parent-only grasp가 길어진
assembly의 중력 모멘트를 지탱하지 못한다는 결과이며, density update 전에
grasp feasibility gate가 필요하다는 근거입니다.

자세한 해석은
[validation report](docs/DRAKE_RB5_CONTACT_FT_CUSTOM_OBJECT_VALIDATION_2026-07-28.md)
를 참고하세요.

## 설치

Ubuntu 22.04와 Python 3.12 기준입니다.

```bash
git clone --recurse-submodules \
  https://github.com/soonawg/drake_validation_test.git
cd drake_validation_test/robot_learning

conda create --prefix .venv-drake-1.54-py312 python=3.12 pip -y
./.venv-drake-1.54-py312/bin/python -m pip install \
  -r requirements/drake.txt
```

이미 clone한 뒤라면 submodule만 초기화할 수 있습니다.

```bash
git submodule update --init --recursive
```

## 실행

2-link headless validation:

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

출력 JSON을 별도로 저장하려면:

```bash
./scripts/run_drake_env.sh \
  python scripts/simulate_drake_rb5_contact_ft_custom_object.py \
  --part-count 3 --output ../results/3link-rerun.json
```

회귀 테스트:

```bash
./scripts/run_drake_env.sh \
  python -m unittest tests.test_drake_rb5_contact_ft_custom_object
```

정지 pose를 정보량으로 선택하면서 2-link 질량·밀도를 순차 추정하려면:

```bash
./scripts/run_drake_env.sh \
  python scripts/run_drake_static_mass_tracking.py \
  --joint-min-deg 20 --joint-max-deg 120 --steps 4
```

이 실험은 CAD 부피·부피중심, 물체별 관절 한계각, FoundationPose를
모사한 정지각과 tare된 F/T만 estimator 입력으로 사용합니다. Drake GT
질량과 sensor bias는 evaluator에서만 결과 검증에 사용합니다.

RB5–PGC 접촉, 3-camera 각도 융합, AFT200 정지 평균까지 연결한 전체
2-link 시뮬레이션:

```bash
./scripts/run_drake_env.sh \
  python scripts/run_drake_contact_mass_pipeline.py \
  --opening-angle-deg 120 --steps 5 \
  --target-relative-95-half-width 0.05 --live
```

여기서 카메라 단계는 실제 FoundationPose 네트워크 대신 각 카메라의
calibrated noise를 GT link pose에 적용한 proxy입니다.
`--steps`는 최대 측정 수이며, 모든 파트 질량의 posterior 95% 구간
상대 반폭이 5% 이하이고 regressor가 full rank이면 더 일찍 종료합니다.

Drake의 3-camera RGB-D·링크 mask·intrinsics를 공식 FoundationPose
scorer/refiner에 넣고, 첫 프레임 등록 후 실시간 tracking 결과를 질량식까지
전달하려면:

```bash
./scripts/run_drake_env.sh \
  python scripts/run_drake_foundationpose_experiment.py
```

이 명령은 기존 `bundlesdf` conda 환경과
`/home/cheon/Desktop/Lab/pipline/repo_chain/FoundationPose`의 weights를
재사용합니다.

3-link를 180° 일자로 테이블에 놓고, hinge를 계속 unlock한 채
`접근 → grasp → lift → 손목 측정`으로 두 관절각과 세 질량을 추정하려면:

```bash
./scripts/run_drake_env.sh \
  python scripts/run_drake_foundationpose_experiment.py \
  --part-count 3 --output-dir results/foundationpose_drake_3link
```

각 hold의 측정식은
`b_i = w_loaded_i - w_empty_i = A_i(q_i, g_i, r_ij) m`이며, 누적된
hold에서 `||W(A m - b)||²`를 최소화한 뒤 질량을 0 이상으로 투영합니다.
상승 중에는 링크별
`f_j/m_j = a_cj - g`,
`τ_j/m_j = r_j × (a_cj - g) + Ī_j α_j + ω_j × Ī_jω_j`를 쌓은
동적 regressor도 낮은 가중치로 결합합니다. 테이블 접촉과 이륙 충격
샘플은 제외합니다.
GT 질량은 추정 입력이 아니라 evaluator와 손목 자세의 사전 선택에만
사용합니다. 총질량 3 kg, grasp drift, F/T 정지, 3-camera 가시성,
wrench innovation과 질량 변화 gate를 통과한 hold만 식에 들어갑니다.

다른 질량 조합은 같은 식에 직접 넣어 검증할 수 있습니다.

```bash
./scripts/run_drake_env.sh \
  python scripts/run_drake_contact_mass_pipeline.py \
  --part-count 3 --vary-internal-angles --steps 3 \
  --opening-angle-deg 180 --initial-opening-angle-deg 180 \
  --wrist-pitch-sequence-deg -20 -10 0 \
  --part-masses-kg 0.55 0.70 0.35
```

손목 자세를 수동 지정하지 않고, 랜덤 질량 calibration과 기존 3-camera
추적 결과에서 안전하고 정보량이 큰 정지 자세를 자동 선택하려면:

```bash
./scripts/run_drake_env.sh \
  python scripts/run_drake_contact_mass_pipeline.py \
  --part-count 3 --vary-internal-angles \
  --opening-angle-deg 180 --initial-opening-angle-deg 180 \
  --auto-plan --planned-steps 3 \
  --camera-calibration-result \
    results/foundationpose_drake_3link_v2/foundationpose_result.json
```

플래너는 후보 자세 조합마다 랜덤 질량의 최대 추정 오차를 먼저
최소화하고, 동률이면 condition number와 log-det 정보량으로 선택합니다.
GT는 이 offline action calibration에만 사용하고 실제 질량 추정 입력에서는
제외합니다.

## 주요 파일

- `robot_learning/scripts/simulate_drake_rb5_contact_ft_custom_object.py`:
  contact/lift/F/T 검증 본체
- `robot_learning/scripts/visualize_drake_rb5_hammer_payload.py`:
  pinned RB5+AFT200+movable PGC model builder
- `robot_learning/configs/experiments/icra_realistic_lab_scene_v1.json`:
  2.3 m × 1.1 m table, robot mount, D435i×2/D456 배치
- `custom_object_cad/drake/`: Parent/Child CAD-derived OBJ
- `results/`: canonical 2/3/4-link 결과
- `docs/`: formulation과 결과 해석

## 주의

- wrist reaction은 raw 값입니다. known tool/gripper dynamics와 F/T bias를
  제거한 뒤 estimator에 넣어야 합니다.
- 3/4-link는 현재 grasp gate를 통과하지 않으므로 density 결과 생성에
  사용하면 안 됩니다.
- 실제 CAD mass, CoM, hinge torque가 측정되면 proxy 값을 교체해야 합니다.
