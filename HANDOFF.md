# 이어받기 — 이 연구와 코드가 무엇인가

새 대화(다른 AI 세션)에서 이 저장소를 이어서 작업할 때 **이 파일 하나만 읽으면**
무슨 연구인지, 코드가 어떻게 짜여 있는지, 지금 어디까지 왔는지 알 수 있도록 썼습니다.

> **AI에게**: 이 파일을 읽은 뒤 [AGENTS.md](AGENTS.md)로 환경을 세우고,
> 자세한 알고리즘은 [my_work/ALGORITHM.md](my_work/ALGORITHM.md),
> 검증 절차는 [my_work/VERIFICATION.md](my_work/VERIFICATION.md)를 보세요.

---

## 1. 한 문장으로

**로봇이 관절 달린 물체를 잡고 자세를 바꿔 가며 손목 힘/토크를 재서, 부위별
밀도(→질량)를 알아내고 시뮬레이션용 URDF를 내놓는다.**

저울로 통째 무게를 재면 **총합**밖에 모릅니다. 물체를 여러 자세로 세워 손목이
받는 힘을 여러 번 읽으면 그 차이에서 **부위별로** 나눌 수 있습니다. 자세를
아무렇게나 고르지 않고, 다음에 어느 자세를 재야 불확실성이 가장 많이 줄어드는지를
계산해서 고릅니다(최적 실험 설계).

```
스캔한 물체 모형 ──> 다음 자세를 고른다 ──> 잡을 수 있나 검사 ──> 손목 힘 측정
                            ^                                        |
                            └──────── 아직 흐릿하면 한 번 더 ─────────┘
                                              |
                                    충분히 또렷하면 URDF 로 내보냄
```

## 2. 장비와 대상

| | |
| --- | --- |
| 팔 | Rainbow Robotics RB5-850E |
| 손목 F/T | AIDIN AFT200-D80-C |
| 그리퍼 | Robotiq 2F-85 (기본) / DH PGC-140-50 |
| 카메라 | Intel RealSense D456 — 관절각은 FoundationPose가 읽음 |
| 시뮬레이터 | Drake 1.54 |
| 대상 물체 | 3-link / 2-link 커스텀 CAD 물체, 실물 스캔한 **데스크 램프** |

데스크 램프는 협업자가 3DGS로 스캔해 만든 배달물입니다
(`assets/desk_lamp_minimal_sim`). GT는 램프를 분해해 저울로 잰 값입니다 —
베이스 396 g, 연결부(Arm) 82 g, Head 84 g.

## 3. 반드시 지켜야 하는 연구 규칙

- **GT(정답)는 채점에만 씁니다.** 탐색·정지 판단에 GT를 쓰면 연구가 무의미해집니다.
  실물에서는 GT를 모르기 때문입니다. `desk_lamp.GROUND_TRUTH`가 쓰이는 곳은
  오차 계산뿐이어야 합니다.
- **정지 조건은 불확실성 기반**입니다(사후 반폭). 치우침 몫을 더할 수 있습니다.
- **주석은 한국어로, "왜"를 적습니다.** 무엇을 하는지는 코드가 이미 말합니다.
- 설계 선택은 **재서** 정합니다. `my_work/study_*.py`가 그 근거입니다.

## 4. 설계 선택과 그 근거 (전부 실험으로 정함)

| 물음 | 답 | 근거 스크립트 |
| --- | --- | --- |
| 후보 자세를 격자에서 고를까, 연속 최적화할까 | **연속 최적화** | `study_continuous.py` |
| 어느 방향으로 기울일 때 정보가 가장 느나 | 직교 3방향이면 회전은 무관 | `study_tilt.py` |
| 최소제곱 vs 총최소제곱 | **TLS** (각도 오차의 치우침 제거, 1.41 % → 0.01 %) | `study_tls.py` |
| D / A / E 최적 중 무엇 | **D-최적** (부위 수를 늘려도 안정) | `study_criterion.py` |
| 언제 멈추나 | 불확실성 + 치우침 몫 | `study_stopping.py` |
| 무거운 힌지(41 g)는 무시해도 되나 | 안 됨 — 부위로 세면 됨 | `study_hinge.py` |
| 각도를 어느 자세에서 읽어야 잘 보이나 | 축이 시선과 직각이면 최악(2.8~6.8배) | `study_startpose.py` |
| 파지점이 어긋나면 | 2 mm에 113 % — **미지수로 함께 풀어야 함** | `study_grasp.py` |

## 5. 코드 지도

```
my_work/
  design_core.py        추정·설계의 핵심 — 후보 선택, TLS, 파지점 추정, 정지 판단
  density_id_drake.py   회귀행렬 A(theta) 와 측정 모형
  density_id_objects.py 물체 정의(ObjectSpec/Part/Joint), Drake plant, 사전분포
  robot_scene.py        RB5+AFT200+그리퍼+테이블 씬, IK, 충돌, 각도 관측성
  path_planning.py      RRT-Connect (관절공간 경로)
  dual_view.py          전체를 묶는 실행 파일 — 화면 2개, sim/deploy 모드
  robot_node.py         실물 로봇 쪽 프로세스 (TCP 반대편)
  pose_bus.py           두 프로세스 사이 JSON 한 줄 규약
  hardware.py           실물 장비 드라이버 자리 (아직 비어 있음)
  desk_lamp.py          스캔 램프를 물체로 물리는 코드 (배달물 구조 자동 인식)
  mesh_props.py         메시에서 부피·도심·관성·색을 뽑는 코드
  grippers.py           PGC-140 / Robotiq 2F-85 정의와 실측표
  calibrate_camera.py   손-눈 캘리브레이션 결과를 파이프라인에 물리는 도구
  study_*.py            설계 선택지 비교 실험 여덟 개
  figures/              study_*.py 가 남기는 그림 (문서가 근거로 참조)
  outputs/              파이프라인 산출물 — URDF·계획·검증 기록 (다시 만들 수 있음)
  scratch/              한 번 쓰고 남긴 진단 스크립트. 아무 데서도 import 안 함
setup/                  bootstrap.sh (설치), doctor.py (12항목 자가 진단)
assets/                 협업자가 준 램프 스캔 배달물
calibration/            카메라 캘리브레이션 결과 (PC마다 다름, git에 안 올림)
third_party/            RB5·PGC·AFT200 (HTD) / Robotiq 2F-85 원본 자산
```

### 핵심 개념 다섯

1. **회귀 모형** `y = A(θ) ρ + ε`. y는 중력 3방향에서 잰 손목 렌치 18개,
   ρ는 부위별 밀도. A는 부위 부피와 도심으로만 만들어집니다.
2. **정보 행렬** `AᵀR⁻¹A`. 다음 자세는 이 행렬식(D-최적)을 가장 키우는 각도.
3. **오차변수(EIV)**: 각도 오차는 A 자체를 틀리게 하므로 잡음이 아니라
   **치우침**입니다. 라운드를 늘려도 안 없어집니다. TLS가 이를 제거합니다.
4. **각도 관측성** `px/deg`: 관절을 1도 돌렸을 때 움직이는 부위가 화면에서
   몇 화소 움직이는가. 이 값이 작으면 FoundationPose가 그 각도를 못 읽습니다.
5. **파지점 어긋남** δ: `τ = τ_model + M·G·(g × δ)`. 총질량 M을 저울로 알면
   δ는 선형 미지수 3개로 모형에 들어옵니다.

## 6. 실행

```bash
cd ~/Desktop/PIVOT
./setup/bootstrap.sh                 # 최초 1회 (파이썬 3.12 + Drake 1.54)
./setup/bootstrap.sh --check         # 12항목 자가 진단

cd my_work
R=../robot_learning/scripts/run_drake_env.sh     # 반드시 이 래퍼로 실행

$R python dual_view.py --mode sim --object desklamp    # 시뮬레이션 검증
$R python dual_view.py --mode sim --object 3link
$R python dual_view.py --mode deploy --object 3link --bus tcp   # 실물, 작업 PC
$R python robot_node.py --host <작업PC IP> --hardware real       # 실물, 로봇 PC
```

화면은 `localhost:7000`(계획·탐색)과 `localhost:7001`(로봇+작업자 UI)입니다.
자세한 것은 [SETUP.md](SETUP.md), 옵션 목록은 [CHANGELOG.md](CHANGELOG.md).

## 7. 지금까지의 결과

**데스크 램프 (시뮬레이션, 실측 GT 대비)**

| 부위 | 실측 질량 | 추정 오차 |
| --- | ---: | ---: |
| 베이스 | 396.0 g | 0.00 % |
| 연결부(Arm) | 82.0 g | 0.18~0.23 % |
| Head | 84.0 g | 0.04 % |

2 라운드에서 목표 불확실성(상대반폭 1 %) 도달.

**커스텀 CAD 물체 (시뮬레이션)** — 둘 다 같은 힌지(실측 41 g)를 부위로 셉니다.

| | 부위 오차 | 힌지 오차 | 라운드 | 후보 | 힌지 토크 여유 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3-link | 0.01~0.06 % | 0.03 % | 2 | 23/25 | 0.37× (joint1) |
| 2-link | 0.03~0.07 % | 0.09 % | 1 | 4/5 | 0.96× |

2-link 는 미지수가 3개(부위 2 + 힌지 1)뿐이라 **1 라운드**에 끝납니다.
힌지 토크 여유가 1 미만이라는 것은 실물에서 그 자세를 그대로 쓰면 관절이
흐른다는 뜻이므로, `--hinge-torque` 와 `--auto-scale` 로 밀도를 낮춰
검증하거나 더 센 힌지를 씁니다.

후보 자세는 그리퍼와 안전 간격에 따라 달라집니다 (여유 10 mm 기준,
램프는 19/25). 자세한 표는 [VERIFICATION.md](my_work/VERIFICATION.md) 4장.

## 8. 최근에 잡은 버그들 (같은 실수를 반복하지 않도록)

전부 **재서** 찾았습니다. 화면만 보고 판단하면 놓칩니다.

| 증상 | 진짜 원인 |
| --- | --- |
| 물체가 손가락 위에 떠 보임 | TCP 높이를 손가락 **몸체 원점**에서 읽음 (패드 면과 39 mm 차이) |
| 죠가 헛돎 | 물체의 **길이 축**이 죠 방향에 놓여 있었음 (x 롤만 주고 있었음) |
| 죠가 활짝 열려 있음 | 개구 각도를 rpy에 구워 넣을 때 **부호가 반대** — 19 mm 요청에 116 mm |
| 링크가 분해돼 보임 | glTF는 **y-up** 규약, 배달물은 z-up → Drake가 90° 돌려 읽음 |
| 로봇이 순간이동 후 거꾸로 움직임 | RRT-Connect가 트리를 맞바꾸며 키우는데 **경로 방향을 안 맞춤** |
| 각도 측정 자세로 가는 이동이 충돌 | 경로를 **지난 라운드 물체 형상**으로 계획 |
| 정지 조건이 영영 만족 안 됨 | 잔차 팽창이 파지점 몫을 예측에 안 넣어 ×105 |
| 새 PC에서 설치가 깨짐 | `requirements/drake.txt`에 **scipy 누락** |

## 9. 남은 일

- **실물 장비 연결**: `hardware.py`의 `Rb5Driver`, `Aft200Sensor`,
  `FoundationPoseSensor`, `run_tare`가 아직 `NotImplementedError`입니다.
  각 클래스가 무엇을 채우면 되는지 스스로 설명합니다.
- **카메라 캘리브레이션**: 실물 실험 전에 `calibrate_camera.py`로 넣어야 합니다.
  안 넣으면 도면상의 명목 위치로 각도 측정 자세를 계산합니다.
- **타어링**: Robotiq만 0.4 kg가 넘습니다(램프 전체가 0.56 kg). 반드시 빼야 합니다.
- **검증 못 한 것**: 스캔 부피 ≠ 재료 부피, 도심 오차, 파지 반복도 —
  전부 시뮬레이션으로는 확인 불가능한 **치우침**입니다.
  [VERIFICATION.md 3장](my_work/VERIFICATION.md)에 정리돼 있습니다.

## 10. 이 저장소

- **https://github.com/LEE9396/PIVOT** (공개)
- 로컬 작업 트리는 `~/Desktop/PIVOT` 자체입니다 (복사본이 아닙니다).
- 팀원 두 명이 작업 PC / 로봇 PC로 씁니다.
- `third_party/`의 PGC-140-50과 AFT200 자산은 재배포 라이선스가 확인되지
  않았습니다. README에 출처와 내림 안내를 적어 두었습니다.
