# 잡아 보고 무게를 알아내기 — 관절 물체의 부위별 밀도 식별

로봇이 관절 달린 물체를 **잡고 자세를 바꿔 가며 손목 힘/토크를 재서**,
부위마다의 밀도(따라서 질량)를 알아내고 시뮬레이션에 바로 쓸 수 있는
URDF 를 내놓는 연구 코드입니다.

저울로 통째 무게를 재면 **총합**밖에 모릅니다. 이 파이프라인은 물체를
여러 자세로 세워 손목이 받는 힘을 여러 번 읽고, 그 차이에서 **부위별로**
나눕니다. 자세를 아무렇게나 고르지 않고, 다음에 어느 자세를 재야 가장 많이
알게 되는지를 계산해서 고릅니다.

```
스캔한 물체 모형 ──> 다음 자세를 고른다 ──> 잡을 수 있나 검사 ──> 손목 힘 측정
                            ^                                        |
                            └──────── 아직 흐릿하면 한 번 더 ─────────┘
                                              |
                                    충분히 또렷하면 URDF 로 내보냄
```

**처음이라면 [my_work/VERIFICATION.md](my_work/VERIFICATION.md) 를 먼저 보세요.**
시뮬레이션 검증과 실물 로봇 검증, 두 길을 그림과 함께 단계별로 설명합니다.

---

## 빨리 시작하기

```bash
git clone https://github.com/LEE9396/PIVOT.git ~/Desktop/PIVOT
cd ~/Desktop/PIVOT
./setup/bootstrap.sh                    # 파이썬 3.12 + Drake 1.54 + 자가 진단

cd my_work
../robot_learning/scripts/run_drake_env.sh python dual_view.py \
    --mode sim --object desklamp        # 브라우저로 localhost:7000, 7001
```

자세한 설치·문제 해결은 [SETUP.md](SETUP.md).
AI 에게 시킬 거라면 [AGENTS.md](AGENTS.md) 를 읽히세요.
`git pull` 뒤 무엇이 달라졌는지는 [CHANGELOG.md](CHANGELOG.md).

---

## 어떻게 돌리나 — 두 가지 검증

같은 알고리즘 코드를 두 가지로 검증합니다. **Drake 가 대신하던 셋**(팔을
움직이는 것 · 힘을 계산하는 것 · 각도를 읽는 것)이 실물로 바뀔 뿐,
**알고리즘 코드는 한 줄도 안 바뀝니다.**

모든 명령은 `my_work/` 에서, 반드시 이 래퍼를 앞에 붙입니다.
그냥 `python` 을 부르면 pydrake 임포트가 조용히 깨집니다.

```bash
cd ~/Desktop/PIVOT/my_work
R=../robot_learning/scripts/run_drake_env.sh
```

### ① 시뮬레이션 검증 — 장비 없이

알고리즘이 맞는지 여기서 걸러냅니다. 정답을 알고 있으므로 채점이 됩니다.

```bash
# 숫자만 보기 (20초, 조작 없음) — 최종 산출물 URDF 까지 만든다
$R python export_urdf.py --object desklamp
$R python export_urdf.py --object 3link

# 전 과정을 화면으로 (3~4분) — 브라우저 탭 두 개를 나란히 연다
$R python dual_view.py --mode sim --object desklamp
$R python dual_view.py --mode sim --object 3link --joint-range-deg 0 180

# 사람 조작 없이 자동으로 (검사용)
$R python dual_view.py --mode sim --object 3link --autostart --auto-adjust
```

화면은 `localhost:7000`(계획·탐색)과 `localhost:7001`(로봇 + 작업자 UI).
원격 PC 라면 `ssh -L 7000:localhost:7000 -L 7001:localhost:7001 사용자@PC`.

설계 선택을 바꿔 보려면 `--select` `--criterion` `--estimator` `--stop-rule`.
각 선택지를 비교한 실험은 `study_*.py` 여덟 개입니다.

### ② 실물 로봇 검증 — PC 한 대면 됩니다

**시작 전에 반드시 끝나 있어야 하는 것 넷.** 이걸 안 하면 알고리즘이
아무리 정확해도 답이 틀립니다.

| | 무엇 | 왜 |
| --- | --- | --- |
| 1 | [카메라 캘리브레이션](calibration/README.md) | 안 하면 도면상 명목 위치로 각도 측정 자세를 계산합니다 |
| 2 | **타어링** | 센서는 그리퍼(2F-85 는 0.4 kg)까지 다 읽습니다. 램프 전체가 0.56 kg 입니다 |
| 3 | `hardware.py` 채우기 | `Rb5Driver` `Aft200Sensor` `FoundationPoseSensor` |
| 4 | 저울로 총무게 | 사전분포와 파지점 어긋남 풀이에 들어갑니다 |

```bash
# 0) 장비 없이 배선만 확인 — 실물 코드가 지나가는 길을 그대로 밟는다
$R python dual_view.py --mode deploy --bus local --hardware sim \
    --object 3link --autostart --auto-adjust

# 1) 카메라 캘리브레이션 (보정판을 그리퍼에 볼트로 고정한 상태)
python3 calib_detect.py --serve --port 5566 --square-mm <자로 잰 값>   # 터미널 1
$R python calibrate_camera.py --run --poses 20                        # 터미널 2

# 2) 실제 실험
$R python dual_view.py --mode deploy --bus local --hardware real \
    --object desklamp --max-rounds 8 --move-duration 8
```

실험 중 순서는 이렇습니다. 로봇은 2~5 단계 동안 **절대 움직이지 않습니다.**

```
1. 작업자가 물체를 그리퍼에 물린다     서보 오프 · 버튼으로 확인
2. 로봇이 시작 자세로 이동             실제 팔 자세를 읽어 경로를 계획한다
3. 작업자가 물체 관절각을 손으로 맞춘다  로봇 정지 · 신호등 초록
4. 각도 확인 → 손 뗌 확인               버튼 두 개
5. FoundationPose 가 실제 각도를 읽는다
6. 천천히 이동하며 중력 3방향 측정      신호등 빨강
   └ 목표 불확실성에 도달할 때까지 3~6 반복 → URDF
```

> 검출기(`calib_detect.py`)만 시스템 python 에서 돕니다. opencv 가 필요한
> 곳이 거기뿐이라 고정된 Drake 환경을 안 건드리려는 것이고,
> FoundationPose 도 같은 구조입니다.

로봇을 계획 프로세스에서 떼고 싶으면 `--bus tcp` + `robot_node.py` 를 쓰지만,
**대개 필요 없습니다.** 이유는 [VERIFICATION.md](my_work/VERIFICATION.md) 2장.

### 산출물

```
my_work/outputs/estimated_desklamp.urdf    <- 최종 산출물
```

램프는 배달물 원본 URDF 를 그대로 두고 **링크별 질량·무게중심·관성텐서만**
바꿔 넣습니다. 형상·관절·충돌메시는 원본 그대로입니다.

```bash
./view_urdf.sh                    # Drake 뷰어로 열어 본다
```

절차의 근거와 세부는 [my_work/VERIFICATION.md](my_work/VERIFICATION.md),
파이프라인이 어떤 순서로 무엇을 하는지는
[my_work/PIPELINE.md](my_work/PIPELINE.md), 알고리즘 자체의 설명은
[my_work/ALGORITHM.md](my_work/ALGORITHM.md).

---

## 지금까지 나온 결과

실물 스캔한 스탠드 램프(3부위, 총 562 g)를 시뮬레이션에서 돌린 결과입니다.
정답은 램프를 분해해 저울로 잰 값이며 **채점에만** 씁니다.

| 부위 | 실측 질량 | 추정 오차 | |
| --- | ---: | ---: | --- |
| 베이스 | 396.0 g | 0.01 % | |
| 연결부(Arm) | 82.0 g | 0.21 % | 로봇이 잡는 곳 |
| Head | 84.0 g | 0.03 % | |

2 라운드 만에 목표 불확실성(상대반폭 1 %)에 도달했습니다.
후보 자세 25 개 중 20 개가 도달·충돌·경로 검사를 모두 통과했습니다.

설계 선택지들은 각각 실험으로 비교했습니다 (`my_work/study_*.py`).

| 물음 | 답 | 근거 |
| --- | --- | --- |
| 후보를 격자에서 고를까, 연속 최적화할까 | **연속 최적화** | `study_continuous.py` |
| 어느 방향으로 기울일 때 정보가 가장 느나 | 직교 3방향이면 회전은 무관 | `study_tilt.py` |
| 최소제곱 vs 총최소제곱 | **TLS** (각도 오차의 치우침을 없앰, 1.41 % → 0.01 %) | `study_tls.py` |
| D / A / E 중 무엇 | **D-최적** (부위 수를 늘려도 안정) | `study_criterion.py` |
| 언제 멈추나 | 불확실성 기반 + 치우침 몫 | `study_stopping.py` |
| 무거운 힌지(41 g)는 무시해도 되나 | 안 됨 — 부위로 세면 됨 | `study_hinge.py` |

---

## 무엇이 들어 있나

```
my_work/          알고리즘·화면·실물 연결 (연구 코드는 전부 여기)
  VERIFICATION.md   두 검증 방법 설명   <- 먼저 읽기
  PIPELINE.md       파이프라인 지도 — 단계별로 왜/무엇을/어디에
  ALGORITHM.md      알고리즘 설명 (고등학생 수준)
  README.md         코드 파일 하나하나의 설명
setup/            새 PC 환경 구성 (bootstrap.sh, doctor.py)
assets/           협업자가 준 스탠드 램프 스캔 배달물
robot_learning/   Drake 환경 실행기와 RB5 씬 유틸
third_party/      RB5·PGC·AFT200 (HTD) / Robotiq 2F-85 원본 자산
docs/, progress/  이전 검증 기록
```

## 장비

| | |
| --- | --- |
| 팔 | Rainbow Robotics RB5-850E |
| 손목 F/T | AIDIN AFT200-D80-C |
| 그리퍼 | Robotiq 2F-85 (기본) / DH PGC-140-50 |
| 카메라 | Intel RealSense D456 (관절각 관측: FoundationPose) |
| 시뮬레이터 | Drake 1.54 |

---

## 제3자 자산에 대하여

`third_party/` 아래 파일들은 이 연구의 산물이 아니라 원저작자의 것입니다.
출처와 라이선스는 [third_party/HTD/THIRD_PARTY_ASSETS.md](third_party/HTD/THIRD_PARTY_ASSETS.md)
에 정리돼 있습니다.

| 자산 | 출처 | 라이선스 |
| --- | --- | --- |
| RB5-850E (`rbpodo_description`) | Rainbow Robotics | Apache-2.0 |
| Robotiq 2F-85 | [a-price/robotiq_arg85_description](https://github.com/a-price/robotiq_arg85_description) | 상단 저장소 참조 |
| DH PGC-140-50 | DH-Robotics `dh_gripper_ros` | 원본에 라이선스 파일 없음 |
| AIDIN AFT200-D80-C | AIDIN 공식 STEP 모델의 시각화 파생물 | 별도 CAD 배포 라이선스 확인 안 됨 |

뒤 두 항목은 재배포 허락이 확인되지 않은 상태입니다. 원저작자의 요청이 있으면
해당 파일을 내리겠습니다. 이 저장소를 쓰실 때 그 자산을 상업적으로 재배포하지
마시고, 필요하면 원저작자에게 직접 받으세요.

## 인용·문의

연구 관련 문의는 이슈로 남겨 주세요. 코드는 연구용으로 자유롭게 참고하셔도 됩니다.
