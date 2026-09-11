# PIVOT 실물 실험 환경 구성 요청

안녕하세요. MeshPCA 쪽 산출물(캘리브레이션 / 타어 / Robotiq / AFT UI / FoundationPose)을
PIVOT 에 이어 붙여서 **실험용 창 4개**를 띄울 수 있게 만들었습니다.

저장소: https://github.com/LEE9396/PIVOT (브랜치 `theory-and-deploy`)
문서: `EXPERIMENT.md` (환경 구성 → 창 4개), `my_work/NAMING.md` (부위 이름 대응)

---

## 지금 상태

**바로 되는 것**
- 환경 구성 (저장소 2개, Drake, venv)
- 자가 진단 4종 (장비 없이)
- 장비 없는 리허설 → 창 1·4 가 실제로 뜹니다 (끝까지 돌려 확인했습니다)
- 타어 로딩, 캘리브레이션 어댑터, 파지점 오버레이, 창 4 라운드 갱신 — 이번에 다 이었습니다

**막혀 있는 것 하나**
- `hardware_real.RbpodoBackend` 가 `NotImplementedError` 입니다.
  `--hardware real` 이 `connect_hardware` 에서 바로 죽어서 **실물 실험이 시작조차 안 됩니다.**
- `~/MeshPCA/pivot/tare_real.py` 가 `hw.Rb5Driver(ip, enable_motion=True, ...)` 와
  `rbpodo.CobotData(ip)` 를 쓰고 있으니, **그쪽 PC 에 도는 구현이 이미 있을 것 같습니다.**
  그 코드를 받을 수 있을까요? `joint_positions / move_to / halt / set_servo` 네 개면 됩니다.

---

## AI 에게 줄 프롬프트

Claude Code 에 아래를 그대로 붙여 넣으시면 됩니다.

```
PIVOT 실물 실험 환경을 이 PC 에 구성해 줘. 캘리브레이션은 이미 끝난 상태야.

저장소 두 개:
  git clone -b theory-and-deploy https://github.com/LEE9396/PIVOT.git ~/Desktop/PIVOT
  git clone https://github.com/Yuseong-Cheon/MeshPCA.git ~/MeshPCA

절차는 ~/Desktop/PIVOT/EXPERIMENT.md 에 전부 있어. 그걸 읽고 따라가 줘.
저장소 규칙은 AGENTS.md, 부위 이름 대응은 my_work/NAMING.md 에 있어.

━━ 가장 먼저 할 일 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

이 PC 에서 RB5 를 실제로 움직이는 코드를 찾아 줘.

~/MeshPCA/pivot/tare_real.py 가 이렇게 부르고 있어:
    hw.Rb5Driver(ip, enable_motion=True, max_speed_deg_s=..., max_accel_deg_s2=...)
    rbpodo.CobotData(ip) / data.request_data(2.0).sdata.jnt_ang[:6]

그런데 PIVOT 의 hardware_real.RbpodoBackend.__init__ 은 NotImplementedError 를
내. 이 PC 에 도는 구현이 있을 테니 찾아서 RbpodoBackend 의 네 함수
(joint_positions / move_to / halt / set_servo)에 옮겨 넣어 줘.
넣은 뒤 hardware_real.py --check 로 확인.

못 찾으면 거기서 멈추고 알려 줘. 이게 없으면 --hardware real 이
connect_hardware 에서 바로 죽어서 실물 실험이 시작조차 안 돼.
(장비 없는 리허설은 이것 없이도 돼.)

━━ 그 다음 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 환경 세 개
   - PIVOT:  ./setup/bootstrap.sh  ->  setup/doctor.py 로 11개 항목 확인
     ★ PIVOT 의 모든 python 실행에는 ../robot_learning/scripts/run_drake_env.sh
       를 앞에 붙여야 해. 맨 python 이면 pydrake 가 조용히 깨져.
   - MeshPCA: python -m venv .venv && pip install -r requirements.txt
   - FoundationPose + SAM3: 별도 체크아웃 (경로만 확인, 없으면 알려 줘)

2. 장비 없이 도는 자가 진단을 전부 돌리고 결과 보고
     gripper_hw.py --check      (9개)
     grasp_overlay.py --check   (7개)
     hardware_real.py --check
     ../setup/doctor.py

3. 캘리브레이션 결과를 PIVOT 형식으로 옮긴다
     cd ~/Desktop/PIVOT/my_work
     $R python import_calibration.py --input <MeshPCA handeye JSON 경로>
   ★ 이걸 안 하면 PIVOT 이 명목 카메라 위치로 조용히 돌아. 라운드를
     늘려도 안 없어지는 치우침이 생겨.

4. 3자세 타어를 잰다 (빈 그리퍼로, 물체 물기 전에)
     PIVOT_WORKDIR=$PWD $R python ~/MeshPCA/pivot/tare_real.py --plan-only
     PIVOT_WORKDIR=$PWD $R python ~/MeshPCA/pivot/tare_real.py \
         --output calibration/aft_tare_current.json --overwrite
   ★ --plan-only 로 먼저 경로를 확인하고, 로봇을 실제로 움직이기 전에
     나한테 확인받아 줘.

5. 창 2 오버레이 패치 적용
     cd ~/MeshPCA && git apply \
       ~/Desktop/PIVOT/my_work/integration/foundationpose_grasp_overlay.patch

6. 런처 설정
     cd ~/Desktop/PIVOT
     cp setup/experiment.conf.example setup/experiment.conf
   이 PC 의 실제 경로·IP·conda 환경 이름으로 채워 줘.

7. 점검
     ./setup/launch_experiment.sh --check
   전부 통과할 때까지 고쳐 줘. 통과하면 알려 줘.

8. 장비 없이 절차 리허설 (창 1·4 가 뜨는지)
     ./setup/launch_experiment.sh --rehearse

━━ 주의 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 로봇을 실제로 움직이는 명령은 반드시 내 확인을 받고 실행해 줘.
- FoundationPose 각도의 부호·영점이 PIVOT 관절각과 아직 안 맞춰졌어
  (my_work/NAMING.md). 실물에서 한 자세를 두 방법으로 읽어 비교해야 해.
  그 전에는 각도 판정이 반대로 나올 수 있어.
- 파지력 기본 205 N 이 램프에 맞는지 확인 필요:
    $R python gripper_hw.py --port /dev/ttyUSB0 --keyboard --plan-mm 44
  물려 놓고 흔들어 보고, 눌린 자국이 남으면 --gripper-force 를 낮춰.
```

---

## GUI 띄우는 코드

```bash
cd ~/Desktop/PIVOT
cp setup/experiment.conf.example setup/experiment.conf   # 경로·IP·물체 설정
$EDITOR setup/experiment.conf

./setup/launch_experiment.sh --check      # 준비물 점검만 (아무것도 안 띄움)
./setup/launch_experiment.sh              # 창 4개 띄우기
./setup/launch_experiment.sh --rehearse   # 장비 없이 창 1·4 만
```

런처는 띄우기 전에 전부 점검하고 **하나라도 없으면 안 띄웁니다.**
틀린 채로 로봇이 움직이는 게 가장 나쁘기 때문입니다. 제 PC 에서 돌리면 이렇게 나옵니다.

```
[실패] RB5 실물 드라이버 (RbpodoBackend)
[실패] 카메라 캘리브레이션
[실패] 3자세 타어
[주의] 파지점이 없습니다 — 지금 만듭니다      <- 자동 생성됩니다
[실패] 창 2 오버레이 패치
6 개 실패 — 위를 고치고 다시 점검하세요
```

창 3 -> 창 2 -> `latest.json` 이 나올 때까지 대기 -> 창 1·4 순으로 띄웁니다.
안 기다리면 창 1 이 각도를 못 읽어 조정 단계에서 헛돕니다.

---

## 창 4개

- **창 1** — 시뮬레이션 탐색. 다음 관절각을 추천하고 파지점(빨간 점)을 표시. Meshcat, 브라우저
- **창 2** — 카메라 뷰. FoundationPose 각도 + 추천 파지점 오버레이. OpenCV 창
- **창 3** — 그리퍼 조작 + F/T 6축 측정값. Tk 창 (MeshPCA `pivot/rb5_ui.py`)
- **창 4** — 밀도 결과. 왼쪽 초기(물 1000) / 오른쪽 탐색 후, 부위별 무지개색 +
  불확실성 막대. 목표 밖이면 "각도를 다시 맞춰 주세요" 표시. Meshcat, 브라우저

## 실험 순서

**1) 파지** — 창1(빨간 점=파지점)과 창2(실물 위 오버레이)를 보며 파지점을 죠 사이에
넣고, 창3 또는 창1 터미널 키보드로 그리퍼를 뭅니다.

```
a / <-  열기 (누르고 있으면 계속)      w / ^  파지력 +
d / ->  닫기 (누르고 있으면 계속)      s / v  파지력 -
```

-> **③ 파지 완료** 를 누릅니다. (무는 명령을 준 적이 없거나, 그리퍼가 "물었다"를
보고하지 않거나, 개구량이 계획과 2mm 넘게 다르면 한 번 더 눌러야 넘어갑니다.)

**2) 각도** — 창1 이 이번 라운드 관절각을 추천합니다. 창2 의 FoundationPose 실측
각도를 보며 물체를 손으로 돌립니다. 허용 오차(±5도) 안에 들면 신호등이 초록으로
바뀝니다. -> **① 각도 확인** -> **② 손 뗐습니다**

각도 판정은 슬라이더가 아니라 **FoundationPose 값(5샘플 중앙값)** 으로 합니다.
트래커가 멈추면 통과시키지 않고 2초마다 이유를 알립니다.

**3) 탐색** — 로봇이 중력 3방향으로 움직이며 측정하고, 창4 오른쪽 열이 갱신됩니다.

- 불확실성이 목표 안 -> `목표 불확실성 도달` 로 정지
- 목표 밖 -> `창 1 의 추천 각도로 다시 맞춰 주세요` -> 2번으로 돌아갑니다

---

## 참고 — 확인해 둔 것

**부위 이름이 배달물 파일명과 다릅니다** (`my_work/NAMING.md`)

- PIVOT `link_1` (파일명 `link_1_head`) = 실제로는 **베이스** = MeshPCA `base`
- PIVOT `link_2` (파일명 `link_2_base`) = 실제로는 **Head** = MeshPCA `head`
- PIVOT `link_3` (파일명 `link_3_support`) = 연결부/팔 = MeshPCA `support` (이건 맞음)

최대 평면 넓이(199.6 vs 62.4 cm²), 도심 순서, 기존 figures 스크립트 셋이 따로 같은
답을 냈습니다.

**부피 오차가 밀도에 미치는 영향** — 부피 10% 오차 기준

- 밀도 오차 약 10% (1:1 로 그대로 옵니다. 라운드를 늘려도 안 없어집니다)
- URDF 질량 오차 약 1.5% (거의 면역입니다)
- URDF 관성 오차 약 6.6%

**파지점 추천** — FoundationPose 가 추적하는 그 메시 위에서 계산합니다. 램프 팔은
전 구간 9~12mm 라 "가장 좁은 자리"에 변별력이 없어서, 물 수 있는 구간이 연달아 가장
길게 이어지는 곳의 한가운데를 잡습니다 (관절 위를 잡으면 파지점이 라운드마다
달라집니다).
