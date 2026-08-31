# 실물 실험 — 창 4개 띄우기

이 문서 하나로 새 PC 에서 환경 구성부터 실험 시작까지 갑니다.
알고리즘 설명은 [README.md](README.md), 저장소 규칙은 [AGENTS.md](AGENTS.md).

---

## 창 4개가 무엇인가

| 창 | 무엇 | 어느 저장소 / 환경 |
| --- | --- | --- |
| **1** | 시뮬레이션 탐색 — 다음 **관절각을 추천**하고 **파지점**을 표시 | PIVOT · Drake 환경 |
| **2** | **카메라 뷰** — FoundationPose 각도 + 추천 파지점 오버레이 | MeshPCA · FoundationPose 환경 |
| **3** | **그리퍼 조작 + F/T 측정값** | MeshPCA · venv (Tk) |
| **4** | **밀도 결과** — 왼쪽 초기(물 1000) / 오른쪽 탐색 후, 부위별 무지개색 + 불확실성 | PIVOT · Drake 환경 |

창 1·4 는 브라우저 탭(Meshcat), 창 2 는 OpenCV 창, 창 3 은 Tk 창입니다.

## 사용 순서

```
1. 창1·창2 를 보며 물체의 파지점을 그리퍼 죠 사이에 넣는다
      -> 창3 (또는 창1 터미널의 키보드) 로 그리퍼를 닫아 문다
2. 파지 완료를 누르면 창1 이 이번 라운드의 관절각을 추천한다
      -> 창2 의 FoundationPose 각도를 보며 물체 관절을 그 각도로 맞춘다
3. 각도 확인을 누르면 로봇이 중력 3방향으로 움직이며 측정한다
      -> 창4 의 '탐색 후' 열이 갱신된다
      -> 불확실성이 목표 안이면 정지, 아니면 2번으로 돌아간다
```

---

## 0. 준비물

- Ubuntu + NVIDIA GPU (FoundationPose 용)
- RB5-850E (`192.168.50.51`), AIDIN AFT200, Robotiq 2F-85 (`/dev/ttyUSB0`)
- RealSense D456
- 저장소 두 개: `LEE9396/PIVOT`, `Yuseong-Cheon/MeshPCA`

## 1. 저장소 받기

```bash
git clone https://github.com/LEE9396/PIVOT.git ~/Desktop/PIVOT
git clone https://github.com/Yuseong-Cheon/MeshPCA.git ~/MeshPCA
```

## 2. 환경 세 개

**① PIVOT (Drake 1.54 고정)** — 창 1·4

```bash
cd ~/Desktop/PIVOT && ./setup/bootstrap.sh
cd my_work && ../robot_learning/scripts/run_drake_env.sh python ../setup/doctor.py
```

11 개 항목이 `[통과]` 면 됩니다. 포트 7000/7001 항목은 다른 화면이 이미
떠 있으면 실패로 나오는데 고장이 아닙니다.

> **모든 PIVOT 실행에 래퍼를 붙입니다.** 편하게 쓰려면:
> ```bash
> cd ~/Desktop/PIVOT/my_work
> export R=../robot_learning/scripts/run_drake_env.sh
> ```
> 맨 `python` 으로 부르면 pydrake 가 조용히 깨집니다.

**② MeshPCA** — 창 3

```bash
cd ~/MeshPCA && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**③ FoundationPose + SAM3** — 창 2

별도 체크아웃이 필요합니다 (MeshPCA 요구사항에 안 들어 있습니다).

- <https://github.com/Yuseong-Cheon/Foundation_pose_edit>
- <https://github.com/facebookresearch/sam3>

자세한 것은 `~/MeshPCA/foundationpose/README.md`.

## 3. 장비 확인 (움직이기 전에)

```bash
cd ~/Desktop/PIVOT/my_work

# 그리퍼 — 키보드로 여닫아 본다 (a/d 개구, w/s 파지력)
$R python gripper_hw.py --port /dev/ttyUSB0 --keyboard --plan-mm 44

# 그리퍼 프레임/환산 자가 진단 (장비 없이)
$R python gripper_hw.py --check

# 파지점 오버레이 투영 계산 (장비 없이)
$R python grasp_overlay.py --check

# 안전 로직 (장비 없이)
$R python hardware_real.py --check
```

## 4. 캘리브레이션 · 타어 (세션 전에 한 번)

```bash
# 1) 카메라 <-> 로봇 (EasyHeC -> MeshPCA 형식)
cd ~/MeshPCA && python calibration/import_easyhec.py ...   # CALIBRATION.md 참고

# 2) 그 결과를 PIVOT 형식으로 옮긴다  ★ 안 하면 PIVOT 은 명목 위치로 돕니다
cd ~/Desktop/PIVOT/my_work
$R python import_calibration.py --input ~/MeshPCA/calibration/handeye_d456.json

# 3) 3자세 타어 — 빈 그리퍼로. 물체를 물기 전에 한다
PIVOT_WORKDIR=$PWD $R python ~/MeshPCA/pivot/tare_real.py --plan-only   # 먼저 계획만
PIVOT_WORKDIR=$PWD $R python ~/MeshPCA/pivot/tare_real.py \
    --output calibration/aft_tare_current.json --overwrite
```

> **타어와 측정의 개구량이 같아야 합니다.** 다르면 그리퍼 무게 몫이 안 빠집니다.
> 그리고 세션에 `--tare-file` 로 그 JSON 을 반드시 넘겨야 합니다 — 안 넘기면
> 첫 측정에서 "missing tare for gravity direction" 으로 멈춥니다.

## 5. 파지점 뽑기 (물체마다 한 번)

FoundationPose 가 **추적하는 그 메시** 위에서 계산해야 카메라 화면에 맞습니다.

```bash
cd ~/Desktop/PIVOT/my_work
$R python grasp_target.py \
    --mesh ~/MeshPCA/foundationpose/assets/lamp/support_metric_watertight.ply \
    --part support --out outputs/grasp_target_lamp.json
```

## 6. 창 2 에 오버레이 패치 붙이기 (한 번)

```bash
cd ~/MeshPCA
git apply ~/Desktop/PIVOT/my_work/integration/foundationpose_grasp_overlay.patch
```

기존 동작은 안 바뀌고 `--grasp-target` 플래그가 생깁니다.

---

## 7. 창 4개 띄우기

**터미널 4개**를 엽니다.

### 창 3 — 그리퍼 + F/T

```bash
cd ~/MeshPCA && source .venv/bin/activate
python pivot/rb5_ui.py \
    --host 192.168.50.51 \
    --port /dev/ttyUSB0 \
    --tare ~/Desktop/PIVOT/my_work/calibration/aft_tare_current.json
```

### 창 2 — 카메라 뷰 + 파지점 오버레이

```bash
conda activate bundlesdf
cd ~/MeshPCA
python foundationpose/run_desk_lamp_live.py \
    --foundationpose-root /path/to/FoundationPose \
    --sam3-root /path/to/sam3 --sam3-python /path/to/sam3/env/bin/python \
    --mesh-dir foundationpose/assets/lamp \
    --init-rgb /tmp/lamp_live_rgb.png --init-depth /tmp/lamp_live_depth_m.npy \
    --intrinsics /tmp/lamp_live_intrinsics.json --masks /tmp/lamp_sam3_masks \
    --output /tmp/lamp_foundationpose_live \
    --grasp-target ~/Desktop/PIVOT/my_work/outputs/grasp_target_lamp.json \
    --grasp-part support
```

각도가 `/tmp/lamp_foundationpose_live/latest.json` 에 계속 쌓입니다.
창 1 이 그 파일을 읽습니다.

### 창 1 + 창 4 — 탐색·파지점 / 밀도 결과

한 명령이 두 화면을 띄웁니다.

```bash
cd ~/Desktop/PIVOT/my_work
$R python dual_view.py \
    --mode deploy --bus local --hardware real \
    --object desklamp --grasp pinch --grasp-part link_3 \
    --prior water --target 0.05 --max-rounds 8 \
    --gripper-port /dev/ttyUSB0 --gripper-force 205 \
    --pose-file /tmp/lamp_foundationpose_live/latest.json \
    --tare-file calibration/aft_tare_current.json \
    --aft-host 192.168.50.51 --robot-host 192.168.50.51
```

출력에 주소가 찍힙니다.

```
  [1] 계획·탐색 화면   http://localhost:7000     <- 창 1
  [4] 밀도 결과 화면   http://localhost:7002     <- 창 4
```

---

## 장비 없이 먼저 리허설

**권장합니다.** 실물 코드가 지나가는 길을 그대로 밟습니다.

```bash
cd ~/Desktop/PIVOT/my_work
$R python dual_view.py --mode deploy --bus local --hardware sim \
    --object 3link --prior water --max-rounds 3 --steps 3 --target 0.05
```

모의 그리퍼·모의 트래커가 붙어서 파지 단계와 각도 단계가 실제와 같은
순서로 돕니다. 창 2·3 은 안 뜹니다 (장비가 없으므로).

---

## 자주 걸리는 것

| 증상 | 원인 |
| --- | --- |
| `ModuleNotFoundError: numpy` | `$R` 를 안 붙였습니다. 2장의 `export R=...` |
| 포트 7000 이 이미 쓰임 | 이전 세션이 살아 있습니다. `pkill -f dual_view.py` |
| `/dev/ttyUSB0` 권한 없음 | `sudo usermod -aG dialout $USER` 후 재로그인 |
| 그리퍼가 "물었다"를 보고 안 함 | 개구량을 물체 단면과 똑같이 준 것입니다. 창의 `②` 버튼(계획 개구로 물기)을 쓰세요 |
| 각도 확인이 안 넘어감 | FoundationPose 값이 목표에서 5도 넘게 벗어나 있습니다. 창 2 를 보며 다시 맞추세요 |
| 창 4 가 계속 빨간 막대 | 아직 불확실성이 목표 밖입니다. 창 1 의 추천 각도로 다시 맞추고 라운드를 더 도세요 |

## 아직 안 끝난 것 — 실물 전에 확인이 필요합니다

1. **로봇을 움직이는 코드가 이 저장소에 없습니다 — 이것이 유일한 실물 차단
   요인입니다.** `hardware_real.RbpodoBackend.__init__` 이 `NotImplementedError`
   를 냅니다. 그래서 `--hardware real` 은 `connect_hardware` 에서 바로 멈추고,
   그 앞의 어떤 것도 실행되지 않습니다. `joint_positions / move_to / halt /
   set_servo` 네 함수만 채우면 됩니다.

   팀원 `~/MeshPCA/pivot/tare_real.py` 는 `rbpodo.CobotData(ip)` 와
   `data.request_data(2.0).sdata.jnt_ang[:6]` 로 관절각을 읽고 있으므로,
   그 PC 에 도는 구현이 이미 있습니다. **그 파일을 받아 오는 것이 가장
   빠릅니다.**
2. **FoundationPose 각도의 부호·영점이 PIVOT 관절각과 안 맞춰졌습니다.**
   대응은 [my_work/NAMING.md](my_work/NAMING.md) 에 정리했지만, 부호는
   실물에서 한 자세를 두 방법으로 읽어 비교해야 확정됩니다. 그 전에는
   각도 판정이 반대로 나올 수 있습니다.
3. **파지력 205 N 이 이 램프에 맞는지 실물 확인이 필요합니다.**
   `gripper_hw.py --keyboard` 로 물려 놓고 흔들어 보세요. 눌린 자국이
   남으면 `--gripper-force` 를 낮추세요.
