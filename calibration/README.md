# 카메라 캘리브레이션

로봇과 카메라를 맞춘 결과를 여기에 둡니다. 파일이 있으면 파이프라인이
**자동으로** 그 값을 씁니다 — 실험실 설정 파일의 명목 위치보다 우선합니다.

```
calibration/camera_cam_d456_front.json      <- 이 이름이면 자동으로 읽힙니다
```

## 왜 중요한가

각도를 읽는 자세는 **카메라가 어디서 보느냐**로 정해집니다. 명목 위치로
계산해 두고 실제 카메라가 10 cm 옆에 있으면, 고른 자세가 실제로는 최적이
아니고 심하면 물체가 화면 밖으로 나갑니다. 카메라는 팔이 부딪힐 수 있는
장애물이기도 해서 충돌 판정도 함께 어긋납니다.

그리고 이 오차는 **라운드를 늘려도 안 없어지는 치우침**입니다.

## 어떤 종류의 캘리브레이션인가

카메라가 실험실에 고정되어 로봇을 바라봅니다 → **eye-to-hand**입니다.
(카메라가 손목에 붙은 eye-in-hand 가 아닙니다)

```
   W  월드 = 로봇 베이스
   E  팔의 tcp 프레임      X_WE 는 관절각 + FK 로 안다
   T  보정판               X_ET 는 모르는 상수 (판을 어디에 붙였나)
   C  카메라               X_WC 는 모르는 상수  <- 구하는 것

   매 자세에서   X_WC · X_CT = X_WE · X_ET
   자세 둘을 빼면   A X = X B     (같은 운동을 두 좌표계에서 본 것)
```

---

## 자동 캘리브레이션 (권장)

보정판을 그리퍼에 물려 두면 로봇이 자세를 순회하며 알아서 잽니다.

### 1. 보정판 만들기

```bash
cd ~/Desktop/PIVOT/my_work
python3 calib_detect.py --make-board board.png      # 시스템 python
```

300 dpi로 인쇄 → **판판한 것(알루미늄판·아크릴)에 붙이기** → 그리퍼에
**볼트로 고정**.

> **인쇄한 뒤 자로 사각형 한 변을 재세요.** 프린터가 배율을 건드리는 일이
> 흔하고, 그 배율 오차는 그대로 거리 오차가 됩니다. 잰 값을 `--square-mm`에
> 넣습니다.

> 판이 캘리브레이션 도중 조금이라도 움직이면 "X_ET 는 상수" 라는 전제가
> 깨져서 답이 통째로 틀어집니다. 테이프로 붙이지 마세요.

### 2. 카메라가 판을 잡는지 확인

```bash
python3 calib_detect.py --preview --square-mm <자로 잰 값>
```

코너 수와 재투영 오차가 화면에 뜹니다. 코너가 24개 다 잡히고 재투영이
1 px 아래면 좋습니다.

### 3. 두 터미널로 실행

검출기만 시스템 python에서 돕니다 (opencv 때문). FoundationPose와 같은 구조입니다.

```bash
# 터미널 1 — 시스템 python (run_drake_env.sh 를 쓰지 않습니다)
python3 calib_detect.py --serve --port 5566 --square-mm <자로 잰 값>

# 터미널 2 — Drake 환경
cd ~/Desktop/PIVOT/my_work
../robot_learning/scripts/run_drake_env.sh python calibrate_camera.py \
    --run --poses 20 --detector-port 5566
```

로봇이 20개 자세를 순회하며 자세마다 판을 찍고, 다 끝나면 풀어서 저장합니다.

### 먼저 장비 없이 돌려보기

```bash
$R python calibrate_camera.py --run --simulate --poses 20
```

정답을 정해 놓고 그 정답을 되찾는지 봅니다. 배선·자세 선정·솔버가 다
맞는지 장비 없이 확인할 수 있습니다.

로봇만 안 움직이고 자세 선정까지만 보려면 `--dry-run`.

### 자세를 몇 개 찍어야 하나

`--simulate` 로 잰 값입니다 (검출 잡음 0.5 mm / 0.05 deg 가정).

| 자세 수 | 카메라 위치 오차 |
| ---: | ---: |
| 8 | 0.77 mm |
| 12 | 0.70 mm |
| **20** | **0.43 mm** |
| 30 | 0.40 mm |

**20개면 충분합니다.** 그 위로는 거의 안 좋아집니다.

### 자세 개수보다 중요한 것 — 회전축 다양성

팔이 비슷한 자리에서만 움직이면 자세를 100개 찍어도 답이 안 나옵니다.
`A X = X B` 가 특이해지기 때문입니다. 코드가 이걸 자동으로 챙기고
(가장 다른 방향을 탐욕적으로 고름), 실행할 때 이렇게 알려줍니다.

```
자세 사이 최대 회전차 180 deg  (충분)
```

`부족` 이라고 나오면 손목을 여러 방향으로 크게 꺾는 자세가 더 필요합니다.

---

## 결과를 믿어도 되는지

실행 끝에 **자기 일관성**이 나옵니다. 구한 `X_WC`, `X_ET` 로
`X_WC·X_CT` 와 `X_WE·X_ET` 를 각각 만들어 얼마나 어긋나는지 잰 값입니다.
**정답을 모르는 실물에서 볼 수 있는 유일한 품질 지표입니다.**

```
자기 일관성  0.78 mm rms / 0.083 deg rms   (최악 1.32 mm)
```

5 mm를 넘으면 셋 중 하나입니다.
- 판이 움직였다 (볼트 확인)
- 자세가 서로 너무 비슷하다 (회전차 확인)
- `--square-mm` 이 실제와 다르다 (자로 다시 재기)

그 다음 명목값과 얼마나 다른지 봅니다.

```bash
$R python calibrate_camera.py --show --check-poses
```

10 cm 넘게 다르면 실험실 설정 파일이 낡았을 수 있으니 사람이 한 번 봅니다.

---

## 손으로 넣기

이미 다른 도구(easy_handeye 등)로 구한 값이 있으면 그대로 넣습니다.

```bash
$R python calibrate_camera.py --matrix 0.42 -0.90 ... --rms-px 0.31
$R python calibrate_camera.py --position 0.35 0.5 1.4 --quat 0.1 0.2 0.3 0.9
$R python calibrate_camera.py --position 0.35 0.5 1.4 --look-at 0 -0.15 1.1
```

## 형식

```json
{
  "id": "cam_d456_front",
  "X_WC": [[r11, r12, r13, tx], [r21, r22, r23, ty],
           [r31, r32, r33, tz], [0, 0, 0, 1]],
  "depth_intrinsics": {"fx": 674.3, "fy": 649.5, "cx": 640.0, "cy": 360.0},
  "resolution": [1280, 720],
  "calibrated_at": "2026-08-14",
  "method": "ChArUco eye-to-hand, 20자세",
  "rms_px": 0.31,
  "residual_mm": 0.78
}
```

`X_WC` 는 **월드(로봇 베이스) 기준 카메라 자세**입니다. Drake 카메라 규약은
z 가 전방, y 가 아래로, OpenCV 와 같습니다.

`depth_intrinsics`·`resolution` 은 안 적으면 실험실 설정값을 그대로 씁니다.

## 주의

- 이 폴더의 `*.json` 은 git 에 올리지 않습니다. **PC마다 다른 값**이기
  때문입니다. 팀원끼리 옮겨 쓰면 안 됩니다.
- 카메라를 옮겼거나 부딪혔으면 다시 캘리브레이션하세요.
- 캘리브레이션 중에는 로봇이 사람 없이 자세를 순회합니다. 첫 순회는
  `--move-duration` 을 넉넉히 주고 옆에서 지켜보세요.

## 코드가 어디 있나

| 파일 | 하는 일 | 어느 환경 |
| --- | --- | --- |
| `my_work/handeye.py` | AX=XB 솔버 (numpy만). 자기 검사 내장 | Drake |
| `my_work/calib_detect.py` | ChArUco 검출 | **시스템 python** (opencv) |
| `my_work/calibrate_camera.py` | 자세 선정·순회·수집·풀이·저장 | Drake |

솔버가 맞는지는 하드웨어 없이 확인할 수 있습니다.

```bash
$R python handeye.py
```
