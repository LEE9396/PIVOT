# 캘리브레이션 결과를 두는 곳

로봇과 카메라를 맞춘 결과(손-눈 캘리브레이션)를 여기에 둡니다.
파일이 있으면 파이프라인이 **자동으로** 그 값을 씁니다 — 실험실 설정 파일의
명목 위치보다 우선합니다.

```
calibration/camera_cam_d456_front.json      <- 이 이름이면 자동으로 읽힙니다
```

## 왜 중요한가

각도를 읽는 자세는 **카메라가 어디서 보느냐**로 정해집니다. 명목 위치로
계산해 두고 실제 카메라가 10 cm 옆에 있으면, 고른 자세가 실제로는 최적이
아니고 심하면 물체가 화면 밖으로 나갑니다. 카메라는 팔이 부딪힐 수 있는
장애물이기도 해서 충돌 판정도 함께 어긋납니다.

## 만드는 법

```bash
cd my_work
R=../robot_learning/scripts/run_drake_env.sh

# 4x4 행렬 (행 우선 16개) — 손-눈 캘리브레이션 결과가 보통 이 형태입니다
$R python calibrate_camera.py --matrix 0.42 -0.90 ... --rms-px 0.31

# ROS tf 처럼 위치 + 쿼터니언(x y z w)
$R python calibrate_camera.py --position 0.35 0.5 1.4 --quat 0.1 0.2 0.3 0.9

# 대충 맞출 때: 위치와 바라보는 점
$R python calibrate_camera.py --position 0.35 0.5 1.4 --look-at 0 -0.15 1.1

# 지금 무엇을 쓰는지, 명목값과 얼마나 다른지, 자세가 어떻게 바뀌는지
$R python calibrate_camera.py --show --check-poses
```

다른 곳에 둔 파일을 쓰려면 `export CAMERA_CALIBRATION=/그/경로.json`.

## 형식

```json
{
  "id": "cam_d456_front",
  "X_WC": [[r11, r12, r13, tx],
           [r21, r22, r23, ty],
           [r31, r32, r33, tz],
           [0, 0, 0, 1]],
  "depth_intrinsics": {"fx": 674.3, "fy": 649.5, "cx": 640.0, "cy": 360.0},
  "resolution": [1280, 720],
  "calibrated_at": "2026-08-14",
  "method": "easy_handeye, 체커보드 24자세",
  "rms_px": 0.31
}
```

`X_WC` 는 **월드(로봇 베이스) 기준 카메라 자세**입니다. Drake 카메라 규약은
z 가 전방, y 가 아래로, OpenCV 와 같습니다.

`depth_intrinsics`·`resolution` 은 안 적으면 실험실 설정값을 그대로 씁니다.

## 주의

- 이 폴더의 `*.json` 은 git 에 올리지 않습니다. **PC마다 다른 값**이기
  때문입니다. 팀원끼리 옮겨 쓰면 안 됩니다.
- 카메라를 옮겼거나 부딪혔으면 다시 캘리브레이션하세요. 캘리브레이션이
  틀리면 각도 측정이 통째로 틀어지고, 그 오차는 라운드를 늘려도 안 없어지는
  치우침입니다.
