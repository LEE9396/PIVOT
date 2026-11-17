# 실물 실험 기록 — session_20260904_1736

2026-09-04 17:36~18:09 KST에 desk lamp를 대상으로 실행한 실제 RB5 탐색 세션이다.
세 번의 측정은 완료됐지만 추정은 수렴하지 않았고, 네 번째 라운드의 시작 자세 경로가
충돌/FOV 검사에서 거부된 뒤 사용자가 실험을 중지했다. 따라서 이 결과를 최종 밀도로
사용하면 안 된다.

## 결과

| 라운드 | 추천 물리각 [deg] | FoundationPose 물리각 [deg] | 밀도 [Arm, Base, Head] kg/m³ | 95% 상대 반폭 |
| --- | --- | --- | --- | --- |
| 1 | 89.7, 60.6 | 88.96, 60.48 | 50.0, 1104.7, 50.0 | 8,124,968.52% |
| 2 | 106.4, 64.0 | 106.28, 63.67 | 50.0, 1420.1, 50.0 | 6,017,126.90% |
| 3 | 92.4, 40.5 | 92.00, 40.36 | 50.0, 1500.2605, 50.0 | 2,743,356.90% |

Arm과 Head가 밀도 하한 50 kg/m³에 붙었고 잔차 팽창도 5527~6507이었다.
원시 18축 F/T 배열은 당시 코드가 파일에 기록하지 않아 프로세스 종료 뒤 복구하지 못했다.
`dual_view.py`에는 이 세션 종료 직후 라운드별 `exploration_round_N.json`을 저장하는 수정이
추가됐다. 이 수정은 표의 결과 계산에는 관여하지 않았다.

원본 산출물:

- `experiment_results.json`: 복원 가능한 전체 세션 요약과 결과 유효성
- `exploration_results.csv`: 라운드별 표
- `posterior_round_0.json`: 마지막 posterior 원본
- `preflight.json`: 시작 전 점검 결과
- `grasp.json`: 실측 파지 변환

## 실행 환경

- RB5-850E, AIDIN AFT200, Robotiq 2F-85, Intel RealSense D456
- Ubuntu 24.04.4 LTS, Linux 7.0.0-30-generic
- NVIDIA GeForce RTX 5080 16 GB, driver 580.173.02
- PIVOT: `real-experiment-ready`, 실행 전 기준 커밋 `8ba1dcd7bf7e92a4e51a75b5f8659a54d892eb67`
- Python 3.12.14, Drake 1.54.0, NumPy 2.5.1, SciPy 1.18.0, rbpodo 0.16.10
- FoundationPose 환경: Python 3.10.14, PyTorch 2.8.0+cu128, pyrealsense2 2.58.3.10794
- SAM3 환경: Python 3.12.13, PyTorch 2.10.0+cu128

자세한 저장소 커밋과 입력 파일 SHA-256은 `environment.json`에 있다. 캘리브레이션 원본과
대형 visual/collision mesh는 PC별·배달물 정책에 따라 커밋하지 않고 해시만 기록했다.

## 실행 당시 설정

핵심값은 `experiment.conf.snapshot`과 같다.

- 물체 `desklamp`, 파지 부위 `link_3`/`support`
- 카메라 640×480 @ 30 Hz, FoundationPose 중앙값 창 10 샘플
- 수동 masking, FOV 가장자리 여유 20 px
- 목표 95% 상대 반폭 5%, 최대 8라운드, 이동시간 8초
- 시작 관절각 `[-73.912750, 13.661336, 124.109833, -133.635742, 83.884857, 93.222351]°`
- 영점 조정은 실험 시작 때 측정한 3방향 값을 세션 동안 재사용
- 외부 장애물과 AFT200 케이블에는 10 mm 여유, RB5 자체 링크는 관통만 금지

## 실제 실행 절차

1. 영점 조정, 카메라/책상/각도 보정, 수동 mask와 FoundationPose가 준비된 상태에서
   `tools/preflight.py`의 11개 항목이 모두 OK인지 확인했다.
2. `my_work/`에서 다음 명령으로 통합 UI를 실행했다.

   ```bash
   ../robot_learning/scripts/run_drake_env.sh python -u pivot_ui.py \
     --conf ../setup/experiment.conf
   ```

3. 창 1의 파지점과 실물 오버레이를 보고 물체의 support를 잡고 `③ 파지 완료`를 눌렀다.
4. 각 라운드에서 추천각에 맞게 물체를 손으로 돌렸다. FoundationPose 5샘플 중앙값이
   ±5° 안에 들어오면 `① 각도 확인`, `② 물체에서 손을 뗐습니다`를 눌렀다.
5. 로봇이 충돌과 FOV를 검사한 경로로 중력 세 방향을 순서대로 이동하며 AFT200을 읽고
   posterior를 갱신했다. 이 과정을 세 라운드 완료했다.
6. 네 번째 라운드는 시작 자세로 가는 경로가 충돌/FOV 검사에서 실패해 이동하지 않았다.
   다음 계획 중 사용자가 중지를 요청했고 `task_stop`을 보내 로봇을 정지했다.

## 외부 코드 재현

실행 당시 PIVOT 밖의 수정도 두 patch로 보존했다.

```bash
git clone https://github.com/Yuseong-Cheon/MeshPCA.git ~/MeshPCA
git -C ~/MeshPCA checkout 4911bef41bf8bd5ffe6bad8b2ea0d4fea28e1107
base64 -d integration/meshpca/session_20260904_1736.patch.b64 | git -C ~/MeshPCA apply -

git clone https://github.com/facebookresearch/sam3.git ~/sam3
git -C ~/sam3 checkout 46957e47805eaa273f4aa7bbbd25a88bca9108ce
base64 -d integration/sam3/session_20260904_1736.patch.b64 | git -C ~/sam3 apply -

git clone https://github.com/NVlabs/FoundationPose.git ~/FoundationPose
git -C ~/FoundationPose checkout df490a5be025e87afe6895b7ba8dcfd4b4034abe
```

그 뒤 `setup/experiment.conf.example`을 복사해 PC 경로와 장비 주소를 맞추고
`setup/launch_experiment.sh`를 실행한다. 실제 로봇을 움직이기 전에는 반드시 현재 환경에서
preflight, 충돌/FOV 계획, 비상정지 동작을 다시 확인한다.
