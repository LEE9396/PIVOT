# 통합 UI 실험 환경을 다른 PC에 구성하기

기준: 2026-09-08 현재 작업 트리. **이 문서가 현재 설치 절차**다.
과거 문서의 “hardware.py 미구현”, “두 PC 필수”, 예전 4창 직접 실행 지시는
현재 통합 UI에 적용하지 않는다. 실물 드라이버는 이미 구현되어 있다.

## 1. 구성과 범위

| 구성 | 현재 기준 |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS, x86_64 |
| GPU | 원 PC RTX 5080 16 GB, NVIDIA 580.173.02, CUDA toolkit 12.8 |
| PIVOT | Python 3.12, Drake 1.54.0, 저장소 내부 venv |
| MeshPCA | 독립 venv, 하드웨어 상태/명령 (socket·POSIX serial) |
| FoundationPose | Python 3.10 conda bundlesdf, torch 2.8.0+cu128 |
| SAM3 | Python 3.12 conda sam3, torch 2.10.0+cu128 |
| 장비 | RB5-850E + AIDIN AFT200 + Robotiq 2F-85 + RealSense D456 |
| 카메라 | 640×480, 30 FPS, 고정 eye-to-hand |
| 화면 | 통합 대시보드 localhost:8080, Meshcat 7000/7001 등 터미널 표시 주소 |

다른 GPU/드라이버 조합은 현 PC와 동일하게 검증된 조합이 아니다.
Drake만 설치하면 시뮬레이션은 가능하지만 실물 통합 UI에는 나머지 환경도 필요하다.
CUDA 빌드·모델 가중치 때문에 디스크 여유를 수십 GB 확보한다.
Linux GUI 세션(Tk 수동 마스크), USB 3 카메라 연결, 로봇 Ethernet 연결이 필요하다.

## 2. PIVOT 설치

아래 명령은 저장소 루트에서 실행한다. 경로에 공백이 없는 구성을 사용한다.

```bash
git clone --branch real-experiment-ready https://github.com/Yuseong-Cheon/PIVOT.git ~/Desktop/PIVOT
cd ~/Desktop/PIVOT
sudo apt update
sudo apt install -y git build-essential cmake ninja-build pkg-config curl \
  python3.12-venv python3-tk libgl1 libglib2.0-0 libusb-1.0-0-dev acl
./setup/bootstrap.sh
```

Drake 실행에는 항상 `robot_learning/scripts/run_drake_env.sh`를 사용한다.
ROS 전역 환경의 PYTHONPATH/LD_LIBRARY_PATH를 제거하는 래퍼다.
Drake 환경에 GPU/영상 패키지를 섞지 않는다.

## 3. 외부 코드 복원 (반드시 고정 커밋 + 현재 패치)

```bash
PIVOT_CHECKOUT="$HOME/Desktop/PIVOT"
git clone https://github.com/Yuseong-Cheon/MeshPCA.git ~/MeshPCA
git -C ~/MeshPCA checkout 4911bef41bf8bd5ffe6bad8b2ea0d4fea28e1107
git -C ~/MeshPCA apply --check "$PIVOT_CHECKOUT/integration/meshpca/workstation_20260908.patch"
git -C ~/MeshPCA apply "$PIVOT_CHECKOUT/integration/meshpca/workstation_20260908.patch"

git clone https://github.com/Yuseong-Cheon/Foundation_pose_edit.git ~/FoundationPose
git -C ~/FoundationPose checkout df490a5be025e87afe6895b7ba8dcfd4b4034abe

git clone https://github.com/facebookresearch/sam3.git ~/sam3
git -C ~/sam3 checkout 46957e47805eaa273f4aa7bbbd25a88bca9108ce
git -C ~/sam3 apply --check "$PIVOT_CHECKOUT/integration/sam3/workstation_20260908.patch"
git -C ~/sam3 apply "$PIVOT_CHECKOUT/integration/sam3/workstation_20260908.patch"
```

현재 패치는 HEAD부터의 **전체 변경분**이다. 09-04의 b64 패치를 먼저 적용하거나
예전 make_desk_lamp_masks.py/tare_real.py를 그 위에 복사하지 않는다.
이미 사용 중인 외부 checkout은 덮어쓰지 말고 새 경로에 위 절차로 구성한다.
FoundationPose 커밋은 NVlabs 원본의 HEAD가 아니라 위 실험용 fork에서 받는다.
실행하는 live/mask 스크립트는 **MeshPCA/foundationpose/** 쪽이다.

## 4. Python 환경과 GPU 빌드

현재 전체 설치 목록은 [locks/](locks/)에 있다. `*.txt`는 실제 설치 버전 목록,
`*-sources.txt`는 Git에서 빌드한 패키지의 원본 커밋,
`*.yml`는 conda 시스템 라이브러리까지 포함한 환경 기록이다.
GPU 환경의 txt 전체를 한 번에 pip install하면 로컬 빌드 패키지를 못 찾을 수 있다.
아래 순서를 따르고 목록은 버전 대조에 사용한다.

### MeshPCA

```bash
python3.12 -m venv ~/MeshPCA/.venv
~/MeshPCA/.venv/bin/python -m pip install -r ~/Desktop/PIVOT/setup/locks/meshpca.txt
~/MeshPCA/.venv/bin/python -c "import tkinter, cv2, numpy"
```

### FoundationPose

Miniforge/conda를 설치하고 해당 셸을 초기화한 뒤 실행한다.
CUDA toolkit은 드라이버와 별개다. `nvcc --version`에서 12.8을 확인한다.
상세 빌드 전제는 고정 커밋의 `~/FoundationPose/readme.md`에 있다.

```bash
conda env create -n bundlesdf -f ~/FoundationPose/environment.yml
conda activate bundlesdf
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install --no-build-isolation -r ~/Desktop/PIVOT/setup/locks/foundationpose-sources.txt
cd ~/FoundationPose
python -m pip install -r requirements.txt
bash build_all_conda.sh
python -m pip install pyrealsense2==2.58.3.10794
python -c "import torch, nvdiffrast.torch, pytorch3d, mycpp; assert torch.cuda.is_available()"
conda deactivate
```

필요한 FoundationPose 가중치는 고정 README의 공식 다운로드 링크에서 받아
`~/FoundationPose/weights/2023-10-28-18-33-37/`(refiner),
`~/FoundationPose/weights/2024-01-11-20-02-45/`(scorer)에 둔다.
두 디렉터리의 config와 checkpoint가 모두 있어야 한다.
가중치 자체는 이 저장소에 재배포하지 않는다. 해시는 `runtime-inputs.json` 참조.

### SAM3

```bash
conda create -n sam3 python=3.12 pip -y
conda activate sam3
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ~/sam3
python -c "import torch, sam3; assert torch.cuda.is_available()"
conda deactivate
```

자동 SAM3 마스크를 사용할 때는 고정 SAM3 README의 Hugging Face
`facebook/sam3` 접근 신청·인증·가중치 다운로드를 먼저 끝낸다.
토큰을 experiment.conf나 Git에 적지 않는다.
현재 기본 `MANUAL_MASK=1`은 사람이 Tk 창에서 부위를 지정한다.
모델 환경은 각각 `locks/foundationpose.txt`, `locks/sam3.txt`와 대조한다.

## 5. 설정·네트워크·장치 권한

```bash
cd ~/Desktop/PIVOT
cp setup/experiment.conf.example setup/experiment.conf
```

`experiment.conf`에서 모든 ROOT/PYTHON 경로를 새 PC에 맞춘다.
특히 `FOUNDATIONPOSE_PYTHON`을 실제 bundlesdf Python으로 명시한다.
`SAM3_PYTHON` 경로에서 추론되는 기본값에 의존하지 않는다.
원 PC 설정의 경로 정규화 기록은 `workstation_20260908.conf.reference`다.
현재 기준은 `GRASP_FRAME=measured`, `TARE_MODE=gravity`,
`ANGLE_FLOOR_DEG=3.0`, `GRASP_SIGMA_MM=10.0`, `ANGLE_MARGIN_DEG=0.0`이다.
오차값은 현 PC의 가정값이며 새 장비 측정으로 검증한다.
`START_ARM_DEG`는 새 셋업에서 확인한 6개 관절각만 설정한다.

RB5와 AFT 스트림은 현재 `192.168.50.51`이다.
PC Ethernet에 충돌하지 않는 같은 대역 주소(예: 192.168.50.10/24)를 설정하고
로봇 컨트롤러 IP, TCP 5000(명령)/5001(상태) 연결, 원격 제어 설정, AFT 출력 설정을 확인한다.
컨트롤러 payload/CoM·F/T 필터·영점 설정을 바꾸면 다시 검증한다.
실측 wrench 경로는 `hardware_real.py` 및 `tools/SIM_REAL_FT_ANALYSIS.md` 참조.

```bash
sudo ./setup/install_gripper_permissions.sh
ls -l /dev/serial/by-id/
```

권한 스크립트의 serial `A9O3H0CZ`는 원 PC USB 어댑터용이다.
다른 어댑터라면 `udevadm info --attribute-walk --name=/dev/ttyUSB0`로
VID/PID/serial을 확인하고 규칙을 맞춘다. 재로그인해 dialout 그룹을 적용한다.
`GRIPPER_PORT`는 실제 장치나 안정적인 `/dev/serial/by-id/...` 경로로 설정한다.

RealSense의 udev 규칙은 librealsense 설치 지침에 따라 설치한다.
`rs-enumerate-devices`(SDK 도구 설치 시) 또는 아래 Python으로 인식 여부를 확인한다.

```bash
~/miniforge3/envs/bundlesdf/bin/python -c "import pyrealsense2 as r; print([(d.get_info(r.camera_info.name), d.get_info(r.camera_info.serial_number)) for d in r.context().devices])"
```

RGB-D를 쓰는 프로그램은 한 번에 하나만 실행한다. 테이블 보정 중에는 추적기를 종료한다.

## 6. 자산·보정·런타임 입력

램프의 URDF, visual_meshes, collision_meshes, convex 조각은 이 저장소에 포함한다.
`LAMP_ASSET_DIR`와 `FP_MESH_DIR`는 **같은 자산 트리**를 가리켜야 한다.
자산 누락/변형 검사는 저장소 루트에서 `sha256sum -c setup/lamp-assets.sha256`로 한다.
부위 대응은 base=link_2, support=link_3, head=link_1이다.
기본 이름표는 저장소 OBJ로 생성하므로 외부 Lab 폴더는 필요 없다.
원 PC의 3DGS 이름표를 그대로 원하면 해당 gaussian PLY를 별도 제공하고
`GAUSSIAN_DIR/GAUSSIAN_FILES`를 설정한다. 이는 추적용 메시를 대체하지 않는다.

| 파일 | 준비 방법 |
| --- | --- |
| calibration/camera_cam_d456_front.json | eye-to-hand 보정, X_WC 및 실제 intrinsics |
| calibration/rb5_table_current.json | 현재 실제 상판의 평면 재측정 |
| calibration/workspace_obstacles_current.json | 받침대 등 장애물을 현재 배치로 실측; workspace_obstacles.py의 형식 참조 |
| calibration/angle_signs.json | 각 관절 각도 방향·영점 검증 |
| calibration/aft_tare_current.json | 물체 없는 그리퍼로 영점 측정, 검산 통과 필요 |
| /tmp/lamp_live_rgb.png, lamp_live_depth_m.npy, lamp_live_intrinsics.json | 추적기 첫 초기화 때 생성 |
| /tmp/lamp_sam3_masks/{base,support,head}.png | 현재 프레임에서 수동 마스크 생성 |
| /tmp/lamp_foundationpose_live/latest.json | 추적기가 계속 갱신하는 자세 |
| /tmp/lamp_foundationpose_live/hardware.json | MeshPCA 하드웨어 상태 프로세스가 생성 |

보정 JSON은 Git에서 제외한다. 장비 배치가 달라지면 원 PC 값을 복사하지 않는다.
PC만 교체하고 물리 배치가 그대로인 경우에도 기존 파일은 별도로 보관·대조하고
새 연결에서 보정과 영점 검증을 통과한 뒤 사용한다.
현재 파일 목록과 SHA256은 `runtime-inputs.json`에 기록했다.

핸드아이 절차: [calibration/README.md](../calibration/README.md).
이미 얻은 EasyHeC 결과는 `tools/import_easyhec.py --help`로 입력 형식을 확인한다.
테이블 보정은 영상 환경에서 실행한다(Drake 환경에는 cv2/RealSense가 없다).

```bash
~/miniforge3/envs/bundlesdf/bin/python my_work/calibrate_table_rgbd.py \
  --serial <새-카메라-serial> \
  --handeye calibration/camera_cam_d456_front.json \
  --output calibration/rb5_table_current.json
./setup/local_ft_check.sh --help
./setup/wrist_tare.sh --plan
```

정지 F/T 검증은 [LOCAL_FT_VALIDATION.md](../tools/LOCAL_FT_VALIDATION.md),
타어·토크 문제는 [SIM_REAL_FT_ANALYSIS.md](../tools/SIM_REAL_FT_ANALYSIS.md),
payload 식별은 [FT_PAYLOAD_IDENTIFICATION.md](../tools/FT_PAYLOAD_IDENTIFICATION.md)를 따른다.
계획이 검증되기 전 자동 이동 명령을 실행하지 않는다.
`wrist_tare.sh --run`은 실제 손목을 움직인다.
`--run-force`로 만든 실패 타어를 정상 실험에 사용하지 않는다.

## 7. 검증 → 통합 UI 실행

장비 연결 전에:

```bash
./setup/bootstrap.sh --check
robot_learning/scripts/run_drake_env.sh python my_work/pivot_ui.py --dry-run --sessions /tmp/pivot-reproduction-check
robot_learning/scripts/run_drake_env.sh python -m unittest discover -s tools -p 'test_*.py'
```

dry-run은 단계 상태 전이 검사이며 실제 대시보드/로봇 시험은 아니다.
`launch_experiment.sh --rehearse`는 기존 dual_view 모의 장비 리허설이다.

장비·보정 준비 후:

```bash
./setup/launch_experiment.sh --check
./setup/launch_experiment.sh
```

check는 import·파일·장치 권한 검사이며 파지점 JSON을 갱신한다.
실제 실험 가능 판정은 추적기가 시작된 뒤 UI 0단계 preflight에서 다시 수행한다.
브라우저 `http://localhost:8080`에서 준비 → 파지 → 각도 → 경로 승인 →
측정/탐색 → 내보내기를 진행한다. 명령 실행 시 터미널과 UI의 상태를 확인한다.
UI를 직접 실행하면 추적기와 하드웨어 상태 프로세스가 시작되지 않으므로
전체 실험은 반드시 런처를 사용한다.

새 프레임으로 마스크를 다시 만들 때:
`./setup/launch_experiment.sh --remask`.
`REUSE_INIT=1`은 물체·카메라·프레임이 동일할 때만 사용한다.
컴퓨터를 옮겼을 때 이전 /tmp 파일은 재사용하지 않는다.

## 8. 결과·로그·문제 해결

- 세션: `my_work/sessions/`. phase.json, grasp.json, 각도/경로/측정/추정 산출물을 세션 단위로 보관한다.
- 별도 진단: `my_work/outputs/`. 검사와 재현에 사용한 conf 및 보정 파일도 실험별로 별도 백업한다.
- 추적기 로그: `/tmp/pivot_win2.log`; 하드웨어 상태 로그: `/tmp/pivot_win3.log`; 지휘자: 실행 터미널.
- 8080 충돌: 기존 UI 확인. 중복 UI는 `/tmp/pivot_ui.lock`으로 차단된다.
- 추적기 종료: Python 경로, CUDA extensions, weights, 세 부위 메시, USB 독점, 마스크를 로그에서 확인한다.
- `import tkinter` 실패: 해당 Python에 Tk 지원을 설치한다(conda 환경이면 conda tk).
- 원격 접속: `ssh -L 8080:localhost:8080 -L 7000:localhost:7000 -L 7001:localhost:7001 user@PC`.
  UI는 로컬 제어용이다. 사용하는 추가 Meshcat 포트가 있으면 함께 전달한다.
- 종료는 지휘자 터미널에서 Ctrl-C. 백그라운드 추적기/하드웨어 프로세스가 남았는지
  명령행과 PID를 확인해 해당 실험 프로세스만 종료하고 카메라/serial을 해제한다.

## 9. 검증 범위

이번 업데이트는 현재 코드·패치·자산과 설치 기록을 보존한다.
새 PC에서 CUDA 빌드, 장치 연결, 실제 UI 조작과 로봇 이동까지 통과했다는 뜻은 아니다.
실행한 검사는 `VALIDATION_20260908.md`에 기록한다.
