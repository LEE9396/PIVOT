#!/usr/bin/env bash
# 실물 실험 창 4개를 띄운다. 먼저 준비물을 전부 점검한다.
#
#   setup/launch_experiment.sh --check      점검만 한다 (아무것도 안 띄운다)
#   setup/launch_experiment.sh --rehearse   장비 없이 창 1·4 만 (모의 장비)
#   setup/launch_experiment.sh              창 4개를 띄운다
#
# 왜 런처가 필요한가. 창마다 파이썬 환경이 다르고(Drake / MeshPCA venv /
# FoundationPose conda) 플래그가 열 개 넘게 붙는다. 손으로 치면 한 군데씩
# 틀리는데, 틀린 채로 로봇이 움직이는 것이 가장 나쁘다. 그래서 **띄우기
# 전에 전부 점검하고, 하나라도 없으면 안 띄운다.**

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${HERE}/experiment.conf"
MODE="${1:-run}"

if [[ ! -f "${CONF}" ]]; then
  echo "설정 파일이 없습니다: ${CONF}"
  echo "  cp setup/experiment.conf.example setup/experiment.conf"
  echo "  그 다음 PC 에 맞게 고치세요."
  exit 1
fi
# shellcheck disable=SC1090
source "${CONF}"

expand() { eval echo "$1"; }
for name in PIVOT_ROOT MESHPCA_ROOT FOUNDATIONPOSE_ROOT SAM3_ROOT \
            MESHPCA_PYTHON SAM3_PYTHON TARE_FILE FP_MESH_DIR \
            FP_INIT_RGB FP_INIT_DEPTH FP_INTRINSICS FP_MASKS FP_OUTPUT \
            LAMP_ASSET_DIR LAPTOP_ASSET_DIR; do
  printf -v "$name" '%s' "$(expand "${!name}")"
done
export DESK_LAMP_DELIVERY="${LAMP_ASSET_DIR}"

WORK="${PIVOT_ROOT}/my_work"
R="${PIVOT_ROOT}/robot_learning/scripts/run_drake_env.sh"
GRASP_JSON="${WORK}/outputs/grasp_target_${OBJECT}.json"
FP_GRASP_MESH="${FP_MESH_DIR}/${FP_GRASP_PART}.ply"
[[ -f "${FP_GRASP_MESH}" ]] || FP_GRASP_MESH="${FP_MESH_DIR}/${FP_GRASP_PART}.obj"

FAIL=0
ok()   { echo "  [통과] $1"; }
bad()  { echo "  [실패] $1"; [[ -n "${2:-}" ]] && echo "         $2"; FAIL=$((FAIL+1)); }
warn() { echo "  [주의] $1"; }

need_file() { [[ -f "$1" ]] && ok "$2" || bad "$2" "없음: $1"; }
need_dir()  { [[ -d "$1" ]] && ok "$2" || bad "$2" "없음: $1"; }

echo "실험 준비 점검"
echo "----------------------------------------------------------------"
need_dir  "${WORK}"                  "PIVOT 체크아웃"
need_file "${R}"                     "Drake 환경 래퍼"
need_dir  "${MESHPCA_ROOT}/pivot"    "MeshPCA 체크아웃"

# --- 창 1·4 가 실제로 뜨는지 (import 까지 해 본다) ---
if "${R}" python -c "import sys; sys.path.insert(0,'${WORK}'); import dual_view, density_view, grasp_target, grasp_overlay" 2>/dev/null; then
  ok "PIVOT 모듈 import (창 1·4)"
else
  bad "PIVOT 모듈 import (창 1·4)" "./setup/bootstrap.sh 를 먼저 돌리세요"
fi

# --- 로봇을 움직일 수 있는가 — 이게 없으면 실물은 시작조차 못 한다 ---
if "${R}" python -c "
import sys; sys.path.insert(0,'${WORK}')
import hardware_real as hr
hr.RbpodoBackend('127.0.0.1')
" 2>/dev/null; then
  ok "RB5 실물 드라이버 (RbpodoBackend)"
else
  bad "RB5 실물 드라이버 (RbpodoBackend)" \
      "hardware_real.py 의 joint_positions/move_to/halt/set_servo 가 비어 있습니다. \
EXPERIMENT.md '아직 안 끝난 것' 1번을 보세요."
fi

# --- 캘리브레이션 · 타어 ---
CAM_JSON="$("${R}" python -c "
import sys; sys.path.insert(0,'${WORK}')
import robot_scene as rs; print(rs.calibration_path())" 2>/dev/null)"
if [[ -n "${CAM_JSON}" && -f "${CAM_JSON}" ]]; then
  ok "카메라 캘리브레이션 ($(basename "${CAM_JSON}"))"
else
  bad "카메라 캘리브레이션" \
      "import_calibration.py --input <MeshPCA handeye JSON> 를 돌리세요"
fi
need_file "${TARE_FILE}" "3자세 타어"

# --- 파지점: 밀도 모델과 같은 mesh/명목점을 매번 내보내 stale JSON을 막는다. ---
mkdir -p "${WORK}/outputs"
if ( cd "${WORK}" && "${R}" python grasp_target.py \
    --mesh "${FP_GRASP_MESH}" \
    --part "${FP_GRASP_PART}" --pivot-part "${GRASP_PART}" \
    --out "${GRASP_JSON}" ); then
  ok "PIVOT 명목 파지점 ($(basename "${GRASP_JSON}"))"
else
  bad "PIVOT 명목 파지점" "밀도 모델과 FoundationPose mesh가 같은지 확인하세요"
fi

# --- 창 2·3 준비물 ---
need_file "${MESHPCA_PYTHON}" "MeshPCA 파이썬 (창 3)"
need_dir  "${FOUNDATIONPOSE_ROOT}" "FoundationPose 체크아웃 (창 2)"
if grep -q "grasp-target" "${MESHPCA_ROOT}/${FP_LIVE_SCRIPT}" 2>/dev/null; then
  ok "창 2 오버레이 패치 적용됨"
else
  bad "창 2 오버레이 패치" \
      "cd ${MESHPCA_ROOT} && git apply ${WORK}/integration/foundationpose_grasp_overlay.patch"
fi
for f in "${FP_INIT_RGB}" "${FP_INIT_DEPTH}" "${FP_INTRINSICS}"; do
  [[ -f "$f" ]] || warn "FoundationPose 초기화 입력 없음: $f (--reuse-init 를 못 씁니다)"
done

echo "----------------------------------------------------------------"
if [[ "${MODE}" == "--check" ]]; then
  [[ ${FAIL} -eq 0 ]] && echo "모두 통과 — 실험을 시작할 수 있습니다" \
                      || echo "${FAIL} 개 실패 — 위를 고치고 다시 점검하세요"
  exit $(( FAIL > 0 ))
fi

# --- 리허설: 장비 없이 창 1·4 만 ---
if [[ "${MODE}" == "--rehearse" ]]; then
  echo "리허설 — 모의 장비로 창 1·4 만 띄웁니다 (로봇은 안 움직입니다)"
  cd "${WORK}" || exit 1
  exec "${R}" python dual_view.py \
    --mode deploy --bus local --hardware sim \
    --object "${OBJECT}" --prior water \
    --target "${TARGET}" --max-rounds "${MAX_ROUNDS}" \
    --steps 3 --move-duration 1
fi

if [[ ${FAIL} -gt 0 ]]; then
  echo "${FAIL} 개가 준비되지 않아 띄우지 않습니다."
  echo "장비 없이 절차만 보려면:  setup/launch_experiment.sh --rehearse"
  exit 1
fi

# --- 창 3: 그리퍼 + F/T ---
echo
echo "[창 3] 그리퍼 + F/T 를 띄웁니다"
( cd "${MESHPCA_ROOT}" && "${MESHPCA_PYTHON}" pivot/rb5_ui.py \
    --host "${AFT_HOST}" --port "${GRIPPER_PORT}" --tare "${TARE_FILE}" \
    >/tmp/pivot_win3.log 2>&1 & )

# --- 창 2 준비: 부위 이름표 그림 ---
#
# 마스크 단계에서 사람이 base/support/head 박스를 그리는데, 어느 덩어리가
# 어느 이름인지 알려 주는 것이 없으면 조용히 틀린다 (my_work/NAMING.md).
# 배달물 메시에서 이름표를 구워 두고 환경변수로 넘긴다.
PART_LEGEND="${PART_LEGEND:-/tmp/pivot_part_legend.png}"
if [[ ! -f "${PART_LEGEND}" ]]; then
  echo "[창 2] 부위 이름표를 굽습니다 -> ${PART_LEGEND}"
  "${R}" python "${PIVOT_ROOT}/tools/make_part_legend.py" "${FP_MESH_DIR}" \
      --parts "${PART_NAMES:-base,support,head}" -o "${PART_LEGEND}" \
      || echo "      [주의] 이름표를 못 구웠습니다 — 이름 없이 진행합니다"
fi
export PIVOT_PART_LEGEND="${PART_LEGEND}"

# --- 창 2: 카메라 뷰 + 파지점 오버레이 ---
echo "[창 2] 카메라 뷰 + 파지점 오버레이를 띄웁니다"
# 글자 프롬프트만으로는 SAM3 가 부위를 못 가른다 (2026-09-02: base 0 픽셀로
# "suspicious lamp masks"). 기본을 사람이 박스를 그리는 쪽으로 둔다.
MASK_FLAG=()
[[ "${MANUAL_MASK:-1}" == "1" ]] && MASK_FLAG=(--manual-mask)
( cd "${MESHPCA_ROOT}" && conda run -n "${FOUNDATIONPOSE_CONDA_ENV}" \
    python "${FP_LIVE_SCRIPT}" \
      "${MASK_FLAG[@]}" \
      --foundationpose-root "${FOUNDATIONPOSE_ROOT}" \
      --sam3-root "${SAM3_ROOT}" --sam3-python "${SAM3_PYTHON}" \
      --mesh-dir "${FP_MESH_DIR}" \
      --width "${FP_CAMERA_WIDTH}" --height "${FP_CAMERA_HEIGHT}" \
      --fps "${FP_CAMERA_FPS}" --angle-window "${FP_ANGLE_WINDOW}" \
      --init-rgb "${FP_INIT_RGB}" --init-depth "${FP_INIT_DEPTH}" \
      --intrinsics "${FP_INTRINSICS}" --masks "${FP_MASKS}" \
      --output "${FP_OUTPUT}" \
      --grasp-target "${GRASP_JSON}" --grasp-part "${FP_GRASP_PART}" \
      --grasp-overlay "${WORK}/grasp_overlay.py" \
    >/tmp/pivot_win2.log 2>&1 & )

# 트래커가 첫 각도를 쓸 때까지 기다린다. 이게 없으면 창 1 이 각도를 못 읽어
# 조정 단계에서 바로 "FoundationPose 를 못 읽습니다" 를 반복한다.
echo "      FoundationPose 가 첫 각도를 낼 때까지 기다립니다 (최대 180초)..."
for _ in $(seq 180); do
  [[ -f "${FP_OUTPUT}/latest.json" ]] && break
  sleep 1
done
if [[ -f "${FP_OUTPUT}/latest.json" ]]; then
  echo "      준비됨: ${FP_OUTPUT}/latest.json"
else
  echo "      [실패] 트래커가 각도를 안 냅니다. /tmp/pivot_win2.log 를 보세요."
  exit 1
fi

# --- 창 1 + 창 4 ---
echo "[창 1·4] 탐색·파지점 / 밀도 결과를 띄웁니다"
echo
cd "${WORK}" || exit 1
# 창 2 가 읽어 주는 각도에는 1~3 도의 오차가 있다.
#
# ANGLE_FLOOR_DEG 는 **정보이득 계산**의 각도 잡음 하한이다. 기본 0.5 도는
# 실측(1~3 도)보다 낙관적이라 작은 각도를 과대평가한다 -> 2.0 으로 올린다.
# 이건 켜도 손해가 없다.
#
# ANGLE_MARGIN_DEG 는 **충돌 검사**에 주는 여유다. 3 도면 head 끝이
# 15.9 mm 움직이는데 지금 실제 여유는 3 mm 라, 켜면 통과하는 자세가
# 하나도 안 남는다. 파지 방향을 고쳐 여유를 10 mm 이상 확보한 **뒤에**
# 올려라. 그전까지는 0.
export PIVOT_ANGLE_MARGIN_DEG="${ANGLE_MARGIN_DEG:-0.0}"
exec "${R}" python dual_view.py \
  --mode deploy --bus local --hardware real \
  --angle-floor-deg "${ANGLE_FLOOR_DEG:-2.0}" \
  --object "${OBJECT}" --grasp pinch --grasp-part "${GRASP_PART}" \
  --prior water --target "${TARGET}" --max-rounds "${MAX_ROUNDS}" \
  --move-duration "${MOVE_DURATION}" \
  --gripper-port "${GRIPPER_PORT}" --gripper-force "${GRIPPER_FORCE}" \
  --pose-file "${FP_OUTPUT}/latest.json" \
  --tare-file "${TARE_FILE}" \
  --meshpca-root "${MESHPCA_ROOT}" \
  --aft-host "${AFT_HOST}" --robot-host "${ROBOT_HOST}"
