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
GAUSSIAN_DIR="${GAUSSIAN_DIR:-}"
GAUSSIAN_FILES="${GAUSSIAN_FILES:-}"

expand() { eval echo "$1"; }
for name in PIVOT_ROOT MESHPCA_ROOT FOUNDATIONPOSE_ROOT SAM3_ROOT \
            MESHPCA_PYTHON SAM3_PYTHON TARE_FILE FP_MESH_DIR \
            FP_INIT_RGB FP_INIT_DEPTH FP_INTRINSICS FP_MASKS FP_OUTPUT \
            LAMP_ASSET_DIR LAPTOP_ASSET_DIR; do
  printf -v "$name" '%s' "$(expand "${!name}")"
done
export DESK_LAMP_DELIVERY="${LAMP_ASSET_DIR}"
[[ -z "${GAUSSIAN_DIR}" ]] || GAUSSIAN_DIR="$(expand "${GAUSSIAN_DIR}")"

WORK="${PIVOT_ROOT}/my_work"
R="${PIVOT_ROOT}/robot_learning/scripts/run_drake_env.sh"
GRASP_JSON="${WORK}/outputs/grasp_target_${OBJECT}.json"
FP_GRASP_MESH="${FP_MESH_DIR}/${FP_GRASP_PART}.obj"
[[ -f "${FP_GRASP_MESH}" ]] || FP_GRASP_MESH="${FP_MESH_DIR}/${FP_GRASP_PART}.ply"

# FoundationPose 가 추적하는 메시는 **밀도 모델이 쓰는 바로 그 메시**여야 한다.
# 다르면 카메라가 읽어 준 물체 자세가 다른 좌표계의 값이 되고, 파지점이
# 통째로 어긋난다. 사람이 두 설정을 손으로 맞춰 두기를 바라면 언젠가 어긋나고,
# 어긋나도 화면에는 아무 표시가 안 난다.
#
# 그래서 설정을 믿지 않고 밀도 모델에게 직접 물어본다. 실제로 어긋나 있었다 —
# FP_MESH_DIR 은 assets/final_objects/lamp/visual_meshes 를 가리키는데 그
# 폴더는 비어 있고, 밀도 모델은 assets/desk_lamp_minimal_sim 을 쓰고 있었다.
PIVOT_MESH="$(DESK_LAMP_DELIVERY="${LAMP_ASSET_DIR}" "${R}" python -c "
import sys
sys.path.insert(0, '${WORK}')
import desk_lamp
print(desk_lamp.visual_mesh_path('${GRASP_PART}'))
" 2>/dev/null | tail -1)"
# 고르는 순서. **사람이 명시한 것이 언제나 이긴다.**
#   1) conf 의 FP_GRASP_MESH        직접 적어 준 경로
#   2) FP_MESH_DIR/FP_GRASP_PART    그 파일이 실제로 있으면
#   3) 밀도 모델이 쓰는 메시         위 둘이 다 없을 때만
#
# 앞서 3) 을 무조건 강제했더니 엉뚱한 메시가 올라왔다. 자동으로 고르는 쪽이
# 맞다는 보장이 없다 — 배달물이 덜 풀려 있으면 밀도 모델도 옛 자산으로
# 내려간다. 그래서 고르지 않고, **다르면 크게 알리고 사람이 정하게 한다.**
if [[ -n "${FP_GRASP_MESH_OVERRIDE:-}" ]]; then
  FP_GRASP_MESH="$(expand "${FP_GRASP_MESH_OVERRIDE}")"
  echo "[알림] FoundationPose 메시를 conf 의 FP_GRASP_MESH 로 씁니다:"
  echo "         ${FP_GRASP_MESH}"
fi
if [[ -n "${PIVOT_MESH}" && "${FP_GRASP_MESH}" != "${PIVOT_MESH}" ]]; then
  echo "[주의] FoundationPose 메시와 밀도 모델 메시가 다릅니다."
  echo "         FoundationPose  ${FP_GRASP_MESH}"
  echo "         밀도 모델        ${PIVOT_MESH}"
  echo "       두 메시는 원점과 축이 다르므로, 카메라가 읽어 준 물체 자세가"
  echo "       다른 좌표계의 값이 되어 파지점이 통째로 어긋납니다."
  echo "       (화면에는 아무 표시도 안 납니다)"
  echo "       고치는 법 — 둘 중 하나:"
  echo "         conf 의 LAMP_ASSET_DIR 를 쓰려는 배달물로 맞춘다"
  echo "         conf 에 FP_GRASP_MESH_OVERRIDE=<경로> 로 직접 지정한다"
fi
if [[ ! -f "${FP_GRASP_MESH}" ]]; then
  if [[ -n "${PIVOT_MESH}" && -f "${PIVOT_MESH}" ]]; then
    echo "[알림] 설정한 메시가 없어 밀도 모델의 것을 씁니다: ${PIVOT_MESH}"
    FP_GRASP_MESH="${PIVOT_MESH}"
  fi
fi
if [[ -f "${FP_GRASP_MESH}" ]]; then
  FP_MESH_DIR="$(dirname "${FP_GRASP_MESH}")"
  FP_GRASP_PART="$(basename "${FP_GRASP_MESH}")"
  FP_GRASP_PART="${FP_GRASP_PART%.*}"
fi
echo "[메시] FoundationPose: ${FP_GRASP_MESH}"
FOUNDATIONPOSE_PYTHON="${FOUNDATIONPOSE_PYTHON:-${SAM3_PYTHON%%/envs/*}/envs/${FOUNDATIONPOSE_CONDA_ENV}/bin/python}"

start_foundationpose() {
  local mask_mode="${1:-auto}" pid_file="${FP_OUTPUT}/foundationpose.pid"
  local old_pid=""
  mkdir -p "${FP_OUTPUT}"
  if [[ -f "${pid_file}" ]]; then
    old_pid="$(<"${pid_file}")"
    if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
      kill -- "-${old_pid}" 2>/dev/null || kill "${old_pid}" 2>/dev/null || true
    fi
  else
    mapfile -t old_pids < <(pgrep -f "[p]ython .*${FP_LIVE_SCRIPT}.*--output ${FP_OUTPUT}" || true)
    ((${#old_pids[@]} == 0)) || kill "${old_pids[@]}" 2>/dev/null || true
  fi
  for _ in $(seq 1 30); do
    pgrep -f "[p]ython .*${FP_LIVE_SCRIPT}.*--output ${FP_OUTPUT}" >/dev/null || break
    sleep 0.1
  done

  local mask_flag=() init_flag=()
  if [[ "${mask_mode}" == "manual" ]]; then
    mask_flag=(--manual-mask)
  elif [[ "${REUSE_INIT:-0}" == "1" && -f "${FP_INIT_RGB}" \
        && -f "${FP_INIT_DEPTH}" && -f "${FP_INTRINSICS}" \
        && -f "${FP_MASKS}/base.png" && -f "${FP_MASKS}/support.png" \
        && -f "${FP_MASKS}/head.png" ]]; then
    init_flag=(--reuse-init)
    echo "      방금 캡처한 초기 프레임·마스크를 재사용합니다"
  elif [[ "${MANUAL_MASK:-1}" == "1" ]]; then
    mask_flag=(--manual-mask)
  fi

  rm -f "${FP_OUTPUT}/latest.json"
  ( cd "${MESHPCA_ROOT}" && exec setsid "${FOUNDATIONPOSE_PYTHON}" "${FP_LIVE_SCRIPT}" \
        "${mask_flag[@]}" \
        "${init_flag[@]}" \
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
      >>/tmp/pivot_win2.log 2>&1 ) &
  echo "$!" >"${pid_file}"
  sleep 2
  kill -0 "$!" 2>/dev/null || return 1
}

if [[ "${MODE}" == "--remask" ]]; then
  PART_LEGEND="${PART_LEGEND:-/tmp/pivot_part_legend_3dgs.png}"
  if [[ ! -f "${PART_LEGEND}" ]]; then
    ( cd "${WORK}" && "${R}" python ../tools/make_part_legend.py \
        "${GAUSSIAN_DIR:-${FP_MESH_DIR}}" --files "${GAUSSIAN_FILES}" \
        --parts "${PART_NAMES:-base,support,head}" -o "${PART_LEGEND}" \
        --title "3DGS part labels for masking" ) || true
  fi
  export PIVOT_PART_LEGEND="${PART_LEGEND}"
  echo "FoundationPose를 재시작하고 새 프레임에서 수동 마스킹을 엽니다."
  start_foundationpose manual
  exit 0
fi

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

# --- 캘리브레이션 · 영점 조정 ---
CAM_JSON="$("${R}" python -c "
import sys; sys.path.insert(0,'${WORK}')
import robot_scene as rs; print(rs.calibration_path())" 2>/dev/null)"
if [[ -n "${CAM_JSON}" && -f "${CAM_JSON}" ]]; then
  ok "카메라 캘리브레이션 ($(basename "${CAM_JSON}"))"
else
  bad "카메라 캘리브레이션" \
      "import_calibration.py --input <MeshPCA handeye JSON> 를 돌리세요"
fi
need_file "${TARE_FILE}" "3자세 영점 조정"

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
if [[ -r "${GRIPPER_PORT}" && -w "${GRIPPER_PORT}" ]]; then
  ok "Robotiq 포트 권한 (${GRIPPER_PORT})"
else
  bad "Robotiq 포트 권한 (${GRIPPER_PORT})" \
      "sudo ${PIVOT_ROOT}/setup/install_gripper_permissions.sh 를 한 번 실행하세요"
fi
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
# 창 3 은 그리퍼와 F/T 만 보여주면 된다. 예전에는 걸쇠 실험 표(Latch
# Measurement.csv)를 늘 같이 띄웠는데, 이 실험과 상관이 없어 화면만 어지럽다.
# conf 의 WIN3_CSV 에 경로를 적었을 때만 띄운다. 비워 두면 안 띄운다.
WIN3_CSV="$(expand "${WIN3_CSV:-}")"
WIN3_ARGS=()
if [[ -n "${WIN3_CSV}" && -f "${WIN3_CSV}" ]]; then
  WIN3_ARGS=(--csv "${WIN3_CSV}")
  echo "         표 같이 띄움: $(basename "${WIN3_CSV}")"
fi
( cd "${MESHPCA_ROOT}" && "${MESHPCA_PYTHON}" pivot/rb5_ui.py \
    --host "${AFT_HOST}" --port "${GRIPPER_PORT}" --tare "${TARE_FILE}" \
    ${WIN3_ARGS[@]+"${WIN3_ARGS[@]}"} \
    --status-file "${FP_OUTPUT}/hardware.json" --headless \
    >/tmp/pivot_win3.log 2>&1 & )

# --- 창 2 준비: 부위 이름표 그림 ---
#
# 마스크 단계에서 사람이 base/support/head 박스를 그리는데, 어느 덩어리가
# 어느 이름인지 알려 주는 것이 없으면 조용히 틀린다 (my_work/NAMING.md).
# 배달물 메시에서 이름표를 구워 두고 환경변수로 넘긴다.
PART_LEGEND="${PART_LEGEND:-/tmp/pivot_part_legend_3dgs.png}"
if [[ ! -f "${PART_LEGEND}" ]]; then
  echo "[창 2] 부위 이름표를 굽습니다 -> ${PART_LEGEND}"
  "${R}" python "${PIVOT_ROOT}/tools/make_part_legend.py" \
      "${GAUSSIAN_DIR:-${FP_MESH_DIR}}" --files "${GAUSSIAN_FILES}" \
      --parts "${PART_NAMES:-base,support,head}" -o "${PART_LEGEND}" \
      --title "3DGS part labels for masking" \
      || echo "      [주의] 이름표를 못 구웠습니다 — 이름 없이 진행합니다"
fi
export PIVOT_PART_LEGEND="${PART_LEGEND}"

# --- 창 2: 카메라 뷰 + 파지점 오버레이 ---
echo "[창 2] 카메라 뷰 + 파지점 오버레이를 띄웁니다"
start_foundationpose

# 통합 UI를 먼저 띄운다. 0단계 화면이 트래커 첫 각도와 최종 점검을 기다린다.
echo "      카메라 초기화는 통합 UI의 0단계에서 확인합니다"

# --- 통합 지휘 UI: 0~5단계, 이후 기존 탐색·밀도 화면을 이어 붙인다. ---
echo "[통합 UI] 0~5단계 지휘 화면을 띄웁니다"
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
export PIVOT_FOV_MARGIN_PX="${CAMERA_FOV_MARGIN_PX:-20}"
export PIVOT_CAMERA_INTRINSICS="${FP_INTRINSICS}"
exec "${R}" python pivot_ui.py --conf "${CONF}"
