#!/usr/bin/env bash
# RB5 손목 전용 3자세 영점 조정. 기본은 무동작 계획 검증이다.
#
#   setup/wrist_tare.sh --plan       J1-J3 고정 경로만 계산
#   setup/wrist_tare.sh --run        J4-J6 자동 영점 조정 실행 (원위치 복귀 없음)
#   setup/wrist_tare.sh --run-force  검산에 떨어져도 저장 (값은 못 믿는다)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${HERE}/experiment.conf"
MODE="${1:---plan}"
shift || true
EXTRA=("$@")          # 뒤에 붙인 것은 tare_real.py 로 그대로 넘긴다
                      #   예) setup/wrist_tare.sh --plan --clearance-mm 15

if [[ ! -f "${CONF}" ]]; then
  echo "설정 파일이 없습니다: ${CONF}"
  echo "  cp setup/experiment.conf.example setup/experiment.conf"
  exit 1
fi
# shellcheck disable=SC1090
source "${CONF}"

expand() { eval echo "$1"; }
PIVOT_ROOT="$(expand "${PIVOT_ROOT}")"
MESHPCA_ROOT="$(expand "${MESHPCA_ROOT}")"
TARE_FILE="$(expand "${TARE_FILE}")"
R="${PIVOT_ROOT}/robot_learning/scripts/run_drake_env.sh"
SCRIPT="${PIVOT_ROOT}/integration/meshpca/tare_real.py"

case "${MODE}" in
  --plan)
    ACTION=(--plan-only)
    ;;
  --run)
    if pgrep -f 'python .*dual_view.py .*--hardware real' >/dev/null; then
      echo "실물 통합 UI가 실행 중입니다. UI를 종료한 뒤 영점 조정을 실행하세요." >&2
      exit 1
    fi
    echo "5초 뒤 손목 J4-J6 자동 영점 조정을 시작합니다. 작업영역에서 물러나세요."
    echo "세 번째 자세에서 저장·종료하며 원위치로 복귀하지 않습니다."
    sleep 5
    ACTION=(--setup --overwrite)
    ;;
  --run-force)
    # 검산에 떨어져도 영점 파일을 **만든다.** 파일이 없으면 통합 UI 가
    # 아예 안 켜지기 때문이다 (tare.apply 가 그 방향 값이 없다고 멈춘다).
    #
    # 이 값으로 나온 밀도를 믿으라는 뜻이 아니다. 그래도 얻는 것이 있다 —
    # 실험을 파지까지 진행하면 grasp.json 이 생기고, 그것이 있어야
    # tools/check_grasp_frames.py 로 토크 기준점(187.7 mm)과 파지
    # 좌표계(137.9 deg) 어긋남을 **실측으로** 잴 수 있다. 그 둘은 아직
    # 한 번도 못 쟀고, 다른 방법도 없다.
    if pgrep -f 'python .*dual_view.py .*--hardware real' >/dev/null; then
      echo "실물 통합 UI가 실행 중입니다. UI를 종료한 뒤 실행하세요." >&2
      exit 1
    fi
    echo "[주의] 검산에 떨어져도 저장합니다."
    echo "       이 영점으로 나온 밀도는 믿을 수 없습니다. 진단용입니다."
    echo "5초 뒤 손목 J4-J6 자동 영점 조정을 시작합니다. 작업영역에서 물러나세요."
    echo "마지막 자세에서 저장·종료하며 원위치로 복귀하지 않습니다."
    sleep 5
    ACTION=(--setup --overwrite --force)
    ;;
  -h|--help)
    sed -n '2,6p' "$0"
    exit 0
    ;;
  *)
    echo "사용법: $0 --plan|--run|--run-force" >&2
    exit 2
    ;;
esac

export PIVOT_WORKDIR="${PIVOT_ROOT}/my_work"
exec "${R}" env PYTHONPATH="${MESHPCA_ROOT}/pivot" python -u "${SCRIPT}" \
  --robot-ip "${ROBOT_HOST}" --output "${TARE_FILE}" "${ACTION[@]}" \
  ${EXTRA[@]+"${EXTRA[@]}"}
