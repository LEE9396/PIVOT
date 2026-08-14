#!/usr/bin/env bash
# URDF/SDF 를 Drake 내장 Meshcat 뷰어로 띄운다.
#
# 왜 별도 뷰어를 안 깔았나: 저장소가 이미 Drake 1.54 를 고정해 두었고,
# pydrake 에 model_visualizer 가 들어 있다. 파서가 시뮬레이터와 "같은" 파서라
# 별도 뷰어에서 멀쩡히 보이던 파일이 Drake 에서만 깨지는 사고가 없다.
#
# 사용:
#   ./view_urdf.sh                        # estimated_desklamp.urdf
#   ./view_urdf.sh estimated_3link.urdf
#   ./view_urdf.sh foo.urdf --visualize_frames
#
# 브라우저 우상단 "Open Controls" 를 열면 관절 슬라이더가 나온다.
# 끄려면 터미널에서 Ctrl-C, 또는 화면의 "Stop Running".
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

model="${1:-estimated_desklamp.urdf}"
[[ $# -gt 0 ]] && shift

exec ../robot_learning/scripts/run_drake_env.sh \
    python -m pydrake.visualization.model_visualizer \
    --open-window "${model}" "$@"
