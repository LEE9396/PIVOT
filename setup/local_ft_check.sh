#!/usr/bin/env bash
# 로봇 연결 후 정지 측정부터 시작한다. 기본 동작은 계획만 하며 이동하지 않는다.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/experiment.conf"
cd "${PIVOT_ROOT}/my_work"
exec "${PIVOT_ROOT}/robot_learning/scripts/run_drake_env.sh" python -B \
  "${PIVOT_ROOT}/tools/local_ft_check.py" \
  --host "${ROBOT_HOST}" --meshpca-root "${MESHPCA_ROOT}" "$@"
