#!/usr/bin/env bash
# 장비 없이 통합 UI 로 데스크 램프 실험을 한 번 끝까지 돌려본다.
#
#   ./setup/quickstart_sim.sh            준비 검사 -> 통합 UI (버튼은 사람이 누른다)
#   ./setup/quickstart_sim.sh --auto     버튼까지 자동 (끝까지 혼자 돈다)
#   ./setup/quickstart_sim.sh --check    준비 검사만
#
# 하는 일
#   1) 브랜치가 들어와 있는지 (pivot_ui 가 --prior weight 를 쓰는지)
#   2) Drake 환경이 있는지 (없으면 bootstrap.sh)
#   3) setup/experiment.sim.conf 가 없으면 예시에서 만든다
#   4) 단계 기계 dry-run
#   5) 통합 UI 를 --rehearse 로 띄운다  ->  http://localhost:8080
#
# 로봇에 아무 명령도 보내지 않는다. ROBOT_HOST 가 비어 있는 conf 만 받는다.
# 그래서 개발 PC 에서도, 로봇 PC 에서도 안전하게 돌릴 수 있다. 실물 세션은
# 로봇 PC 에서 setup/launch_experiment.sh 로 한다 (TEAMMATE_CHECKLIST.md §6-0).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
R="${ROOT}/robot_learning/scripts/run_drake_env.sh"
CONF="${HERE}/experiment.sim.conf"
MODE="${1:-run}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\033[31m[실패]\033[0m %s\n' "$*"; exit 1; }

cd "${ROOT}"

say "[1/5] 코드 — 브랜치 확인"
if ! grep -q '"--prior", "weight"' my_work/pivot_ui.py; then
  fail "my_work/pivot_ui.py 가 아직 --prior water 입니다. torque-only-given-mass 브랜치를 먼저 가져오세요:
  git remote add lee https://github.com/LEE9396/PIVOT.git; git fetch lee torque-only-given-mass
  git cherry-pick 2dd7611..lee/torque-only-given-mass"
fi
grep -q -- '--rehearse' my_work/pivot_ui.py || fail "pivot_ui.py 에 --rehearse 가 없습니다 (브랜치가 오래됐습니다)"
echo "  OK  $(git log --oneline -1)"

say "[2/5] Drake 환경"
if [[ ! -x "${ROOT}/robot_learning/.venv-drake-1.54-py312/bin/python" ]]; then
  echo "  없음 -> setup/bootstrap.sh 를 돌립니다 (수 분)"
  "${HERE}/bootstrap.sh"
fi
"${R}" python -c "import pydrake, scipy" 2>/dev/null || fail "Drake 환경 import 실패. ./setup/bootstrap.sh --check"
echo "  OK"

say "[3/5] 설정 — ${CONF}"
if [[ ! -f "${CONF}" ]]; then
  cp "${HERE}/experiment.sim.conf.example" "${CONF}"
  sed -i "s#^PIVOT_ROOT=.*#PIVOT_ROOT=${ROOT}#; s#~/Desktop/PIVOT#${ROOT}#g" "${CONF}"
  echo "  예시에서 만들었습니다"
fi
# 실물로 갈 수 있는 값이 들어 있으면 여기서 멈춘다.
if grep -qE '^ROBOT_HOST=.+' "${CONF}"; then
  fail "${CONF} 에 ROBOT_HOST 가 있습니다. 리허설 conf 는 ROBOT_HOST 를 비워야 합니다"
fi
grep -E '^(OBJECT|GRASP_FRAME|TOTAL_MASS_KG|GRASP_SIGMA_MM|MAX_ROUNDS)=' "${CONF}" | sed 's/^/  /'
[[ -d "$(grep -E '^LAMP_ASSET_DIR=' "${CONF}" | cut -d= -f2 | sed "s#~#${HOME}#")" ]] \
  || fail "LAMP_ASSET_DIR 가 없습니다 (램프 자산이 저장소에 있어야 합니다)"

say "[4/5] 단계 기계 dry-run"
"${R}" python my_work/pivot_ui.py --dry-run --sessions /tmp/pivot-quickstart-dry >/dev/null \
  && echo "  OK" || fail "pivot_ui --dry-run 실패"
"${R}" python tools/preflight.py --conf "${CONF}" | grep -E "토크 기준점|저울 총질량|파지 사전평균|파지 불확실성" | sed 's/^/  /' || true
echo "  (시뮬이라 위 항목은 WARN 이어도 진행합니다)"

if [[ "${MODE}" == "--check" ]]; then
  say "준비 끝. 띄우려면:  ./setup/quickstart_sim.sh"
  exit 0
fi

say "[5/5] 통합 UI — 리허설"
echo "  브라우저에서 http://localhost:8080  (창 1 Meshcat 은 터미널에 링크가 뜹니다)"
echo "  단계: 0 준비 -> 1 파지 -> 2 각도 -> 3 경로 -> 4 탐색 -> 5 내보내기"
echo "  로봇은 움직이지 않습니다. 파지는 짐작값, 렌치·자세는 모의입니다."
ARGS=(--conf "${CONF}" --rehearse --sessions "${ROOT}/my_work/sessions")
[[ "${MODE}" == "--auto" ]] && ARGS+=(--auto)
# -u: 단계 안내가 버퍼에 갇히지 않고 바로 터미널에 뜬다 (처음 보는 사람이 따라가야 한다)
exec "${R}" python -u my_work/pivot_ui.py "${ARGS[@]}"
