#!/usr/bin/env bash
# 로봇 PC 에서: 램프 무게 한 줄로 실물 실험 준비를 끝내고 통합 UI 를 띄운다.
#
#   ./setup/quickstart_real.sh --mass 0.571          # 저울 kg, 힌지 포함
#   ./setup/quickstart_real.sh --mass 0.571 --check  # 준비만 점검. 로봇을 절대 움직이지 않는다
#
# 순서 (런북 §1~§8 중 사람 손이 안 가는 것 전부)
#   1) 코드   — torque-only-given-mass 브랜치가 없으면 가져온다 (cherry-pick)
#   2) 설정   — 기존 setup/experiment.conf 의 키만 고친다 (경로·IP 는 그대로)
#               TOTAL_MASS_KG=<mass> GRASP_FRAME=measured TARE_MODE=gravity
#               GRASP_SIGMA_MM>=15 GRASP_MU_MM=(비움)
#   3) 자산   — desk_lamp.py 파지점 정의를 세션 노트에 남긴다
#   4) 장애물 — workspace_obstacles_current.json 이 validated 인지 (타어 장면에도 필요)
#   5) 타어   — 파일이 없거나 tare_check 에 떨어지면 setup/wrist_tare.sh --run 이
#               장면(책상·장애물·케이블 원통)으로 경로를 계획해 8방향을 자동으로 잰다
#               (5초 카운트다운 뒤 손목이 움직인다)
#   6) 점검   — launch_experiment.sh --check + preflight
#   7) 실행   — launch_experiment.sh (창 4개 + 대시보드 :8080)
#
# 탐색 1라운드가 끝나면 통합 UI 가 tools/round_check.py 로 검산 3개를 찍는다.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
R="${ROOT}/robot_learning/scripts/run_drake_env.sh"
CONF="${HERE}/experiment.conf"
MASS=""; MODE="run"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mass) MASS="${2:-}"; shift 2 ;;
    --check) MODE="check"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "모르는 인자: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\033[31m[실패]\033[0m %s\n' "$*"; exit 1; }

[[ -n "${MASS}" ]] || fail "--mass <저울 kg, 힌지 포함> 이 필요합니다. 예) --mass 0.571"
python3 - "${MASS}" <<'PY' || fail "--mass 는 0.05~5 kg 사이 숫자여야 합니다 (g 가 아니라 kg)"
import sys; m=float(sys.argv[1]); sys.exit(0 if 0.05 <= m <= 5.0 else 1)
PY
cd "${ROOT}"

say "[1/6] 코드 — torque-only-given-mass 브랜치"
if ! grep -q '"--prior", "weight"' my_work/pivot_ui.py; then
  echo "  브랜치가 없습니다. LEE9396/PIVOT 에서 가져옵니다."
  git remote add lee https://github.com/LEE9396/PIVOT.git 2>/dev/null || true
  git fetch lee torque-only-given-mass
  # 작업 트리가 더러워도 된다 — 가져올 커밋이 건드리는 파일에 미커밋 수정이 있을 때만
  # 막는다. 로봇 PC 에는 진행 중인 분석 파일과 미추적 산출물이 늘 있기 마련이다.
  BASE="$(git merge-base HEAD lee/torque-only-given-mass)"
  CLASH="$(comm -12 <(git diff --name-only "${BASE}" lee/torque-only-given-mass | sort) \
                    <(git diff --name-only HEAD | sort))"
  [[ -z "${CLASH}" ]] || fail "가져올 커밋이 고치는 파일에 미커밋 수정이 있습니다. 먼저 커밋하거나 stash 하세요:
${CLASH}"
  # 범위를 공통 조상부터 잡는다. 이 체크아웃이 브랜치 부모(2dd7611)보다 더 나가
  # 있어도, 브랜치가 그 위로 rebase 돼 있으면 중복 커밋 없이 붙는다.
  git cherry-pick "$(git merge-base HEAD lee/torque-only-given-mass)..lee/torque-only-given-mass" \
    || fail "cherry-pick 충돌. 'git status' 로 파일을 보고 팀에 알리세요 (git cherry-pick --abort 로 되돌릴 수 있음)"
fi
grep -q -- '--rehearse' my_work/pivot_ui.py || fail "브랜치가 오래됐습니다 (pivot_ui 에 --rehearse 없음). git fetch lee 후 다시"
echo "  OK  $(git log --oneline -1)"

say "[2/6] 설정 — ${CONF} (키만 고칩니다)"
[[ -f "${CONF}" ]] || fail "setup/experiment.conf 가 없습니다. 이 PC 의 경로·IP 가 든 conf 가 먼저 있어야 합니다:
  cp setup/experiment.conf.example setup/experiment.conf  후 PIVOT_ROOT/ROBOT_HOST/GRIPPER_PORT 등을 채우세요"
"${R}" python tools/conf_set.py "${CONF}" \
    "TOTAL_MASS_KG=${MASS}" GRASP_FRAME=measured TARE_MODE=gravity GRASP_MU_MM= \
    --min GRASP_SIGMA_MM=15
ROBOT_HOST="$("${R}" python tools/conf_set.py "${CONF}" --show ROBOT_HOST)"
[[ -n "${ROBOT_HOST}" ]] || fail "conf 에 ROBOT_HOST 가 없습니다. 이 스크립트는 로봇 PC 용입니다 (장비 없이는 quickstart_sim.sh)"
TARE_FILE="$("${R}" python tools/conf_set.py "${CONF}" --show TARE_FILE)"
echo "  ROBOT_HOST=${ROBOT_HOST}  TARE_FILE=${TARE_FILE:-(없음)}"

say "[3/6] 자산 — 파지점 정의"
N="$(grep -c 'geometry\[root\]\["centroid"\]' my_work/desk_lamp.py || true)"
echo "  desk_lamp.py centroid 줄 = ${N}  ($( [[ "${N}" == "0" ]] && echo '핀치 점' || echo '부피 도심' ) 정의 — measured 모드라 일관되면 됩니다)"
sha256sum -c setup/lamp-assets.sha256 --quiet 2>/dev/null && echo "  자산 sha256 OK" || echo "  [주의] 자산 sha256 불일치 — 메시가 다른 빌드일 수 있습니다 (TEAMMATE_CHECKLIST §2)"

say "[4/6] 장애물 파일 — 타어와 실험의 충돌 장면에 둘 다 필요"
OBST="${ROOT}/calibration/workspace_obstacles_current.json"
if [[ ! -f "${OBST}" ]]; then
  echo "  [필요] ${OBST} 가 없습니다 — 받침대·단차 같은 추가 장애물을 한 번 재서 적어야 합니다."
  echo "         예시: calibration/workspace_obstacles_example.json 을 복사해 center_m/size_m 를"
  echo "         로봇 베이스 기준 m 단위로 채우고 status 를 validated 로 바꾸세요 (없으면 [] 로 두고 validated)."
  fail "장애물 파일 없이는 충돌 장면이 완성되지 않습니다"
fi
OBST_STATUS="$("${R}" python -c "
import sys; sys.path.insert(0,'my_work'); import workspace_obstacles as w
d=w.load('${OBST}'); print(d['status'], len(d['boxes']), '|', ', '.join(d['unresolved']))" 2>&1 | tail -1)"
echo "  ${OBST_STATUS}"
[[ "${OBST_STATUS}" == validated* ]] || fail "장애물 파일이 validated 가 아닙니다 (${OBST_STATUS}). 실측을 채우고 status 를 validated 로, unresolved 를 [] 로 바꾸세요. 장애물이 없으면 boxes 도 []"

say "[5/6] 타어 — 빈 그리퍼 영점 (이 PC·이 배선)"
tare_ok=0
if [[ -n "${TARE_FILE}" && -f "${TARE_FILE}" ]]; then
  if "${R}" python my_work/tare_check.py "${TARE_FILE}" >/tmp/pivot_tare_check.log 2>&1; then
    tare_ok=1; grep -E "잔차|축" /tmp/pivot_tare_check.log | sed 's/^/  /' | head -4
    echo "  OK  $(stat -c '%y' "${TARE_FILE}" | cut -c1-16) 에 잰 파일"
  else
    echo "  tare_check 실패:"; tail -6 /tmp/pivot_tare_check.log | sed 's/^/    /'
  fi
else
  echo "  타어 파일이 없습니다."
fi
if [[ ${tare_ok} -eq 0 && "${MODE}" == "check" ]]; then
  echo "  [--check] 타어가 필요합니다. --check 에서는 손목을 움직이지 않습니다."
  echo "           --check 없이 실행하면 이 자리에서 자동 영점 조정(5초 카운트다운 뒤 손목 이동)을 합니다."
  TARE_PENDING=1
elif [[ ${tare_ok} -eq 0 ]]; then
  echo
  echo "  자동 영점 조정을 시작합니다. 장면(책상·장애물·케이블 원통)으로 충돌 없는"
  echo "  경로를 계획해 8방향을 스스로 돕니다. 그리퍼는 비어 있어야 하고, 케이블은"
  echo "  팔에 고정돼 센서를 가로지르지 않아야 합니다. (5초 뒤 손목이 움직입니다)"
  "${HERE}/wrist_tare.sh" --run \
    || fail "타어 실패. 힘 잔차가 0.5 N 을 넘으면 센서가 아니라 케이블을 다시 고정하고 재실행"
  "${R}" python my_work/tare_check.py "${TARE_FILE}" || fail "tare_check 불합격 — 케이블 고정 후 다시"
fi

say "[6/7] 점검"
"${HERE}/launch_experiment.sh" --check || { [[ "${MODE}" == "check" && "${TARE_PENDING:-0}" == 1 ]] && echo "  (타어가 없어 실패한 항목은 실행 때 자동으로 채워집니다)" || fail "런처 점검 실패 — 위 항목을 고치고 다시"; }
"${R}" python tools/preflight.py --conf "${CONF}" | grep -E "실패|주의|토크 기준점|저울 총질량|파지 사전평균|파지 불확실성" | sed 's/^/  /' || true
echo
echo "  사람이 확인할 것 1개 (한 번만):"
echo "    - AFT200 렌치 기준면이 ft_mount 원통 중심과 같은가 (데이터시트, 최대 ~26 mm)"
echo "  파지 좌표계 검사(check_grasp_frames)는 통합 UI 가 파지 직후 자동으로 합니다."

if [[ "${MODE}" == "check" ]]; then
  if [[ "${TARE_PENDING:-0}" == 1 ]]; then
    say "준비 점검 끝 — 타어만 남았습니다. 실행하면 먼저 손목을 움직여 8방향을 잽니다:  ./setup/quickstart_real.sh --mass ${MASS}"
  else
    say "준비 끝. 띄우려면:  ./setup/quickstart_real.sh --mass ${MASS}"
  fi
  exit 0
fi

say "[7/7] 통합 UI — 실물"
echo "  대시보드 http://localhost:8080 · 단계 0 준비 -> 1 파지 -> 2 각도 -> 3 경로 -> 4 탐색 -> 5 내보내기"
echo "  탐색 1라운드가 끝나면 검산 3개(힘 크기 / 잔차팽창 / 파지 오프셋)가 자동으로 찍힙니다."
exec "${HERE}/launch_experiment.sh"
