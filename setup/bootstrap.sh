#!/usr/bin/env bash
# 새 PC 에서 이 한 줄이면 끝난다.
#
#   ./setup/bootstrap.sh            환경을 만들고 진단까지
#   ./setup/bootstrap.sh --check    이미 있는 환경을 진단만
#
# 하는 일
#   1) robot_learning/.venv-drake-1.54-py312 에 파이썬 3.12 환경을 만든다
#   2) requirements/drake.txt 를 그대로 설치한다
#   3) setup/doctor.py 로 자산·임포트·씬 생성을 확인한다
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${repo_root}/robot_learning/.venv-drake-1.54-py312"
requirements="${repo_root}/robot_learning/requirements/drake.txt"
runner="${repo_root}/robot_learning/scripts/run_drake_env.sh"

check_only=0
[[ "${1:-}" == "--check" ]] && check_only=1

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 파이썬 3.12
make_env() {
    if [[ -x "${venv_dir}/bin/python" ]]; then
        say "[1/3] 파이썬 환경이 이미 있습니다: ${venv_dir}"
        return
    fi
    say "[1/3] 파이썬 3.12 환경을 만듭니다"
    if command -v conda >/dev/null 2>&1; then
        conda create --prefix "${venv_dir}" python=3.12 pip -y
    elif python3.12 -c "import venv" >/dev/null 2>&1; then
        python3.12 -m venv "${venv_dir}"
    else
        cat >&2 <<'EOF'
파이썬 3.12 를 만들 방법이 없습니다. 둘 중 하나를 설치하세요.

  (가) conda  — https://docs.conda.io/en/latest/miniconda.html
  (나) 우분투 패키지
        sudo apt update
        sudo apt install python3.12 python3.12-venv

설치한 뒤 이 스크립트를 다시 실행하세요.
EOF
        exit 2
    fi
}

install_requirements() {
    say "[2/3] 필요한 꾸러미를 설치합니다 (Drake 1.54 포함, 5~10분)"
    "${venv_dir}/bin/python" -m pip install --upgrade pip
    "${venv_dir}/bin/python" -m pip install -r "${requirements}"
}

if [[ ${check_only} -eq 0 ]]; then
    make_env
    install_requirements
else
    say "[진단만 합니다 — 설치는 건너뜁니다]"
    if [[ ! -x "${venv_dir}/bin/python" ]]; then
        echo "파이썬 환경이 없습니다. --check 없이 다시 실행하세요." >&2
        exit 2
    fi
fi

say "[3/3] 자가 진단"
cd "${repo_root}/my_work"
"${runner}" python "${repo_root}/setup/doctor.py"
