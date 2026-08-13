#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
default_venv_dir="${repo_root}/.venv-drake-1.54-py312"
venv_dir="${DRAKE_VENV_DIR:-${default_venv_dir}}"

if [[ "${venv_dir}" != /* ]]; then
    venv_dir="${repo_root}/${venv_dir}"
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    echo "Drake environment not found: ${venv_dir}" >&2
    echo "Create it with:" >&2
    echo "  conda create --prefix ${default_venv_dir} python=3.12 pip -y" >&2
    echo "  ${default_venv_dir}/bin/python -m pip install -r requirements/drake.txt" >&2
    exit 2
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args ...]" >&2
    echo "Example: $0 python tests/test_drake_environment.py" >&2
    exit 2
fi

# The workstation sources ROS 2 globally. Its Python and shared-library paths
# leak into ordinary virtual environments, so keep this baseline process clean.
export VIRTUAL_ENV="${venv_dir}"
export PATH="${venv_dir}/bin:${PATH}"
export XDG_CACHE_HOME="${repo_root}/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
unset PYTHONPATH
unset LD_LIBRARY_PATH
unset PYTHONHOME
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
mkdir -p "${MPLCONFIGDIR}" "${PIP_CACHE_DIR}"

exec "$@"
