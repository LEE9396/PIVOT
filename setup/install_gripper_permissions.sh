#!/usr/bin/env bash
# Robotiq FT232R가 재연결되어도 현재 로그인 사용자가 읽고 쓸 수 있게 한다.
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "sudo $0 로 한 번만 실행하세요."
  exit 1
fi

PIVOT_USER="${SUDO_USER:-${1:-}}"
if [[ -z "${PIVOT_USER}" || "${PIVOT_USER}" == root ]]; then
  echo "적용할 일반 사용자를 찾지 못했습니다. sudo $0 <사용자>"
  exit 1
fi

usermod -aG dialout "${PIVOT_USER}"
printf '%s\n' \
  'SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", ATTRS{serial}=="A9O3H0CZ", GROUP="dialout", MODE="0660", TAG+="uaccess", SYMLINK+="pivot-robotiq"' \
  > /etc/udev/rules.d/70-pivot-robotiq.rules
udevadm control --reload-rules
udevadm trigger --subsystem-match=tty
[[ ! -e /dev/ttyUSB0 ]] || setfacl -m "u:${PIVOT_USER}:rw" /dev/ttyUSB0

echo "완료: ${PIVOT_USER}를 dialout에 추가하고 /dev/pivot-robotiq 규칙을 설치했습니다."
echo "현재 세션은 즉시 사용 가능하며, 다음 로그인부터 그룹 권한도 유지됩니다."
