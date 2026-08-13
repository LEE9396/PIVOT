#!/usr/bin/env bash
# GitHub 에 **비공개** 저장소를 만들고 올린다. 처음 한 번만 쓰면 된다.
#
#   ./setup/push_to_github.sh                 이름 기본값으로
#   ./setup/push_to_github.sh 다른이름         이름을 정해서
#
# 미리 해야 할 것 (사람이 직접, 한 번만):
#   gh auth login          -> GitHub.com / HTTPS / 브라우저 로그인
#                             (브라우저에 로그인돼 있으면 화면에 뜨는
#                              8자리 코드만 붙여 넣으면 됩니다)
#
# 처음에는 비공개로 만든다. 공개로 돌리려면 만든 뒤에
#   gh repo edit <계정>/<이름> --visibility public --accept-visibility-change-consequences
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
name="${1:-artman-density-id}"
gh_bin="$(command -v gh || echo "${HOME}/.local/bin/gh")"

cd "${repo_root}"

if [[ ! -x "${gh_bin}" ]]; then
    echo "gh 를 못 찾았습니다. 설치: https://cli.github.com" >&2
    exit 2
fi

if ! "${gh_bin}" auth status >/dev/null 2>&1; then
    cat >&2 <<EOF
GitHub 로그인이 안 돼 있습니다. 이 터미널에서 먼저 실행하세요.

    ${gh_bin} auth login

  - What account do you want to log into?   GitHub.com
  - Preferred protocol                      HTTPS
  - Authenticate Git with your credentials? Yes
  - How would you like to authenticate?     Login with a web browser

화면에 8자리 코드가 나옵니다. 브라우저에서 그 코드를 넣으면 끝입니다.
그 다음 이 스크립트를 다시 실행하세요.
EOF
    exit 2
fi

account="$("${gh_bin}" api user --jq .login)"
echo "GitHub 계정: ${account}"
echo "만들 저장소: ${account}/${name}  (비공개)"

if "${gh_bin}" repo view "${account}/${name}" >/dev/null 2>&1; then
    echo "이미 있습니다. 원격만 잇고 올립니다."
    git remote get-url origin >/dev/null 2>&1 \
        || git remote add origin "https://github.com/${account}/${name}.git"
    git push -u origin HEAD
else
    "${gh_bin}" repo create "${name}" \
        --private \
        --source=. \
        --remote=origin \
        --description "관절 물체의 부위별 밀도를 손목 F/T 로 식별하는 파이프라인 (RB5 + AFT200 + Drake)" \
        --push
fi

echo
echo "완료했습니다.  https://github.com/${account}/${name}"
echo
echo "팀원 초대 (아이디를 넣으세요):"
echo "    ${gh_bin} repo add-collaborator ${account}/${name} <팀원_깃허브_아이디> --permission push"
echo
echo "팀원 쪽에서 할 일:"
echo "    git clone https://github.com/${account}/${name}.git ~/Desktop/PIVOT"
echo "    cd ~/Desktop/PIVOT && ./setup/bootstrap.sh"
