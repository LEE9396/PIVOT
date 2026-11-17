#!/usr/bin/env python3
"""experiment.conf 의 KEY=VALUE 를 제자리에서 고친다. 주석과 순서는 그대로 둔다.

    python tools/conf_set.py setup/experiment.conf TOTAL_MASS_KG=0.571 GRASP_MU_MM=
    python tools/conf_set.py setup/experiment.conf --min GRASP_SIGMA_MM=15

KEY=VALUE      그 키를 그 값으로. 없으면 파일 끝에 붙인다.
--min KEY=N    지금 값이 N 보다 작거나 없을 때만 N 으로 올린다.
--show KEY     현재 값을 찍는다 (없으면 빈 줄).

preflight.read_conf 와 같은 형식(KEY=VALUE, # 주석, 따옴표 벗김)만 다룬다.
셸을 실행하지 않는다.
"""

import re
import sys
from pathlib import Path

KEY = re.compile(r"^([A-Z_][A-Z0-9_]*)=(.*)$")


def set_keys(path, assignments, minimums=()):
    lines = path.read_text().splitlines() if path.is_file() else []
    todo = dict(assignments)
    floors = dict(minimums)
    out, changed = [], []
    for line in lines:
        m = KEY.match(line.strip())
        if m and (m.group(1) in todo or m.group(1) in floors):
            key, current = m.group(1), m.group(2).strip().strip("'\"")
            if key in todo:
                value = todo.pop(key)
            else:
                floor = floors.pop(key)
                try:
                    value = floor if float(current or "nan") < float(floor) else current
                except ValueError:
                    value = floor
            if value != current:
                changed.append((key, current, value))
            out.append(f"{key}={value}")
        else:
            out.append(line)
    for key, value in list(todo.items()) + list(floors.items()):
        out.append(f"{key}={value}")
        changed.append((key, None, value))
    path.write_text("\n".join(out) + "\n")
    return changed


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    path = Path(argv[1]).expanduser()
    assignments, minimums, shows = [], [], []
    args = iter(argv[2:])
    for token in args:
        if token == "--min":
            key, _, value = next(args).partition("=")
            minimums.append((key, value))
        elif token == "--show":
            shows.append(next(args))
        else:
            key, sep, value = token.partition("=")
            if not sep:
                sys.exit(f"KEY=VALUE 형식이 아닙니다: {token}")
            assignments.append((key, value))
    if shows:
        from preflight import read_conf
        conf = read_conf(path)
        for key in shows:
            print(conf.get(key, ""))
        return
    for key, before, after in set_keys(path, assignments, minimums):
        arrow = "추가" if before is None else f"{before!r} ->"
        print(f"  {key}: {arrow} {after!r}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main(sys.argv)
