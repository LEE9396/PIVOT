#!/usr/bin/env python
"""세션 폴더와 단계(phase) 상태 — 창 4개가 공유하는 하나의 진실.

왜 필요한가
-----------
실험은 2 <-> 4 단계를 여러 번 오간다 (불확실성이 목표 밖이면 각도를 다시
조정하고 또 탐색). 단계마다의 결과를 남기지 않으면 **재탐색 때마다 파지부터
다시** 해야 한다. 그리고 창 2·3 은 지금 단계가 무엇인지 알아야 표시를 바꾼다
(파지점 오버레이를 끄고, 렌치 표시로 넘어가고).

그래서 세션 폴더 하나를 진실로 둔다. 창 1 이 쓰고, 창 2·3 은 읽는다.
파일 하나라 프로세스가 달라도, 파이썬 환경이 달라도(창 2 는 conda,
창 3 은 venv) 그냥 된다. 메시지 버스를 새로 놓을 이유가 없다.

    session_YYYYMMDD_HHMM/
      phase.json          지금 단계. 창 2·3 이 폴링한다
      preflight.json      0단계 점검 결과
      grasp.json          X_G_O(그리퍼 기준 물체), 파지 부위, 개구량
      angle_round_N.json  추천 θ, 확정 θ, 제외 목록
      path_round_N.json   q 시작·경유·끝, 충돌 검사 결과
      wrench_round_N.csv  렌치 원시값
      posterior_round_N.json
      export/             URDF + 메시 (sim-ready)
"""

import json
import os
import time
from pathlib import Path

PHASES = [
    ("preflight", "0 준비"),
    ("grasp",     "1 파지점"),
    ("angle",     "2 각도 조정"),
    ("path",      "3 경로"),
    ("explore",   "4 탐색"),
    ("export",    "5 내보내기"),
]
PHASE_KEYS = [key for key, _ in PHASES]


class Session:
    """세션 폴더 하나. 창 1 이 쓰고 창 2·3 이 읽는다."""

    def __init__(self, root, create=True):
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "export").mkdir(exist_ok=True)

    # -- 만들기 / 찾기 ---------------------------------------------------
    @classmethod
    def new(cls, base, stamp=None):
        stamp = stamp or time.strftime("%Y%m%d_%H%M")
        return cls(Path(base) / f"session_{stamp}")

    @classmethod
    def latest(cls, base):
        folders = sorted(Path(base).glob("session_*"))
        if not folders:
            return None
        return cls(folders[-1], create=False)

    @classmethod
    def from_env(cls, base=None):
        """PIVOT_SESSION 환경변수를 본다. 창 2·3 이 이걸로 붙는다."""
        env = os.environ.get("PIVOT_SESSION")
        if env:
            return cls(env, create=False)
        return cls.latest(base or ".") if base else None

    # -- 단계 -------------------------------------------------------------
    def set_phase(self, key, round_index=0, note="", **extra):
        if key not in PHASE_KEYS:
            raise ValueError(f"모르는 단계: {key}")
        payload = dict(phase=key, index=PHASE_KEYS.index(key),
                       label=dict(PHASES)[key], round=int(round_index),
                       note=note, updated_at=time.time(), **extra)
        self.write("phase.json", payload)
        return payload

    def phase(self):
        """지금 단계. 창 2·3 이 폴링한다. 없으면 0단계로 본다."""
        data = self.read("phase.json")
        if not data:
            return dict(phase="preflight", index=0, label="0 준비", round=0)
        return data

    def wait_phase(self, keys, timeout_s=None, poll_s=0.3):
        """지정한 단계 중 하나가 될 때까지 기다린다. 창 2·3 용."""
        keys = {keys} if isinstance(keys, str) else set(keys)
        end = None if timeout_s is None else time.time() + timeout_s
        while end is None or time.time() < end:
            current = self.phase()
            if current.get("phase") in keys:
                return current
            time.sleep(poll_s)
        return None

    # -- 파일 -------------------------------------------------------------
    def path(self, name):
        return self.root / name

    def write(self, name, payload):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # 창 2·3 이 읽는 중에 반쪽짜리를 보면 안 된다. 임시 파일에 쓰고 옮긴다.
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                        default=float) + "\n")
        temporary.replace(target)
        return target

    def read(self, name, default=None):
        target = self.root / name
        if not target.is_file():
            return default
        try:
            return json.loads(target.read_text())
        except json.JSONDecodeError:
            return default          # 쓰는 중일 수 있다. 다음 폴링에 다시 본다

    def append_csv(self, name, row, header=None):
        target = self.root / name
        new = not target.is_file()
        with open(target, "a") as handle:
            if new and header:
                handle.write(",".join(header) + "\n")
            handle.write(",".join(f"{v}" for v in row) + "\n")
        return target

    def rounds(self, prefix="posterior_round_"):
        return sorted(self.root.glob(f"{prefix}*.json"))


def describe(session):
    """터미널 한 줄 요약."""
    current = session.phase()
    bar = []
    for index, (key, label) in enumerate(PHASES):
        if index < current["index"]:
            bar.append(f"✓{label}")
        elif index == current["index"]:
            bar.append(f"[{label}]")
        else:
            bar.append(label)
    return "  ".join(bar) + f"   라운드 {current.get('round', 0)}"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="my_work/sessions")
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--set", default=None, help=f"단계로 옮긴다 {PHASE_KEYS}")
    ap.add_argument("--round", type=int, default=0)
    args = ap.parse_args()

    session = Session.new(args.base) if args.new else Session.latest(args.base)
    if session is None:
        raise SystemExit(f"{args.base} 에 세션이 없습니다 (--new 로 만드세요)")
    if args.set:
        session.set_phase(args.set, args.round)
    print(session.root)
    print(describe(session))
