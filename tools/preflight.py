#!/usr/bin/env python
"""0단계 준비 점검 — 조용히 틀리는 것들을 소리나게 만든다.

이 파일이 있는 이유
-------------------
2026-09-02 에 실물 실험이 하루 날아갔다. 원인 두 개가 **둘 다 조용했다**.

  - 배달물에 볼록 분해가 없어 충돌 메시가 통짜 OBJ 하나로 폴백했다.
    Drake 가 그걸 볼록 껍질로 감싸 형상이 2~3 배 부풀고, 파지점이 팔
    한가운데로 옮겨가 head 가 F/T 마운트를 41 mm 파고들었다.
  - 창 2 의 SAM3 마스크가 base 를 0 픽셀로 잡아 FoundationPose 가 시작조차
    못 했다. 런처는 180 초를 기다리다 조용히 끝났다.

둘 다 "돌려 보기 전에는 알 수 없는" 것이 아니었다. 파일 몇 개만 보면
알 수 있었다. 그래서 그것들을 여기 모았다.

판정
----
  OK    이대로 가도 된다
  WARN  돌아가지만 결과가 조용히 나빠질 수 있다 (명목값으로 도는 것들)
  FAIL  여기서 멈춰야 한다. 진행하면 반드시 사고가 난다

실행
    $R python tools/preflight.py --conf setup/experiment.conf
    $R python tools/preflight.py --conf setup/experiment.conf --json out.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

OK, WARN, FAIL = "OK", "WARN", "FAIL"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, level, name, detail, fix=None):
        self.rows.append(dict(level=level, name=name, detail=detail, fix=fix))
        return level

    def worst(self):
        for level in (FAIL, WARN):
            if any(r["level"] == level for r in self.rows):
                return level
        return OK

    def show(self):
        mark = {OK: "  OK ", WARN: " 주의", FAIL: "*실패"}
        width = max(len(r["name"]) for r in self.rows) if self.rows else 10
        for row in self.rows:
            print(f"{mark[row['level']]}  {row['name']:<{width}}  {row['detail']}")
            if row["fix"] and row["level"] != OK:
                for line in row["fix"].split("\n"):
                    print(f"        -> {line}")
        counts = {level: sum(1 for r in self.rows if r["level"] == level)
                  for level in (OK, WARN, FAIL)}
        print(f"\n  통과 {counts[OK]} / 주의 {counts[WARN]} / 실패 {counts[FAIL]}")
        if counts[FAIL]:
            print("  실패가 있습니다. 고치기 전에는 다음 단계로 못 갑니다.")
        elif counts[WARN]:
            print("  진행할 수 있지만, 주의 항목은 결과를 조용히 나쁘게 합니다.")
        else:
            print("  전부 통과. 1단계(파지점 확정)로 갈 수 있습니다.")


def read_conf(path):
    """experiment.conf 의 KEY=VALUE 를 읽는다 (셸을 실행하지 않는다)."""
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            continue
        value = value.strip().strip("'\"")
        values[key] = os.path.expanduser(value)
    return values


def bbox_signature(mesh_dir, names):
    """부위 메시의 bbox 로 '어느 빌드인가' 를 나타내는 지문."""
    signature = {}
    for name in names:
        for suffix in (".obj", ".ply", ".stl"):
            path = Path(mesh_dir) / f"{name}{suffix}"
            if not path.is_file() or suffix != ".obj":
                continue
            lo = [float("inf")] * 3
            hi = [float("-inf")] * 3
            for line in open(path, errors="ignore"):
                if not line.startswith("v "):
                    continue
                xyz = [float(t) for t in line.split()[1:4]]
                lo = [min(a, b) for a, b in zip(lo, xyz)]
                hi = [max(a, b) for a, b in zip(hi, xyz)]
            if lo[0] != float("inf"):
                signature[name] = [round(v, 4) for v in lo + hi]
            break
    return signature


# ---------------------------------------------------------------------------
def check_asset(report, conf):
    """자산 경로와 볼록 분해 — 이번 사고의 1번 원인."""
    root = conf.get("LAMP_ASSET_DIR") or os.environ.get("DESK_LAMP_DELIVERY")
    if not root:
        return report.add(FAIL, "자산 경로", "LAMP_ASSET_DIR 이 없습니다",
                          "setup/experiment.conf 에 배달물 폴더를 적으세요")
    root = Path(root)
    if not root.is_dir():
        return report.add(FAIL, "자산 경로", f"없는 폴더: {root}")
    report.add(OK, "자산 경로", str(root))

    collisions = root / "collision_meshes"
    if not collisions.is_dir():
        collisions = root / "drake" / "collisions"
    if not collisions.is_dir():
        return report.add(FAIL, "충돌 메시", f"{root} 에 collision_meshes 가 없습니다")

    parts = sorted({p.stem for p in collisions.glob("*.obj")})
    if not parts:                              # minimal_v2 배치
        parts = sorted(d.name for d in collisions.iterdir() if d.is_dir())
    missing = []
    for name in parts:
        pieces = sorted((collisions / "convex" / name).glob("part_*.obj"))
        if not pieces:
            pieces = sorted((collisions / name).glob("part_*.obj"))
        if not pieces:
            missing.append(name)
    if missing:
        report.add(FAIL, "볼록 분해",
                   f"{', '.join(missing)} 에 볼록 조각이 없습니다"
                   f" -> 통짜 메시가 볼록 껍질 하나로 쓰입니다",
                   f"python tools/make_convex.py {collisions}\n"
                   "형상이 2~3 배 부풀고 파지점이 부위 한가운데로 갑니다.\n"
                   "2026-09-02 에 head 가 F/T 마운트를 41 mm 파고든 원인입니다.")
    else:
        report.add(OK, "볼록 분해", f"{len(parts)} 부위 모두 조각 있음")
    return collisions


def check_same_build(report, conf):
    """LAMP_ASSET_DIR 과 FP_MESH_DIR 이 같은 빌드인가 — 사고의 2번 원인 후보."""
    asset = conf.get("LAMP_ASSET_DIR")
    fp = conf.get("FP_MESH_DIR")
    if not (asset and fp):
        return report.add(WARN, "빌드 일치", "경로가 없어 확인 못 했습니다")
    asset, fp = Path(asset), Path(fp)
    if fp.parent == asset:
        return report.add(OK, "빌드 일치", "충돌·시각 메시가 같은 자산")
    names = sorted({p.stem for p in fp.glob("*.obj")})[:3]
    a = bbox_signature(asset / "collision_meshes", names)
    b = bbox_signature(fp, names)
    common = sorted(set(a) & set(b))
    if not common:
        return report.add(WARN, "빌드 일치",
                          f"서로 다른 폴더인데 비교할 부위가 없습니다\n"
                          f"        충돌 {asset}\n        시각 {fp}")
    worst = 0.0
    for name in common:
        worst = max(worst, max(abs(x - y) for x, y in zip(a[name], b[name])))
    if worst > 0.01:
        return report.add(FAIL, "빌드 일치",
                          f"충돌 메시와 시각 메시가 최대 {1000*worst:.0f} mm 어긋납니다"
                          f" — 다른 빌드입니다",
                          "LAMP_ASSET_DIR 과 FP_MESH_DIR 을 같은 자산으로 맞추세요.\n"
                          "한쪽만 바꾸면 충돌 기하와 창 2 오버레이가 다른 좌표계가 됩니다.")
    return report.add(OK, "빌드 일치", f"메시 차이 {1000*worst:.1f} mm")


def check_calibration(report, conf, work):
    """핸드아이·타어·책상·각도부호 — 없으면 조용히 명목값으로 돈다."""
    calib = work / "calibration"
    camera = sorted(calib.glob("camera_*.json"))
    if camera:
        report.add(OK, "핸드아이", f"{camera[0].name}")
    else:
        report.add(FAIL, "핸드아이", f"{calib}/camera_*.json 이 없습니다",
                   "import_calibration.py 로 옮기세요.\n"
                   "없으면 PIVOT 이 **명목 카메라 위치**로 조용히 돕니다 —\n"
                   "라운드를 늘려도 안 없어지는 치우침이 생깁니다.")

    tare = conf.get("TARE_FILE")
    if tare and Path(tare).is_file():
        report.add(OK, "3자세 타어", Path(tare).name)
    else:
        report.add(FAIL, "3자세 타어", f"없습니다: {tare}",
                   "MeshPCA pivot/tare_real.py 로 빈 그리퍼 타어를 재세요")

    table = calib / "rb5_table_current.json"
    if table.is_file():
        data = json.loads(table.read_text())
        if data.get("status") == "valid":
            plane = data["plane_in_robot_base"]
            report.add(OK, "책상 실측",
                       f"기울기 {plane.get('tilt_deg', 0):.2f} deg,"
                       f" rms {data.get('quality', {}).get('rms_mm', 0):.2f} mm")
        else:
            report.add(WARN, "책상 실측",
                       f"status={data.get('status')} — 안 씁니다 (명목값으로 갑니다)")
    else:
        report.add(WARN, "책상 실측", "없습니다 — 도면 명목값(윗면 750 mm)으로 돕니다",
                   "calibrate_table_rgbd.py 로 재세요.\n"
                   "명목이 실제보다 낮으면 부딪히는 자세를 통과시킵니다.")

    signs = calib / "angle_signs.json"
    if signs.is_file():
        data = json.loads(signs.read_text())
        rms = max(v.get("residual_rms_deg", 0) for v in data.values())
        report.add(OK, "각도 부호·영점",
                   f"{len(data)} 관절, 잔차 최대 {rms:.2f} deg"
                   f" (ANGLE_FLOOR_DEG 에 쓰세요)")
    else:
        report.add(WARN, "각도 부호·영점", "없습니다 — 각도 판정이 반대로 나올 수 있습니다",
                   "tools/angle_signs.py 로 관절마다 한 번씩 쓸어담으세요")


def check_tracker(report, conf):
    """FoundationPose 가 첫 각도를 냈는가 — 이게 없으면 창 1 이 헛돕니다."""
    output = conf.get("FP_OUTPUT")
    if not output:
        return report.add(WARN, "FoundationPose", "FP_OUTPUT 이 없습니다")
    latest = Path(output) / "latest.json"
    if not latest.is_file():
        return report.add(WARN, "FoundationPose",
                          f"{latest} 가 아직 없습니다 (창 2 를 먼저 띄우세요)",
                          "마스크가 자동으로 안 잡히면 MANUAL_MASK=1 로 사람이 그리세요.\n"
                          "2026-09-02: 글자 프롬프트가 base 를 0 픽셀로 잡아 실패했습니다.")
    try:
        data = json.loads(latest.read_text())
    except Exception as exc:                                   # noqa: BLE001
        return report.add(FAIL, "FoundationPose", f"latest.json 을 못 읽습니다: {exc}")
    angles = [k for k in data if k.endswith("_deg")]
    return report.add(OK, "FoundationPose",
                      f"각도 수신 {', '.join(angles) if angles else list(data)[:3]}")


def check_masks(report, conf):
    masks = conf.get("FP_MASKS")
    if not masks:
        return report.add(WARN, "SAM3 마스크", "FP_MASKS 가 없습니다")
    folder = Path(masks)
    summary = folder / "summary.json"
    if not summary.is_file():
        return report.add(WARN, "SAM3 마스크", "아직 안 만들었습니다")
    data = json.loads(summary.read_text())
    pixels = data.get("pixels", {})
    thin = [k for k, v in pixels.items() if v < 500]
    if thin:
        return report.add(FAIL, "SAM3 마스크",
                          f"픽셀이 너무 적은 부위: {thin} ({pixels})",
                          "MANUAL_MASK=1 로 사람이 박스를 그리세요.\n"
                          "부위 이름표는 tools/make_part_legend.py 가 굽습니다.")
    return report.add(OK, "SAM3 마스크", f"{pixels}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=Path, default=Path("setup/experiment.conf"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    conf = read_conf(args.conf.expanduser())
    if not conf:
        print(f"[주의] {args.conf} 를 못 읽었습니다 — 환경변수만 봅니다\n")
    work = Path(conf.get("PIVOT_ROOT", ".")).expanduser() / "my_work"

    print(f"PIVOT 0단계 준비 점검   ({args.conf})\n")
    report = Report()
    check_asset(report, conf)
    check_same_build(report, conf)
    check_calibration(report, conf, work)
    check_masks(report, conf)
    check_tracker(report, conf)
    report.show()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            dict(conf=str(args.conf), verdict=report.worst(), rows=report.rows),
            indent=2, ensure_ascii=False) + "\n")
        print(f"\n  {args.json} 에 저장했습니다.")
    sys.exit({OK: 0, WARN: 0, FAIL: 1}[report.worst()])


if __name__ == "__main__":
    main()
