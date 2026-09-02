"""추천한 파지점을 **실제 카메라 화면의 물체 위에** 겹쳐 그린다.

무엇을 잇는가
-------------
    grasp_target.py   파지점을 메시 좌표로 고른다        (PIVOT, Drake 환경)
            |         JSON 하나로 넘긴다
            v
    FoundationPose    그 메시의 자세를 카메라 좌표로 준다 (팀원, bundlesdf 환경)
            |
            v
    이 파일           둘을 곱해 화소로 투영하고 그린다

**왜 이 파일에는 Drake 도 pydrake 도 없나.** 카메라를 쥐고 있는 것은
FoundationPose 쪽 프로세스다 (RealSense 는 한 프로세스만 열 수 있다). 그러니
그리는 일은 그쪽 환경에서 일어나야 한다. 그쪽에는 pydrake 가 없고 이쪽에는
cv2 가 없다. 그래서 이 파일은 **numpy 만으로 계산**하고, 그리기(cv2)는
있을 때만 쓴다. 양쪽 환경 어디서 import 해도 깨지지 않는다.

좌표 규약
---------
    X_CM  카메라 <- 메시  (4x4). FoundationPose 가 주는 그 값이다.
    K     3x3 핀홀 내부행렬 (fx, fy, cx, cy). RealSense 컬러 내부행렬.
    점은 메시 좌표계 [m] 에서 온다 (grasp_target.py 가 그렇게 낸다).

자가 검사 (카메라도 로봇도 없이):
    ../robot_learning/scripts/run_drake_env.sh python grasp_overlay.py --check
"""

import argparse
import json
from pathlib import Path

import numpy as np

# 화면 색 (BGR). 파지점은 빨강 — Meshcat 파지 안내와 같은 뜻이 되도록 맞춘다.
COLOR_POINT = (60, 60, 235)
COLOR_JAW = (40, 200, 255)
COLOR_LONG = (200, 200, 200)
COLOR_TEXT = (255, 255, 255)


def load_target(path):
    """grasp_target.py 가 낸 JSON 을 읽어 넘파이로 바꾼다."""
    data = json.loads(Path(path).read_text())
    for key in ("point", "jaw_axis", "long_axis"):
        data[key] = np.asarray(data[key], dtype=float)
    return data


def intrinsic_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def intrinsics_from_json(path):
    """팀원 캘리브레이션 JSON(import_easyhec.py 출력)에서 K 를 꺼낸다."""
    data = json.loads(Path(path).read_text())
    values = data["intrinsics"] if "intrinsics" in data else data
    return intrinsic_matrix(values["fx"], values["fy"],
                            values["cx"], values["cy"])


def transform(X_CM, points):
    """메시 좌표의 점들을 카메라 좌표로 옮긴다. points 는 (N,3)."""
    X_CM = np.asarray(X_CM, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    return points @ X_CM[:3, :3].T + X_CM[:3, 3]


def project(K, points_camera, min_depth_m=1e-4):
    """카메라 좌표 점을 화소로. 돌려주는 것: (화소 (N,2), 보이는가 (N,))

    카메라 뒤(z<=0)에 있는 점은 투영하면 안 된다. 그냥 나누면 화면 반대편에
    엉뚱한 자리로 찍히는데, 그게 '물체 위에 파지점이 있다' 처럼 보이면
    사람이 거기에 손을 넣는다.
    """
    points_camera = np.atleast_2d(np.asarray(points_camera, dtype=float))
    depth = points_camera[:, 2]
    visible = depth > min_depth_m
    safe = np.where(visible, depth, 1.0)
    normalized = points_camera / safe[:, None]
    pixels = normalized @ np.asarray(K, dtype=float).T
    return pixels[:, :2], visible


def target_geometry(target, X_CM, pad_half_len_m=0.018):
    """파지 목표를 그릴 점들로 펼친다 (전부 메시 좌표 -> 카메라 좌표).

    돌려주는 것: dict
        point       파지점 하나
        pads        죠 두 장의 중심 (개구량만큼 벌어진 자리)
        pad_bars    패드마다 장축을 따라 그은 선분의 양 끝 (2, 2, 3)
        long_bar    장축 방향 안내선의 양 끝
    """
    point = np.asarray(target["point"], dtype=float)
    jaw = np.asarray(target["jaw_axis"], dtype=float)
    jaw = jaw / np.linalg.norm(jaw)
    long_axis = np.asarray(target["long_axis"], dtype=float)
    long_axis = long_axis - (long_axis @ jaw) * jaw
    long_axis = long_axis / np.linalg.norm(long_axis)
    half = 0.5 * float(target["opening_m"])

    pads = np.stack([point + half * jaw, point - half * jaw])
    pad_bars = np.stack([
        np.stack([pad - pad_half_len_m * long_axis,
                  pad + pad_half_len_m * long_axis]) for pad in pads])
    long_bar = np.stack([point - 2.0 * pad_half_len_m * long_axis,
                         point + 2.0 * pad_half_len_m * long_axis])

    return dict(
        point=transform(X_CM, point)[0],
        pads=transform(X_CM, pads),
        pad_bars=np.stack([transform(X_CM, bar) for bar in pad_bars]),
        long_bar=transform(X_CM, long_bar),
    )


def draw(image, target, X_CM, K, label=None, thickness=2):
    """image 위에 파지 안내를 그린다. 그린 그림을 돌려준다.

    파지점이 카메라 뒤이거나 화면 밖이면 **아무것도 안 그리고** 왜 안
    그렸는지 글자로 알린다. 조용히 안 그리면 사람은 추천이 없는 줄 안다.
    """
    import cv2                       # 그릴 때만 필요하다

    view = image.copy()
    geometry = target_geometry(target, X_CM)
    height, width = view.shape[:2]

    points = np.vstack([geometry["point"][None, :], geometry["pads"],
                        geometry["long_bar"],
                        geometry["pad_bars"].reshape(-1, 3)])
    pixels, visible = project(K, points)
    if not visible[0]:
        cv2.putText(view, "grasp point is behind the camera", (24, height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_POINT, 2)
        return view

    def xy(index):
        return tuple(int(round(v)) for v in pixels[index])

    # 장축 안내선 -> 패드 막대 -> 죠 사이 선 -> 파지점 순으로 겹쳐 그린다.
    if visible[3] and visible[4]:
        cv2.line(view, xy(3), xy(4), COLOR_LONG, 1, cv2.LINE_AA)
    for k in range(2):
        a, b = 5 + 2 * k, 6 + 2 * k
        if visible[a] and visible[b]:
            cv2.line(view, xy(a), xy(b), COLOR_JAW, thickness + 1, cv2.LINE_AA)
    if visible[1] and visible[2]:
        cv2.line(view, xy(1), xy(2), COLOR_JAW, 1, cv2.LINE_AA)
    cv2.circle(view, xy(0), 7, COLOR_POINT, -1, cv2.LINE_AA)
    cv2.circle(view, xy(0), 12, COLOR_POINT, thickness, cv2.LINE_AA)

    text = label or (f"grasp {1000*target['width_m']:.0f} mm"
                     f" / open {1000*target['opening_m']:.0f} mm")
    origin = (min(max(xy(0)[0] + 18, 8), width - 260), max(xy(0)[1] - 14, 22))
    cv2.putText(view, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(view, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                COLOR_TEXT, 1, cv2.LINE_AA)
    if not (0 <= xy(0)[0] < width and 0 <= xy(0)[1] < height):
        cv2.putText(view, "grasp point is outside the image",
                    (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COLOR_POINT, 2)
    return view


# ---------------------------------------------------------------------------
def self_check(verbose=True):
    failures = []

    def case(name, fn):
        try:
            fn()
            if verbose:
                print(f"  [통과] {name}")
        except Exception as exc:                       # noqa: BLE001
            failures.append((name, exc))
            print(f"  [실패] {name}: {exc}")

    K = intrinsic_matrix(600.0, 600.0, 320.0, 240.0)

    def t_center():
        # 광축 위 1 m 앞의 점은 주점에 찍힌다.
        pixels, visible = project(K, [[0.0, 0.0, 1.0]])
        assert visible[0]
        assert np.allclose(pixels[0], [320.0, 240.0]), pixels

    def t_scale():
        # x 로 10 cm, 깊이 1 m -> 0.1*600 = 60 화소 옆.
        pixels, _ = project(K, [[0.1, 0.0, 1.0]])
        assert np.allclose(pixels[0], [380.0, 240.0]), pixels
        # 두 배 멀면 절반만 벌어진다.
        pixels, _ = project(K, [[0.1, 0.0, 2.0]])
        assert np.allclose(pixels[0], [350.0, 240.0]), pixels

    def t_behind():
        _, visible = project(K, [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
        assert not visible.any(), "카메라 뒤/원점의 점이 보인다고 나왔다"

    def t_transform():
        X = np.eye(4)
        X[:3, 3] = [0.0, 0.0, 0.5]
        assert np.allclose(transform(X, [[0.0, 0.0, 0.5]]), [[0.0, 0.0, 1.0]])
        # 회전도 제대로 걸리는지: z 축 90도
        X = np.eye(4)
        X[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        assert np.allclose(transform(X, [[1.0, 0.0, 0.0]]), [[0.0, 1.0, 0.0]])

    def t_geometry():
        target = dict(point=[0.0, 0.0, 1.0], jaw_axis=[1.0, 0.0, 0.0],
                      long_axis=[0.0, 1.0, 0.0], width_m=0.010,
                      opening_m=0.040)
        geometry = target_geometry(target, np.eye(4))
        # 패드 두 장은 개구량만큼 떨어져 있어야 한다.
        gap = np.linalg.norm(geometry["pads"][0] - geometry["pads"][1])
        assert abs(gap - 0.040) < 1e-9, gap
        # 파지점은 두 패드의 한가운데다.
        assert np.allclose(geometry["pads"].mean(axis=0), geometry["point"])

    def t_orthogonalize():
        # 장축이 죠 축과 직교하지 않아도 결과는 직교해야 한다.
        target = dict(point=[0.0, 0.0, 1.0], jaw_axis=[1.0, 0.0, 0.0],
                      long_axis=[0.7, 0.7, 0.0], width_m=0.01, opening_m=0.04)
        geometry = target_geometry(target, np.eye(4))
        jaw = geometry["pads"][0] - geometry["pads"][1]
        along = geometry["long_bar"][1] - geometry["long_bar"][0]
        assert abs(float(jaw @ along)) < 1e-12, jaw @ along

    def t_round_trip():
        # 메시 좌표의 파지점을 카메라 60 cm 앞에 두면 주점 근처에 찍힌다.
        target = dict(point=[0.2, 0.45, 0.25], jaw_axis=[1.0, 0.0, 0.0],
                      long_axis=[0.0, 0.0, 1.0], width_m=0.01, opening_m=0.04)
        X = np.eye(4)
        X[:3, 3] = -np.asarray(target["point"]) + np.array([0.0, 0.0, 0.6])
        geometry = target_geometry(target, X)
        pixels, visible = project(K, geometry["point"][None, :])
        assert visible[0]
        assert np.allclose(pixels[0], [320.0, 240.0], atol=1e-6), pixels

    case("광축 위의 점은 주점에", t_center)
    case("깊이에 따라 화소가 줄어든다", t_scale)
    case("카메라 뒤의 점은 안 보인다", t_behind)
    case("자세 변환", t_transform)
    case("패드 간격 = 개구량", t_geometry)
    case("장축을 죠 축에 직교화", t_orthogonalize)
    case("메시 좌표 -> 화소 왕복", t_round_trip)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="카메라 없이 투영 계산을 검증한다")
    parser.add_argument("--target", type=Path, default=None)
    args = parser.parse_args()
    if args.target and not args.check:
        target = load_target(args.target)
        print(f"{target.get('part', '?')}  단면"
              f" {1000*target['width_m']:.1f} mm  개구"
              f" {1000*target['opening_m']:.1f} mm")
        return 0
    print("grasp_overlay 자가 진단")
    failures = self_check()
    print(f"\n{'모두 통과' if not failures else f'{len(failures)} 개 실패'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
