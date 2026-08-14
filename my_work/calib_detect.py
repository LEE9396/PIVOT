"""보정판 검출기 — 카메라 영상에서 ChArUco 보드의 자세를 읽어 준다.

**이 파일만 Drake 환경 밖에서 돕니다.**

왜 떼어 놓나
------------
opencv 가 필요한 곳은 태그를 '보는' 일 하나뿐이다. 그것 때문에 고정된
Drake 1.54 환경에 opencv 를 밀어 넣으면 팀원 전원이 bootstrap 을 다시
돌려야 한다. FoundationPose 도 같은 이유로 별도 노드다 — 같은 구조를 쓴다.

    [Drake 환경]  calibrate_camera.py --run
                     팔을 자세로 보내고, FK 로 X_WE 를 만들고, AX=XB 를 푼다
                          |
                          |  {"cmd": "detect"}  /  {"ok":.., "X_CT":..}
                          v
    [시스템 python]  calib_detect.py --serve
                     카메라에서 한 장 찍어 ChArUco 자세를 낸다

실행 (두 터미널, 같은 PC)
-------------------------
    # 터미널 1 — 시스템 python (opencv 필요, run_drake_env.sh 를 쓰지 않는다)
    python3 calib_detect.py --serve --port 5566 \
        --squares-x 7 --squares-y 5 --square-mm 35 --marker-mm 26

    # 터미널 2 — Drake 환경
    ../robot_learning/scripts/run_drake_env.sh python calibrate_camera.py \
        --run --detector-port 5566

먼저 확인할 것
--------------
    python3 calib_detect.py --preview      # 카메라가 보이나, 보드가 잡히나
    python3 calib_detect.py --make-board board.png    # 인쇄용 보드 그림

보드를 만들 때 주의
-------------------
- **인쇄한 뒤 자로 사각형 한 변을 재서** --square-mm 에 넣는다. 프린터가
  배율을 건드리는 일이 흔하고, 그 배율 오차는 그대로 거리 오차가 된다.
- 판판한 것(알루미늄판, 아크릴)에 붙인다. 종이가 휘면 자세가 흔들린다.
- 그리퍼에 **볼트로** 고정한다. 캘리브레이션 도중 조금이라도 움직이면
  X_ET 가 상수라는 전제가 깨져 답이 통째로 틀어진다.
"""

import argparse
import json
import socket
import sys

import numpy as np

try:
    import cv2
except ImportError:                       # pragma: no cover - 환경 안내
    sys.exit("opencv 가 없습니다.  pip install opencv-contrib-python\n"
             "이 파일은 Drake 환경이 아니라 **시스템 python** 으로 돌립니다.")


DEFAULT_DICT = "DICT_5X5_100"


def build_board(args):
    """ChArUco 보드. opencv 4.7 에서 이름이 바뀌어 양쪽을 다 받는다."""
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, args.dictionary))
    square_m = args.square_mm * 1e-3
    marker_m = args.marker_mm * 1e-3
    size = (args.squares_x, args.squares_y)
    if hasattr(cv2.aruco.CharucoBoard, "create"):        # 4.6 이하
        board = cv2.aruco.CharucoBoard_create(
            size[0], size[1], square_m, marker_m, dictionary)
    else:                                                # 4.7 이상
        board = cv2.aruco.CharucoBoard(size, square_m, marker_m, dictionary)
    return board, dictionary


def open_camera(args):
    """RealSense 가 있으면 그것, 없으면 평범한 UVC 장치.

    RealSense 를 쓰면 공장 내부파라미터를 그대로 받을 수 있어 정확하다.
    없으면 --fx --fy --cx --cy 로 직접 줘야 한다.
    """
    try:
        import pyrealsense2 as rs2
    except ImportError:
        rs2 = None

    if rs2 is not None and not args.uvc:
        pipeline = rs2.pipeline()
        config = rs2.config()
        config.enable_stream(rs2.stream.color, args.width, args.height,
                             rs2.format.bgr8, 30)
        profile = pipeline.start(config)
        intr = (profile.get_stream(rs2.stream.color)
                .as_video_stream_profile().get_intrinsics())
        K = np.array([[intr.fx, 0.0, intr.ppx],
                      [0.0, intr.fy, intr.ppy],
                      [0.0, 0.0, 1.0]])
        dist = np.array(intr.coeffs, dtype=float)
        print(f"[검출] RealSense 색상 스트림  fx {intr.fx:.1f} fy {intr.fy:.1f}"
              f"  cx {intr.ppx:.1f} cy {intr.ppy:.1f}")

        def grab():
            frames = pipeline.wait_for_frames()
            return np.asanyarray(frames.get_color_frame().get_data())

        return grab, K, dist, pipeline.stop

    capture = cv2.VideoCapture(args.device)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        sys.exit(f"카메라 {args.device} 를 못 열었습니다")
    if args.fx is None:
        sys.exit("pyrealsense2 가 없으면 --fx --fy --cx --cy 를 직접 줘야 합니다\n"
                 "  (pip install pyrealsense2 하면 자동으로 읽습니다)")
    K = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy],
                  [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    print(f"[검출] UVC 장치 {args.device}  fx {args.fx} fy {args.fy}")

    def grab():
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("프레임을 못 받았습니다")
        return frame

    return grab, K, dist, capture.release


def detect(frame, board, dictionary, K, dist, n_min=6):
    """한 장에서 보드 자세를 낸다. (X_CT 4x4, 잡은 코너 수, 재투영 rms) 또는 None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):              # 4.7 이상
        corners, ids, _ = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) == 0:
        return None

    n, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board)
    if n is None or n < n_min:
        return None

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        ch_corners, ch_ids, board, K, dist, None, None)
    if not ok:
        return None

    # 재투영 오차 — 이 자세를 믿어도 되는지 알려주는 유일한 지표다.
    # 코너 좌표를 얻는 이름이 opencv 4.7 에서 속성 -> 메서드로 바뀌었다.
    all_corners = (board.getChessboardCorners()
                   if hasattr(board, "getChessboardCorners")
                   else board.chessboardCorners)
    object_points = np.array([all_corners[int(i)] for i in ch_ids.ravel()],
                             dtype=np.float32)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum(
        (projected.reshape(-1, 2) - ch_corners.reshape(-1, 2)) ** 2, axis=1))))

    X = np.eye(4)
    X[:3, :3] = cv2.Rodrigues(rvec)[0]
    X[:3, 3] = tvec.ravel()
    return X, int(n), rms


def average_detection(grab, board, dictionary, K, dist, n_frames, n_min):
    """여러 장을 잡아 **평행이동만** 평균낸다.

    회전은 단순 평균이 회전행렬이 아니게 되므로 재투영이 가장 좋은 한 장을
    고른다. 정지 상태에서 몇 장 찍는 것이므로 자세 차이는 원래 작다.
    """
    results = []
    for _ in range(n_frames):
        got = detect(grab(), board, dictionary, K, dist, n_min)
        if got is not None:
            results.append(got)
    if not results:
        return None
    best = min(results, key=lambda r: r[2])
    X = best[0].copy()
    X[:3, 3] = np.mean([r[0][:3, 3] for r in results], axis=0)
    return X, int(np.mean([r[1] for r in results])), best[2], len(results)


def serve(args):
    board, dictionary = build_board(args)
    grab, K, dist, close = open_camera(args)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[검출] 포트 {args.port} 에서 기다립니다. "
          f"calibrate_camera.py --run 을 실행하세요.")
    sock, addr = server.accept()
    print(f"[검출] {addr} 연결됨")
    reader = sock.makefile("r", encoding="utf-8")
    try:
        while True:
            line = reader.readline()
            if not line:
                break
            request = json.loads(line)
            if request.get("cmd") == "quit":
                break
            if request.get("cmd") == "intrinsics":
                reply = dict(ok=True, K=K.tolist(), dist=np.asarray(dist).tolist(),
                             resolution=[args.width, args.height])
            else:
                got = average_detection(grab, board, dictionary, K, dist,
                                        args.frames, args.min_corners)
                if got is None:
                    reply = dict(ok=False,
                                 reason="보드를 못 찾았습니다 (가림·흐림·화면 밖)")
                else:
                    X, n_corner, rms, n_ok = got
                    reply = dict(ok=True, X_CT=X.tolist(), corners=n_corner,
                                 rms_px=rms, frames=n_ok)
                    print(f"[검출] 코너 {n_corner}개  재투영 {rms:.3f} px")
            sock.sendall((json.dumps(reply) + "\n").encode("utf-8"))
    finally:
        reader.close(); sock.close(); server.close(); close()
        print("[검출] 종료")


def preview(args):
    """카메라와 보드가 제대로 잡히는지 눈으로 본다. q 로 종료."""
    board, dictionary = build_board(args)
    grab, K, dist, close = open_camera(args)
    print("[검출] 창에서 q 를 누르면 끝납니다.")
    try:
        while True:
            frame = grab()
            got = detect(frame, board, dictionary, K, dist, args.min_corners)
            if got is None:
                cv2.putText(frame, "board not found", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                X, n_corner, rms = got
                cv2.drawFrameAxes(frame, K, dist,
                                  cv2.Rodrigues(X[:3, :3])[0], X[:3, 3], 0.05)
                cv2.putText(frame,
                            f"corners {n_corner}  rms {rms:.2f}px  "
                            f"z {X[2, 3]:.3f}m", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
            cv2.imshow("calib preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        close(); cv2.destroyAllWindows()


def make_board(args):
    board, _ = build_board(args)
    px_per_m = args.dpi / 0.0254
    size = (int(args.squares_x * args.square_mm * 1e-3 * px_per_m),
            int(args.squares_y * args.square_mm * 1e-3 * px_per_m))
    image = (board.generateImage(size) if hasattr(board, "generateImage")
             else board.draw(size))
    cv2.imwrite(args.make_board, image)
    print(f"저장 -> {args.make_board}  ({args.dpi} dpi 로 인쇄하면 "
          f"사각형 한 변이 {args.square_mm} mm 입니다)")
    print("인쇄한 뒤 자로 한 변을 재서 --square-mm 을 실제 값으로 고치세요.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--make-board", default=None, metavar="PNG")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5566)
    ap.add_argument("--dictionary", default=DEFAULT_DICT)
    ap.add_argument("--squares-x", type=int, default=7)
    ap.add_argument("--squares-y", type=int, default=5)
    ap.add_argument("--square-mm", type=float, default=35.0,
                    help="인쇄한 뒤 자로 실제로 잰 값을 넣을 것")
    ap.add_argument("--marker-mm", type=float, default=26.0)
    ap.add_argument("--frames", type=int, default=10,
                    help="자세 하나당 몇 장을 평균낼지")
    ap.add_argument("--min-corners", type=int, default=6)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--uvc", action="store_true",
                    help="RealSense 가 있어도 평범한 UVC 로 연다")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    if args.make_board:
        make_board(args)
    elif args.preview:
        preview(args)
    elif args.serve:
        serve(args)
    else:
        ap.error("--serve / --preview / --make-board 중 하나가 필요합니다")


if __name__ == "__main__":
    main()
