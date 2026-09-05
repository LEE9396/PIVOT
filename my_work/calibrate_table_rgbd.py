#!/usr/bin/env python3
"""D456 실측 depth와 손-눈 변환으로 RB5 기준 테이블 평면을 구한다."""

import argparse
import json
from pathlib import Path
import tkinter as tk

import cv2
import numpy as np
import pyrealsense2 as rs


def fit_plane(points, threshold_m=0.006, trials=400, seed=0):
    """RANSAC 뒤 SVD로 평면을 다시 맞춘다."""
    points = np.asarray(points, dtype=float)
    if len(points) < 500:
        raise ValueError("유효한 테이블 depth 점이 500개보다 적습니다.")
    rng = np.random.default_rng(seed)
    sample = points[rng.choice(len(points), min(20000, len(points)), replace=False)]
    best = None
    for _ in range(trials):
        tri = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        offset = -normal @ tri[0]
        mask = np.abs(sample @ normal + offset) < threshold_m
        if best is None or mask.sum() > best.sum():
            best = mask
    if best is None or best.sum() < 500:
        raise RuntimeError("선택 영역에서 테이블 평면을 찾지 못했습니다.")
    cloud = sample[best]
    center = cloud.mean(axis=0)
    _, _, vt = np.linalg.svd(cloud - center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    offset = float(-normal @ center)
    residual = np.abs(points @ normal + offset)
    inliers = residual < threshold_m
    rms_mm = float(np.sqrt(np.mean(residual[inliers] ** 2)) * 1000)
    return normal, offset, inliers, rms_mm


def select_table(image):
    points = []
    root = tk.Tk()
    root.title("Table calibration - left click, Enter=fit, Backspace=undo, R=reset")
    ppm = cv2.imencode(".ppm", image)[1].tobytes()
    photo = tk.PhotoImage(data=ppm)
    canvas = tk.Canvas(root, width=image.shape[1], height=image.shape[0],
                       highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, image=photo, anchor="nw")
    cancelled = [False]

    def redraw():
        canvas.delete("selection")
        if len(points) >= 2:
            coords = [value for point in points for value in point]
            canvas.create_line(*coords, fill="#00ffff", width=3,
                               tags="selection")
        if len(points) >= 3:
            canvas.create_line(*points[-1], *points[0], fill="#00ffff",
                               width=3, tags="selection")
        for x, y in points:
            canvas.create_oval(x - 5, y - 5, x + 5, y + 5,
                               fill="#ff3030", outline="", tags="selection")
        canvas.create_text(18, 18, anchor="nw", fill="white",
                           font=("Sans", 16, "bold"),
                           text="LEFT: table boundary   ENTER: fit   "
                                "BACKSPACE: undo   R: reset",
                           tags="selection")

    def click(event):
        points.append((event.x, event.y))
        redraw()

    def accept(_event=None):
        if len(points) >= 3:
            root.quit()

    def cancel(_event=None):
        cancelled[0] = True
        root.quit()

    canvas.bind("<Button-1>", click)
    root.bind("<Return>", accept)
    root.bind("<BackSpace>", lambda _event: (points.pop(), redraw()) if points else None)
    root.bind("r", lambda _event: (points.clear(), redraw()))
    root.bind("R", lambda _event: (points.clear(), redraw()))
    root.bind("<Escape>", cancel)
    root.protocol("WM_DELETE_WINDOW", cancel)
    redraw()
    canvas.focus_set()
    root.mainloop()
    root.destroy()
    if cancelled[0]:
        raise KeyboardInterrupt("테이블 선택을 취소했습니다.")
    return np.asarray(points, np.int32)


def capture(serial, frames):
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depths = []
    try:
        for index in range(30 + frames):
            aligned = align.process(pipeline.wait_for_frames(10000))
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            if index >= 30:
                depths.append(np.asanyarray(depth_frame.get_data()).copy())
            image = np.asanyarray(color_frame.get_data()).copy()
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    finally:
        pipeline.stop()
    if len(depths) < frames:
        raise RuntimeError(f"D456 depth를 {len(depths)}/{frames}장만 받았습니다.")
    return image, np.median(np.stack(depths), axis=0) * scale, intrinsics, scale


def calibrate(args):
    calibration = json.loads(args.handeye.read_text())
    if calibration.get("status") != "valid":
        raise RuntimeError("유효한 EasyHeC 손-눈 보정이 아닙니다.")
    base_from_camera = np.asarray(calibration["base_from_camera"], dtype=float)
    if base_from_camera.shape != (4, 4):
        raise ValueError("base_from_camera는 4x4여야 합니다.")

    image, depth, intrinsics, scale = capture(args.serial, args.frames)
    polygon = select_table(image)
    mask = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    yy, xx = np.nonzero(mask & np.isfinite(depth) & (depth > 0.2) & (depth < 3.0))
    if len(xx) < 500:
        raise RuntimeError("선택 영역에 유효한 depth가 부족합니다.")
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(xx), min(30000, len(xx)), replace=False)
    xx, yy = xx[chosen], yy[chosen]
    camera_points = np.asarray([
        rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)],
                                        float(depth[y, x]))
        for x, y in zip(xx, yy)
    ])
    base_points = (camera_points @ base_from_camera[:3, :3].T
                   + base_from_camera[:3, 3])
    normal, offset, inliers, rms_mm = fit_plane(
        base_points, args.threshold_mm / 1000)
    height = float(-offset / normal[2])
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
    inlier_fraction = float(inliers.mean())
    passed = (inlier_fraction >= args.min_inlier_fraction
              and rms_mm <= args.max_rms_mm and tilt_deg <= args.max_tilt_deg)

    result = {
        "status": "valid" if passed else "invalid",
        "method": "D456 aligned metric depth + manual table mask + RANSAC/SVD",
        "camera": {"model": "Intel RealSense D456", "serial": args.serial,
                   "resolution": [1280, 720], "depth_scale_m": scale},
        "handeye_path": str(args.handeye.resolve()),
        "polygon_px": polygon.tolist(),
        "samples": int(len(base_points)),
        "plane_in_robot_base": {
            "equation": [*normal.tolist(), offset],
            "normal": normal.tolist(),
            "height_at_base_origin_m": height,
            "tilt_deg": tilt_deg,
        },
        "selected_xy_bounds_in_robot_base_m": {
            "min": base_points[inliers, :2].min(axis=0).tolist(),
            "max": base_points[inliers, :2].max(axis=0).tolist(),
        },
        "quality": {"inlier_fraction": inlier_fraction, "rms_mm": rms_mm,
                    "threshold_mm": args.threshold_mm},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    overlay = image.copy()
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 180
    selected = mask.astype(bool)
    overlay[selected] = (0.65 * overlay[selected] + 0.35 * tint[selected]).astype(np.uint8)
    cv2.polylines(overlay, [polygon], True, (0, 255, 255), 3)
    cv2.putText(overlay, f"{result['status'].upper()}  h(base)={height:.4f}m  "
                f"tilt={tilt_deg:.2f}deg  rms={rms_mm:.2f}mm",
                (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 5,
                cv2.LINE_AA)
    cv2.putText(overlay, f"{result['status'].upper()}  h(base)={height:.4f}m  "
                f"tilt={tilt_deg:.2f}deg  rms={rms_mm:.2f}mm",
                (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.imwrite(str(args.output.with_suffix(".png")), overlay)
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError("테이블 평면 품질 기준을 통과하지 못했습니다.")


def self_test():
    rng = np.random.default_rng(4)
    xy = rng.uniform(-1, 1, (4000, 2))
    z = 0.42 + 0.02 * xy[:, 0] - 0.01 * xy[:, 1] + rng.normal(0, 0.001, 4000)
    points = np.column_stack([xy, z])
    points[:300] = rng.uniform(-1, 1, (300, 3))
    normal, offset, inliers, rms = fit_plane(points, 0.004)
    assert inliers.mean() > 0.85 and rms < 2 and normal[2] > 0.99
    assert abs(-offset / normal[2] - 0.42) < 0.003
    print("self-test: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="333422300364")
    parser.add_argument("--handeye", type=Path,
                        default=Path("calibration/rb5_d456_easyhec_current.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("calibration/rb5_table_current.json"))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--threshold-mm", type=float, default=6.0)
    parser.add_argument("--min-inlier-fraction", type=float, default=0.75)
    parser.add_argument("--max-rms-mm", type=float, default=4.0)
    parser.add_argument("--max-tilt-deg", type=float, default=5.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        calibrate(args)


if __name__ == "__main__":
    main()
