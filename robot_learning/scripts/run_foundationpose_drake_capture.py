#!/usr/bin/env python3
"""Run the installed FoundationPose model on a Drake RGB-D capture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import trimesh


FOUNDATIONPOSE = Path(
    "/home/cheon/Desktop/Lab/pipline/repo_chain/FoundationPose"
)
PHANTOM_V3_ASSET_DIR = (
    Path(__file__).resolve().parents[1] / "assets/phantom_v3"
)
PARENT_HINGE_XYZ = np.array((0.083, 0.0, 0.010))
CHILD_HINGE_XYZ = np.array((0.106, 0.0, 0.0))
STATIC_ANGLE_SPAN_DEG = 5.0
MINIMUM_EDGE_MARGIN_PIXELS = 10
MAXIMUM_MASK_DEPTH_CENTROID_RESIDUAL_M = 0.08
sys.path[:0] = [
    str(FOUNDATIONPOSE),
    str(FOUNDATIONPOSE / "mycpp/build"),
]

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr


def _translation(xyz):
    matrix = np.eye(4)
    matrix[:3, 3] = xyz
    return matrix


def _colored_box(extents, xyz, rgb):
    mesh = trimesh.creation.box(
        extents=extents,
        transform=_translation(xyz),
    )
    mesh.visual.vertex_colors = np.tile((*rgb, 255), (len(mesh.vertices), 1))
    return mesh


def _desk_lamp_meshes():
    arm = trimesh.util.concatenate(
        (
            _colored_box(
                (0.40, 0.035, 0.035),
                (0.20, 0.0, 0.0),
                (13, 158, 148),
            ),
            _colored_box(
                (0.070, 0.008, 0.004),
                (0.12, 0.012, 0.0195),
                (242, 51, 31),
            ),
            _colored_box(
                (0.025, 0.012, 0.004),
                (0.29, -0.010, 0.0195),
                (242, 209, 31),
            ),
        )
    )
    head_center = np.array((-0.283 / 2.0, 0.0, 0.0))
    head = trimesh.util.concatenate(
        (
            _colored_box(
                (0.283, 0.055, 0.025),
                head_center,
                (242, 166, 20),
            ),
            _colored_box(
                (0.25, 0.035, 0.004),
                head_center + np.array((0.0, 0.0, -0.0155)),
                (255, 235, 140),
            ),
            _colored_box(
                (0.055, 0.010, 0.004),
                (-0.075, 0.020, 0.0145),
                (204, 31, 184),
            ),
        )
    )
    base = trimesh.util.concatenate(
        (
            _colored_box(
                (0.30, 0.20, 0.035),
                (0.0, 0.0, 0.0),
                (20, 71, 173),
            ),
            _colored_box(
                (0.035, 0.05, 0.07),
                (-0.1325, 0.0, 0.0525),
                (20, 71, 173),
            ),
            _colored_box(
                (0.085, 0.018, 0.004),
                (0.075, 0.055, 0.0195),
                (31, 230, 209),
            ),
        )
    )
    return {"parent": base, "child_1": arm, "child_2": head}


def _jewelry_box_meshes():
    return {
        "parent": trimesh.util.concatenate(
            (
                _colored_box((0.28, 0.20, 0.045), (0.0, 0.0, 0.0), (46, 82, 140)),
                _colored_box((0.245, 0.165, 0.006), (0.0, 0.0, 0.0265), (199, 204, 219)),
                _colored_box((0.050, 0.018, 0.003), (0.070, -0.055, 0.0315), (242, 184, 31)),
            )
        ),
        "child_1": trimesh.util.concatenate(
            (
                _colored_box(
                    (0.28, 0.20, 0.015),
                    (0.0, -0.10, 0.0105),
                    (158, 71, 148),
                ),
                _colored_box(
                    (0.045, 0.014, 0.003),
                    (-0.075, -0.055, 0.0195),
                    (26, 184, 209),
                ),
            )
        ),
    }


def _meshes(part_count, object_profile=None):
    if object_profile == "desk_lamp":
        return _desk_lamp_meshes()
    if object_profile == "jewelry_box":
        return _jewelry_box_meshes()
    if object_profile == "phantom_v3":
        return {
            link: trimesh.load_mesh(
                PHANTOM_V3_ASSET_DIR / filename,
                process=False,
            ).apply_scale(0.001)
            for link, filename in zip(
                ("parent", "child_1", "child_2"),
                (
                    "v3_part0_root.obj",
                    "v3_part1_elbow.obj",
                    "v3_part2_tip.obj",
                ),
                strict=True,
            )
        }
    parent_main = _colored_box((0.120, 0.040, 0.020), (0.020, 0.0, 0.0), (20, 61, 158))
    parent_neck = _colored_box((0.035, 0.035, 0.020), (-0.0575, 0.0, 0.0), (20, 61, 158))
    parent_flange = _colored_box((0.005, 0.040, 0.020), (-0.0775, 0.0, 0.0), (20, 61, 158))
    parent_markers = (
        _colored_box((0.025, 0.008, 0.0002), (0.030, 0.012, 0.0101), (235, 51, 31)),
        _colored_box((0.008, 0.012, 0.0002), (0.020, 0.008, 0.0101), (235, 51, 31)),
    )
    child_center = np.array((0.053, 0.0, -0.010))
    child_colors = ((20, 148, 117), (219, 112, 20), (140, 56, 191))
    marker_colors = ((235, 204, 26), (20, 190, 220), (235, 65, 180))
    meshes = {
        "parent": trimesh.util.concatenate(
            (parent_main, parent_neck, parent_flange, *parent_markers)
        )
    }
    for index in range(1, part_count):
        marker_side = -1.0 if index % 2 else 1.0
        meshes[f"child_{index}"] = trimesh.util.concatenate(
            (
                _colored_box(
                    (0.100, 0.040, 0.020),
                    child_center,
                    child_colors[index - 1],
                ),
                _colored_box(
                    (0.025, 0.008, 0.0002),
                    child_center
                    + (0.015 * marker_side, 0.012 * marker_side, 0.0101),
                    marker_colors[index - 1],
                ),
                _colored_box(
                    (0.008, 0.012, 0.0002),
                    child_center
                    + (0.005 * marker_side, 0.008 * marker_side, 0.0101),
                    marker_colors[index - 1],
                ),
            )
        )
    meshes["child"] = meshes["child_1"]
    return meshes


def _estimator(mesh, scorer, refiner, glctx, debug_dir):
    return FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=0,
        debug_dir=str(debug_dir),
    )


def _set_tracking_prior(estimator, pose):
    estimator.pose_last = torch.as_tensor(
        pose
        @ np.linalg.inv(
            estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
        ),
        device="cuda",
        dtype=torch.float,
    )


def _masked_depth_centroid(depth, mask, K):
    rows, columns = np.nonzero(mask & (depth > 0.0))
    if len(rows) < 50:
        return None
    z = depth[rows, columns]
    points = np.column_stack(
        (
            (columns - K[0, 2]) * z / K[0, 0],
            (rows - K[1, 2]) * z / K[1, 1],
            z,
        )
    )
    return np.median(points, axis=0)


def _register_nearest(
    estimator,
    *,
    K,
    rgb,
    depth,
    mask,
    predicted_pose,
    iterations,
):
    pose = estimator.register(
        K=K,
        rgb=rgb,
        depth=depth,
        ob_mask=mask,
        iteration=iterations,
    )
    if predicted_pose is None:
        return pose
    centered = estimator.poses.detach().cpu().numpy()
    to_mesh = estimator.get_tf_to_centered_mesh().detach().cpu().numpy()
    candidates = centered @ to_mesh
    costs = [
        _rotation_distance(predicted_pose, candidate)
        + 20.0
        * np.linalg.norm(
            predicted_pose[:3, 3] - candidate[:3, 3]
        )
        for candidate in candidates
    ]
    best = int(np.argmin(costs))
    estimator.pose_last = torch.as_tensor(
        centered[best],
        device="cuda",
        dtype=torch.float,
    )
    return candidates[best]


def _rotation_distance(a, b):
    relative = a[:3, :3].T @ b[:3, :3]
    return math.acos(
        np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    )


def _joint_constrained_child_pose(
    parent,
    opening_deg,
    joint_index,
    object_profile=None,
):
    if object_profile == "jewelry_box":
        angle = math.radians(-opening_deg)
        c, s = math.cos(angle), math.sin(angle)
        pose = np.eye(4)
        pose[:3, :3] = parent[:3, :3] @ np.array(
            ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
        )
        pose[:3, 3] = (
            parent[:3, :3] @ np.array((0.0, 0.10, 0.0225))
            + parent[:3, 3]
        )
        return pose
    if object_profile == "phantom_v3":
        axes = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
        origins = ((0.150, 0.0, 0.0), (0.110, 0.0, 0.0))
        axis = np.asarray(axes[joint_index - 1])
        skew = np.array(
            (
                (0.0, -axis[2], axis[1]),
                (axis[2], 0.0, -axis[0]),
                (-axis[1], axis[0], 0.0),
            )
        )
        angle = math.radians(opening_deg)
        rotation = (
            np.eye(3)
            + math.sin(angle) * skew
            + (1.0 - math.cos(angle)) * (skew @ skew)
        )
        pose = np.eye(4)
        pose[:3, :3] = parent[:3, :3] @ rotation
        pose[:3, 3] = (
            parent[:3, :3] @ np.asarray(origins[joint_index - 1])
            + parent[:3, 3]
        )
        return pose
    pose = np.eye(4)
    bend = math.radians(
        -opening_deg
        if object_profile == "desk_lamp"
        else opening_deg - 180.0
    )
    c, s = math.cos(bend), math.sin(bend)
    pose[:3, :3] = parent[:3, :3] @ np.array(
        ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    )
    if object_profile == "desk_lamp":
        offset = (
            np.array((-0.1325, 0.0, 0.108))
            if joint_index == 1
            else np.array((0.40, 0.0, -0.033))
        )
    else:
        offset = PARENT_HINGE_XYZ if joint_index == 1 else CHILD_HINGE_XYZ
    pose[:3, 3] = parent[:3, :3] @ offset + parent[:3, 3]
    return pose


def _fuse(poses):
    distances = np.array(
        [
            [
                _rotation_distance(a, b)
                + 20.0 * np.linalg.norm(a[:3, 3] - b[:3, 3])
                for b in poses
            ]
            for a in poses
        ]
    )
    medoid = poses[int(np.argmin(distances.sum(axis=1)))]
    inliers = [
        pose
        for pose in poses
        if _rotation_distance(medoid, pose) <= math.radians(15.0)
        and np.linalg.norm(medoid[:3, 3] - pose[:3, 3]) <= 0.05
    ]
    mean = np.mean([pose[:3, :3] for pose in inliers], axis=0)
    u, _, vt = np.linalg.svd(mean)
    fused = np.eye(4)
    fused[:3, :3] = u @ np.diag((1.0, 1.0, np.linalg.det(u @ vt))) @ vt
    fused[:3, 3] = np.median([pose[:3, 3] for pose in inliers], axis=0)
    return fused, len(inliers)


def _opening_angle(parent, child, object_profile=None, joint_index=1):
    relative = parent[:3, :3].T @ child[:3, :3]
    if object_profile == "jewelry_box":
        return -math.degrees(
            math.atan2(relative[2, 1], relative[1, 1])
        )
    if object_profile == "phantom_v3":
        return math.degrees(
            math.atan2(
                relative[1, 0],
                relative[0, 0],
            )
            if joint_index == 1
            else math.atan2(
                relative[0, 2],
                relative[0, 0],
            )
        )
    raw = math.degrees(
        math.atan2(relative[0, 2], relative[0, 0])
    )
    if object_profile == "desk_lamp":
        return (-raw) % 360.0
    return (
        180.0 + raw
    ) % 360.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--track-iterations", type=int, default=1)
    parser.add_argument("--initialize-from-known-pose", action="store_true")
    parser.add_argument(
        "--time-window",
        type=float,
        nargs=2,
        action="append",
        default=[],
    )
    args = parser.parse_args()
    output = args.output or args.capture / "foundationpose_result.json"
    metadata = json.loads((args.capture / "metadata.json").read_text())
    if args.time_window:
        metadata["frames"] = [
            frame
            for frame in metadata["frames"]
            if any(
                start <= frame["time_s"] <= end
                for start, end in args.time_window
            )
        ]
    if not metadata["frames"]:
        raise ValueError("no RGB-D frames remain in the selected time window")
    first_frame_index = metadata["frames"][0]["index"]
    links = tuple(metadata["body_labels"])
    object_profile = metadata.get("object_profile")
    meshes = _meshes(len(links), metadata.get("object_profile"))
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimators = {
        (camera, link): _estimator(
            meshes[link],
            scorer,
            refiner,
            glctx,
            args.capture / "debug" / camera / link,
        )
        for camera in metadata["cameras"]
        for link in links
    }
    rows = []
    previous_world = {}
    previous_openings = []
    bad_frame_counts = {link: 0 for link in links}
    recovery_agreement_counts = {link: 0 for link in links}
    recovering_links = set()
    failed_streams = set()
    recovery_events = []
    for frame in metadata["frames"]:
        world_poses = {link: [] for link in links}
        world_pose_by_stream = {}
        camera_rows = {}
        frame_started = time.perf_counter()
        static_frame = any(
            hold_end_s - 1.0
            <= frame["time_s"]
            <= hold_end_s + 1e-6
            for hold_end_s in metadata.get("hold_end_times_s", [])
        )
        for camera, camera_data in metadata["cameras"].items():
            stem = f"{frame['index']:04d}"
            directory = args.capture / camera
            rgb = cv2.cvtColor(
                cv2.imread(str(directory / "rgb" / f"{stem}.png")),
                cv2.COLOR_BGR2RGB,
            )
            depth = np.load(
                directory / "depth_m" / f"{stem}.npy"
            ).astype(np.float32)
            depth[~np.isfinite(depth)] = 0.0
            K = np.loadtxt(directory / "K.txt").astype(np.float32)
            x_world_camera = np.asarray(camera_data["X_world_camera"])
            x_camera_world = np.linalg.inv(x_world_camera)
            camera_rows[camera] = {}
            camera_world_poses = {}
            for link_index, link in enumerate(links):
                estimator = estimators[(camera, link)]
                started = time.perf_counter()
                mask = cv2.imread(
                    str(
                        directory
                        / f"{link}_mask"
                        / f"{stem}.png"
                    ),
                    cv2.IMREAD_GRAYSCALE,
                ) > 0
                gt = np.asarray(
                    frame["cameras"][camera][
                        "gt_object_in_camera"
                    ][link]
                )
                re_registered = False
                kinematic_residual_m = 0.0
                constraint_projected = False
                if (
                    frame["index"] == first_frame_index
                    and args.initialize_from_known_pose
                ):
                    _set_tracking_prior(estimator, gt)
                    pose = estimator.track_one(
                        rgb=rgb,
                        depth=depth,
                        K=K,
                        iteration=args.track_iterations,
                    )
                elif frame["index"] == first_frame_index:
                    predicted_world = (
                        _joint_constrained_child_pose(
                            camera_world_poses[links[link_index - 1]],
                            metadata["initial_opening_angles_deg"][
                                link_index - 1
                            ],
                            link_index,
                            object_profile,
                        )
                        if link_index > 0
                        else None
                    )
                    predicted_pose = (
                        x_camera_world @ predicted_world
                        if predicted_world is not None
                        else None
                    )
                    if np.count_nonzero(mask) >= 50:
                        pose = _register_nearest(
                            estimator,
                            K=K,
                            rgb=rgb,
                            depth=depth,
                            mask=mask,
                            predicted_pose=predicted_pose,
                            iterations=args.iterations,
                        )
                    elif world_poses[link]:
                        if predicted_pose is None:
                            predicted_world, _ = _fuse(world_poses[link])
                            predicted_pose = x_camera_world @ predicted_world
                        _set_tracking_prior(estimator, predicted_pose)
                        pose = estimator.track_one(
                            rgb=rgb,
                            depth=depth,
                            K=K,
                            iteration=args.track_iterations,
                        )
                    else:
                        raise RuntimeError(
                            f"{camera}/{link} is invisible in the first "
                            "frame and no other camera supplied a prior"
                        )
                else:
                    predicted_world = (
                        _joint_constrained_child_pose(
                            camera_world_poses[links[link_index - 1]],
                            previous_openings[link_index - 1],
                            link_index,
                            object_profile,
                        )
                        if link_index > 0
                        else previous_world[link]
                    )
                    predicted_pose = x_camera_world @ predicted_world
                    _set_tracking_prior(estimator, predicted_pose)
                    if (
                        static_frame
                        and (camera, link) in failed_streams
                        and np.count_nonzero(mask) >= 50
                    ):
                        pose = _register_nearest(
                            estimator,
                            K=K,
                            rgb=rgb,
                            depth=depth,
                            mask=mask,
                            predicted_pose=predicted_pose,
                            iterations=args.iterations,
                        )
                        re_registered = True
                        failed_streams.discard((camera, link))
                    else:
                        pose = estimator.track_one(
                            rgb=rgb,
                            depth=depth,
                            K=K,
                            iteration=args.track_iterations,
                        )
                    kinematic_residual_m = (
                        np.linalg.norm(
                            (x_world_camera @ pose)[:3, 3]
                            - predicted_world[:3, 3]
                        )
                        if link_index > 0
                        else 0.0
                    )
                    if link_index > 0 and kinematic_residual_m > 0.01:
                        pose[:3, 3] = predicted_pose[:3, 3]
                        _set_tracking_prior(estimator, pose)
                        constraint_projected = True
                world_pose = x_world_camera @ pose
                camera_world_poses[link] = world_pose
                world_pose_by_stream[(camera, link)] = world_pose
                observed_centroid = _masked_depth_centroid(depth, mask, K)
                centroid_residual_m = (
                    float(
                        np.linalg.norm(
                            (
                                pose
                                @ np.append(meshes[link].centroid, 1.0)
                            )[:3]
                            - observed_centroid
                        )
                    )
                    if observed_centroid is not None
                    else math.inf
                )
                edge_margin = frame["cameras"][camera].get(
                    "edge_margin_pixels",
                    {},
                ).get(link, MINIMUM_EDGE_MARGIN_PIXELS)
                local_bad = bool(
                    np.count_nonzero(mask) < 50
                    or edge_margin < MINIMUM_EDGE_MARGIN_PIXELS
                    or centroid_residual_m
                    > MAXIMUM_MASK_DEPTH_CENTROID_RESIDUAL_M
                    or (
                        static_frame
                        and (
                            _rotation_distance(predicted_pose, pose)
                            > math.radians(20.0)
                            or np.linalg.norm(
                                predicted_pose[:3, 3] - pose[:3, 3]
                            )
                            > 0.05
                            or kinematic_residual_m > 0.01
                        )
                    )
                    if frame["index"] != first_frame_index
                    else np.count_nonzero(mask) < 50
                    or edge_margin < MINIMUM_EDGE_MARGIN_PIXELS
                )
                camera_rows[camera][link] = {
                    "rotation_error_deg": math.degrees(
                        _rotation_distance(gt, pose)
                    ),
                    "translation_error_m": float(
                        np.linalg.norm(gt[:3, 3] - pose[:3, 3])
                    ),
                    "inference_ms": (
                        time.perf_counter() - started
                    )
                    * 1000.0,
                    "re_registered": (
                        re_registered
                        if frame["index"] != first_frame_index
                        else False
                    ),
                    "joint_origin_residual_m": (
                        kinematic_residual_m
                        if frame["index"] != first_frame_index
                        and link_index > 0
                        else 0.0
                    ),
                    "joint_constraint_projected": (
                        constraint_projected
                        if frame["index"] != first_frame_index
                        else False
                    ),
                    "tracking_bad": local_bad,
                    "mask_depth_centroid_residual_m": centroid_residual_m,
                }
                world_poses[link].append(world_pose)
        fused = {}
        inliers = {}
        for link in links:
            fused[link], inliers[link] = _fuse(world_poses[link])
        safe_hold_requested = False
        for link in links:
            bad_cameras = {
                camera
                for camera in metadata["cameras"]
                if camera_rows[camera][link]["tracking_bad"]
                or _rotation_distance(
                    fused[link],
                    world_pose_by_stream[(camera, link)],
                )
                > math.radians(15.0)
                or np.linalg.norm(
                    fused[link][:3, 3]
                    - world_pose_by_stream[(camera, link)][:3, 3]
                )
                > 0.05
            }
            if link in recovering_links:
                recovery_agreement_counts[link] = (
                    recovery_agreement_counts[link] + 1
                    if len(bad_cameras) < 2
                    else 0
                )
                if recovery_agreement_counts[link] >= 3:
                    recovering_links.remove(link)
                    recovery_agreement_counts[link] = 0
                    recovery_events.append(
                        {
                            "frame_index": frame["index"],
                            "link": link,
                            "event": "tracking_resumed",
                        }
                    )
            else:
                bad_frame_counts[link] = (
                    bad_frame_counts[link] + 1
                    if len(bad_cameras) >= 2
                    else 0
                )
                if bad_frame_counts[link] >= 5:
                    recovering_links.add(link)
                    failed_streams.update(
                        (camera, link) for camera in bad_cameras
                    )
                    bad_frame_counts[link] = 0
                    safe_hold_requested = True
                    recovery_events.append(
                        {
                            "frame_index": frame["index"],
                            "link": link,
                            "event": "safe_hold_requested",
                            "failed_cameras": sorted(bad_cameras),
                        }
                    )
        previous_world = fused
        openings = [
            _opening_angle(
                fused[links[index - 1]],
                fused[links[index]],
                object_profile,
                index,
            )
            for index in range(1, len(links))
        ]
        if object_profile == "jewelry_box":
            references = (
                previous_openings
                or metadata["initial_opening_angles_deg"]
            )
            openings = [
                min(
                    (value, 180.0 - value),
                    key=lambda candidate: abs(candidate - reference),
                )
                for value, reference in zip(
                    openings,
                    references,
                    strict=True,
                )
            ]
        if metadata.get("opening_limit_deg", 360.0) <= 180.0:
            openings = [min(value, 360.0 - value) for value in openings]
        joint_limits = metadata.get(
            "joint_opening_limits_deg",
            [metadata.get("opening_limit_deg", 360.0)] * len(openings),
        )
        joint_bounds = metadata.get(
            "joint_angle_bounds_deg",
            [[0.0, limit] for limit in joint_limits],
        )
        openings = [
            float(np.clip(value, lower, upper))
            for value, (lower, upper) in zip(
                openings,
                joint_bounds,
                strict=True,
            )
        ]
        previous_openings = openings
        actual_openings = frame.get(
            "actual_opening_angles_deg",
            [frame["actual_opening_deg"]],
        )
        errors = [
            (opening - actual + 180.0) % 360.0 - 180.0
            for opening, actual in zip(
                openings,
                actual_openings,
                strict=True,
            )
        ]
        rows.append(
            {
                "index": frame["index"],
                "time_s": frame["time_s"],
                "actual_opening_deg": frame["actual_opening_deg"],
                "actual_opening_angles_deg": actual_openings,
                "foundationpose_opening_deg": openings[0],
                "foundationpose_opening_angles_deg": openings,
                "angle_error_deg": errors[0],
                "angle_errors_deg": errors,
                "camera_inliers": inliers,
                "safe_hold_requested": safe_hold_requested,
                "recovering_links": sorted(recovering_links),
                "fused_world_poses": {
                    link: fused[link].tolist() for link in links
                },
                "frame_inference_ms": (
                    time.perf_counter() - frame_started
                )
                * 1000.0,
                "per_camera": camera_rows,
                "visible_pixels": {
                    camera: frame["cameras"][camera]["visible_pixels"]
                    for camera in metadata["cameras"]
                },
                "edge_margin_pixels": {
                    camera: frame["cameras"][camera].get(
                        "edge_margin_pixels",
                        {
                            link: MINIMUM_EDGE_MARGIN_PIXELS
                            for link in links
                        },
                    )
                    for camera in metadata["cameras"]
                },
            }
        )
        print(
            f"{frame['index'] + 1}/{len(metadata['frames'])} "
            f"GT={frame['actual_opening_deg']:.2f} "
            f"FP={np.round(openings, 2).tolist()} "
            f"error={np.round(errors, 2).tolist()}"
        )
    errors = np.abs(
        np.asarray([row["angle_errors_deg"] for row in rows])
    )
    track_ms = np.asarray(
        [row["frame_inference_ms"] for row in rows[1:]]
    )
    static_angles = np.asarray(
        [row["foundationpose_opening_angles_deg"] for row in rows[-5:]]
    )
    static_holds = []
    minimum_visible_pixels = 50
    for hold_end_s in metadata.get("hold_end_times_s", []):
        hold_rows = [
            row
            for row in rows
            if hold_end_s - 1.0 <= row["time_s"] <= hold_end_s + 1e-6
        ]
        if not hold_rows:
            continue
        trusted_rows = [
            row
            for row in hold_rows
            if all(
                count >= 2 for count in row["camera_inliers"].values()
            )
        ] or hold_rows
        angles = np.asarray(
            [
                row["foundationpose_opening_angles_deg"]
                for row in trusted_rows
            ]
        )
        visibility = {
            camera: {
                link: min(
                    row["visible_pixels"][camera][link]
                    for row in hold_rows
                )
                for link in links
            }
            for camera in metadata["cameras"]
        }
        edge_margins = {
            camera: {
                link: min(
                    row["edge_margin_pixels"][camera][link]
                    for row in hold_rows
                )
                for link in links
            }
            for camera in metadata["cameras"]
        }
        static_holds.append(
            {
                "hold_end_s": hold_end_s,
                "frame_indices": [row["index"] for row in trusted_rows],
                "filtered_opening_angles_deg": np.median(
                    angles,
                    axis=0,
                ).tolist(),
                "angle_span_deg": np.ptp(angles, axis=0).tolist(),
                "stationary": bool(
                    np.all(
                        np.ptp(angles, axis=0)
                        <= STATIC_ANGLE_SPAN_DEG
                    )
                ),
                "minimum_visible_pixels_per_camera": visibility,
                "minimum_edge_margin_pixels_per_camera": edge_margins,
                "all_links_visible": all(
                    sum(
                        visibility[camera][link]
                        >= minimum_visible_pixels
                        and edge_margins[camera][link]
                        >= MINIMUM_EDGE_MARGIN_PIXELS
                        for camera in visibility
                    )
                    >= 2
                    for link in links
                ),
            }
        )
    if static_holds:
        static_angles = np.asarray(
            [static_holds[-1]["filtered_opening_angles_deg"]]
        )
    result = {
        "schema": "drake-rgbd-official-foundationpose-tracking-v1",
        "object_profile": object_profile,
        "mass_order_links": metadata.get(
            "mass_order_links",
            list(links),
        ),
        "model": str(FOUNDATIONPOSE),
        "weights": {
            "scorer": "2024-01-11-20-02-45",
            "refiner": "2023-10-28-18-33-37",
        },
        "initialization": (
            "known grasp pose then track_one"
            if args.initialize_from_known_pose
            else "first-frame masks then track_one"
        ),
        "frames": rows,
        "summary": {
            "frame_count": len(rows),
            "angle_mae_deg": float(errors.mean()),
            "angle_p95_deg": float(np.percentile(errors, 95)),
            "angle_max_deg": float(errors.max()),
            "median_tracking_frame_ms": float(
                np.median(track_ms)
            ),
            "tracking_fps": float(
                1000.0 / np.mean(track_ms)
            ),
            "median_tracking_frame_ms_three_cameras_two_links": float(
                np.median(track_ms)
            ),
            "tracking_fps_three_cameras_two_links": float(
                1000.0 / np.mean(track_ms)
            ),
            "filtered_static_opening_deg": float(
                np.median(static_angles[:, 0])
            ),
            "filtered_static_opening_angles_deg": np.median(
                static_angles,
                axis=0,
            ).tolist(),
            "static_window_span_deg": float(
                np.max(np.ptp(static_angles, axis=0))
            ),
            "stationary": (
                static_holds[-1]["stationary"]
                if static_holds
                else bool(
                    np.all(
                        np.ptp(static_angles, axis=0)
                        <= STATIC_ANGLE_SPAN_DEG
                    )
                )
            ),
            "static_holds": static_holds,
            "static_angle_span_threshold_deg": STATIC_ANGLE_SPAN_DEG,
            "minimum_visible_pixels_threshold": minimum_visible_pixels,
            "minimum_edge_margin_pixels_threshold": (
                MINIMUM_EDGE_MARGIN_PIXELS
            ),
            "maximum_mask_depth_centroid_residual_m": (
                MAXIMUM_MASK_DEPTH_CENTROID_RESIDUAL_M
            ),
            "re_registration_count": int(
                sum(
                    item["re_registered"]
                    for row in rows
                    for camera in row["per_camera"].values()
                    for item in camera.values()
                )
            ),
            "recovery_events": recovery_events,
            "recovery_policy": (
                "hold after two-camera failure for five captured frames; "
                "re-register failed streams only while static; resume after "
                "two-camera agreement for three frames"
            ),
        },
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
