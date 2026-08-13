#!/usr/bin/env python3
"""Run Drake RGB-D, real FoundationPose tracking, then static mass fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import run_drake_contact_mass_pipeline as mass_pipeline
import simulate_drake_rb5_contact_ft_custom_object as contact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument("--part-count", type=int, choices=(2, 3), default=2)
    parser.add_argument("--part-masses-kg", type=float, nargs="+")
    parser.add_argument("--opening-angle-deg", type=float, default=180.0)
    parser.add_argument("--initial-opening-angle-deg", type=float, default=180.0)
    parser.add_argument("--initial-opening-angles-deg", type=float, nargs="+")
    parser.add_argument("--wrist-pitch-sequence-deg", type=float, nargs="+")
    parser.add_argument("--wrist-roll-sequence-deg", type=float, nargs="+")
    parser.add_argument("--action-plan", type=Path)
    parser.add_argument("--auto-plan", action="store_true")
    parser.add_argument("--camera-calibration-result", type=Path)
    parser.add_argument("--mass-output", type=Path)
    args = parser.parse_args()
    args.output_dir = args.output_dir or (
        ROOT / f"results/foundationpose_drake_{args.part_count}link"
    )
    masses = tuple(
        args.part_masses_kg or (0.8, 0.4, 0.25)[: args.part_count]
    )
    action_plan = None
    if args.action_plan is not None:
        action_plan = json.loads(args.action_plan.read_text()).get(
            "action_plan"
        )
    elif args.auto_plan:
        action_plan = mass_pipeline.plan_wrist_actions(
            camera_calibration_result=args.camera_calibration_result,
        )
    wrist_pitch = tuple(
        (
            action_plan["selected"]["wrist_pitch_sequence_deg"]
            if action_plan is not None
            else args.wrist_pitch_sequence_deg
        )
        or (
            (40.0, 80.0, 100.0)
            if args.part_count == 2
            else mass_pipeline.WRIST_PITCH_FREE_3LINK_DEG
        )
    )
    wrist_roll = tuple(
        args.wrist_roll_sequence_deg
        or (
            mass_pipeline.WRIST_ROLL_FREE_3LINK_DEG
            if args.part_count == 3
            and wrist_pitch == mass_pipeline.WRIST_PITCH_FREE_3LINK_DEG
            else (0.0,) * len(wrist_pitch)
        )
    )
    initial_angles = (
        tuple(args.initial_opening_angles_deg)
        if args.initial_opening_angles_deg
        else None
    )
    contact.simulate(
        args.part_count,
        initial_opening_angle_deg=args.initial_opening_angle_deg,
        initial_opening_angles_deg=initial_angles,
        initial_wrist_pitch_deg=0.0,
        opening_angle_deg=args.opening_angle_deg,
        wrist_pitch_sequence_deg=wrist_pitch,
        wrist_roll_sequence_deg=wrist_roll,
        free_hinges=True,
        grasp_offset_m=contact.PARENT_END_GRASP_OFFSET_M,
        pgc_controller_kp=mass_pipeline.PGC_CALIBRATED_KP,
        part_masses_kg=masses,
        foundationpose_capture_dir=args.output_dir,
        foundationpose_capture_fps=5.0,
    )
    pose_result = args.output_dir / "foundationpose_result.json"
    with (args.output_dir / "inference.log").open("w") as log:
        subprocess.run(
            [
                "/home/cheon/miniforge3/envs/bundlesdf/bin/python",
                str(SCRIPT_DIR / "run_foundationpose_drake_capture.py"),
                str(args.output_dir),
            ],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    result = mass_pipeline.run(
        opening_angle_deg=args.opening_angle_deg,
        initial_opening_angle_deg=args.initial_opening_angle_deg,
        initial_opening_angles_deg=initial_angles,
        part_count=args.part_count,
        vary_internal_angles=args.part_count == 3,
        part_masses_kg=masses,
        wrist_pitch_sequence_deg=wrist_pitch,
        wrist_roll_sequence_deg=wrist_roll,
        steps=len(wrist_pitch),
        seed=20260728,
        foundationpose_result=pose_result,
    )
    if action_plan is not None:
        result["action_plan"] = action_plan
    mass_output = args.mass_output or (
        ROOT
        / f"results/contact_mass_pipeline_foundationpose_{args.part_count}link.json"
    )
    mass_output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "foundationpose": json.loads(pose_result.read_text())[
                    "summary"
                ],
                "mass": result["final"],
                "termination": result["termination"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
