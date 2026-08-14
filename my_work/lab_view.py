"""실험실 씬을 Drake Meshcat 에서 직접 확인하는 뷰어.

배치(테이블, 받침대, D456 카메라, 안전 촬영 영역), 계획에 담긴 자세들,
그리고 물체 관절각을 손으로 돌려보며 눈으로 검증하기 위한 도구다.

화면 요소
  - 노란 선   D456 카메라 시야 절두체. 카메라가 어디를 보는지
  - 파란 선   안전 촬영 영역 상자. 측정 자세는 이 안에만 놓인다
  - 버튼      시작 자세 / 라운드별 측정 자세로 즉시 이동
  - 슬라이더  물체 관절각을 직접 돌려본다 (충돌·간섭 눈으로 확인)
  - 버튼      "카메라 시점으로 보기" 를 누르면 화면이 D456 위치로 이동

실행:
    cd ~/Desktop/PIVOT/my_work
    ../robot_learning/scripts/run_drake_env.sh python lab_view.py \
        --plan outputs/plan_3link.json
    # 계획 없이 배치만 보려면
    ../robot_learning/scripts/run_drake_env.sh python lab_view.py --object 3link
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pydrake.geometry import Rgba, StartMeshcat
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization

import density_id_objects as obj
import robot_scene as rs


def frustum_segments(camera, near_m=0.35, far_m=1.8):
    """카메라 절두체를 선분 쌍으로. 시야각은 설정 파일의 depth_fov_deg."""
    pose = rs.look_at_pose(np.array(camera["position_xyz_m"]),
                           np.array(camera["look_at_xyz_m"]))
    fov_h, fov_v = np.deg2rad(camera["depth_fov_deg"])
    starts, ends = [], []

    def corners(distance):
        half_x = distance * np.tan(fov_h / 2.0)
        half_y = distance * np.tan(fov_v / 2.0)
        local = [np.array([sx * half_x, sy * half_y, distance])
                 for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        return [pose @ point for point in local]

    near, far = corners(near_m), corners(far_m)
    eye = np.array(camera["position_xyz_m"])
    for point in far:                       # 카메라에서 뻗는 네 모서리
        starts.append(eye)
        ends.append(point)
    for ring in (near, far):                # 근/원 평면 사각형
        for index in range(4):
            starts.append(ring[index])
            ends.append(ring[(index + 1) % 4])
    return np.array(starts).T, np.array(ends).T


def box_segments(lower, upper):
    """축 정렬 상자의 12개 모서리."""
    corners = np.array([[x, y, z] for x in (lower[0], upper[0])
                        for y in (lower[1], upper[1])
                        for z in (lower[2], upper[2])])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    starts = np.array([corners[a] for a, _ in edges]).T
    ends = np.array([corners[b] for _, b in edges]).T
    return starts, ends


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--object", choices=tuple(obj.OBJECTS), default="3link")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text()) if args.plan else None
    key = plan["object"] if plan else args.object
    spec = obj.OBJECTS[key]
    limits = ([(np.deg2rad(a), np.deg2rad(b))
               for a, b in plan["joint_range_deg"]] if plan else
              [j.limits_rad for j in spec.joints])
    densities = plan["density_gt"] if plan else [p.rho_gt for p in spec.parts]

    meshcat = StartMeshcat()
    builder = DiagramBuilder()
    scene = rs.build_scene(spec, densities, limits, builder=builder,
                           include_visuals=True)
    plant = scene["plant"]
    arm_joints = [plant.GetJointByName(n, scene["arm"])
                  for n in rs.ARM_JOINT_NAMES]
    finger_joints = [plant.GetJointByName(n, scene["gripper"])
                     for n in ("finger1_joint", "finger2_joint")]
    object_joints = [plant.GetJointByName(j.name, scene["payload"])
                     for j in spec.joints]
    AddDefaultVisualization(builder, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # --- 참고선: 카메라 절두체와 안전 촬영 영역 ---
    starts, ends = frustum_segments(rs.CAMERA)
    meshcat.SetLineSegments("/lab/camera_fov", starts, ends,
                            line_width=1.5, rgba=Rgba(0.95, 0.80, 0.15, 1.0))
    starts, ends = box_segments(rs.WORKSPACE_LOWER_M, rs.WORKSPACE_UPPER_M)
    meshcat.SetLineSegments("/lab/capture_box", starts, ends,
                            line_width=1.5, rgba=Rgba(0.25, 0.60, 0.95, 1.0))

    q = plant.GetPositions(plant_context).copy()
    finger = (plan["finger_position_m"] if plan
              else rs.finger_position_for(spec))
    for joint in finger_joints:
        q[joint.position_start()] = finger

    def show():
        plant.SetPositions(plant_context, q)
        diagram.ForcedPublish(context)

    def set_arm_from(full_q):
        for joint in arm_joints:
            q[joint.position_start()] = full_q[joint.position_start()]

    # --- 자세 버튼 ---
    poses = []
    if plan:
        poses.append(("시작 자세", np.array(plan["arm_q_start"])))
        for entry in plan["rounds"]:
            deg = np.round(entry["object_joint_deg"], 0)
            for index, (arm_q, g) in enumerate(zip(entry["arm_q_measure"],
                                                   entry["gravity_dirs"]), 1):
                poses.append((f"round {entry['round']} q={deg} 중력{index}",
                              np.array(arm_q)))
    buttons = {}
    for label, _ in poses:
        meshcat.AddButton(label)
        buttons[label] = meshcat.GetButtonClicks(label)

    camera_button = "카메라(D456) 시점으로 보기"
    meshcat.AddButton(camera_button)
    camera_clicks = meshcat.GetButtonClicks(camera_button)
    reset_button = "전경 시점으로 돌아가기"
    meshcat.AddButton(reset_button)
    reset_clicks = meshcat.GetButtonClicks(reset_button)

    # --- 물체 관절 슬라이더 ---
    sliders = []
    for joint, (lo, hi) in zip(spec.joints, limits):
        name = f"{joint.name} [deg]"
        meshcat.AddSlider(name, min=np.degrees(lo), max=np.degrees(hi),
                          step=0.5, value=np.degrees(lo))
        sliders.append(name)

    if poses:
        set_arm_from(poses[0][1])
    show()

    print(f"\n{spec.label}")
    print(f"Meshcat: {meshcat.web_url()}")
    print(f"  테이블 {rs.TABLE_SIZE_M} m, 상판 z = {rs.TABLE_TOP_Z_M} m")
    print(f"  로봇 베이스 {rs.ROBOT_BASE_XYZ_M} m,"
          f" yaw {rs.ROBOT_BASE_RPY_DEG[2]:.0f} deg")
    print(f"  카메라 {rs.CAMERA['model']} @ {rs.CAMERA['position_xyz_m']}"
          f" -> {rs.CAMERA['look_at_xyz_m']}")
    print(f"  노란 선 = 카메라 시야, 파란 선 = 안전 촬영 영역")
    if plan:
        print(f"  버튼 {len(poses)}개로 계획된 자세를 확인할 수 있습니다")
    print(f"  슬라이더로 물체 관절각을 돌려 간섭을 직접 확인하세요")
    print(f"  Ctrl-C 로 종료합니다")

    try:
        while True:
            changed = False
            for label, arm_q in poses:
                clicks = meshcat.GetButtonClicks(label)
                if clicks != buttons[label]:
                    buttons[label] = clicks
                    set_arm_from(arm_q)
                    print(f"  -> {label}")
                    changed = True
            if meshcat.GetButtonClicks(camera_button) != camera_clicks:
                camera_clicks = meshcat.GetButtonClicks(camera_button)
                meshcat.SetCameraPose(
                    np.array(rs.CAMERA["position_xyz_m"]),
                    np.array(rs.CAMERA["look_at_xyz_m"]))
                print("  -> D456 시점")
            if meshcat.GetButtonClicks(reset_button) != reset_clicks:
                reset_clicks = meshcat.GetButtonClicks(reset_button)
                meshcat.SetCameraPose(np.array([1.8, -2.0, 1.9]),
                                      np.array([0.0, -0.2, 0.95]))
                print("  -> 전경 시점")
            values = [meshcat.GetSliderValue(n) for n in sliders]
            for joint, value in zip(object_joints, values):
                index = joint.position_start()
                if abs(q[index] - np.deg2rad(value)) > 1e-9:
                    q[index] = np.deg2rad(value)
                    changed = True
            if changed:
                show()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n종료합니다.")


if __name__ == "__main__":
    main()
