# Third-party assets

This repository preserves only the third-party files needed for the Drake baseline.

- **RB5-850E:** `rbpodo_description` URDF and DAE meshes, plus OBJ visual and
  collision meshes converted from the package's DAE/STL files. The Apache License
  2.0 supplied with that package is included at
  `assets/rbpodo_description/LICENSE`.
- **Intel RealSense D435i:** expanded from `realsense2_description`; the OBJ
  visual is converted from its DAE mesh. The
  upstream RealSense ROS repository is distributed under Apache-2.0:
  <https://github.com/realsenseai/realsense-ros>.
- **DH PGC-140-50:** URDF-derived geometry from the official
  `DH-Robotics/dh_gripper_ros` repository at commit
  `f59f9c2f4bc8eb116448b1d798791424bf64e337`. No license file was present in
  the locally obtained source. Confirm redistribution permission before
  forwarding these files beyond the intended research recipient.
- **AIDIN AFT200-D80-C:** `aft200_visual.obj` is a visualization derivative of
  an official AIDIN STEP model. No separate CAD redistribution license was
  found locally. Confirm permission before forwarding it beyond the intended
  research recipient.

Product and company names remain the property of their respective owners.
