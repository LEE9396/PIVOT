# RB5 Drake viewer

[![Drake checks](https://github.com/Yuseong-Cheon/HTD/actions/workflows/check.yml/badge.svg)](https://github.com/Yuseong-Cheon/HTD/actions/workflows/check.yml)

Portable Meshcat and contact-physics baseline for:

```text
RB5-850E -> AFT200-KIT-RB -> DH PGC-140-50
          + Intel RealSense D435i + lab table
```

## Requirements

- Linux x86-64
- Python 3.12
- A web browser

## Setup

```bash
git clone https://github.com/Yuseong-Cheon/HTD.git
cd HTD
sudo apt-get update
sudo apt-get install -y python3.12-venv libegl1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python drake_rb5_scene.py --check
python drake_rb5_scene.py
python drake_rb5_scene.py --physics --check
python drake_rb5_scene.py --full-check
python drake_rb5_scene.py --physics
```

Both views provide six RB5 joint sliders and a PGC jaw slider. Physics mode
also drops a 200 g box onto the table. Open the printed Meshcat URL and stop
with `Ctrl-C`.

Expected check result:

```text
[DrakeRB5] PASS positions=8 collisions=15 base_clearance=0.024m table_top=0.300m PGC=25mm/jaw D435i_K=(446.803,432.971,424,240)
[DrakeRB5] RGBD PASS color=848x480 valid_depth=334104
[DrakeRB5] GRAVITY PASS hold_error=0.0000rad max_compensation=17.3Nm
[DrakeRB5] AFT200 PASS force=9.81N torque=0.388Nm error=...N/...Nm
[DrakeRB5] PHYSICS PASS box_z=0.330m
[DrakeRB5] MOTOR CONTACT PASS max_penetration=1.4mm
[DrakeRB5] CONTROL PASS contact_speed=0.00050rad/s jaw_speed=5.04e-08rad/s
[DrakeRB5] FLOOR PASS box_z=0.030m
[DrakeRB5] COLLISION AUDIT PASS safe=4 samples=250 ...
[DrakeRB5] 60S STABILITY PASS ...
[DrakeRB5] REPEATABILITY PASS runs=10 ...
[DrakeRB5] TIMESTEP PASS 0.5ms:... 1ms:... 2ms:...
[DrakeRB5] OBJECT SWEEP PASS light:... baseline:... heavy:... thin:... fast:...
[DrakeRB5] D435I PRECISION PASS expected=0.470m measured=0.470m ...
[DrakeRB5] FULL PASS elapsed=...s
```

Robot motion in the normal view is position-set kinematics. Physics mode uses
Drake's native discrete PD actuators and gravity compensation so the robot reacts to
table contact. Table and floor use rigid hydroelastic contact, while test objects use
compliant hydroelastic contact with the `kSimilar` discrete approximation. Slider
commands are clipped to joint limits and rejected when the requested pose has robot
self-collision; table and object contact remain allowed.

`--check` covers assembly and actual RGB-D output. `--physics --check` covers control,
contact, floor, and AFT200 reaction direction/magnitude. `--full-check` adds 60-second
stability, 10-run repeatability, 0.5/1/2 ms sensitivity, light/heavy/thin/fast objects,
250 deterministic collision samples, and a known-range D435i target.

A 4x4 m fixed floor catches objects outside the table.

## Validation boundary

- D435i uses nominal intrinsics and exact simulated extrinsics, not device EEPROM
  calibration, noise, or latency.
- AFT200 reports the Drake weld-joint reaction in the child joint frame, not real CAN
  hardware or a 1000 Hz electronics model.
- PGC collision uses measured-contact-area pad boxes; replace them after measured
  gripper collision geometry is available.
- Automated grasping, IK, trajectory optimization, Isaac USD, and real hardware control
  are not included.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ensurepip is not available` | Install `python3.12-venv`, then recreate `.venv`. |
| `could not initialize EGL` | Install `libegl1`. |
| `SELF-COLLISION REJECTED` | The unsafe slider target was rejected and reset to the last safe pose. |
| Object sinks or tunnels after physics changes | Run `--full-check`; all 0.5/1/2 ms and object cases must pass before sharing. |

See `THIRD_PARTY_ASSETS.md` before redistributing the included meshes.
