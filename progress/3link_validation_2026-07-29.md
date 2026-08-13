# Articulated mass-estimation handoff — updated 2026-07-30

## Verdict

The earlier nominal simulation passed, but the new systematic-error audit
shows that none of the three household objects currently meets the 6% robust
mass/density target. Do not claim real-world part-wise identification from the
nominal results below. The robot still uses one continuous grasp: it never
places the object back on the table between holds.

The current error envelope, excluding grasp slip, is:

- body-frame link COM offset: `±5 mm` per axis;
- FoundationPose joint-angle bias: `±1°` per joint, constant over one run;
- residual AFT200 bias after known-tool compensation:
  `±0.07 N / ±0.007 Nm`;
- link volume scale error: `±3%`.

Each estimate uses 400 block/residual bootstrap refits with the same sampled
systematic bias held across every pose. The estimator now requires every
internal joint to move by at least `10°` before it can declare part-wise
convergence. `--robust-error-scale 0` reproduces the nominal model;
the default `1` enables the complete envelope.

## Current robust results

| Object | Internal-joint range | Maximum robust 95% half-width | Result |
|---|---:|---:|---|
| jewelry box | `0.035°` | `20.6%` | safe-hold failure |
| desk lamp | `87.74° / 0.050°` | `68.1%` | insufficient upper-joint excitation |
| phantom_v3 | `42.55° / 41.03°` | `72.0%` | grasp-rotation margin exhausted |

The jewelry-box control comparison is important:

- nominal (`--robust-error-scale 0`): passes after three holds with `0.99%`;
- robust (`--robust-error-scale 1`): fails with `20.6%`;
- both runs move the hinge only about `0.035°`.

This proves that the old success depended on treating mesh-centroid geometry
as exact. The desk lamp confirms the physical observability issue: the first
joint can move substantially, but the last joint remains at its stop, so its
connected link masses cannot be separated robustly. Phantom excites both
joints but needs either better calibrated error bounds, safer additional
information, or a different physical protocol.

Representative results:

- `robot_learning/results/contact_mass_pipeline_jewelry_box_nominal_v6.json`
- `robot_learning/results/contact_mass_pipeline_jewelry_box_robust_excitation_v6.json`
- `robot_learning/results/contact_mass_pipeline_desk_lamp_robust_excitation_v6.json`
- `robot_learning/results/contact_mass_pipeline_phantom_v3_robust_excitation_v6.json`

## Pose-difficulty and joint-response update — 2026-07-30

The GT-free adaptive objective now subtracts a normalized pose-difficulty
cost from log-determinant information. The cost includes absolute wrist
pitch/roll, deviation from a 15-degree increment, predicted grasp torque, and
proximity to the 20-degree grasp-rotation limit. After a non-responsive
action, the controller keeps the same wrist axis and uses one easier
confirmation action instead of increasing difficulty or changing axes.

For every target joint, predicted excitation is compared with the measured
FoundationPose angle change. An action is informative above `0.03 Nm`; a
response must be at least `1 degree`. Two consecutive informative actions
without that response terminate with
`joint_not_responsive_under_single_grasp`.

| Object | Holds | Joint range | Max robust 95% half-width | Max wrist | Max grasp rotation | Result |
|---|---:|---:|---:|---:|---:|---|
| jewelry box | 3 | `0.035°` | `18.45%` | `15°` | `1.87°` | joint not responsive |
| desk lamp | 3 | `0.035° / 0.043°` | `78.03%` | `15°` | `0.57°` | joints not responsive |
| phantom_v3 | 5 | `27.55° / 36.96°` | `78.21%` | `30°` | `15.82°` | no safe geometry action |

None passes the unchanged 6% target. Jewelry box and desk lamp now stop after
the required two measured non-responses; the desk lamp never approaches its
old `pitch=60°` posture. Phantom keeps both joint ranges above 10 degrees and
actual grasp rotation below 20 degrees, but its final condition number is
`85.42`, and every remaining candidate violates a predicted safety or geometry
gate.

New results:

- `robot_learning/results/contact_mass_pipeline_jewelry_box_pose_response_v7.json`
- `robot_learning/results/contact_mass_pipeline_desk_lamp_pose_response_v7.json`
- `robot_learning/results/contact_mass_pipeline_phantom_v3_pose_response_v7.json`

The earlier regression checkpoint had one legacy 3-link failure at `21.25%`
maximum mass error versus its old `12%` threshold. Its root cause and corrected
gate are recorded below.

## Error-source decomposition and protocol decision — 2026-07-30

The 400-refit bootstrap now reports each source separately using the same
resampled blocks and residuals. Percentages below are maximum mass/density
relative 95% half-widths:

| Object | Nominal | F/T bias | Angle bias | COM offset | Volume scale | Combined |
|---|---:|---:|---:|---:|---:|---:|
| jewelry box | `0.70%` | `3.04%` | `2.48%` | `16.84%` | `2.95%` | `18.45%` |
| desk lamp | `10.66%` | `12.80%` | `66.56%` | `52.43%` | `11.10%` | `78.03%` |
| phantom_v3 | `28.01%` | `42.66%` | `40.62%` | `75.93%` | `28.01%` | `78.21%` |

COM geometry is the dominant systematic source for every object. Desk lamp
also has a severe angle-bias/observability problem. Phantom remains above 6%
even nominally, so calibration alone cannot make its current five-hold
trajectory pass.

New source-decomposition results:

- `robot_learning/results/contact_mass_pipeline_jewelry_box_error_sources_v8.json`
- `robot_learning/results/contact_mass_pipeline_desk_lamp_error_sources_v8.json`
- `robot_learning/results/contact_mass_pipeline_phantom_v3_error_sources_v8.json`

Physical protocol decision:

- No brake, indexing fixture, documented static lock, or second manipulator is
  currently part of the realizable experiment.
- Therefore jewelry box and desk lamp must report only total mass and
  whole-object COM. Their per-link estimates remain diagnostic simulation
  values and are not accepted measurements.
- Part-wise identification may resume only after a real mechanical
  indexing/brake fixture can hold at least three known static joint
  configurations while the robot retains the grasp. Direct joint
  teleportation or simulator-only locking remains prohibited.

Follow-up checks:

- Known-tool-compensated maximum fit residuals pass the modeled AFT200
  resolution: jewelry box `0.076 N / 0.0079 Nm`; desk lamp
  `0.084 N / 0.0075 Nm`.
- Phantom 12-second motion validation passes all three initial conditions with
  zero penetration and stationary final windows:
  `robot_learning/results/phantom_v3_motion_validation_12s_v8.json`.
- The FoundationPose GPU end-to-end run completed 317 frames, issued three
  safe-hold requests, resumed all three links, and performed nine
  re-registrations. Recovery behavior works, but pose accuracy fails
  (`11.16°` MAE, `74.42°` p95); the second joint spans only `4.38°`, so the
  mass result is rejected with `insufficient_internal_joint_excitation`.
  Artifacts:
  `robot_learning/results/foundationpose_drake_3link_e2e_v8/foundationpose_result.json`
  and
  `robot_learning/results/contact_mass_pipeline_foundationpose_3link_e2e_v8.json`.
- The legacy generic 3-link regression exposed the same root cause: its second
  joint spans only `0.144°`. The 10-degree excitation gate now applies to
  every 3-link run, and the regression checks rejection instead of relaxing
  the old 12% error threshold.
- Final regression: all 23 tests pass, and `git diff --check` passes.

## Next-session work plan

Proceed in this order and do not relax the 6% target merely to obtain a pass.

1. Add a pose-difficulty term to the GT-free action objective.
   - Penalize absolute wrist pitch/roll, transition size, predicted grasp
     torque, and proximity to the `20°` grasp-rotation limit.
   - Prefer `15°` incremental moves.
   - Reject the observed desk-lamp `pitch=60°` posture unless no easier safe
     action can provide comparable information.
2. Add an excitation-response check after each hold.
   - Compare predicted joint excitation with measured joint-angle change.
   - If a target joint changes by less than a calibrated minimum after two
     informative wrist actions, stop with
     `joint_not_responsive_under_single_grasp`.
   - Do not keep increasing wrist difficulty when only another joint moves.
3. Separate the uncertainty contribution of each error source.
   - Report nominal repeatability, F/T bias, angle bias, COM offset, and volume
     scale individually as well as combined.
   - Use this to determine which physical calibration would materially reduce
     uncertainty; do not tune bounds using mass GT.
4. Re-run the three objects headless.
   - Jewelry box must not pass without real hinge excitation.
   - Desk lamp must not pass while joint 2 remains near `0°`.
   - Phantom must keep both joint ranges above `10°`, remain within grasp
     limits, and reach combined mass/density half-width `<=6%`.
5. Decide the physical protocol for non-responsive passive joints.
   - Preferred options are a real brake/indexing fixture, manual reposition
     with a documented static lock, or a second manipulator.
   - Do not simulate direct joint teleportation or locking unless the same
     mechanism is realizable in hardware.
   - If no such mechanism is adopted, report only total mass and whole-object
     COM for jewelry box/desk lamp, not per-link masses.
6. After the estimator protocol is physically defensible, re-run:
   - known-tool-wrench regressions on jewelry box and desk lamp;
   - phantom mesh collision and 12-second stationarity checks;
   - FoundationPose GPU end-to-end safe-hold/re-registration;
   - the complete unit suite and `git diff --check`.

Current regression status is recorded in the update above. The remaining
legacy failure predates the robust audit path and must be diagnosed rather
than hidden or threshold-relaxed.

Resume command:

```bash
cd /home/cheon/drake_validation_test
git status --short
git log -1 --oneline
```

Suggested next-chat prompt:

> `/home/cheon/drake_validation_test에서 이어서 진행해. 2026-07-30 강건성
> 감사에서 jewelry/desk/phantom 모두 6% 실패했고, 단일 파지와 10° 내부관절
> excitation gate가 적용됐다. progress/3link_validation_2026-07-29.md의
> Next-session work plan부터 시작해서 자세 난이도 비용과 joint-response 조기중단을
> 구현한 뒤 세 물체를 다시 검증해.`

The remaining sections document the earlier nominal preflight and are kept as
historical context.

## Final action

- Start with all three links straight at 180° on the table.
- Keep both hinges unlocked.
- Grasp at the parent end, lift 8 cm, then retract 30 cm while lifting 45 cm.
- Collect eight stationary holds:
  - wrist pitch: `[-20, -15, -10, -5, 5, 10, 15, 20]°`
  - wrist roll: `[-20, 20, -20, 20, -20, 20, -20, 20]°`
- Require all eight holds to pass contact, slip, stationarity, camera
  visibility, F/T consistency, rank, and condition-number gates.

Measurement is adaptive rather than fixed at eight holds:

- Start evaluating after three accepted static holds.
- After every new hold, run 400 block-bootstrap mass refits with AFT200
  quantization noise.
- Stop immediately when rank is 3, condition number is at most 50, the maximum
  relative 95% half-width is at most the requested target, and mass change is
  at most 2%.
- If the safe candidate pool is exhausted first, return
  `next_action=add_safe_pose_or_regrasp` without accepting a mass.

On the official FoundationPose sequence, the normal 5% target stopped at pose
8 with a 4.44% maximum half-width. A 6% target stopped early at pose 5 with a
5.89% maximum half-width, confirming that the controller stop is data-driven.

The eight-pose sequence is now the validated fallback, not a geometry-specific
constant. For a new object, `--auto-plan --geometry-action-input FILE` selects
the safest informative subset from candidate pitch/roll poses without using
mass GT. Each candidate supplies either a finite `6×3 mass_regressor`, or:

```json
{
  "wrist_pitch_deg": -20,
  "wrist_roll_deg": -20,
  "gravity_sensor_m_s2": [0, 0, -9.81],
  "part_com_sensor_m": [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]],
  "grasp_safe": true,
  "all_links_visible": true,
  "stationary": true
}
```

The RGB-D/CAD geometry and robot FK produce these COM positions for each
candidate. The planner rejects non-finite, non-`6×3`, unsafe, invisible, or
rank-deficient inputs. If no subset has rank 3, or if the measured condition
number/uncertainty gate fails, it returns no mass and requests a different
grasp or additional safe poses.

Planner-only random-geometry regression varied three link lengths and COM
fractions across 100 synthetic graspable geometries without using masses. All
100 produced full-rank plans, five different action subsets were selected, and
the worst condition number was `41.73` (`<= 50`). A deliberately degenerate
regressor is rejected instead of producing a mass.

## Estimator

For each stationary hold `k`, in the AFT200 sensor frame:

```text
b_k = w_loaded,k - w_empty,k
A_k[:,j] = [-g_k ; -(r_kj × g_k)]
m_hat = max(0, argmin_m Σ_k ||W(A_k m - b_k)||²)
```

Only static gravity wrenches are used for part-wise mass. Dynamic acceleration
regressors are disabled, removing simulator-acceleration leakage and making the
estimate independent of authored link inertia.

## Validation results

### F/T equation and frames

- Static maximum residual: `0.121 N / 0.0115 Nm`.
- Integrated excitation residual: `0.0123 N / 0.0027 Nm`.
- Both are inside the modeled AFT200 resolution (`0.15 N / 0.015 Nm`).

### Unseen mass combinations

The eight-pose design had condition number about `10.8–11.4`.

| GT masses (kg) | Maximum relative error |
|---|---:|
| `[0.8, 0.4, 0.25]` | 3.36% |
| `[0.55, 0.7, 0.35]` | 1.04% |
| `[1.1, 0.3, 0.6]` | 0.62% |
| `[0.35, 0.45, 0.25]` | 0.61% |

### Camera degradation

- Stress model: fused joint-angle noise `σ=0.5°`, with two of eight poses
  randomly missing.
- Hardest mass set p95 maximum error: `7.50%`.
- Stress criterion: p95 `<= 10%`; passed.

### Friction and hinge damping

- Nominal and high-friction accepted runs: maximum mass error `0.62%` and
  `3.86%`.
- Low-friction run slipped `7.5 mm` and was rejected before estimation by the
  `6 mm` grasp-drift gate.

### Payload boundary

- RB5-850E payload: `5 kg`.
- AFT200 + PGC modeled tool mass: `1.635 kg`.
- PGC-140-50 recommended object limit: `3 kg`.
- At a `3 kg` object, total flange load is `4.635 kg`.
- Low-force grasp (`KP=3000`) is attempted first. If lift/slip fails, the
  object is lowered and regrasped at `KP=6000`, still under the modeled
  per-finger `140 N` actuator limit.
- The 3 kg boundary run passed after one regrasp.

### Official FoundationPose

New eight-pose capture:

- 317 RGB-D frames, three cameras, three links.
- Every link visible in every static hold.
- Angle MAE `0.472°`, p95 `1.245°`.
- Final estimate `[0.8015, 0.3960, 0.2539] kg` for
  GT `[0.8, 0.4, 0.25] kg`.
- Maximum mass error `1.55%`.
- Design condition number `10.73`.
- `validation.passed: true`.

### Regression

- All 10 software tests pass, including the arbitrary-geometry action-input
  safety/selection contract.

## Artifacts

- Official pose result:
  `robot_learning/results/foundationpose_drake_3link_8pose_preflight/foundationpose_result.json`
- Official-pose mass result:
  `results/contact_mass_pipeline_foundationpose_3link_8pose_preflight.json`
- 3 kg boundary result:
  `results/contact_mass_pipeline_3link_3kg_preflight.json`

## Required on the physical system

1. Load the individual D435i intrinsics and camera-to-robot transforms.
2. Load the individual AFT200 6×6 calibration matrix and bias.
3. Verify camera, encoder, and F/T timestamps on one clock.
4. Run empty-tool tare and a known calibration weight.
5. Execute the same trajectory at reduced speed with hardware collision and
   wrench stops enabled.
6. Accept real operation only if the same eight-hold gates pass.
