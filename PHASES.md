# Conveyor bottle inspection and reject

Plan: [`PLAN.md`](PLAN.md)

## Phase A: conveyor cell physics and ground-truth picking (DONE 2026-09-27)

Everything in this phase is local. NVIDIA's asset server was unreachable from this machine during the work, so:
- the conveyor, bottles, reject bin and pedestal are procedural (`cell_scene.py`);
- the Franka is converted from the URDF bundled with Isaac Sim's URDF importer
  (`import_franka_local.py` -> `assets/panda_arm_hand/`).

| File | What it does |
|---|---|
| `cell_scene.py` | builds the cell: belt driven by PhysX surface velocity, low-friction side guides, frame and legs, reject bin on a stand, inspection camera and presentation camera. Also holds the procedural bottles: a lathe-mesh body, shoulder, neck, lip and hollow mouth, an optional cap, and cylinder colliders |
| `robot_setup.py` | Franka setup shared by all scripts: gravity compensation, 70 N fingers, finger friction, measured fingertip offset, neck-grasp height |
| `phaseA_belt_test.py` | belt stop-and-go test without a robot |
| `grasp_debug.py` | single-bottle grasp and lift check (finger force, grasp height) |
| `phaseA_pick_test.py` | acceptance test: the belt stops each capless bottle at the pick station; the robot grasps the neck, lifts it and drops it in the reject bin |

Run a test with: `cd <isaac-sim>; ./kit/kit apps/isaacsim.exp.base.python.kit --no-window --exec phaseA_pick_test.py`

### Results
- **Pick test: 20/20 picks into the reject bin, 20 in a row.** No drops, nothing fell off the belt, no capless bottle
  escaped, and 6 capped bottles passed to the exit. Contact error 1.1-1.2 mm. The fingers closed to 2.8-3.2 mm under
  the neck diameter, which means a firm grip. Video: `logs/conveyor/phaseA_pick_test/phaseA_pick_test.mp4`.
- **Belt test: 24 bottles (8 capless stopped and removed, 16 passed).** Largest bottle lean was 1.1° and none tipped.
  Stop position was within 2.4 mm of the pick point.
- In the inspection camera, a missing cap is plainly visible as an open mouth
  (`logs/conveyor/phaseA_belt_test/inspection_view.png`).

### Findings that later phases depend on
1. **PhysX surface velocity carries objects at 2.10x the value set** (linear, measured at 0.075, 0.10 and 0.15 m/s).
   `set_belt_speed()` takes the real transport speed and divides by `BELT_GAIN`. Stops were 49-54 mm late until this
   was fixed. Phase D tracks bottles using belt speed, so this factor must be re-checked in the GUI run.
2. **Use simulation time, not a count of app updates.** One app update can advance physics by several steps, so the
   belt ramp uses `timeline.get_current_time()`.
3. **Finger force:** the URDF's 20 N let a 0.45 kg bottle slip out of a neck grasp. At the real Franka Hand's 70 N
   the bottle lifts 0.2 m with no slip. A body grasp fails because the fingers hit the shoulder.
4. **Belt stop logic:** the belt must stop when a reject reaches the station, even while the robot is still busy.
   Checking only between picks let the next reject slip past.
5. **The arm returns to a joint-space ready pose before every approach.** Differential IK from the bin pose could not
   reach back to the belt.

### Open items for Phase B+
- The warehouse surroundings need the asset server (the scene supports `build(warehouse=True, assets_root=...)`).
- Bottles have no labels yet. Phase B will randomize label textures, colors, shapes, belt material, lighting and distractors.

## Phase B: synthetic dataset (DONE 2026-09-27)
`sdg_generate.py` renders from the deployment inspection camera (±2-3 cm pose jitter, ±1.5° FOV, roll of a few
degrees) with domain randomization: bottle shape, body colour and roughness, procedural label textures, cap size,
colour and shape, caps the same colour as the body (12%), belt, guide, rail and floor materials, light intensity,
colour temperature and position, and a fill light. It places 1-6 bottles per frame, only when fully inside the image.
Output is COCO, 0-indexed: 0 = bottle_ok, 1 = bottle_nocap.

| split | frames | bottle_ok | bottle_nocap |
|---|---|---|---|
| train | 24,000 | 24,466 | 25,528 |
| val | 3,000 | 3,107 | 3,132 |
| test_unseen | 2,000 | 2,120 | 2,102 |

`test_unseen` uses body and cap hues 0.42-0.58 (never in train), a conical shoulder and held-out label textures.
Generation took 53 min for 2.6 GB (`sdg_data/`, not in git).

## Phase C: training (DONE)
RT-DETRv2-r18 (`phase4_perception_training/rtdetr_upstream/.../configs/rtdetrv2/rtdetrv2_r18vd_bottlecap.yml`).
It is fine-tuned (`-t`) from the Phase 4 pill/box detector because GitHub is unreachable for ImageNet or COCO weights.
Training ran 16 epochs with no zoom or crop augmentation (fixed deployment scale), at the 250 W cap, in 1 h 47 min.
Val mAP50 0.995 and mAP50:95 0.9945 (best at epoch 3). Model: `models/bottlecap_best.onnx`.

`eval_bottlecap.py` on **test_unseen** (per bottle): **no-cap recall 100% (2102/2102), false rejects 0/2120,
0 missed**, at every threshold from 0.1 to 0.7 (targets: >= 99%, <= 2%). The epoch-0 model already scored 100% and
1/2120. Its one error, a pale cap the same colour as the body, is the case the depth check exists for.

## Phase D: closed loop, stop-and-pick, perception only (DONE)
`conveyor_demo.py` (+ `bottle_detect.py` server, `inspection.py`):
1. Inspect every 0.25 s.
2. RT-DETR returns P(ok) and P(no-cap) per bottle.
3. The depth check looks for an open mouth: a hole >= 8 mm below the rim at the bottle axis.
4. Each bottle is tracked in belt coordinates using odometry.
5. A bottle is rejected only if the detector (mean P(no-cap) > 0.5) and the depth majority both agree.
6. The belt stops when the reject reaches the pick station, and the robot grasps it at the perceived position.

Ground truth is used only for scoring.

| seed | bottles | defects -> bin | escaped | false rejects | pick failures | max pick-position error |
|---|---|---|---|---|---|---|
| 1 | 50 | 13/13 | 0 | 0 | 0 | 5.6 mm |
| 2 | 50 | 10/10 | 0 | 0 | 0 | 5.7 mm |
| 3 | 50 | 17/17 | 0 | 0 | 0 | 5.7 mm |

**150 bottles: all 40 defects rejected into the bin, 110/110 good bottles passed.** The depth check agreed with the
detector on every track: open mouth in 14-17 of 14-17 views for every defect, and 0 views for every capped bottle.
Logs: `logs/conveyor/phaseD_seed{1,2,3}/`.

## Phase F: live video (DONE)
`./run_conveyor_demo.sh --record [--seed N]` (env `N` = bottles, default 24) opens Isaac Sim with the presentation
camera in the viewport and the "Inspection" panel docked right (`live_panel.py`). The panel shows the live inspection
image with per-bottle verdicts, the belt state, inspected/passed/rejected counts, a ground-truth audit and the decision
log. The script records the window in real time with ffmpeg. The surroundings are built locally (`build_room`:
racks, safety lines, camera gantry). NVIDIA's warehouse placed the cell on an empty stretch of floor, so it looked
emptier.
- Take (seed 1, 24 bottles): 7/7 defects rejected, 17/17 good passed, 0 missed, 0 false rejects.
- `logs/conveyor/linkedin/conveyor_inspection_realtime.mp4`: 6 min 9 s, unedited. In the GUI the
  simulation runs slower than real time (137 s of sim time over 369 s, ~18-30 FPS).
- `logs/conveyor/linkedin/conveyor_inspection_90s.mp4`: 88 s cut, real-time open and close, 2x/7x in
  between, speed labelled on screen.
- Post text: `logs/conveyor/linkedin/linkedin_post_draft.md`.

## Phase E: pick from the moving belt (IN PROGRESS, paused 2026-09-27 for a workstation restart)
`DEMO_MOVING=1` in `conveyor_demo.py`. The belt runs at 0.10 m/s (`DEMO_BELT_SPEED`) with a bottle every 2.4 s
(`DEMO_SPAWN_S`), and it never stops. For each reject, the arm:
1. follows the predicted position (camera track + belt odometry), with an integral lag correction along the belt;
2. descends, closes and lifts while following (the fingers close along the belt direction);
3. drops the bottle in the bin from above the rim.

Bursts: while the arm is busy and another reject is queued, the belt slows to 0.03 m/s instead of stopping.
- 12-bottle debug run: 3/3 back-to-back rejects picked, 0 misses, grasps 2-5 mm from ground truth,
  6.5-6.9 s per cycle, 2 slowdowns, belt never stopped.
- **Acceptance (3 seeds x 50) was interrupted.** Seed 1 had reached bottle 44 with 11 picks, and no summary was
  written. To resume:
  `cd <isaac-sim>; for s in 1 2 3; do DEMO_MOVING=1 DEMO_N=50 DEMO_SEED=$s DEMO_OUT=logs/conveyor/phaseE_seed$s ./kit/kit apps/isaacsim.exp.base.python.kit --no-window --exec conveyor_demo.py; done`
  Then read the `SUMMARY` line in each `demo_log.txt` (target: 0 escaped defects, 0 false rejects, 0 belt stops).
