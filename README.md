# Conveyor Bottle-Cap Inspection & Robotic Reject in NVIDIA Isaac Sim (real-time)

**Author:** Vishal Bhapkar · **Date:** Sep 2026 · **Status:** PoC complete (simulation only). Phases A–D and F are done; Phase E (moving-belt pick) is a stretch goal and only partly done
**Developed on:** Isaac Sim 6.0.1 standalone, Ubuntu · Detailed phase notes: [`PHASES.md`](PHASES.md)
**Hardware:** RTX 5070 Ti (training capped at 250 W) · Ryzen 9 5950X · Ubuntu

---

## 1. Summary

Bottles ride a conveyor under an overhead RGB-D inspection camera. An RT-DETR detector trained **only on synthetic data**
decides whether each bottle has its cap. A depth check independently confirms an open mouth. A bottle with a missing cap is tracked
along the belt. The belt stops when that bottle reaches the robot, and a Franka Panda grasps it by the neck and drops it into a reject bin.
Good bottles carry on to the end of the line.

**Headline results:**
- **Detector, on a held-out set of unseen colours and shapes:** 100% no-cap recall (2102/2102) and 0 false rejects (0/2120).
- **Closed loop, 150 bottles over 3 seeds:** 40/40 defects rejected into the bin, 110/110 good bottles passed, 0 failed picks, pick-position error ≤5.7 mm.

This project followed the pills/cartons pick-and-place PoC and applied its lessons. Training data was generated from the exact
deployment camera pose and pixel scale, and every decision is confirmed in 3D.

---

## 2. End-to-end pipeline: from synthetic data to a working robot

```
 [A] Conveyor cell + GT picking ─▶ [B] SDG (Replicator, 29k frames) ─▶ [C] Train RT-DETR ─▶ ONNX
                                                                                    │
 [F] Real-time video ◀── [D] Closed loop: detect + depth + track + stop + pick ◀────┘
                         [E] (stretch) pick from moving belt
```

### Phase A: Conveyor cell physics and ground-truth picking
Everything was built locally, because NVIDIA's asset server and GitHub were unreachable from the network at the time.
- **`cell_scene.py`:**
  - A belt driven by PhysX surface velocity, with low-friction side guides, frame and legs.
  - A reject bin on a stand, an inspection camera and a presentation camera.
  - Procedural bottles: a lathe-mesh body, shoulder, neck, lip and hollow mouth, an optional cap, and cylinder colliders.
- **Franka:** converted from the URDF bundled with Isaac Sim (`import_franka_local.py`). Shared setup is in `robot_setup.py`: gravity compensation, 70 N fingers, finger friction and the neck-grasp height.
- **Acceptance:** the robot picked bottles using ground-truth positions to prove the physics before any perception was added.
  - **20/20 consecutive picks** into the bin with no drops. Contact error 1.1–1.2 mm.
  - Belt test: 24 bottles, maximum lean 1.1°, none tipped, stop position within 2.4 mm.

**Physics findings:**
1. PhysX surface velocity moves objects at **2.10×** the set value, so a `BELT_GAIN` correction was added.
2. Use timeline sim time, not app-update counts.
3. The URDF's 20 N finger force let a 0.45 kg bottle slip, so it was raised to the real Franka Hand's **70 N**.
4. The belt must stop when a reject reaches the station even while the robot is busy.
5. The arm returns to a joint-space ready pose before every approach.

### Phase B: Synthetic data generation (SDG)
Script: `sdg_generate.py` (Omniverse Replicator), rendered from the **deployment inspection camera** with small pose jitter
(±2–3 cm position, ±1.5° FOV, a few degrees of roll), so the pixel scale matches the live system.

**Domain randomization:**
- Bottle shape, body colour and roughness.
- Procedural label textures.
- Cap size, colour and shape. In 12% of bottles the cap is the same colour as the body.
- Belt, guide, rail and floor materials.
- Light intensity, colour temperature and position, plus a fill light.
- 1–6 bottles per frame, placed only when fully inside the image.

Output is COCO with 2 classes: `0 = bottle_ok`, `1 = bottle_nocap` (balanced ~50/50).

| Split | Frames | bottle_ok | bottle_nocap |
|---|---|---|---|
| train | 24,000 | 24,466 | 25,528 |
| val | 3,000 | 3,107 | 3,132 |
| **test_unseen** | 2,000 | 2,120 | 2,102 |
| **Total** | **29,000** | | |

`test_unseen` is a true generalization test. It uses **body and cap hues 0.42–0.58 that never appear in train**, a conical shoulder
shape and held-out label textures. Generation took 53 min and produced 2.6 GB.

### Phase C: Detector training
- **RT-DETRv2-r18vd**, fine-tuned from the earlier pill/box detector, because ImageNet and COCO weights couldn't be downloaded (GitHub was blocked).
- 16 epochs with no zoom or crop augmentation (fixed deployment scale), on a 250 W GPU cap, in **1 h 47 min**.
- Val **mAP50 0.995**, mAP50:95 0.9945. Exported to `models/bottlecap_best.onnx`. Training config: `training/configs/rtdetrv2_r18vd_bottlecap.yml`. Eval output: `training/eval_bottlecap_best_test_unseen.json`.
- **`eval_bottlecap.py` on test_unseen:** no-cap recall **100% (2102/2102)**, false rejects **0/2120**, 0 missed detections, at **every** reject threshold from 0.1 to 0.7. The targets were ≥99% recall and ≤2% false rejects.
- Even the epoch-0 model had only one error: a pale cap the same colour as the body. That is exactly the case the depth check below was designed for.

### Phase D: Closed loop, stop-and-pick, perception only
`conveyor_demo.py` + `bottle_detect.py` (detector server) + `inspection.py`:
1. Inspect every 0.25 s of sim time.
2. RT-DETR returns P(ok) and P(no-cap) for each bottle.
3. **Depth check:** look for an open mouth, a hole ≥ 8 mm below the rim at the bottle axis.
4. Track each bottle in belt coordinates using belt odometry. Each bottle is judged across ~15 views.
5. **Reject only if both agree:** mean P(no-cap) > 0.5 AND a depth majority shows an open mouth.
6. Stop the belt when the reject reaches the pick station. The Franka grasps the neck at the **perceived** position and drops the bottle into the bin.

Ground truth is used only for scoring.

| Seed | Bottles | Defects → bin | Escaped | False rejects | Pick failures | Max pick-position error |
|---|---|---|---|---|---|---|
| 1 | 50 | 13/13 | 0 | 0 | 0 | 5.6 mm |
| 2 | 50 | 10/10 | 0 | 0 | 0 | 5.7 mm |
| 3 | 50 | 17/17 | 0 | 0 | 0 | 5.7 mm |
| **Total** | **150** | **40/40** | **0** | **0** | **0** | |

The depth check agreed with the detector on every track. Every defect showed an open mouth in all of its 14–17 views, and capped bottles showed it in 0 views.

### Phase F: Real-time video
`./run_conveyor_demo.sh --record [--seed N]` opens Isaac Sim with the presentation camera in the viewport and an
**"Inspection" panel** docked on the right (`live_panel.py`). The panel shows:
- the live inspection image with per-bottle verdicts;
- the belt state;
- inspected, passed and rejected counts;
- a ground-truth audit;
- the decision log.

The surrounding room (racks, safety lines, camera gantry) is built locally. ffmpeg records the window in real time.

- Recorded take (seed 1, 24 bottles): **7/7 defects rejected, 17/17 good passed**, 0 missed, 0 false rejects.
- In the GUI the sim runs slower than wall-clock time (137 s of sim time over 369 s, ~18–30 FPS), because it is rendering while recording.

### Phase E: Pick from a moving belt (stretch, partly done)
`DEMO_MOVING=1`: the belt runs continuously at 0.10 m/s with a new bottle every 2.4 s. For each reject, the arm:
1. follows the predicted position (camera track + belt odometry, with an integral lag correction);
2. descends, closes and lifts while following;
3. drops the bottle in the bin.

During a burst of rejects the belt slows to 0.03 m/s instead of stopping.
- 12-bottle debug run: **3/3 back-to-back rejects picked**, grasp error 2–5 mm, cycle 6.5–6.9 s, the belt never stopped.
- The 3 × 50-bottle acceptance run was interrupted by a workstation restart and has **not** been completed.

---

## 3. Key files

| File | Role |
|---|---|
| `cell_scene.py` | Conveyor cell, procedural bottles, cameras, room |
| `robot_setup.py`, `import_franka_local.py` | Franka from the local URDF, gripper force and friction, grasp offsets |
| `phaseA_belt_test.py`, `phaseA_pick_test.py`, `grasp_debug.py` | Phase A physics acceptance |
| `sdg_generate.py` | Phase B Replicator dataset |
| `eval_bottlecap.py` | Phase C evaluation on test_unseen |
| `bottle_detect.py`, `inspection.py`, `conveyor_demo.py` | Phase D/E closed loop |
| `live_panel.py`, `run_conveyor_demo.sh` | Phase F live GUI panel + recording |

---

## 4. Engineering decisions

- **Lessons carried over from the pill project:** train at the same camera pose and pixel scale as deployment, confirm every decision in 3D, and evaluate on shapes and colours never seen in training.
- **Two independent checks (vision AND depth):** they guard against look-alikes, such as a cap the same colour as the bottle. A false reject wastes product, and a missed defect ships it, so both checks are required.
- **Neck grasp with the parallel gripper:** a vacuum cup cannot seal on an open neck, and a body grasp collides with the shoulder.
- **Stop-and-pick first, moving pick second:** this proved correctness before adding tracking dynamics.

---

## 5. Limitations

- **Simulation only.** No real camera, belt or robot yet. Sim depth is perfect, so real sensor noise will affect the open-mouth check.
- Opaque (HDPE-like) bottles only. Clear PET and loose or tilted caps are future work.
- Phase E moving-belt acceptance (3 × 50 bottles) is not complete.
- The cell is procedural. NVIDIA's warehouse assets were unavailable during the build.

---

## 6. Setup notes

- **Isaac Sim 6.0.1 standalone** is required. Scripts run through the kit binary, for example `./kit/kit apps/isaacsim.exp.base.python.kit --no-window --exec conveyor_demo.py`. `run_conveyor_demo.sh --record` runs the live GUI with recording.
- This is a PoC snapshot: **absolute paths from the original workstation are hard-coded** (Isaac Sim install, repo root, the detector's Python venv, output folders). Edit them for your machine.
- `assets/panda_arm_hand/`: the Franka Panda converted from the URDF bundled with Isaac Sim's URDF importer (`import_franka_local.py`). The whole cell is procedural, so no asset server is needed.
- The detector server (`bottle_detect.py`) needs `onnxruntime-gpu`, `numpy` and `opencv-python`.
- Training: clone [RT-DETR](https://github.com/lyuwenyu/RT-DETR) (`rtdetrv2_pytorch`) and put the config in `configs/rtdetrv2/`. The 29k-frame dataset (2.6 GB) is not included. Regenerate it with `sdg_generate.py`.
- `models/` uses **Git LFS**. Run `git lfs install` before cloning.

## 7. Demo videos

- Full 6-minute **unedited real-time** Isaac Sim screen recording (24 bottles). The cell is on the left and the live Inspection panel is on the right.
- 88-second cut: real-time open and close, with the middle sped up and labelled on screen.

**Stack:** Isaac Sim 6.0.1 · Omniverse Replicator · PhysX · RT-DETRv2 · ONNX Runtime (CUDA) · Python

**Next steps:** finish the moving-belt acceptance run, add clear-bottle and loose-cap defect classes, and test on real hardware (RealSense + belt) to measure the sim-to-real gap.
