# Next phase: conveyor bottle inspection and reject (approved 2026-09-27)

Bottles travel along a conveyor in the warehouse cell. An overhead RGB-D inspection camera classifies each bottle. A bottle
with a **missing cap** is picked off the belt by the Franka and dropped into a reject bin. Capped bottles continue to the
end of the line.

## Decisions (user approved the recommendations)
1. Opaque (HDPE-like) bottles first. Clear PET is a later extension.
2. Defect class: missing cap only (loose or tilted cap may come later).
3. Stop-and-pick first (Phase D). Picking from a moving belt is a stretch goal (Phase E).
4. Franka parallel gripper grasping the bottle neck (~2.5-3 cm). The vacuum cup cannot seal on an open neck.

## Lessons from the pill work
- Generate training data from the **same camera pose and pixel scale** as the demo; the pill scale mismatch caused most of the detection trouble.
- Confirm every decision in 3D: a missing cap makes the bottle ~1.5-2 cm shorter, which depth can measure. Reject only when the detector and depth agree.
- Evaluate on bottle shapes and colours never seen in training, not only on new seeds.

## Phases

| Phase | Work | Done when |
|---|---|---|
| A. Conveyor cell (no training) | warehouse + conveyor, inspection camera, reject bin, end-of-line stop; generated bottles (body, neck, separate cap); Franka hand grasps the neck; picking driven by simulator ground truth to prove the physics | 20 consecutive picks with no drops, no tipping or falling through the belt |
| B. Synthetic dataset | Replicator from the inspection camera. Randomize shapes, sizes, colours, labels, cap colours, belt material, lighting, spacing, arm occlusion and distractors. ~50/50 cap/no-cap, 20-30k frames | stats logged; labels checked on samples; pixel size matches the demo camera |
| C. Training | RT-DETR r18 (existing venv), COCO init; classes `bottle_ok`, `bottle_nocap`; GPU capped at 250 W | on unseen shapes: no-cap recall >= 99%, false rejects <= 2% |
| D. Closed loop, stop-and-pick | detect -> depth height check -> track with belt speed -> stop in the pick zone -> pick -> reject bin | 50+ bottles, 0 missed defects, 0 false rejects, several seeds |
| E. Moving-belt pick (stretch) | arm follows the bottle while the belt keeps running | same as D with the belt never stopping |
| F. Video and write-up | real-time recording with a live panel (OK/REJECT, counts) plus a 90 s cut | videos + README |

## Risks
- Tall bottles tipping on the belt. Mitigation: realistic mass and friction, CCD, Phase A checks.
- Sim-to-real gap for the detector (out of scope; the unseen-shape test set gives an early signal).
- GPU power: set the 250 W cap (`sudo nvidia-smi -pl 250`) before training.

Code: ``
