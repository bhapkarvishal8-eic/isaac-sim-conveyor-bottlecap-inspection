"""Phase A pick test: ground-truth-driven reject picks from the conveyor (no perception yet).

The belt stops each uncapped bottle at PICK_Y. The Franka (local URDF import, parallel gripper)
grasps the neck, lifts it and drops it into the reject bin, then the belt restarts. Capped bottles
run to the exit. Passes when TARGET_PICKS consecutive picks land in the bin with no drops, no
tipping, and nothing falls off the belt.
Run: kit --no-window --exec phaseA_pick_test.py   (env: TEST_SEED, TARGET_PICKS, TEST_OUT, TEST_FRAMES=1)
"""
import json
import math
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, "/home/vishal/poc/conveyor_inspection")
import omni.kit.app
import omni.timeline
import omni.usd
import isaacsim.core.experimental.utils.app as app_utils

from isaacsim.core.simulation_manager import SimulationManager
import omni.replicator.core as rep
from PIL import Image
from pxr import PhysxSchema, Usd, UsdPhysics

import cell_scene as C
import robot_setup as R

SEED = int(os.environ.get("TEST_SEED", "0"))
TARGET_PICKS = int(os.environ.get("TARGET_PICKS", "20"))
OUT = os.environ.get("TEST_OUT", "/home/vishal/poc/logs/conveyor/phaseA_pick_test")
FRAMES = os.environ.get("TEST_FRAMES", "1") == "1"
os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
UNCAPPED_FRACTION = 0.75
SPAWN_EVERY_S = 1.8
DOWN = R.DOWN
READY_Q = R.READY_Q
LOG = open(os.path.join(OUT, "log.txt"), "w")


def log(m):
    print("[PICK] " + m, flush=True)
    LOG.write(m + "\n")
    LOG.flush()


def quat_angle(a, b):
    return 2 * math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b))))))


try:
    omni.usd.get_context().new_stage()
    app_utils.update_app(steps=2)
    SimulationManager.set_physics_sim_device("cpu")
    scene = C.build(seed=SEED)
    stage = omni.usd.get_context().get_stage()
    rng = scene["rng"]
    R.add_franka(stage)
    app_utils.update_app(steps=2)
    robot = R.make_controller()

    pres = None
    if FRAMES:
        rp = rep.create.render_product(scene["presentation_camera"], (1280, 720))
        pres = rep.AnnotatorRegistry.get_annotator("rgb")
        pres.attach([rp])

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app_utils.update_app(steps=5)
    t0 = timeline.get_current_time()
    q0 = np.concatenate([READY_Q, [0.04, 0.04]])
    robot.set_dof_positions(q0)
    robot.set_dof_position_targets(q0)

    st = {"speed": 0.0, "target": C.BELT_SPEED, "t": 0.0, "frame": 0, "step": 0,
          "next_spawn": 0.0, "spawned": 0, "uncapped_spawned": 0, "queued": None}
    bottles = {}

    def sim_step(n=1):
        """Advance the sim; runs belt ramp, spawning, exit removal and bottle monitoring."""
        for _ in range(n):
            app_utils.update_app(steps=1)
            t_now = timeline.get_current_time() - t0
            dt, st["t"] = t_now - st["t"], t_now
            st["step"] += 1
            dv = C.BELT_SPEED * dt / C.BELT_RAMP_S
            s = st["speed"]
            st["speed"] = min(st["target"], s + dv) if s < st["target"] else max(st["target"], s - dv)
            C.set_belt_speed(stage, st["speed"])
            if (st["uncapped_spawned"] < TARGET_PICKS and st["t"] >= st["next_spawn"]
                    and st["speed"] > 0.9 * C.BELT_SPEED):
                capped = rng.random() >= UNCAPPED_FRACTION
                spec = C.random_bottle_spec(rng, capped)
                path = f"/World/Bottles/b{st['spawned']:03d}"
                C.spawn_bottle(stage, path, spec, C.BELT_X + rng.uniform(-0.02, 0.02), C.SPAWN_Y)
                bottles[path] = {"spec": spec, "state": "on_belt", "max_tilt": 0.0}
                st["spawned"] += 1
                st["uncapped_spawned"] += 0 if capped else 1
                st["next_spawn"] = st["t"] + SPAWN_EVERY_S
            for path, b in bottles.items():
                if b["state"] != "on_belt":
                    continue
                pos, tilt = C.bottle_pose(stage, path)
                b["max_tilt"] = max(b["max_tilt"], tilt)
                if pos[2] < -0.05:
                    b["state"] = "FELL"
                    log(f"{path} FELL off the belt")
                elif pos[1] > C.EXIT_Y:
                    b["state"] = "passed" if b["spec"]["capped"] else "ESCAPED"
                    stage.GetPrimAtPath(path).SetActive(False)
                elif (not b["spec"]["capped"]) and st["queued"] is None \
                        and pos[1] >= C.PICK_Y - C.BELT_SPEED * C.BELT_RAMP_S / 2:
                    # a reject reached the pick station: stop the belt now, even if the robot is busy
                    st["queued"] = path
                    st["target"] = 0.0
                    b["state"] = "queued"
            if pres is not None and st["step"] % 6 == 0:
                d = pres.get_data()
                if getattr(d, "ndim", 0) == 3:
                    Image.fromarray(np.asarray(d)[:, :, :3]).save(
                        os.path.join(OUT, "frames", f"f_{st['frame']:05d}.jpg"), quality=85)
                    st["frame"] += 1

    def ee():
        _, p, q = robot.get_current_state()
        return p[0].astype(float), q[0].astype(float)

    def move_joints(q_goal, seconds=1.5):
        qs = robot.get_dof_positions().numpy()[0][:7].astype(float)
        t_start = st["t"]
        while st["t"] - t_start < seconds:
            s = min(1.0, (st["t"] - t_start) / seconds)
            s = 0.5 - 0.5 * math.cos(math.pi * s)
            robot.set_dof_position_targets((qs + s * (q_goal - qs)).reshape(1, -1), dof_indices=list(range(7)))
            sim_step()
        sim_step(10)

    def move_to(goal, speed=0.30, tol=0.003, settle=120):
        p0, _ = ee()
        goal = np.asarray(goal, float)
        dur = max(0.25, float(np.linalg.norm(goal - p0)) / speed)
        t_start = st["t"]
        while st["t"] - t_start < dur:
            s = min(1.0, (st["t"] - t_start) / dur)
            s = 0.5 - 0.5 * math.cos(math.pi * s)
            robot.set_end_effector_pose(position=p0 + s * (goal - p0), orientation=DOWN)
            sim_step()
        ok = 0
        for _ in range(settle):
            robot.set_end_effector_pose(position=goal, orientation=DOWN)
            sim_step()
            p, q = ee()
            ok = ok + 1 if (np.linalg.norm(p - goal) < tol and quat_angle(q, DOWN) < 2) else 0
            if ok >= 6:
                break
        return float(np.linalg.norm(ee()[0] - goal))

    def gripper(width, steps=40):
        robot.set_dof_position_targets(np.array([[width / 2, width / 2]]), dof_indices=[7, 8])
        sim_step(steps)
        return float(np.sum(robot.get_dof_positions().numpy()[0][7:9]))

    (bx0, by0, bz0), (bx1, by1, bz1) = scene["reject_bin_range"]
    bin_c = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2])
    picks, consecutive, best_streak, failures = [], 0, 0, 0
    move_joints(READY_Q, 1.0)
    gripper(0.08, 20)
    HAND_TO_TIP = R.hand_to_fingertip(stage, ee()[0][2])
    log(f"robot ready, ee at {np.round(ee()[0], 3)}, hand->fingertip {1000 * HAND_TO_TIP:.1f} mm")

    while len(picks) < TARGET_PICKS and failures < 5 and st["t"] < 1500:
        # wait for the next uncapped bottle to reach the stop trigger (ground truth in Phase A)
        cand = st["queued"]
        if cand is None:
            sim_step()
            continue
        while st["speed"] > 0.0:
            sim_step()
        sim_step(10)
        b = bottles[cand]
        b["state"] = "picking"
        spec = b["spec"]
        pos, tilt = C.bottle_pose(stage, cand)
        grasp_z = R.neck_grasp_hand_z(pos[2], spec, HAND_TO_TIP)
        gx, gy = pos[0], pos[1]
        move_joints(READY_Q, 1.2)          # start every approach from the same joint configuration
        gripper(0.08, 10)
        move_to([gx, gy, grasp_z + 0.12])
        err = move_to([gx, gy, grasp_z], speed=0.12, tol=0.002)
        width = gripper(0.0, 45)
        held_ok = width > 0.6 * 2 * spec["neck_r"]
        move_to([gx, gy, grasp_z + 0.16], speed=0.20)
        lifted = C.bottle_pose(stage, cand)[0][2] > pos[2] + 0.10
        # carry to the reject bin; restart the belt as soon as the bottle is clear of it
        st["queued"] = None
        st["target"] = C.BELT_SPEED
        move_to([bin_c[0], bin_c[1], 0.40])
        move_to([bin_c[0], bin_c[1], 0.32], speed=0.15)
        gripper(0.08, 40)
        sim_step(40)
        fpos, ftilt = C.bottle_pose(stage, cand)
        in_bin = bx0 < fpos[0] < bx1 and by0 < fpos[1] < by1 and fpos[2] < bz1
        move_to([bin_c[0], bin_c[1], 0.42])
        move_joints(READY_Q, 1.2)
        ok = held_ok and lifted and in_bin
        if not in_bin:
            stage.GetPrimAtPath(cand).SetActive(False)   # clear a failed bottle so the test continues
        b["state"] = "in_bin" if in_bin else "DROPPED"
        consecutive = consecutive + 1 if ok else 0
        best_streak = max(best_streak, consecutive)
        failures += 0 if ok else 1
        picks.append({"bottle": cand, "ok": bool(ok), "contact_err_mm": round(1000 * err, 1),
                      "grip_width_mm": round(1000 * width, 1), "neck_d_mm": round(2000 * spec["neck_r"], 1),
                      "lifted": bool(lifted), "in_bin": bool(in_bin)})
        log(f"pick {len(picks)}: {cand} {'OK' if ok else 'FAIL'}  contact err {1000 * err:.1f} mm, "
            f"grip {1000 * width:.1f} mm on {2000 * spec['neck_r']:.1f} mm neck, lifted={lifted}, in_bin={in_bin}"
            + ("" if in_bin else f"  final pos {np.round(fpos, 3)}"))

    states = [b["state"] for b in bottles.values()]
    summary = {"seed": SEED, "picks_attempted": len(picks), "picks_ok": int(sum(p["ok"] for p in picks)),
               "max_consecutive_ok": best_streak, "passed_capped": states.count("passed"),
               "escaped_uncapped": states.count("ESCAPED"), "fell_off_belt": states.count("FELL"),
               "dropped": states.count("DROPPED"),
               "max_tilt_on_belt_deg": round(max((b["max_tilt"] for b in bottles.values()), default=0), 2),
               "sim_time_s": round(st["t"], 1)}
    json.dump({"summary": summary, "picks": picks}, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    log("SUMMARY " + json.dumps(summary))
except Exception:
    log("FAILED\n" + traceback.format_exc())
finally:
    LOG.close()
    omni.kit.app.get_app().post_quit()
