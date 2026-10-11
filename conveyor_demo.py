"""Phase D: closed-loop conveyor inspection and reject (stop-and-pick), perception only.

Pipeline (ground truth is read ONLY for scoring and the log):
  1. INSPECT  the inspection camera renders RGB + depth every INSPECT_PERIOD_S of sim time; the RT-DETR
              model (bottle_detect.py server) returns one entry per bottle with P(ok), P(no-cap)
  2. VERIFY   depth: localize the bottle axis and check for an open mouth (inspection.py)
  3. TRACK    each bottle is tracked in belt coordinates (s = y - belt odometry); votes accumulate
              over the frames in which it is fully in view
  4. DECIDE   when the track leaves the decision window: REJECT if mean P(no-cap) > 0.5 AND the depth
              check saw an open mouth in the majority of views; otherwise PASS
  5. PICK     a REJECT track is followed by odometry; the belt stops when it reaches the pick station,
              the Franka grasps the neck at the perceived (x, predicted y), drops it in the reject bin
Metrics: correct rejects, escaped defects (no-cap reached the exit), false rejects (capped bottle picked),
pick failures.

Env: DEMO_LIVE (0 = headless), DEMO_N (bottles), DEMO_SEED, DEMO_OUT, DEMO_MODEL, DEMO_FRAMES (1 = save
presentation frames), DEMO_NOCAP_P (defect rate), DEMO_QUIT_WHEN_DONE
"""
import json
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = "/home/vishal/poc/conveyor_inspection"
sys.path.insert(0, HERE)
import omni.kit.app
import omni.timeline
import omni.usd
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
import omni.replicator.core as rep
from PIL import Image, ImageDraw, ImageFont

import cell_scene as C
import inspection as I
import robot_setup as R

LIVE = os.environ.get("DEMO_LIVE", "0") == "1"
N_BOTTLES = int(os.environ.get("DEMO_N", "30"))
SEED = int(os.environ.get("DEMO_SEED", "1"))
OUT = os.environ.get("DEMO_OUT", "/home/vishal/poc/logs/conveyor/phaseD_latest")
MODEL = os.environ.get("DEMO_MODEL", f"{HERE}/models/bottlecap_best.onnx")
FRAMES = os.environ.get("DEMO_FRAMES", "0") == "1"
NOCAP_P = float(os.environ.get("DEMO_NOCAP_P", "0.3"))
QUIT_WHEN_DONE = os.environ.get("DEMO_QUIT_WHEN_DONE", "0" if LIVE else "1") == "1"
VENV_PY = "/home/vishal/poc/stage1_perception_arm/phase4_perception_training/rtdetr_venv/bin/python"

SPAWN_EVERY_S = 1.8
# Phase E: pick from the moving belt (the belt never stops)
MOVING = os.environ.get("DEMO_MOVING", "0") == "1"
if MOVING:
    C.BELT_SPEED = float(os.environ.get("DEMO_BELT_SPEED", "0.10"))
    SPAWN_EVERY_S = float(os.environ.get("DEMO_SPAWN_S", "2.4"))
Y_GRASP_MAX = 0.28        # the grasp (fingers closed) must happen before the bottle passes this y
SLOW_Y = -0.12             # a queued reject past this point while the arm is busy -> slow the belt
SLOW_SPEED = 0.03

INSPECT_PERIOD_S = 0.25
DECIDE_Y = C.INSPECT_Y + 0.28      # a track is decided when its predicted y passes this
MIN_VIEWS = 2
W, H = C.INSPECT_CAM_RES

os.makedirs(os.path.join(OUT, "inspect"), exist_ok=True)
if FRAMES:
    os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
LOG = open(os.path.join(OUT, "demo_log.txt"), "w")


def log(m):
    line = f"[CONV t={time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


class Track:
    _next = 0

    def __init__(self, s, x):
        self.id = Track._next
        Track._next += 1
        self.s, self.x = s, x          # belt coordinate (y at odometry 0) and lateral position
        self.p_nocap, self.p_ok, self.mouth = [], [], []
        self.top_z = []
        self.state = "tracking"        # tracking -> PASS | REJECT -> queued -> picked / pick_failed
        self.gt_path = None

    def add(self, s, x, b, loc):
        n = len(self.p_nocap)
        self.s = (self.s * n + s) / (n + 1)
        self.x = (self.x * n + x) / (n + 1)
        self.p_nocap.append(b["p_nocap"])
        self.p_ok.append(b["p_ok"])
        self.mouth.append(loc["open_mouth"])
        self.top_z.append(loc["top_z"])

    def decide(self):
        if len(self.p_nocap) < MIN_VIEWS:
            return None
        det = float(np.mean(self.p_nocap)) > 0.5 and float(np.mean(self.p_nocap)) > float(np.mean(self.p_ok))
        depth = sum(self.mouth) > len(self.mouth) / 2
        self.state = "REJECT" if (det and depth) else "PASS"
        return self.state


def main():
    omni.usd.get_context().new_stage()
    app_utils.update_app(steps=2)
    SimulationManager.set_physics_sim_device("cpu")
    scene = C.build(seed=SEED, room=LIVE or os.environ.get("DEMO_ROOM", "0") == "1")
    stage = omni.usd.get_context().get_stage()
    rng = scene["rng"]
    R.add_franka(stage)
    app_utils.update_app(steps=2)
    robot = R.make_controller()

    rp = rep.create.render_product(scene["inspection_camera"], (W, H))
    ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    ann_depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    ann_rgb.attach([rp])
    ann_depth.attach([rp])
    pres = None
    if FRAMES:
        rp2 = rep.create.render_product(scene["presentation_camera"], (1280, 720))
        pres = rep.AnnotatorRegistry.get_annotator("rgb")
        pres.attach([rp2])
    K = I.intrinsics(W, H, C.INSPECT_CAM_HFOV)
    from pxr import Usd, UsdGeom
    M = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(scene["inspection_camera"])).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()))

    panel = None
    if LIVE:
        from live_panel import LivePanel
        panel = LivePanel(scene["presentation_camera"])
    det_proc = subprocess.Popen([VENV_PY, os.path.join(HERE, "bottle_detect.py"), MODEL], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=HERE,
                                env={"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/home/vishal")})
    log(f"detector: {det_proc.stdout.readline().strip()}  model={MODEL}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app_utils.update_app(steps=5)
    t0 = timeline.get_current_time()
    q0 = np.concatenate([R.READY_Q, [0.04, 0.04]])
    robot.set_dof_positions(q0)
    robot.set_dof_position_targets(q0)

    st = {"speed": 0.0, "target": C.BELT_SPEED, "t": 0.0, "odo": 0.0, "step": 0, "frame": 0,
          "next_spawn": 0.0, "spawned": 0, "next_inspect": 0.0, "queued": None, "n_inspect": 0, "pickq": []}
    bottles = {}      # GT (scoring only)
    tracks = []

    def gt_nearest(x, y, max_d=0.05):
        best, bd = None, max_d
        for p, b in bottles.items():
            if b["state"] != "on_belt":
                continue
            pos, _ = C.bottle_pose(stage, p)
            d = math.hypot(pos[0] - x, pos[1] - y)
            if d < bd:
                best, bd = p, d
        return best, bd

    def inspect():
        rgb = ann_rgb.get_data()
        depth = ann_depth.get_data()
        if getattr(rgb, "ndim", 0) != 3 or depth is None:
            return
        depth = np.asarray(depth).squeeze()
        k = st["n_inspect"]
        st["n_inspect"] += 1
        img = os.path.join(OUT, "inspect", "cur.png")
        Image.fromarray(np.asarray(rgb)[:, :, :3]).save(img)
        det_proc.stdin.write(f"{img} {OUT}/inspect/cur.json 0.2\n")
        det_proc.stdin.flush()
        if not det_proc.stdout.readline().startswith("OK"):
            return
        dets = json.load(open(f"{OUT}/inspect/cur.json"))["bottles"]
        for b in dets:
            x1, y1, x2, y2 = b["xyxy"]
            if x1 < 5 or x2 > W - 5 or y1 < 5 or y2 > H - 5:        # decide only on fully visible bottles
                continue
            loc = I.localize_bottle(b["xyxy"], depth, K, M)
            if loc is None:
                continue
            s = loc["y"] - st["odo"]
            tr = min(tracks, key=lambda t: abs(t.s - s) + abs(t.x - loc["x"]), default=None)
            if tr is None or tr.state != "tracking" or abs(tr.s - s) > 0.035 or abs(tr.x - loc["x"]) > 0.035:
                if tr is not None and tr.state != "tracking" and abs(tr.s - s) <= 0.035:
                    continue               # already decided
                tr = Track(s, loc["x"])
                tracks.append(tr)
                tr.gt_path, err = gt_nearest(loc["x"], loc["y"])
                tr.loc_err_mm = None if tr.gt_path is None else round(1000 * err, 1)
            tr.add(s, loc["x"], b, loc)
            b["_track"], b["_loc"] = tr, loc
        if panel is not None:
            ov = []
            for b in dets:
                tr, loc = b.get("_track"), b.get("_loc")
                bad_rgb = b["p_nocap"] > b["p_ok"]
                if loc is None:
                    ov.append((b["xyxy"], "entering view", None))
                    continue
                bad = bad_rgb and loc["open_mouth"]
                txt = (f"#{tr.id} NO CAP {b['p_nocap']:.2f} | mouth open" if bad else
                       f"#{tr.id} OK {b['p_ok']:.2f}" if not bad_rgb else f"#{tr.id} ? rgb/depth disagree")
                ov.append((b["xyxy"], txt, bad))
            panel.set_image(np.asarray(rgb)[:, :, :3], ov)
        if k % 20 == 0:
            vis = Image.fromarray(np.asarray(rgb)[:, :, :3])
            d = ImageDraw.Draw(vis)
            for b in dets:
                col = (230, 40, 40) if b["p_nocap"] > b["p_ok"] else (40, 200, 60)
                d.rectangle(b["xyxy"], outline=col, width=3)
                d.text((b["xyxy"][0] + 3, b["xyxy"][1] + 3), f"ok {b['p_ok']:.2f} / nocap {b['p_nocap']:.2f}", fill=col)
            vis.save(os.path.join(OUT, "inspect", f"insp_{k:05d}.jpg"), quality=85)

    def sim_step(n=1):
        for _ in range(n):
            app_utils.update_app(steps=1)
            t_now = timeline.get_current_time() - t0
            dt, st["t"] = t_now - st["t"], t_now
            st["step"] += 1
            st["odo"] += st["speed"] * dt           # belt odometry ("encoder")
            if MOVING:
                # burst handling: never stop the belt; slow it while the arm is busy (and not following a bottle)
                # if a queued reject would otherwise leave the pick window
                slow = st.get("busy") and (len(st["pickq"]) > 0 or st.get("hold_slow"))
                if slow and st["target"] != SLOW_SPEED:
                    st["slowdowns"] = st.get("slowdowns", 0) + 1
                st["target"] = SLOW_SPEED if slow else C.BELT_SPEED
            dv = C.BELT_SPEED * dt / C.BELT_RAMP_S
            s = st["speed"]
            st["speed"] = min(st["target"], s + dv) if s < st["target"] else max(st["target"], s - dv)
            C.set_belt_speed(stage, st["speed"])
            if st["t"] > 3.0:
                st["min_speed"] = min(st.get("min_speed", 9.0), st["speed"])
            # spawn (random defects)
            if st["spawned"] < N_BOTTLES and st["t"] >= st["next_spawn"] and st["speed"] > 0.9 * C.BELT_SPEED:
                capped = rng.random() >= NOCAP_P
                spec = C.random_bottle_spec(rng, capped)
                path = f"/World/Bottles/b{st['spawned']:03d}"
                C.spawn_bottle(stage, path, spec, C.BELT_X + rng.uniform(-0.02, 0.02), C.SPAWN_Y)
                bottles[path] = {"spec": spec, "state": "on_belt", "capped": capped}
                st["spawned"] += 1
                st["next_spawn"] = st["t"] + SPAWN_EVERY_S
            # exits (GT bookkeeping)
            for path, b in bottles.items():
                if b["state"] == "on_belt":
                    pos, _ = C.bottle_pose(stage, path)
                    if pos[1] > C.EXIT_Y:
                        b["state"] = "passed_exit"
                        stage.GetPrimAtPath(path).SetActive(False)
                    elif pos[2] < -0.05:
                        b["state"] = "fell"
            # perception
            if st["t"] >= st["next_inspect"]:
                st["next_inspect"] = st["t"] + INSPECT_PERIOD_S
                inspect()
            # decisions + stop trigger
            for tr in tracks:
                y_pred = tr.s + st["odo"]
                if tr.state == "tracking" and y_pred > DECIDE_Y:
                    res = tr.decide()
                    if res is None:
                        tr.state = "PASS_undecided"
                    gt = bottles.get(tr.gt_path, {})
                    if panel is not None:
                        panel.add_decision(
                            f"bottle #{tr.id}: " + ("NO CAP -> REJECT" if tr.state == "REJECT" else "cap OK -> pass")
                            + f"   (P(no cap) {np.mean(tr.p_nocap or [0]):.2f}, open mouth in {sum(tr.mouth)}/{len(tr.mouth)} views)")
                    log(f"track {tr.id}: {tr.state}  views={len(tr.p_nocap)} P(nocap)={np.mean(tr.p_nocap or [0]):.2f} "
                        f"open_mouth={sum(tr.mouth)}/{len(tr.mouth)}  | GT {tr.gt_path} "
                        f"{'capped' if gt.get('capped') else 'NO CAP' if gt else '?'}  loc err {tr.loc_err_mm} mm")
                if MOVING and tr.state == "REJECT":
                    tr.state = "queued"
                    st["pickq"].append(tr)
                if tr.state == "REJECT" and st["queued"] is None and y_pred >= C.PICK_Y - C.BELT_SPEED * C.BELT_RAMP_S / 2:
                    tr.state = "queued"
                    st["queued"] = tr
                    st["target"] = 0.0
            if panel is not None and st["step"] % 6 == 0:
                q = st["queued"]
                if st["speed"] >= C.BELT_SPEED * 0.99:
                    belt = f"BELT RUNNING  {C.BELT_SPEED:.2f} m/s"
                elif st["target"] == 0.0:
                    belt = "BELT STOPPED  -  picking reject" if st["speed"] == 0 else "BELT STOPPING  -  reject at pick station"
                else:
                    belt = "BELT STARTING"
                detail = (f"reject #{q.id} at pick station, grasp at ({q.x:.3f}, {q.s + st['odo']:.3f}) m from the camera track"
                          if q is not None else "every bottle is inspected in ~14 views as it passes the camera")
                dec = [t for t in tracks if t.state not in ("tracking",)]
                n_pass = sum(t.state.startswith("PASS") for t in dec)
                n_rej = sum(t.state == "picked" for t in dec)
                esc = sum((not b["capped"]) and b["state"] == "passed_exit" for b in bottles.values())
                fr = sum(b["capped"] and b["state"] in ("rejected_bin", "pick_failed") for b in bottles.values())
                panel.update(belt, detail, len(dec), n_pass, n_rej,
                             f"audit vs simulator ground truth:  missed defects {esc}   |   false rejects {fr}")
            if pres is not None and st["step"] % 4 == 0:
                d = pres.get_data()
                if getattr(d, "ndim", 0) == 3:
                    Image.fromarray(np.asarray(d)[:, :, :3]).save(os.path.join(OUT, "frames", f"f_{st['frame']:05d}.jpg"),
                                                                  quality=85)
                    st["frame"] += 1

    def ee():
        _, p, q = robot.get_current_state()
        return p[0].astype(float), q[0].astype(float)

    def move_joints(q_goal, seconds=1.2):
        qs = robot.get_dof_positions().numpy()[0][:7].astype(float)
        ts = st["t"]
        while st["t"] - ts < seconds:
            s = 0.5 - 0.5 * math.cos(math.pi * min(1.0, (st["t"] - ts) / seconds))
            robot.set_dof_position_targets((qs + s * (q_goal - qs)).reshape(1, -1), dof_indices=list(range(7)))
            sim_step()
        sim_step(8)

    def move_to(goal, speed=0.30, tol=0.003, settle=120):
        p0, _ = ee()
        goal = np.asarray(goal, float)
        dur = max(0.25, float(np.linalg.norm(goal - p0)) / speed)
        ts = st["t"]
        while st["t"] - ts < dur:
            s = 0.5 - 0.5 * math.cos(math.pi * min(1.0, (st["t"] - ts) / dur))
            robot.set_end_effector_pose(position=p0 + s * (goal - p0), orientation=R.DOWN)
            sim_step()
        ok = 0
        for _ in range(settle):
            robot.set_end_effector_pose(position=goal, orientation=R.DOWN)
            sim_step()
            ok = ok + 1 if np.linalg.norm(ee()[0] - goal) < tol else 0
            if ok >= 6:
                break
        return float(np.linalg.norm(ee()[0] - goal))

    def gripper(width, steps=40):
        robot.set_dof_position_targets(np.array([[width / 2, width / 2]]), dof_indices=[7, 8])
        sim_step(steps)
        return float(np.sum(robot.get_dof_positions().numpy()[0][7:9]))

    move_joints(R.READY_Q, 1.0)
    gripper(0.08, 20)
    hand_to_tip = R.hand_to_fingertip(stage, ee()[0][2])
    (bx0, by0, _), (bx1, by1, bz1) = scene["reject_bin_range"]
    bin_c = ((bx0 + bx1) / 2, (by0 + by1) / 2)
    log(f"ready: seed={SEED} bottles={N_BOTTLES} defect rate={NOCAP_P} hand->tip {1000 * hand_to_tip:.1f} mm")

    def finish_drop(tr, width, extra):
        """Carry the held bottle to the reject bin, release, score (GT only for the log), go home."""
        if MOVING:     # faster cycle: drop from above the rim, loose tolerances away from the belt
            move_to([bin_c[0], bin_c[1], 0.36], speed=0.55, tol=0.02, settle=20)
            gripper(0.08, 22)
            sim_step(12)
        else:
            move_to([bin_c[0], bin_c[1], 0.40])
            move_to([bin_c[0], bin_c[1], 0.32], speed=0.15)
            gripper(0.08, 40)
            sim_step(30)
        in_bin = False
        if tr.gt_path:
            fpos = C.bottle_pose(stage, tr.gt_path)[0]
            in_bin = bool(bx0 < fpos[0] < bx1 and by0 < fpos[1] < by1 and fpos[2] < bz1)
            bottles[tr.gt_path]["state"] = "rejected_bin" if in_bin else "pick_failed"
            if not in_bin:
                stage.GetPrimAtPath(tr.gt_path).SetActive(False)
        tr.state = "picked" if in_bin else "pick_failed"
        if MOVING:
            move_joints(R.READY_Q, 0.8)
        else:
            move_to([bin_c[0], bin_c[1], 0.42])
            move_joints(R.READY_Q, 0.9)
        gt_capped = bottles.get(tr.gt_path, {}).get("capped")
        rec = {"track": tr.id, "gt": tr.gt_path, "gt_capped": gt_capped, "in_bin": in_bin, "grip_mm": round(1000 * width, 1)}
        rec.update(extra)
        return rec

    def pick_moving(tr):
        """Grasp a bottle while the belt keeps running: follow its predicted position (camera track + belt
        odometry), with an integral correction for the arm's tracking lag, close, lift, then carry to the bin."""
        t_start = st["t"]
        st["busy"], st["following"] = True, True
        ypred = lambda: tr.s + st["odo"]
        v = C.BELT_SPEED
        # the fingers close along the belt direction (world y), so along-belt timing errors are tolerated best
        top = float(np.median(tr.top_z))
        gz = top - C.NECK_H + R.NECK_TIP_ABOVE_SHOULDER + hand_to_tip
        z_pre = gz + 0.10
        needed = 1.0 + 0.4 + 0.9 + 0.5          # approach, sync, descend, close (s)
        if ypred() + v * needed > Y_GRASP_MAX:
            if ypred() + SLOW_SPEED * needed * 1.5 <= Y_GRASP_MAX:
                st["hold_slow"] = True          # too late at full speed: keep the belt slow until this grasp
                log(f"MOVING PICK track {tr.id}: late (y={ypred():+.3f}) -> belt held at {SLOW_SPEED} m/s for this grasp")
        if ypred() + (SLOW_SPEED if st.get("hold_slow") else v) * needed > Y_GRASP_MAX:
            st["busy"], st["following"], st["hold_slow"] = False, False, False
            tr.state = "missed_window"
            log(f"MOVING PICK track {tr.id}: robot free too late (bottle at y={ypred():+.3f}), skipped")
            return None
        yoff = [0.0]

        def follow(z_from, z_to, dur, until=None):
            p0 = ee()[0]
            ts = st["t"]
            while True:
                s_ = min(1.0, (st["t"] - ts) / dur) if dur > 0 else 1.0
                s_ = 0.5 - 0.5 * math.cos(math.pi * s_)
                z = z_from + s_ * (z_to - z_from)
                tgt = np.array([tr.x, ypred() + yoff[0], z])
                robot.set_end_effector_pose(position=tgt, orientation=R.DOWN)
                sim_step()
                e_y = ypred() - ee()[0][1]
                yoff[0] += 0.08 * e_y                    # integral lag correction along the belt
                if st["t"] - ts >= dur and (until is None or until(e_y)):
                    return e_y
                if st["t"] - ts > dur + 1.5:
                    return e_y

        gripper_open = np.array([[0.04, 0.04]])
        robot.set_dof_position_targets(gripper_open, dof_indices=[7, 8])
        # approach: blend from the current pose onto the moving target
        p0 = ee()[0]
        ts = st["t"]
        while st["t"] - ts < 1.0:
            s_ = 0.5 - 0.5 * math.cos(math.pi * min(1.0, (st["t"] - ts) / 1.0))
            tgt = np.array([tr.x, ypred() + st["speed"] * 0.15, z_pre])
            robot.set_end_effector_pose(position=p0 + s_ * (tgt - p0), orientation=R.DOWN)
            sim_step()
        yoff[0] = st["speed"] * 0.15
        ok_n = [0]

        def synced(e):
            ok_n[0] = ok_n[0] + 1 if abs(e) < 0.004 else 0
            return ok_n[0] >= 5
        follow(z_pre, z_pre, 0.2, until=synced)
        follow(z_pre, gz, 0.9)
        e_close = follow(gz, gz, 0.1)
        gt_close = C.bottle_pose(stage, tr.gt_path)[0] if tr.gt_path else None
        ee_close = ee()[0]
        robot.set_dof_position_targets(np.array([[0.0, 0.0]]), dof_indices=[7, 8])
        follow(gz, gz, 0.5)
        width = float(np.sum(robot.get_dof_positions().numpy()[0][7:9]))
        follow(gz, gz + 0.14, 0.6)                           # lift clear while still moving with the belt
        st["following"], st["hold_slow"] = False, False
        grasp_err = None if gt_close is None else round(1000 * math.hypot(gt_close[0] - ee_close[0], gt_close[1] - ee_close[1]), 1)
        rec = finish_drop(tr, width, {"mode": "moving", "sync_err_mm": round(1000 * e_close, 1),
                                      "grasp_xy_err_vs_gt_mm": grasp_err, "cycle_s": None,
                                      "y_at_grasp": round(float(ee_close[1]), 3)})
        rec["cycle_s"] = round(st["t"] - t_start, 1)
        st["busy"] = False
        log(f"MOVING PICK track {tr.id}: {'in bin' if rec['in_bin'] else 'FAILED'}  belt {v:.2f} m/s, "
            f"hand-bottle sync err {1000 * e_close:+.1f} mm, grasp xy err vs GT {grasp_err} mm at y={ee_close[1]:+.3f}, "
            f"grip {1000 * width:.1f} mm, cycle {rec['cycle_s']} s")
        return rec

    picks = []
    t_end_guard = N_BOTTLES * SPAWN_EVERY_S + 400
    while st["t"] < t_end_guard:
        if MOVING:
            if st["pickq"]:
                tr = st["pickq"].pop(0)
                res = pick_moving(tr)
                if res is not None:
                    picks.append(res)
                continue
            sim_step()
            if st["spawned"] >= N_BOTTLES and all(b["state"] != "on_belt" for b in bottles.values()):
                break
            continue
        tr = st["queued"]
        if tr is None:
            sim_step()
            if st["spawned"] >= N_BOTTLES and all(b["state"] != "on_belt" for b in bottles.values()):
                break
            continue
        while st["speed"] > 0.0:
            sim_step()
        sim_step(10)
        # perceived pick position: lateral x from inspection, along-belt y from odometry
        gx, gy = tr.x, tr.s + st["odo"]
        top = float(np.median(tr.top_z))
        # neck grasp from the perceived top of the (open) neck: fingertips 4 mm above the neck's lower end
        grasp_z = top - C.NECK_H + R.NECK_TIP_ABOVE_SHOULDER + hand_to_tip
        gt_pos = C.bottle_pose(stage, tr.gt_path)[0] if tr.gt_path else None
        pick_err = None if gt_pos is None else round(1000 * math.hypot(gt_pos[0] - gx, gt_pos[1] - gy), 1)
        move_joints(R.READY_Q, 1.0)
        gripper(0.08, 10)
        move_to([gx, gy, grasp_z + 0.12])
        err = move_to([gx, gy, grasp_z], speed=0.12, tol=0.002)
        width = gripper(0.0, 45)
        move_to([gx, gy, grasp_z + 0.16], speed=0.20)
        st["queued"] = None
        st["target"] = C.BELT_SPEED
        move_to([bin_c[0], bin_c[1], 0.40])
        move_to([bin_c[0], bin_c[1], 0.32], speed=0.15)
        gripper(0.08, 40)
        sim_step(40)
        in_bin = False
        if tr.gt_path:
            fpos = C.bottle_pose(stage, tr.gt_path)[0]
            in_bin = bool(bx0 < fpos[0] < bx1 and by0 < fpos[1] < by1 and fpos[2] < bz1)
            gt = bottles[tr.gt_path]
            gt["state"] = "rejected_bin" if in_bin else "pick_failed"
            if not in_bin:
                stage.GetPrimAtPath(tr.gt_path).SetActive(False)
        tr.state = "picked" if in_bin else "pick_failed"
        move_to([bin_c[0], bin_c[1], 0.42])
        move_joints(R.READY_Q, 1.0)
        gt_capped = bottles.get(tr.gt_path, {}).get("capped")
        picks.append({"track": tr.id, "gt": tr.gt_path, "gt_capped": gt_capped, "in_bin": in_bin,
                      "pick_pos_err_mm": pick_err, "grip_mm": round(1000 * width, 1)})
        log(f"PICK track {tr.id}: {'in bin' if in_bin else 'FAILED'}  pick-position error vs GT {pick_err} mm, "
            f"grip {1000 * width:.1f} mm, GT {'capped (FALSE REJECT)' if gt_capped else 'no cap'}")

    # ---- scoring (ground truth)
    B = list(bottles.values())
    n_def = sum(not b["capped"] for b in B)
    summary = {
        "seed": SEED, "bottles": len(B), "defects": n_def, "good": len(B) - n_def,
        "defects_rejected_into_bin": sum((not b["capped"]) and b["state"] == "rejected_bin" for b in B),
        "defects_escaped": sum((not b["capped"]) and b["state"] == "passed_exit" for b in B),
        "false_rejects": sum(b["capped"] and b["state"] in ("rejected_bin", "pick_failed") for b in B),
        "good_passed": sum(b["capped"] and b["state"] == "passed_exit" for b in B),
        "pick_failures": sum(b["state"] == "pick_failed" for b in B),
        "fell": sum(b["state"] == "fell" for b in B),
        "tracks": len(tracks), "sim_time_s": round(st["t"], 1),
        "pick_pos_err_mm_max": max((p.get("pick_pos_err_mm") or p.get("grasp_xy_err_vs_gt_mm") or 0 for p in picks), default=None),
        "mode": "moving" if MOVING else "stop_and_pick", "belt_speed": C.BELT_SPEED,
        "belt_stops": 0 if MOVING else sum(1 for p in picks),
        "missed_window": sum(t.state == "missed_window" for t in tracks),
        "belt_slowdowns": st.get("slowdowns", 0), "min_belt_speed": round(st.get("min_speed", C.BELT_SPEED), 3),
        "max_cycle_s": max((p.get("cycle_s") or 0 for p in picks), default=None),
    }
    json.dump({"summary": summary, "picks": picks}, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("SUMMARY " + json.dumps(summary))
    if panel is not None:
        panel.update("DONE  -  line cleared", f"{summary['bottles']} bottles: {summary['defects_rejected_into_bin']}/"
                     f"{summary['defects']} defects rejected into the bin, {summary['good_passed']}/{summary['good']} good bottles passed",
                     len(tracks), summary["good_passed"], summary["defects_rejected_into_bin"],
                     f"audit vs simulator ground truth:  missed defects {summary['defects_escaped']}   |   "
                     f"false rejects {summary['false_rejects']}")
        for _ in range(240):
            app_utils.update_app(steps=1)
    det_proc.stdin.close()
    det_proc.terminate()


try:
    main()
except Exception:
    log("FAILED\n" + traceback.format_exc())
finally:
    LOG.close()
    if QUIT_WHEN_DONE and LIVE:
        t_end = time.time() + 5.0
        while time.time() < t_end:
            app_utils.update_app(steps=1)
    if QUIT_WHEN_DONE:
        import carb.settings
        carb.settings.get_settings().set("/app/file/ignoreUnsavedOnExit", True)
        omni.kit.app.get_app().post_quit()
