"""Phase A physics check (no robot, no downloads): belt stop-and-go with procedural bottles.

Spawns bottles (30% without cap), runs the belt, stops each uncapped bottle at PICK_Y with a
speed ramp, "picks" it by removing it after a hold, restarts the belt, and removes capped bottles
at the exit. Reports tipping, falling through the belt, and stop-position error.
Run: kit --no-window --exec phaseA_belt_test.py   (env: TEST_N, TEST_SEED, TEST_OUT)
"""
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/home/vishal/poc/conveyor_inspection")
import omni.kit.app
import omni.timeline
import omni.usd
import isaacsim.core.experimental.utils.app as app_utils
import omni.replicator.core as rep
from PIL import Image

import cell_scene as C

N = int(os.environ.get("TEST_N", "24"))
if os.environ.get("CMD_SPEED"):
    C.BELT_SPEED = float(os.environ["CMD_SPEED"])
SEED = int(os.environ.get("TEST_SEED", "0"))
OUT = os.environ.get("TEST_OUT", "/home/vishal/poc/logs/conveyor/phaseA_belt_test")
os.makedirs(os.path.join(OUT, "frames"), exist_ok=True)
DT = 1.0 / 60.0
SPAWN_EVERY_S = 1.7
HOLD_S = 3.0


def log(m):
    print("[BELT] " + m, flush=True)


try:
    omni.usd.get_context().new_stage()
    app_utils.update_app(steps=2)
    scene = C.build(seed=SEED)
    stage = omni.usd.get_context().get_stage()
    rng = scene["rng"]
    rp = rep.create.render_product(scene["presentation_camera"], (1280, 720))
    pres = rep.AnnotatorRegistry.get_annotator("rgb")
    pres.attach([rp])
    rp2 = rep.create.render_product(scene["inspection_camera"], C.INSPECT_CAM_RES)
    insp = rep.AnnotatorRegistry.get_annotator("rgb")
    insp.attach([rp2])

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    app_utils.update_app(steps=5)
    t_start = timeline.get_current_time()

    bottles = {}          # path -> dict(spec, max_tilt, state)
    speed, target = 0.0, C.BELT_SPEED
    t, next_spawn, spawned, frame = 0.0, 0.0, 0, 0
    holding = None        # (path, release_time)
    stop_errors = []
    step = 0
    while True:
        # spawn
        if spawned < N and t >= next_spawn and speed > 0.5 * C.BELT_SPEED:
            capped = rng.random() >= 0.30
            spec = C.random_bottle_spec(rng, capped)
            path = f"/World/Bottles/b{spawned:03d}"
            C.spawn_bottle(stage, path, spec, C.BELT_X + rng.uniform(-0.02, 0.02), C.SPAWN_Y)
            bottles[path] = {"spec": spec, "max_tilt": 0.0, "state": "on_belt", "min_z": 0.0}
            spawned += 1
            next_spawn = t + SPAWN_EVERY_S

        # belt speed ramp, on simulation time (an app update can advance physics by several steps)
        app_utils.update_app(steps=1)
        t_now = timeline.get_current_time() - t_start
        dt, t = t_now - t, t_now
        step += 1
        dv = C.BELT_SPEED * dt / C.BELT_RAMP_S
        speed = min(target, speed + dv) if speed < target else max(target, speed - dv)
        C.set_belt_speed(stage, speed)

        if step in (120, 240) and "/World/Bottles/b000" in bottles:
            yb = C.bottle_pose(stage, "/World/Bottles/b000")[0][1]
            if step == 120:
                meas = (yb, t)
            else:
                log(f"MEASURE bottle speed {(yb - meas[0]) / (t - meas[1]):.4f} m/s over {t - meas[1]:.2f} sim-s "
                    f"({step - 120} updates); commanded {speed:.3f}")
        # monitor bottles
        for path, b in bottles.items():
            if b["state"] not in ("on_belt", "held"):
                continue
            pos, tilt = C.bottle_pose(stage, path)
            b["max_tilt"] = max(b["max_tilt"], tilt)
            b["min_z"] = min(b["min_z"], pos[2])
            if pos[2] < -0.05:
                b["state"] = "fell"
                log(f"{path} FELL z={pos[2]:.3f}")
            elif pos[1] > C.EXIT_Y:
                b["state"] = "passed" if b["spec"]["capped"] else "ESCAPED"
                stage.GetPrimAtPath(path).SetActive(False)
            elif (not b["spec"]["capped"]) and b["state"] == "on_belt" and holding is None \
                    and pos[1] >= C.PICK_Y - C.BELT_SPEED * C.BELT_RAMP_S / 2:
                target = 0.0                     # stop so this bottle comes to rest at PICK_Y
                holding = [path, None]
                b["state"] = "held"

        if holding is not None and speed == 0.0:
            if holding[1] is None:
                app_utils.update_app(steps=10)   # settle
                pos, tilt = C.bottle_pose(stage, holding[0])
                stop_errors.append(float(pos[1] - C.PICK_Y))
                log(f"{holding[0]} stopped at y={pos[1]:+.4f} (err {1000 * (pos[1] - C.PICK_Y):+.1f} mm), tilt {tilt:.1f}")
                holding[1] = t + HOLD_S
            elif t >= holding[1]:
                bottles[holding[0]]["state"] = "picked"
                stage.GetPrimAtPath(holding[0]).SetActive(False)
                holding = None
                target = C.BELT_SPEED

        if step % 4 == 0:
            d = pres.get_data()
            if getattr(d, "ndim", 0) == 3:
                Image.fromarray(np.asarray(d)[:, :, :3]).save(os.path.join(OUT, "frames", f"f_{frame:05d}.jpg"), quality=88)
                frame += 1
        if step == 900:
            d = insp.get_data()
            if getattr(d, "ndim", 0) == 3:
                Image.fromarray(np.asarray(d)[:, :, :3]).save(os.path.join(OUT, "inspection_view.png"))

        if spawned == N and all(b["state"] not in ("on_belt", "held") for b in bottles.values()):
            break
        if t > N * SPAWN_EVERY_S + 120:
            log("TIMEOUT")
            break

    states = [b["state"] for b in bottles.values()]
    tilts = [b["max_tilt"] for b in bottles.values()]
    summary = {
        "bottles": N, "uncapped": sum(not b["spec"]["capped"] for b in bottles.values()),
        "picked": states.count("picked"), "passed": states.count("passed"), "escaped": states.count("ESCAPED"),
        "fell": states.count("fell"), "max_tilt_deg": round(max(tilts), 2),
        "tipped_over_10deg": sum(x > 10 for x in tilts),
        "stop_err_mm_mean": round(1000 * float(np.mean(stop_errors)), 1) if stop_errors else None,
        "stop_err_mm_absmax": round(1000 * float(np.max(np.abs(stop_errors))), 1) if stop_errors else None,
        "sim_time_s": round(t, 1),
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    log("SUMMARY " + json.dumps(summary))
except Exception:
    log("FAILED\n" + traceback.format_exc())
finally:
    omni.kit.app.get_app().post_quit()
