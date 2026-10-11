"""Render the video scene (room + cell + robot + a few bottles) from candidate presentation cameras."""
import os, sys, traceback, math
import numpy as np
sys.path.insert(0, "/home/vishal/poc/conveyor_inspection")
import omni.kit.app, omni.timeline, omni.usd
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.simulation_manager import SimulationManager
import omni.replicator.core as rep
from PIL import Image
from pxr import UsdGeom, Gf
import cell_scene as C, robot_setup as R
OUT = "/home/vishal/poc/logs/conveyor/room_preview"; os.makedirs(OUT, exist_ok=True)
CANDS = {"down": ((2.1, 1.2, 1.25), (0.2, -0.2, 0.0)), "down2": ((1.9, 1.6, 1.0), (0.3, -0.3, 0.05)),
         "front": ((2.4, -0.3, 1.3), (0.1, -0.1, 0.0))}
WH = os.environ.get("WAREHOUSE", "0") == "1"
OUT = OUT + ("/warehouse" if WH else "/room"); os.makedirs(OUT, exist_ok=True)
try:
    omni.usd.get_context().new_stage(); app_utils.update_app(steps=2)
    SimulationManager.set_physics_sim_device("cpu")
    from isaacsim.storage.native import get_assets_root_path
    sc = C.build(seed=1, room=not WH, warehouse=WH, assets_root=get_assets_root_path() if WH else None); st = omni.usd.get_context().get_stage()
    R.add_franka(st); app_utils.update_app(steps=2); robot = R.make_controller()
    rng = sc["rng"]
    for i, y in enumerate([-0.9, -0.62, -0.35, -0.05, 0.3, 0.6]):
        C.spawn_bottle(st, f"/World/Bottles/p{i}", C.random_bottle_spec(rng, capped=(i % 3 != 1)), C.BELT_X, y)
    anns = {}
    for k, (e, t) in CANDS.items():
        cam = UsdGeom.Camera.Define(st, f"/World/Cand_{k}")
        cam.CreateHorizontalApertureAttr(20.955); cam.CreateVerticalApertureAttr(20.955 * 9 / 16)
        cam.CreateFocalLengthAttr(16.0); cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200))
        UsdGeom.Xformable(cam).AddTransformOp().Set(C.look_at(e, t))
        rp = rep.create.render_product(f"/World/Cand_{k}", (1280, 720))
        a = rep.AnnotatorRegistry.get_annotator("rgb"); a.attach([rp]); anns[k] = a
    omni.timeline.get_timeline_interface().play(); app_utils.update_app(steps=5)
    q0 = np.concatenate([R.READY_Q, [0.04, 0.04]]); robot.set_dof_positions(q0); robot.set_dof_position_targets(q0)
    app_utils.update_app(steps=240 if WH else 90)
    for k, a in anns.items():
        Image.fromarray(np.asarray(a.get_data())[:, :, :3]).save(f"{OUT}/{k}.jpg", quality=92)
    print("PREVIEW OK", flush=True)
except Exception:
    print("PREVIEW FAILED\n" + traceback.format_exc(), flush=True)
finally:
    omni.kit.app.get_app().post_quit()
