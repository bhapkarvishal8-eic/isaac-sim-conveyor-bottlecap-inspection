"""Grasp-only check: one uncapped bottle at the pick point, belt stopped. Logs finger width and bottle
height through close and lift, for several grasp heights. Env: FINGER_MU (finger friction)."""
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

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.robot.experimental.manipulators.examples", True)
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.manipulators.examples.franka.franka import Franka
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

import cell_scene as C

FRANKA_USD = "/home/vishal/poc/conveyor_inspection/assets/panda_arm_hand/panda_arm_hand.usda"
HAND = "/World/robot/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/panda_link4/panda_link5/panda_link6/panda_link7/panda_link8/panda_hand"
DOWN = np.array([0.0, 1.0, 0.0, 0.0])
READY_Q = np.array([0.0, -0.30, 0.0, -2.2, 0.0, 1.95, 0.785])
MU = float(os.environ.get("FINGER_MU", "1.0"))
FORCE = float(os.environ.get("FINGER_FORCE", "20"))
GRASP = os.environ.get("GRASP", "neck")


def log(m):
    print("[GD] " + m, flush=True)


try:
    omni.usd.get_context().new_stage()
    app_utils.update_app(steps=2)
    SimulationManager.set_physics_sim_device("cpu")
    scene = C.build(seed=0)
    stage = omni.usd.get_context().get_stage()
    rp = stage.DefinePrim("/World/robot", "Xform")
    rp.GetReferences().AddReference(FRANKA_USD)
    app_utils.update_app(steps=2)
    for p in Usd.PrimRange(rp):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
    fm = C._physics_material(stage, "/World/PhysicsMaterials/Finger", MU, MU)
    for f in ("left", "right"):
        C._bind_physics(stage.GetPrimAtPath(HAND + f"/panda_{f}finger"), fm)
    for j in ("panda_finger_joint1", "panda_finger_joint2"):
        UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"/World/robot/Physics/{j}"), "linear").GetMaxForceAttr().Set(FORCE)
    robot = Franka(robot_path="/World/robot", create_robot=False, end_effector_link=RigidPrim(HAND))
    spec = C.random_bottle_spec(scene["rng"], capped=False)
    spec["mass"] = 0.45
    C.spawn_bottle(stage, "/World/Bottles/t0", spec, C.BELT_X, 0.0)
    tl = omni.timeline.get_timeline_interface()
    tl.play()
    app_utils.update_app(steps=5)
    q0 = np.concatenate([READY_Q, [0.04, 0.04]])
    robot.set_dof_positions(q0)
    robot.set_dof_position_targets(q0)
    app_utils.update_app(steps=30)

    def ee():
        _, p, q = robot.get_current_state()
        return p[0].astype(float)

    def go(goal, n=90, settle=60):
        p0 = ee()
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / n)
            robot.set_end_effector_pose(position=p0 + s * (np.asarray(goal) - p0), orientation=DOWN)
            app_utils.update_app(steps=1)
        for _ in range(settle):
            robot.set_end_effector_pose(position=np.asarray(goal), orientation=DOWN)
            app_utils.update_app(steps=1)
        return np.linalg.norm(ee() - goal)

    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide", "proxy"])
    tip = min(bb.ComputeWorldBound(stage.GetPrimAtPath(HAND + f"/panda_{f}finger")).ComputeAlignedRange().GetMin()[2]
              for f in ("left", "right"))
    hand_to_tip = ee()[2] - tip
    pos, _ = C.bottle_pose(stage, "/World/Bottles/t0")
    neck_top, _ = C.bottle_heights(spec)
    log(f"hand->tip {1000 * hand_to_tip:.1f} mm; neck d {2000 * spec['neck_r']:.1f} mm, neck top z {neck_top:.3f}, "
        f"body d {2000 * spec['body_r']:.1f}, finger mu {MU}, force {FORCE} N, grasp {GRASP}")
    # tips at: 4 mm above the shoulder end (= neck bottom + 4 mm)
    tip_z = pos[2] + neck_top - spec["neck_h"] + 0.004
    if GRASP == "body":
        tip_z = pos[2] + spec["body_h"] - 0.035
    gz = tip_z + hand_to_tip
    log(f"approach err {1000 * go([pos[0], pos[1], gz + 0.10]):.1f} mm")
    log(f"descend err {1000 * go([pos[0], pos[1], gz], n=120):.1f} mm")
    robot.set_dof_position_targets(np.array([[0.0, 0.0]]), dof_indices=[7, 8])
    for k in range(60):
        app_utils.update_app(steps=1)
        if k % 15 == 14:
            w = robot.get_dof_positions().numpy()[0][7:9]
            log(f"closing t{k}: finger dofs {np.round(1000 * w, 1)} mm, bottle z {C.bottle_pose(stage, '/World/Bottles/t0')[0][2]:.4f}")
    for dz in (0.02, 0.06, 0.12, 0.20):
        e = go([pos[0], pos[1], gz + dz], n=60, settle=30)
        bp, tilt = C.bottle_pose(stage, "/World/Bottles/t0")
        w = robot.get_dof_positions().numpy()[0][7:9]
        log(f"lift +{dz:.2f}: ee err {1000 * e:.1f} mm, bottle z {bp[2]:+.4f} (want {pos[2] + dz:+.4f}), tilt {tilt:.1f}, "
            f"fingers {np.round(1000 * w, 1)}")
except Exception:
    log("FAILED\n" + traceback.format_exc())
finally:
    omni.kit.app.get_app().post_quit()
