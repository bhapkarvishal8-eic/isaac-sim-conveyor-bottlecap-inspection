"""Franka for the conveyor cell: local URDF import (no network), parallel gripper.

Settings validated in grasp_debug.py (2026-09-27):
- arm links have gravity disabled: the real Franka compensates gravity in its controller, and the
  position drives (400 Nm/rad) then hold IK targets within a few mm;
- finger drives at 70 N (the real Franka Hand's continuous grasp force; the URDF says 20 N, which
  let a 0.45 kg bottle slip out of a neck grasp), friction 1.0 on the fingers;
- a neck grasp with the fingertips 4 mm above the neck's lower end holds through a 0.2 m lift.
"""
import numpy as np
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

FRANKA_USD = "/home/vishal/poc/conveyor_inspection/assets/panda_arm_hand/panda_arm_hand.usda"
ROBOT_PATH = "/World/robot"
HAND = (ROBOT_PATH + "/Geometry/panda_link0/panda_link1/panda_link2/panda_link3/panda_link4/panda_link5"
        "/panda_link6/panda_link7/panda_link8/panda_hand")
DOWN = np.array([0.0, 1.0, 0.0, 0.0])      # wxyz, gripper pointing down
READY_Q = np.array([0.0, -0.30, 0.0, -2.2, 0.0, 1.95, 0.785])
FINGER_FORCE = 70.0
FINGER_MU = 1.0
NECK_TIP_ABOVE_SHOULDER = 0.004


def add_franka(stage):
    """Reference the local Franka and apply the cell's physics settings. Call before timeline.play()."""
    import cell_scene as C
    rp = stage.DefinePrim(ROBOT_PATH, "Xform")
    rp.GetReferences().AddReference(FRANKA_USD)
    for p in Usd.PrimRange(rp):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
    for j in ("panda_finger_joint1", "panda_finger_joint2"):
        UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(f"{ROBOT_PATH}/Physics/{j}"), "linear").GetMaxForceAttr().Set(FINGER_FORCE)
    fm = C._physics_material(stage, "/World/PhysicsMaterials/Finger", FINGER_MU, FINGER_MU)
    for f in ("left", "right"):
        C._bind_physics(stage.GetPrimAtPath(HAND + f"/panda_{f}finger"), fm)


def make_controller():
    import omni.kit.app
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.robot.experimental.manipulators.examples", True)
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.robot.experimental.manipulators.examples.franka.franka import Franka
    return Franka(robot_path=ROBOT_PATH, create_robot=False, end_effector_link=RigidPrim(HAND))


def hand_to_fingertip(stage, hand_z):
    """Vertical distance hand origin -> lowest fingertip point (hand pointing down)."""
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide", "proxy"])
    tip = min(bb.ComputeWorldBound(stage.GetPrimAtPath(HAND + f"/panda_{f}finger")).ComputeAlignedRange().GetMin()[2]
              for f in ("left", "right"))
    return float(hand_z - tip)


def neck_grasp_hand_z(bottle_base_z, spec, hand_to_tip):
    import cell_scene as C
    neck_top, _ = C.bottle_heights(spec)
    return bottle_base_z + neck_top - spec["neck_h"] + NECK_TIP_ABOVE_SHOULDER + hand_to_tip
