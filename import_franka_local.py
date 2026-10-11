"""Convert the Franka URDF bundled with Isaac Sim's URDF importer into a local USD (no network needed).
Run: kit --no-window --exec import_franka_local.py"""
import os, traceback
import omni.kit.app
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
ISAAC = "/home/vishal/Downloads/isaac-sim-standalone-6.0.1-linux-x86_64"
DESC = ISAAC + "/exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description"
OUT = "/home/vishal/poc/conveyor_inspection/assets"
try:
    cfg = URDFImporterConfig(
        urdf_path=DESC + "/robots/panda_arm_hand.urdf", usd_path=OUT,
        ros_package_paths=[{"name": "franka_description", "path": DESC}],
        fix_base=True, merge_fixed_joints=False, allow_self_collision=False,
        joint_drive_type="force", joint_target_type="position",
        override_joint_stiffness={"panda_joint[1-7]": 400.0, "panda_finger_joint.*": 2000.0},
        override_joint_damping={"panda_joint[1-7]": 80.0, "panda_finger_joint.*": 100.0},
    )
    out = URDFImporter(cfg).import_urdf()
    print("IMPORT OK", out, flush=True)
except Exception:
    print("IMPORT FAILED\n" + traceback.format_exc(), flush=True)
finally:
    omni.kit.app.get_app().post_quit()
