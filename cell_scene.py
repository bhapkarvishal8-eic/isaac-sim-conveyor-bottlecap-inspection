"""Conveyor inspection cell: belt, procedural bottles, reject bin, cameras (Isaac Sim 6.0.1, kit --exec).

World frame: z = 0 is the belt's top surface. The Franka base sits at the origin (on a pedestal),
the belt runs along +y at x = BELT_X, the inspection camera looks down on the belt upstream of the
pick zone, and the reject bin stands beside the robot.

Everything here is procedural (no asset download needed). build(warehouse=True) additionally
references the NVIDIA warehouse around the cell.
"""

from __future__ import annotations

import math
import random

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

# ---------------------------------------------------------------- layout
BELT_X = 0.52              # belt centerline (robot base at origin)
BELT_Y0, BELT_Y1 = -1.10, 1.00
BELT_WIDTH = 0.22
BELT_THICK = 0.03
GUIDE_GAP = 0.15           # distance between the inner faces of the side guides
GUIDE_H = 0.035            # guide height above the belt
FLOOR_Z = -0.86            # belt top at a typical 86 cm conveyor height
SPAWN_Y = -1.00
EXIT_Y = 0.92              # bottles past this are removed and counted as "passed"
PICK_Y = 0.0               # bottles to reject are stopped here
INSPECT_Y = -0.55          # inspection camera looks at this belt position
BELT_SPEED = 0.15          # m/s
BELT_RAMP_S = 0.6          # start/stop ramp (abrupt changes tip tall bottles)

REJECT_BIN_CENTER = (0.02, 0.50)     # downstream side: out of the inspection camera view
REJECT_BIN_SIZE = (0.44, 0.36, 0.30)   # outer x, y, height (holds ~20 bottles of up to 0.23 m)
REJECT_BIN_TOP_Z = 0.02    # bin rim just above belt height, on a stand

INSPECT_CAM_RES = (1280, 720)
INSPECT_CAM_HFOV = 58.0
# eye offset across the belt (+x) -> the belt runs along the image's long axis; ~15 deg from vertical
# so the camera sees the cap / open mouth and a little of the bottle's side
INSPECT_CAM_EYE = (BELT_X + 0.19, INSPECT_Y, 0.75)
INSPECT_CAM_TARGET = (BELT_X, INSPECT_Y, 0.0)
PRES_EYE = (2.10, 1.20, 1.25)       # from the downstream side: bottles flow toward the viewer past the robot
PRES_TARGET = (0.20, -0.20, 0.00)

# ---------------------------------------------------------------- bottles (opaque HDPE-like)
BODY_R = (0.029, 0.036)
BODY_H = (0.13, 0.17)
SHOULDER_H = (0.025, 0.040)
NECK_R = (0.0125, 0.0150)
NECK_H = 0.022
CAP_H = 0.018
CAP_EXTRA_R = 0.003        # cap radius = neck radius + this
BOTTLE_MASS = (0.30, 0.45)  # kg (filled)
BODY_COLORS = [(0.92, 0.92, 0.90), (0.15, 0.35, 0.75), (0.85, 0.55, 0.15), (0.20, 0.55, 0.30),
               (0.75, 0.15, 0.20), (0.55, 0.30, 0.65), (0.95, 0.85, 0.30)]
CAP_COLORS = [(0.9, 0.1, 0.1), (0.1, 0.2, 0.8), (0.95, 0.95, 0.95), (0.1, 0.1, 0.1), (0.1, 0.6, 0.2), (0.95, 0.75, 0.1)]


def look_at(eye, target):
    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()


def _material(stage, path, rgb, roughness=0.5, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def _physics_material(stage, path, static_f, dynamic_f, restitution=0.0):
    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(static_f)
    pm.CreateDynamicFrictionAttr(dynamic_f)
    pm.CreateRestitutionAttr(restitution)
    return mat


def _bind_physics(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path, center, size, mat=None, collide=True):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(*center))
    c.AddScaleOp().Set(Gf.Vec3f(*size))
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if mat is not None:
        _bind(c.GetPrim(), mat)
    return c.GetPrim()


def _cyl(stage, path, center, radius, height, mat=None, axis="Z", collide=True):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(radius)
    c.CreateHeightAttr(height)
    c.CreateAxisAttr(axis)
    c.AddTranslateOp().Set(Gf.Vec3d(*center))
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if mat is not None:
        _bind(c.GetPrim(), mat)
    return c.GetPrim()


# ---------------------------------------------------------------- conveyor
def build_conveyor(stage):
    m_belt = _material(stage, "/World/Looks/Belt", (0.07, 0.07, 0.08), roughness=0.85)
    m_alu = _material(stage, "/World/Looks/Aluminium", (0.70, 0.72, 0.74), roughness=0.35, metallic=0.8)
    m_frame = _material(stage, "/World/Looks/Frame", (0.25, 0.27, 0.30), roughness=0.5, metallic=0.5)
    pm_belt = _physics_material(stage, "/World/PhysicsMaterials/Belt", 0.9, 0.8)
    pm_guide = _physics_material(stage, "/World/PhysicsMaterials/Guide", 0.15, 0.10)

    length = BELT_Y1 - BELT_Y0
    cy = (BELT_Y0 + BELT_Y1) / 2
    belt = _box(stage, "/World/Conveyor/Belt", (BELT_X, cy, -BELT_THICK / 2), (BELT_WIDTH, length, BELT_THICK), m_belt)
    rb = UsdPhysics.RigidBodyAPI.Apply(belt)
    rb.CreateKinematicEnabledAttr(True)
    sv = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(belt)
    sv.CreateSurfaceVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    sv.CreateSurfaceVelocityEnabledAttr(True)
    _bind_physics(belt, pm_belt)

    # low-friction side guides keep bottles on the centerline (real bottling lines have them)
    for side, s in (("L", -1), ("R", 1)):
        x = BELT_X + s * (GUIDE_GAP / 2 + 0.006)
        g = _box(stage, f"/World/Conveyor/Guide{side}", (x, cy, 0.012 + GUIDE_H / 2), (0.012, length, GUIDE_H), m_alu)
        _bind_physics(g, pm_guide)
        # guide posts (visual)
        for k in range(6):
            y = BELT_Y0 + 0.1 + k * (length - 0.2) / 5
            _box(stage, f"/World/Conveyor/Post{side}{k}", (BELT_X + s * (BELT_WIDTH / 2 + 0.02), y, 0.01),
                 (0.012, 0.012, 0.05), m_alu, collide=False)
    # frame, end rollers, legs (visual only)
    for side, s in (("L", -1), ("R", 1)):
        _box(stage, f"/World/Conveyor/Rail{side}", (BELT_X + s * (BELT_WIDTH / 2 + 0.02), cy, -0.05),
             (0.04, length + 0.08, 0.10), m_frame, collide=False)
    for end, y in (("A", BELT_Y0), ("B", BELT_Y1)):
        _cyl(stage, f"/World/Conveyor/Roller{end}", (BELT_X, y, -BELT_THICK / 2), BELT_THICK / 2 + 0.005,
             BELT_WIDTH + 0.02, m_alu, axis="X", collide=False)
    for k, y in enumerate((BELT_Y0 + 0.08, cy, BELT_Y1 - 0.08)):
        for side, s in (("L", -1), ("R", 1)):
            _box(stage, f"/World/Conveyor/Leg{side}{k}", (BELT_X + s * (BELT_WIDTH / 2 + 0.02), y, (FLOOR_Z - 0.10) / 2),
                 (0.04, 0.04, -FLOOR_Z - 0.10), m_frame, collide=False)
    return belt


# Measured (phaseA_belt_test, 2026-09-27): bottles travel at 2.10x the PhysX surface-velocity value,
# linear at 0.075/0.10/0.15 m/s. set_belt_speed() takes the real transport speed (m/s).
BELT_GAIN = 2.10


def set_belt_speed(stage, v):
    """v = desired transport speed of objects on the belt, m/s."""
    PhysxSchema.PhysxSurfaceVelocityAPI(stage.GetPrimAtPath("/World/Conveyor/Belt")).GetSurfaceVelocityAttr().Set(
        Gf.Vec3f(0.0, float(v) / BELT_GAIN, 0.0))


# ---------------------------------------------------------------- reject bin + robot pedestal
def build_reject_bin(stage):
    m = _material(stage, "/World/Looks/RejectBin", (0.80, 0.12, 0.10), roughness=0.55)
    m_stand = _material(stage, "/World/Looks/Stand", (0.30, 0.32, 0.35), roughness=0.5, metallic=0.4)
    cx, cy = REJECT_BIN_CENTER
    sx, sy, h = REJECT_BIN_SIZE
    t = 0.008
    z0 = REJECT_BIN_TOP_Z - h
    _box(stage, "/World/RejectBin/Floor", (cx, cy, z0 + t / 2), (sx, sy, t), m)
    _box(stage, "/World/RejectBin/WallN", (cx, cy + sy / 2 - t / 2, z0 + h / 2), (sx, t, h), m)
    _box(stage, "/World/RejectBin/WallS", (cx, cy - sy / 2 + t / 2, z0 + h / 2), (sx, t, h), m)
    _box(stage, "/World/RejectBin/WallE", (cx + sx / 2 - t / 2, cy, z0 + h / 2), (t, sy, h), m)
    _box(stage, "/World/RejectBin/WallW", (cx - sx / 2 + t / 2, cy, z0 + h / 2), (t, sy, h), m)
    _box(stage, "/World/RejectBin/Stand", (cx, cy, (FLOOR_Z + z0) / 2), (sx - 0.04, sy - 0.04, z0 - FLOOR_Z), m_stand)
    # robot pedestal (visual; the Franka base is fixed at the origin)
    _box(stage, "/World/Pedestal", (0.0, 0.0, (FLOOR_Z - 0.002) / 2), (0.26, 0.26, -FLOOR_Z - 0.002), m_stand, collide=False)
    return ((cx - sx / 2, cy - sy / 2, z0), (cx + sx / 2, cy + sy / 2, REJECT_BIN_TOP_Z))


# ---------------------------------------------------------------- bottles
def random_bottle_spec(rng: random.Random, capped: bool):
    neck_r = rng.uniform(*NECK_R)
    return {
        "body_r": rng.uniform(*BODY_R), "body_h": rng.uniform(*BODY_H), "shoulder_h": rng.uniform(*SHOULDER_H),
        "neck_r": neck_r, "neck_h": NECK_H, "cap_r": neck_r + CAP_EXTRA_R, "cap_h": CAP_H, "capped": capped,
        "mass": rng.uniform(*BOTTLE_MASS), "body_rgb": rng.choice(BODY_COLORS), "cap_rgb": rng.choice(CAP_COLORS),
    }


def bottle_heights(spec):
    """z of the neck top (open mouth) and of the overall top, relative to the bottle base."""
    neck_top = spec["body_h"] + spec["shoulder_h"] + spec["neck_h"]
    return neck_top, neck_top + (spec["cap_h"] - 0.004 if spec["capped"] else 0.0)


def _profile(spec):
    """(r, z) outline of the bottle wall, bottom to mouth, then down the inside of the neck."""
    br, bh, sh, nr, nh = spec["body_r"], spec["body_h"], spec["shoulder_h"], spec["neck_r"], spec["neck_h"]
    pts = [(0.0, 0.0), (br - 0.006, 0.0)]
    for k in range(1, 5):                                   # rounded bottom edge
        a = k / 4 * math.pi / 2
        pts.append((br - 0.006 + 0.006 * math.sin(a), 0.006 - 0.006 * math.cos(a)))
    pts.append((br, bh))
    for k in range(1, 9):                                   # smooth shoulder (cosine ease)
        t = k / 8
        pts.append((nr + (br - nr) * 0.5 * (1 + math.cos(math.pi * t)), bh + sh * t))
    top = bh + sh + nh
    pts += [(nr, top - 0.004), (nr + 0.0018, top - 0.003), (nr + 0.0018, top), (nr - 0.0022, top),
            (nr - 0.0022, top - 0.030), (0.0, top - 0.030)]
    return pts


def _lathe_mesh(stage, path, profile, mat, n=40):
    rings = len(profile)
    pts, normals = [], []
    for i, (r, z) in enumerate(profile):
        # profile tangent -> outward normal in the (r, z) plane
        r0, z0 = profile[max(0, i - 1)]
        r1, z1 = profile[min(rings - 1, i + 1)]
        tr, tz = r1 - r0, z1 - z0
        nr_, nz_ = tz, -tr
        ln = math.hypot(nr_, nz_) or 1.0
        nr_, nz_ = nr_ / ln, nz_ / ln
        for j in range(n):
            a = 2 * math.pi * j / n
            pts.append(Gf.Vec3f(r * math.cos(a), r * math.sin(a), z))
            normals.append(Gf.Vec3f(nr_ * math.cos(a), nr_ * math.sin(a), nz_))
    counts, idx = [], []
    for i in range(rings - 1):
        for j in range(n):
            a, b = i * n + j, i * n + (j + 1) % n
            idx += [a, b, b + n, a + n]
            counts.append(4)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    rmax = max(r for r, _ in profile)
    zmax = max(z for _, z in profile)
    mesh.CreateExtentAttr([Gf.Vec3f(-rmax, -rmax, 0), Gf.Vec3f(rmax, rmax, zmax)])
    _bind(mesh.GetPrim(), mat)
    return mesh.GetPrim()


def _collider_only(prim):
    UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)   # invisible, still collides


def spawn_bottle(stage, path, spec, x, y):
    """One rigid body: a lathe mesh for looks, plain cylinders as colliders. Base on the belt (z=0)."""
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0015))
    root.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(spec["mass"])
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, spec["body_h"] * 0.42))   # filled: low center of mass
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    prb.CreateEnableCCDAttr(True)
    prb.CreateSolverPositionIterationCountAttr(16)
    prb.CreateSleepThresholdAttr(0.0)   # a sleeping bottle would ignore the belt restarting

    key = path.split("/")[-1]
    m_body = _material(stage, f"/World/Looks/Bottles/{key}_body", spec["body_rgb"], roughness=0.42)
    _lathe_mesh(stage, path + "/visual", _profile(spec), m_body)
    br, bh, sh, nr, nh = spec["body_r"], spec["body_h"], spec["shoulder_h"], spec["neck_r"], spec["neck_h"]
    _collider_only(_cyl(stage, path + "/col_body", (0, 0, bh / 2), br, bh))
    _collider_only(_cyl(stage, path + "/col_shoulder", (0, 0, bh + sh / 2), (br + nr) / 2, sh))
    _collider_only(_cyl(stage, path + "/col_neck", (0, 0, bh + sh + nh / 2), nr, nh))
    if spec["capped"]:
        m_cap = _material(stage, f"/World/Looks/Bottles/{key}_cap", spec["cap_rgb"], roughness=0.35)
        z0, cr, ch = bh + sh + nh - 0.004, spec["cap_r"], spec["cap_h"]
        cap_prof = [(0.0, z0), (cr - 0.001, z0), (cr, z0 + 0.001), (cr, z0 + ch - 0.002),
                    (cr - 0.002, z0 + ch), (0.0, z0 + ch)]
        cap = _lathe_mesh(stage, path + "/cap", cap_prof, m_cap, n=48)
        _collider_only(_cyl(stage, path + "/col_cap", (0, 0, z0 + ch / 2), cr, ch))
    pm = _physics_material(stage, f"/World/PhysicsMaterials/Bottles/{key}", 0.8, 0.7)
    _bind_physics(prim, pm)
    return prim



# ---------------------------------------------------------------- procedural warehouse (for videos)
def build_room(stage, seed=0):
    """Warehouse surroundings built locally (NVIDIA's asset server is unreachable from this network):
    concrete floor with safety markings, pallet racks with cartons and pallets, walls, high-bay lights,
    and a camera gantry over the belt so viewers can see where the inspection camera sits.
    Purely visual (no colliders) except the floor, which already exists."""
    rng = random.Random(seed + 99)
    Z = FLOOR_Z
    m_yel = _material(stage, "/World/Looks/Room/Yellow", (0.95, 0.75, 0.05), roughness=0.6)
    m_wall = _material(stage, "/World/Looks/Room/Wall", (0.78, 0.80, 0.82), roughness=0.9)
    m_wall2 = _material(stage, "/World/Looks/Room/WallBand", (0.20, 0.33, 0.55), roughness=0.8)
    m_up = _material(stage, "/World/Looks/Room/Upright", (0.92, 0.38, 0.08), roughness=0.5, metallic=0.3)
    m_beam = _material(stage, "/World/Looks/Room/Beam", (0.10, 0.28, 0.62), roughness=0.5, metallic=0.3)
    m_deck = _material(stage, "/World/Looks/Room/Deck", (0.55, 0.57, 0.60), roughness=0.6, metallic=0.5)
    m_pal = _material(stage, "/World/Looks/Room/Pallet", (0.62, 0.47, 0.30), roughness=0.9)
    m_cams = _material(stage, "/World/Looks/Room/CamBody", (0.12, 0.12, 0.14), roughness=0.4, metallic=0.4)
    m_lens = _material(stage, "/World/Looks/Room/Lens", (0.02, 0.02, 0.03), roughness=0.1, metallic=0.9)
    m_grey = _material(stage, "/World/Looks/Room/Grey", (0.55, 0.57, 0.60), roughness=0.4, metallic=0.7)
    cards = [_material(stage, f"/World/Looks/Room/Card{k}", c, roughness=0.9) for k, c in
             enumerate([(0.72, 0.55, 0.36), (0.66, 0.50, 0.32), (0.78, 0.62, 0.42), (0.60, 0.46, 0.30), (0.85, 0.85, 0.82)])]
    R_ = "/World/Room"
    # floor: large concrete slab + painted walkway lines around the cell
    m_conc = _material(stage, "/World/Looks/Room/Concrete", (0.46, 0.47, 0.48), roughness=0.85)
    _box(stage, R_ + "/Slab", (0, 0, Z - 0.012), (40, 40, 0.02), m_conc, collide=False)
    for k, (c, sz) in enumerate([((0.25, -1.45, Z + 0.001), (3.6, 0.07, 0.002)), ((0.25, 1.45, Z + 0.001), (3.6, 0.07, 0.002)),
                                 ((-1.55, 0.0, Z + 0.001), (0.07, 2.9, 0.002)), ((2.05, 0.0, Z + 0.001), (0.07, 2.9, 0.002))]):
        _box(stage, R_ + f"/Line{k}", c, sz, m_yel, collide=False)
    # walls
    _box(stage, R_ + "/WallBack", (-7.5, 0, Z + 4.0), (0.2, 30, 8.0), m_wall, collide=False)
    _box(stage, R_ + "/WallBackBand", (-7.38, 0, Z + 1.0), (0.05, 30, 2.0), m_wall2, collide=False)
    _box(stage, R_ + "/WallSide", (0, 9.5, Z + 4.0), (30, 0.2, 8.0), m_wall, collide=False)
    _box(stage, R_ + "/WallSideBand", (0, 9.38, Z + 1.0), (30, 0.05, 2.0), m_wall2, collide=False)
    # pallet racks: rows parallel to y behind the robot, and one row across the far end
    def rack(name, x0, y0, bays, along="y"):
        bay, depth, levels = 2.7, 1.1, (0.0, 1.6, 3.2)
        for b in range(bays + 1):
            for d in (0.0, depth):
                px = x0 + (d if along == "y" else b * bay)
                py = y0 + (b * bay if along == "y" else d)
                _box(stage, f"{name}/Up{b}_{int(d * 10)}", (px, py, Z + 2.3), (0.09, 0.09, 4.6), m_up, collide=False)
        for li, lz in enumerate(levels):
            for d in (0.0, depth):
                if along == "y":
                    c, sz = (x0 + d, y0 + bays * bay / 2, Z + lz + 0.12), (0.06, bays * bay, 0.12)
                else:
                    c, sz = (x0 + bays * bay / 2, y0 + d, Z + lz + 0.12), (bays * bay, 0.06, 0.12)
                _box(stage, f"{name}/Beam{li}_{int(d * 10)}", c, sz, m_beam, collide=False)
            for b in range(bays):
                cx = x0 + depth / 2 if along == "y" else x0 + (b + 0.5) * bay
                cy = y0 + (b + 0.5) * bay if along == "y" else y0 + depth / 2
                if lz > 0:
                    _box(stage, f"{name}/Deck{li}_{b}", (cx, cy, Z + lz + 0.19), (depth if along == "y" else bay - 0.1,
                         bay - 0.1 if along == "y" else depth, 0.02), m_deck, collide=False)
                for s_ in (-0.62, 0.62):          # two pallet loads per bay
                    if rng.random() < 0.15:
                        continue
                    px = cx if along == "y" else cx + s_
                    py = cy + s_ if along == "y" else cy
                    base = Z + lz + (0.20 if lz > 0 else 0.0)
                    _box(stage, f"{name}/Pal{li}_{b}_{int(s_ > 0)}", (px, py, base + 0.07), (1.0, 1.0, 0.14) if along == "y"
                         else (1.0, 1.0, 0.14), m_pal, collide=False)
                    h = rng.uniform(0.5, 1.2)
                    nx, ny = rng.choice([(2, 2), (2, 3), (3, 3), (1, 2)])
                    for ix in range(nx):
                        for iy in range(ny):
                            bw, bd = 0.98 / nx, 0.98 / ny
                            bh = h * rng.uniform(0.85, 1.0)
                            _box(stage, f"{name}/Box{li}_{b}_{int(s_ > 0)}_{ix}{iy}",
                                 (px - 0.49 + (ix + 0.5) * bw, py - 0.49 + (iy + 0.5) * bd, base + 0.14 + bh / 2),
                                 (bw - 0.015, bd - 0.015, bh), rng.choice(cards), collide=False)
    rack(R_ + "/RackA", -3.4, -6.0, 5)
    rack(R_ + "/RackB", -5.8, -6.0, 5)
    rack(R_ + "/RackC", -3.0, 5.2, 3, along="x")
    # a few floor pallets with wrapped stock near the cell
    for k, (px, py) in enumerate([(-2.3, 3.6), (3.2, 3.4), (-2.4, -3.4)]):
        _box(stage, R_ + f"/FPal{k}", (px, py, Z + 0.07), (1.0, 1.2, 0.14), m_pal, collide=False)
        _box(stage, R_ + f"/FLoad{k}", (px, py, Z + 0.14 + 0.45), (0.95, 1.15, 0.9), rng.choice(cards), collide=False)
    # camera gantry over the inspection station
    gy = INSPECT_Y
    for s_, gx in (("L", BELT_X - 0.20), ("R", BELT_X + 0.30)):
        _box(stage, R_ + f"/Gantry{s_}", (gx, gy, (Z + 0.86) / 2), (0.04, 0.04, 0.86 - Z), m_grey, collide=False)
    _box(stage, R_ + "/GantryTop", (BELT_X + 0.05, gy, 0.86), (0.54, 0.04, 0.04), m_grey, collide=False)
    ex, ey, ez = INSPECT_CAM_EYE
    _box(stage, R_ + "/GantryArm", ((ex + BELT_X + 0.30) / 2, gy, 0.84), (abs(ex - BELT_X - 0.30) + 0.04, 0.03, 0.03), m_grey, collide=False)
    _box(stage, R_ + "/CamBody", (ex, ey, ez + 0.035), (0.09, 0.025, 0.028), m_cams, collide=False)
    _cyl(stage, R_ + "/CamLensL", (ex - 0.025, ey, ez + 0.02), 0.006, 0.01, m_lens, collide=False)
    _cyl(stage, R_ + "/CamLensR", (ex + 0.025, ey, ez + 0.02), 0.006, 0.01, m_lens, collide=False)
    # high-bay lights
    for k, (lx, ly) in enumerate([(0.0, 0.0), (-4.0, -3.0), (-4.0, 3.0), (3.0, 4.0), (3.0, -4.0)]):
        L = UsdLux.RectLight.Define(stage, R_ + f"/HighBay{k}")
        L.CreateWidthAttr(1.2)
        L.CreateHeightAttr(0.4)
        L.CreateIntensityAttr(9000 if k else 6000)
        L.CreateColorTemperatureAttr(5200)
        L.CreateEnableColorTemperatureAttr(True)
        UsdGeom.Xformable(L).AddTranslateOp().Set(Gf.Vec3d(lx, ly, Z + 7.0))
        _box(stage, R_ + f"/Fixture{k}", (lx, ly, Z + 7.06), (1.3, 0.5, 0.08), m_grey, collide=False)

# ---------------------------------------------------------------- scene
def build(seed: int = 0, warehouse: bool = False, assets_root: str | None = None, room: bool = False):
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    ps = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    ps.CreateEnableCCDAttr(True)

    if warehouse and assets_root:
        wh = stage.DefinePrim("/World/Warehouse", "Xform")
        wh.GetReferences().AddReference(assets_root + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd")
        xf = UsdGeom.Xformable(wh)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp(opSuffix="cell").Set(Gf.Vec3d(0, 0, FLOOR_Z))
    else:
        m_floor = _material(stage, "/World/Looks/Floor", (0.33, 0.36, 0.40), roughness=0.8)
        _box(stage, "/World/Floor", (0, 0, FLOOR_Z - 0.01), (12, 12, 0.02), m_floor)
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(800)
    light = UsdLux.RectLight.Define(stage, "/World/CellLight")
    light.CreateWidthAttr(1.2)
    light.CreateHeightAttr(2.2)
    light.CreateIntensityAttr(2500)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(0.40, -0.1, 1.5))

    build_conveyor(stage)
    bin_range = build_reject_bin(stage)
    if room and not warehouse:
        build_room(stage, seed)

    h_ap = 20.955
    cam = UsdGeom.Camera.Define(stage, "/World/InspectionCamera")
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateVerticalApertureAttr(h_ap * INSPECT_CAM_RES[1] / INSPECT_CAM_RES[0])
    cam.CreateFocalLengthAttr(h_ap / 2.0 / math.tan(math.radians(INSPECT_CAM_HFOV) / 2.0))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 30.0))
    UsdGeom.Xformable(cam).AddTransformOp().Set(look_at(INSPECT_CAM_EYE, INSPECT_CAM_TARGET))
    pres = UsdGeom.Camera.Define(stage, "/World/PresentationCamera")
    pres.CreateHorizontalApertureAttr(h_ap)
    pres.CreateVerticalApertureAttr(h_ap * 9 / 16)
    pres.CreateFocalLengthAttr(16.0)
    pres.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
    UsdGeom.Xformable(pres).AddTransformOp().Set(look_at(PRES_EYE, PRES_TARGET))

    return {"inspection_camera": "/World/InspectionCamera", "presentation_camera": "/World/PresentationCamera",
            "reject_bin_range": bin_range, "rng": random.Random(seed)}


def bottle_pose(stage, path):
    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = m.ExtractTranslation()
    up = m.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))
    return np.array([t[0], t[1], t[2]]), tilt
