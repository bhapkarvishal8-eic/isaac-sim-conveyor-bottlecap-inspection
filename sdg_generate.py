"""Phase B: synthetic training data for cap / no-cap bottle detection, from the inspection camera.

Renders the Phase A conveyor cell from the deployment inspection camera (small pose jitter only, so
the pixel scale matches the demo) with domain randomization, and writes COCO detection datasets:
    <OUT>/<split>/images/000000.jpg,  <OUT>/<split>/annotations.json
Classes (0-indexed, same convention as the Phase 3/4 RT-DETR pipeline): 0 = bottle_ok, 1 = bottle_nocap.

Splits:
    train, val    -- same distribution
    test_unseen   -- body/cap colours from a hue band never used in training, labels from a held-out
                     texture set, and a different (conical) shoulder shape: measures generalization.

Run: kit --no-window --exec sdg_generate.py   (env: SDG_OUT, SDG_TRAIN, SDG_VAL, SDG_TEST, SDG_SEED)
"""
import colorsys
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/home/vishal/poc/conveyor_inspection")
import carb.settings
import omni.kit.app
import omni.timeline
import omni.usd
import isaacsim.core.experimental.utils.app as app_utils
import omni.replicator.core as rep
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdSemantics, UsdShade

import cell_scene as C

OUT = os.environ.get("SDG_OUT", "/home/vishal/poc/conveyor_inspection/sdg_data")
COUNTS = {"train": int(os.environ.get("SDG_TRAIN", "24000")), "val": int(os.environ.get("SDG_VAL", "3000")),
          "test_unseen": int(os.environ.get("SDG_TEST", "2000"))}
SEED = int(os.environ.get("SDG_SEED", "1234"))
MAX_BOTTLES = 6
RENDER_UPDATES = 4          # app updates per frame so RTX temporal effects settle on the new scene
W, H = C.INSPECT_CAM_RES
CLASSES = ["bottle_ok", "bottle_nocap"]
HELD_OUT_HUE = (0.42, 0.58)  # cyan/teal body and cap colours appear only in test_unseen
TEX_DIR = os.path.join(OUT, "_label_textures")
N_TEX_TRAIN, N_TEX_TEST = 60, 16
LOG_EVERY = 500


def log(m):
    print(f"[SDG {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------- label textures
def make_label_textures(rng):
    os.makedirs(TEX_DIR, exist_ok=True)
    try:
        fonts = [ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{f}", s) for f, s in
                 [("DejaVuSans-Bold.ttf", 54), ("DejaVuSans.ttf", 40), ("DejaVuSerif-Bold.ttf", 48)]]
    except Exception:
        fonts = [ImageFont.load_default()]
    words = ["AQUA", "PURE", "FRESH", "CLEAN", "SOFT", "ULTRA", "NATURAL", "ORGANIC", "CLASSIC", "ACTIVE", "MILD",
             "SHAMPOO", "LOTION", "WASH", "OIL", "JUICE", "SODA", "DET", "PLUS", "MAX", "LITE", "GOLD", "ZERO"]
    paths = []
    for i in range(N_TEX_TRAIN + N_TEX_TEST):
        bg = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.0, 0.8), rng.uniform(0.5, 1.0)))
        im = Image.new("RGB", (1024, 256), bg)
        d = ImageDraw.Draw(im)
        for _ in range(rng.randint(0, 4)):     # stripes / bands
            y0 = rng.randint(0, 230)
            col = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.2, 1), rng.uniform(0.2, 1)))
            d.rectangle([0, y0, 1024, y0 + rng.randint(6, 40)], fill=col)
        for _ in range(rng.randint(1, 3)):     # text
            txt = " ".join(rng.choice(words) for _ in range(rng.randint(1, 2)))
            col = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(rng.random(), rng.uniform(0, 1), rng.uniform(0, 0.6)))
            d.text((rng.randint(10, 600), rng.randint(10, 180)), txt, font=rng.choice(fonts), fill=col)
        if rng.random() < 0.6:                 # barcode
            x = rng.randint(700, 900)
            for k in range(40):
                if rng.random() < 0.55:
                    d.rectangle([x + 3 * k, 150, x + 3 * k + rng.choice([1, 2]), 235], fill=(10, 10, 10))
        p = os.path.join(TEX_DIR, f"label_{i:03d}.png")
        im.save(p)
        paths.append(p)
    return paths[:N_TEX_TRAIN], paths[N_TEX_TRAIN:]


# ----------------------------------------------------------------------------- geometry
def profile(spec, conical=False):
    br, bh, sh, nr, nh = spec["body_r"], spec["body_h"], spec["shoulder_h"], spec["neck_r"], spec["neck_h"]
    pts = [(0.0, 0.0), (br - 0.006, 0.0)]
    for k in range(1, 5):
        a = k / 4 * math.pi / 2
        pts.append((br - 0.006 + 0.006 * math.sin(a), 0.006 - 0.006 * math.cos(a)))
    pts.append((br, bh))
    for k in range(1, 9):
        t = k / 8
        f = (1 - t) if conical else 0.5 * (1 + math.cos(math.pi * t))
        pts.append((nr + (br - nr) * f, bh + sh * t))
    top = bh + sh + nh
    lip = spec["lip"]
    pts += [(nr, top - 0.004), (nr + lip, top - 0.003), (nr + lip, top), (nr - 0.0022, top),
            (nr - 0.0022, top - 0.030), (0.0, top - 0.030)]
    return pts


def lathe_arrays(prof, n, uv_z=None):
    """Points/normals/indices for a surface of revolution. uv_z=(z0,z1) adds a seam column + st coords."""
    cols = n + 1 if uv_z else n
    pts, nrm, st = [], [], []
    rings = len(prof)
    for i, (r, z) in enumerate(prof):
        r0, z0 = prof[max(0, i - 1)]
        r1, z1 = prof[min(rings - 1, i + 1)]
        nr_, nz_ = (z1 - z0), -(r1 - r0)
        ln = math.hypot(nr_, nz_) or 1.0
        nr_, nz_ = nr_ / ln, nz_ / ln
        for j in range(cols):
            a = 2 * math.pi * j / n
            pts.append(Gf.Vec3f(r * math.cos(a), r * math.sin(a), z))
            nrm.append(Gf.Vec3f(nr_ * math.cos(a), nr_ * math.sin(a), nz_))
            if uv_z:
                st.append(Gf.Vec2f(j / n, (z - uv_z[0]) / (uv_z[1] - uv_z[0])))
    counts, idx = [], []
    for i in range(rings - 1):
        for j in range(n):
            a = i * cols + j
            b = i * cols + (j + 1) % cols if not uv_z else i * cols + j + 1
            idx += [a, b, b + cols, a + cols]
            counts.append(4)
    rmax = max(r for r, _ in prof)
    ext = [Gf.Vec3f(-rmax, -rmax, min(z for _, z in prof)), Gf.Vec3f(rmax, rmax, max(z for _, z in prof))]
    return pts, nrm, counts, idx, st, ext


def material(stage, path):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat, sh


def textured_material(stage, path):
    mat, sh = material(stage, path)
    reader = UsdShade.Shader.Define(stage, path + "/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, path + "/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
    return mat, sh, tex


class BottleRig:
    """One reusable bottle: body, label sleeve, cap. Re-shaped / recoloured every frame."""

    def __init__(self, stage, i):
        self.path = f"/World/SDG/Bottle{i}"
        self.stage = stage
        root = UsdGeom.Xform.Define(stage, self.path)
        self.t_op = root.AddTranslateOp()
        self.r_op = root.AddOrientOp()
        self.root = root.GetPrim()
        self.body = UsdGeom.Mesh.Define(stage, self.path + "/body")
        self.label = UsdGeom.Mesh.Define(stage, self.path + "/label")
        self.cap = UsdGeom.Mesh.Define(stage, self.path + "/cap")
        for m in (self.body, self.label, self.cap):
            m.CreateSubdivisionSchemeAttr("none")
            m.CreateDoubleSidedAttr(True)
        self.body_mat, self.body_sh = material(stage, self.path + "/Looks/body")
        self.cap_mat, self.cap_sh = material(stage, self.path + "/Looks/cap")
        self.label_mat, self.label_sh, self.label_tex = textured_material(stage, self.path + "/Looks/label")
        for m, mat in ((self.body, self.body_mat), (self.cap, self.cap_mat), (self.label, self.label_mat)):
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

    @staticmethod
    def _set_mesh(mesh, arrays, with_st=False):
        pts, nrm, counts, idx, st, ext = arrays
        mesh.GetPointsAttr().Set(pts) if mesh.GetPointsAttr() else mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr().Set(counts)
        mesh.CreateFaceVertexIndicesAttr().Set(idx)
        mesh.CreateNormalsAttr().Set(nrm)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateExtentAttr().Set(ext)
        if with_st:
            pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(st)

    def hide(self):
        UsdGeom.Imageable(self.root).MakeInvisible()

    def apply(self, spec, x, y, yaw, tilt):
        UsdGeom.Imageable(self.root).MakeVisible()
        self.t_op.Set(Gf.Vec3d(x, y, 0.0))
        q = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw) * Gf.Rotation(Gf.Vec3d(1, 0, 0), tilt[0]) * Gf.Rotation(Gf.Vec3d(0, 1, 0), tilt[1])
        qq = q.GetQuat()
        self.r_op.Set(Gf.Quatf(qq.GetReal(), *[float(v) for v in qq.GetImaginary()]))
        self._set_mesh(self.body, lathe_arrays(profile(spec, spec["conical"]), 48))
        self.body_sh.GetInput("diffuseColor").Set(Gf.Vec3f(*spec["body_rgb"]))
        self.body_sh.GetInput("roughness").Set(spec["body_rough"])
        if spec["label"]:
            UsdGeom.Imageable(self.label).MakeVisible()
            z0, z1 = spec["label_z"]
            r = spec["body_r"] + 0.0008
            prof = [(r, z0 + k * (z1 - z0) / 6) for k in range(7)]
            self._set_mesh(self.label, lathe_arrays(prof, 48, uv_z=(z0, z1)), with_st=True)
            self.label_tex.GetInput("file").Set(Sdf.AssetPath(spec["label_tex"]))
            self.label_sh.GetInput("roughness").Set(spec["label_rough"])
        else:
            UsdGeom.Imageable(self.label).MakeInvisible()
        if spec["capped"]:
            UsdGeom.Imageable(self.cap).MakeVisible()
            base = spec["body_h"] + spec["shoulder_h"] + spec["neck_h"] - 0.004
            cr, ch = spec["cap_r"], spec["cap_h"]
            prof = [(0.0, base), (cr - 0.001, base), (cr, base + 0.001), (cr, base + ch - 0.002),
                    (cr - spec["cap_chamfer"], base + ch), (0.0, base + ch)]
            self._set_mesh(self.cap, lathe_arrays(prof, 48))
            self.cap_sh.GetInput("diffuseColor").Set(Gf.Vec3f(*spec["cap_rgb"]))
            self.cap_sh.GetInput("roughness").Set(spec["cap_rough"])
        else:
            UsdGeom.Imageable(self.cap).MakeInvisible()
        import omni.replicator.core.functional as F
        F.modify.semantics(self.root, {"class": [CLASSES[0] if spec["capped"] else CLASSES[1]]}, mode="replace")


# ----------------------------------------------------------------------------- randomization
def rand_color(rng, unseen, sat=(0.0, 0.9), val=(0.15, 1.0)):
    if unseen:
        h = rng.uniform(*HELD_OUT_HUE)
        return colorsys.hsv_to_rgb(h, rng.uniform(0.35, 0.9), rng.uniform(0.3, 1.0))
    while True:
        h = rng.random()
        if not (HELD_OUT_HUE[0] <= h <= HELD_OUT_HUE[1]):
            return colorsys.hsv_to_rgb(h, rng.uniform(*sat), rng.uniform(*val))


def rand_bottle(rng, unseen, textures):
    br = rng.uniform(0.026, 0.038)
    nr = rng.uniform(0.011, 0.016)
    bh = rng.uniform(0.11, 0.18)
    capped = rng.random() < 0.5
    cap_same_as_body = rng.random() < 0.12          # hard case: cap colour == body colour
    body_rgb = rand_color(rng, unseen)
    return {
        "body_r": br, "body_h": bh, "shoulder_h": rng.uniform(0.02, 0.045), "neck_r": nr,
        "neck_h": rng.uniform(0.016, 0.028), "lip": rng.uniform(0.0008, 0.0025),
        "conical": unseen, "capped": capped,
        "cap_r": nr + rng.uniform(0.0015, 0.0045), "cap_h": rng.uniform(0.012, 0.026),
        "cap_chamfer": rng.uniform(0.0005, 0.003),
        "body_rgb": body_rgb, "body_rough": rng.uniform(0.2, 0.7),
        "cap_rgb": body_rgb if cap_same_as_body else rand_color(rng, unseen, val=(0.05, 1.0)),
        "cap_rough": rng.uniform(0.2, 0.6),
        "label": rng.random() < 0.7, "label_tex": rng.choice(textures), "label_rough": rng.uniform(0.3, 0.8),
        "label_z": (bh * rng.uniform(0.15, 0.35), bh * rng.uniform(0.65, 0.9)),
    }


def bottle_top_z(spec):
    top = spec["body_h"] + spec["shoulder_h"] + spec["neck_h"]
    return top + (spec["cap_h"] - 0.004 if spec["capped"] else 0.0)


# ----------------------------------------------------------------------------- camera
def camera_matrix(eye, target):
    return np.array(C.look_at(eye, target))


def project(p, K, M):
    pc = np.append(np.asarray(p, float), 1.0) @ np.linalg.inv(M)
    d = -pc[2]
    return K["cx"] + pc[0] * K["fx"] / d, K["cy"] - pc[1] * K["fy"] / d


def main():
    rng = random.Random(SEED)
    tex_train, tex_test = make_label_textures(rng)
    omni.usd.get_context().new_stage()
    app_utils.update_app(steps=2)
    scene = C.build(seed=SEED)
    stage = omni.usd.get_context().get_stage()
    # extra randomizable materials on belt / guides / floor
    def rebind(prim_path, mat_path):
        mat, sh = material(stage, mat_path)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path)).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
        return sh
    belt_sh = rebind("/World/Conveyor/Belt", "/World/Looks/SDG_Belt")
    guide_sh = [rebind(f"/World/Conveyor/Guide{s}", f"/World/Looks/SDG_Guide{s}") for s in "LR"]
    floor_sh = rebind("/World/Floor", "/World/Looks/SDG_Floor")
    rail_sh = [rebind(f"/World/Conveyor/Rail{s}", f"/World/Looks/SDG_Rail{s}") for s in "LR"]
    light = UsdLux.RectLight(stage.GetPrimAtPath("/World/CellLight"))
    light_xf = UsdGeom.Xformable(light.GetPrim()).GetOrderedXformOps()[0]
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/DomeLight"))
    light.CreateEnableColorTemperatureAttr(True)
    fill = UsdLux.SphereLight.Define(stage, "/World/SDG_Fill")
    fill.CreateRadiusAttr(0.15)
    fill_op = UsdGeom.Xformable(fill).AddTranslateOp()
    rigs = [BottleRig(stage, i) for i in range(MAX_BOTTLES)]
    cam = UsdGeom.Camera(stage.GetPrimAtPath(scene["inspection_camera"]))
    cam_op = UsdGeom.Xformable(cam).GetOrderedXformOps()[0]
    h_ap = cam.GetHorizontalApertureAttr().Get()

    settings = carb.settings.get_settings()
    settings.set("/rtx/post/dlss/execMode", 0)
    rp = rep.create.render_product(scene["inspection_camera"], (W, H))
    ann_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    ann_bb = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    ann_rgb.attach([rp])
    ann_bb.attach([rp])
    omni.timeline.get_timeline_interface().play()
    app_utils.update_app(steps=30)

    stats = {}
    for split, n_frames in COUNTS.items():
        if n_frames <= 0:
            continue
        unseen = split == "test_unseen"
        textures = tex_test if unseen else tex_train
        img_dir = os.path.join(OUT, split, "images")
        os.makedirs(img_dir, exist_ok=True)
        coco = {"images": [], "annotations": [],
                "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)],
                "info": {"description": f"conveyor bottle cap/no-cap, split={split}, seed={SEED}"}}
        n_ann = [0, 0]
        dropped = 0
        t0 = time.time()
        fid = 0
        while fid < n_frames:
            # ---- scene randomization
            belt_sh.GetInput("diffuseColor").Set(Gf.Vec3f(*rand_color(rng, False, sat=(0, 0.5), val=(0.04, 0.45))))
            belt_sh.GetInput("roughness").Set(rng.uniform(0.4, 0.95))
            g = rng.uniform(0.45, 0.9)
            for s in guide_sh:
                s.GetInput("diffuseColor").Set(Gf.Vec3f(g, g, g * rng.uniform(0.95, 1.05)))
                s.GetInput("metallic").Set(rng.uniform(0.0, 0.9))
                s.GetInput("roughness").Set(rng.uniform(0.2, 0.6))
            rc = rand_color(rng, False, sat=(0, 0.6), val=(0.1, 0.8))
            for s in rail_sh:
                s.GetInput("diffuseColor").Set(Gf.Vec3f(*rc))
            floor_sh.GetInput("diffuseColor").Set(Gf.Vec3f(*rand_color(rng, False, sat=(0, 0.5), val=(0.15, 0.7))))
            light.GetIntensityAttr().Set(rng.uniform(1200, 4500))
            light.CreateColorTemperatureAttr().Set(rng.uniform(3500, 7500))
            light_xf.Set(Gf.Vec3d(C.BELT_X + rng.uniform(-0.5, 0.3), C.INSPECT_Y + rng.uniform(-0.5, 0.5), rng.uniform(1.1, 2.0)))
            dome.GetIntensityAttr().Set(rng.uniform(300, 1400))
            fill.CreateIntensityAttr().Set(rng.uniform(0, 30000) if rng.random() < 0.5 else 0.0)
            fill_op.Set(Gf.Vec3d(C.BELT_X + rng.uniform(-0.8, 0.8), C.INSPECT_Y + rng.uniform(-0.8, 0.8), rng.uniform(0.4, 1.2)))
            # camera: deployment pose + small jitter (keeps the pixel scale the detector will see)
            eye = np.array(C.INSPECT_CAM_EYE) + np.array([rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.uniform(-0.03, 0.03)])
            # target moves with the eye (+-1 cm extra) -> view roll stays within a few degrees
            tgt = np.array(C.INSPECT_CAM_TARGET) + np.array([eye[0] - C.INSPECT_CAM_EYE[0], eye[1] - C.INSPECT_CAM_EYE[1], 0.0]) \
                + np.array([rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), 0.0])
            M = camera_matrix(tuple(eye), tuple(tgt))
            cam_op.Set(Gf.Matrix4d(*M.flatten().tolist()))
            hfov = C.INSPECT_CAM_HFOV + rng.uniform(-1.5, 1.5)
            fl = h_ap / 2.0 / math.tan(math.radians(hfov) / 2.0)
            cam.GetFocalLengthAttr().Set(fl)
            K = {"fx": W * fl / h_ap, "fy": W * fl / h_ap, "cx": W / 2.0, "cy": H / 2.0}
            # bottles: 1..6 along the visible belt, spaced (sometimes touching), inside the guides
            nb = rng.choices([1, 2, 3, 4, 5, 6], weights=[1, 3, 4, 4, 2, 1])[0]
            ys = sorted(rng.uniform(C.INSPECT_Y - 0.50, C.INSPECT_Y + 0.50) for _ in range(nb))
            placed, specs = [], []
            for y in ys:
                spec = rand_bottle(rng, unseen, textures)
                if placed and y - placed[-1][1] < placed[-1][2] + spec["body_r"] + 0.002:
                    y = placed[-1][1] + placed[-1][2] + spec["body_r"] + rng.uniform(0.002, 0.03)
                x = C.BELT_X + rng.uniform(-1, 1) * max(0.0, C.GUIDE_GAP / 2 - spec["body_r"] - 0.003)
                # keep only bottles fully inside the frame: a partly visible bottle would otherwise be an
                # unlabeled "background" bottle (label noise); at run time the decision is made mid-frame
                r, top = spec["body_r"], bottle_top_z(spec)
                corners = [(x + dx, y + dy, z) for dx in (-r, r) for dy in (-r, r) for z in (0.0, top)]
                uv = [project(c, K, M) for c in corners]
                if not all(3 <= u <= W - 3 and 3 <= v <= H - 3 for u, v in uv):
                    continue
                placed.append((x, y, spec["body_r"]))
                specs.append(spec)
            if not specs:
                continue
            for i, rig in enumerate(rigs):
                if i < len(specs):
                    x, y, _ = placed[i]
                    tilt = (rng.gauss(0, 0.8), rng.gauss(0, 0.8))
                    rig.apply(specs[i], x, y, rng.uniform(0, 360), tilt)
                else:
                    rig.hide()
            app_utils.update_app(steps=RENDER_UPDATES)
            rgb = ann_rgb.get_data()
            bb = ann_bb.get_data()
            if getattr(rgb, "ndim", 0) != 3 or bb is None:
                continue
            # ---- annotations
            if os.environ.get("SDG_DEBUG") and fid < 2:
                log(f"BBDEBUG keys={list(bb.keys())} info={ {k: (str(v)[:300]) for k, v in bb['info'].items()} } data={str(bb['data'])[:400]} dtype={getattr(bb['data'], 'dtype', None)}")
            info = bb["info"]
            paths = info.get("primPaths", [])
            anns = []
            for k, row in enumerate(bb["data"]):
                path = paths[k] if k < len(paths) else ""
                if not path.startswith("/World/SDG/Bottle"):
                    continue
                bi = int(path.split("Bottle")[1].split("/")[0])
                if bi >= len(specs):
                    continue
                spec = specs[bi]
                x0, y0, x1, y1 = float(row["x_min"]), float(row["y_min"]), float(row["x_max"]), float(row["y_max"])
                occ = float(row["occlusionRatio"])
                # the cap / mouth must be in view for the class to be decidable
                bx, by, _ = placed[bi]
                u, v = project((bx, by, bottle_top_z(spec)), K, M)
                if occ > 0.6 or not (4 <= u <= W - 4 and 4 <= v <= H - 4) or (x1 - x0) < 12 or (y1 - y0) < 12:
                    dropped += 1
                    continue
                cls = 0 if spec["capped"] else 1
                n_ann[cls] += 1
                anns.append({"id": len(coco["annotations"]) + len(anns), "image_id": fid, "category_id": cls,
                             "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1], "area": (x1 - x0 + 1) * (y1 - y0 + 1),
                             "iscrowd": 0, "occlusion": round(occ, 3)})
            fname = f"{fid:06d}.jpg"
            Image.fromarray(np.asarray(rgb)[:, :, :3]).save(os.path.join(img_dir, fname), quality=94)
            coco["images"].append({"id": fid, "file_name": fname, "width": W, "height": H})
            coco["annotations"] += anns
            fid += 1
            if fid % LOG_EVERY == 0:
                el = time.time() - t0
                log(f"{split}: {fid}/{n_frames} frames, {el / fid:.3f} s/frame, ETA {(n_frames - fid) * el / fid / 60:.1f} min, "
                    f"ok={n_ann[0]} nocap={n_ann[1]} dropped={dropped}")
                json.dump(coco, open(os.path.join(OUT, split, "annotations.json"), "w"))
        json.dump(coco, open(os.path.join(OUT, split, "annotations.json"), "w"))
        stats[split] = {"frames": fid, "bottle_ok": n_ann[0], "bottle_nocap": n_ann[1], "dropped_undecidable": dropped,
                        "minutes": round((time.time() - t0) / 60, 1)}
        log(f"SPLIT DONE {split}: {json.dumps(stats[split])}")
    json.dump({"seed": SEED, "splits": stats, "held_out_hue": HELD_OUT_HUE, "camera_eye": C.INSPECT_CAM_EYE,
               "camera_target": C.INSPECT_CAM_TARGET, "hfov": C.INSPECT_CAM_HFOV},
              open(os.path.join(OUT, "dataset_stats.json"), "w"), indent=1)
    log("SDG COMPLETE " + json.dumps(stats))


try:
    main()
except Exception:
    log("SDG FAILED\n" + traceback.format_exc())
finally:
    omni.kit.app.get_app().post_quit()
