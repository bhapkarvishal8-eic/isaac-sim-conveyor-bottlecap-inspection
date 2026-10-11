"""Depth-based bottle localization and open-mouth check for the inspection camera (numpy only).

For a detected bottle: back-project the depth pixels inside its box, keep points above the belt,
take the top of the bottle (highest points), and estimate
  * the bottle axis position (x, y) = centroid of the top region (the neck / cap),
  * top height,
  * open_mouth: with no cap the top is a ring (the lip) around a hole whose inside is >= 8 mm lower;
    with a cap the top is a solid disc. We look at points inside a small radius around the top
    centroid: if a clear fraction of them lies well below the top, the mouth is open.
This is independent of the RGB detector, so a bottle is rejected only when both agree.
"""
from __future__ import annotations

import numpy as np


def intrinsics(width, height, hfov_deg):
    fx = width / 2.0 / np.tan(np.radians(hfov_deg) / 2.0)
    return {"fx": fx, "fy": fx, "cx": width / 2.0, "cy": height / 2.0}


def backproject(u, v, d, K):
    """USD camera looks down -Z with +Y up. d = distance to image plane."""
    return np.stack([(u - K["cx"]) * d / K["fx"], -(v - K["cy"]) * d / K["fy"], -d], axis=-1)


def to_world(p_cam, M):
    ph = np.concatenate([p_cam, np.ones((len(p_cam), 1))], axis=1)
    return (ph @ M)[:, :3]


def localize_bottle(xyxy, depth, K, M, belt_z=0.0, top_band=0.006, mouth_r=0.007, drop=0.008):
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth.shape[1] - 1, x2), min(depth.shape[0] - 1, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    vv, uu = np.mgrid[y1:y2 + 1, x1:x2 + 1]
    d = depth[y1:y2 + 1, x1:x2 + 1]
    ok = np.isfinite(d) & (d > 0.05) & (d < 5.0)
    if ok.sum() < 30:
        return None
    pts = to_world(backproject(uu[ok].astype(float), vv[ok].astype(float), d[ok], K), M)
    above = pts[pts[:, 2] > belt_z + 0.05]
    if len(above) < 20:
        return None
    top_z = float(np.percentile(above[:, 2], 99))
    top = above[above[:, 2] > top_z - top_band]
    cx, cy = np.median(top[:, 0]), np.median(top[:, 1])
    # all points near the axis (within mouth_r), any height in the top 40 mm
    r = np.hypot(above[:, 0] - cx, above[:, 1] - cy)
    core = above[(r < mouth_r) & (above[:, 2] > top_z - 0.040)]
    low_frac = float((core[:, 2] < top_z - drop).mean()) if len(core) >= 5 else 0.0
    return {"x": float(cx), "y": float(cy), "top_z": top_z, "n_top": int(len(top)), "n_core": int(len(core)),
            "low_frac": round(low_frac, 3), "open_mouth": bool(low_frac > 0.25)}
