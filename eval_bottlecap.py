#!/usr/bin/env python3
"""Phase C acceptance: per-bottle defect metrics for the cap/no-cap RT-DETR model (ONNX, deployment path).

For every ground-truth bottle, the prediction with the highest IoU (>= 0.5) above the score threshold
decides its class. Metrics (the plan's acceptance criteria):
    nocap_recall      = no-cap bottles classified no-cap / all no-cap bottles      (target >= 0.99)
    false_reject_rate = capped bottles classified no-cap / all capped bottles      (target <= 0.02)
    missed            = bottles with no matching detection at all
A "reject" decision is made when P(no-cap) >= reject_thresh, so the threshold sweep shows the trade-off.

Run in phase4_perception_training/rtdetr_venv:
    eval_bottlecap.py model.onnx <split_dir> [--thresh 0.3] [--out report.json] [--vis N]
"""
import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def detect(sess, bgr):
    h, w = bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (640, 640)).astype(np.float32).transpose(2, 0, 1)[None]
    labels, boxes, scores = sess.run(None, {"images": x, "orig_target_sizes": np.array([[w, h]], dtype=np.int64)})
    return labels[0], boxes[0], scores[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("split_dir")
    ap.add_argument("--thresh", type=float, default=0.3, help="min detection score")
    ap.add_argument("--out")
    ap.add_argument("--vis", type=int, default=0, help="save N visualizations of errors")
    a = ap.parse_args()
    sess = ort.InferenceSession(a.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    coco = json.load(open(os.path.join(a.split_dir, "annotations.json")))
    gts = {}
    for an in coco["annotations"]:
        x, y, w, h = an["bbox"]
        gts.setdefault(an["image_id"], []).append((an["category_id"], [x, y, x + w, y + h]))
    # per GT bottle: (gt_class, best matched detections per class -> max score)
    records = []
    vis_dir = os.path.join(os.path.dirname(a.out or "."), "eval_errors") if a.vis else None
    n_vis = 0
    for im in coco["images"]:
        bgr = cv2.imread(os.path.join(a.split_dir, "images", im["file_name"]))
        labels, boxes, scores = detect(sess, bgr)
        keep = scores >= 0.05
        labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
        err = False
        for cls, gb in gts.get(im["id"], []):
            p = {0: 0.0, 1: 0.0}
            for l, b, s in zip(labels, boxes, scores):
                if iou(gb, b) >= 0.5:
                    p[int(l)] = max(p[int(l)], float(s))
            records.append((cls, p[0], p[1]))
            decided = 1 if p[1] >= a.thresh and p[1] >= p[0] else (0 if p[0] >= a.thresh else -1)
            err |= decided != cls
        if err and vis_dir and n_vis < a.vis:
            os.makedirs(vis_dir, exist_ok=True)
            v = bgr.copy()
            for cls, gb in gts.get(im["id"], []):
                cv2.rectangle(v, tuple(map(int, gb[:2])), tuple(map(int, gb[2:])), (255, 255, 255), 1)
            for l, b, s in zip(labels, boxes, scores):
                if s >= a.thresh:
                    col = (0, 200, 0) if int(l) == 0 else (0, 0, 255)
                    cv2.rectangle(v, tuple(map(int, b[:2])), tuple(map(int, b[2:])), col, 2)
                    cv2.putText(v, f"{['ok', 'NOCAP'][int(l)]} {s:.2f}", (int(b[0]), int(b[1]) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.imwrite(os.path.join(vis_dir, im["file_name"]), v)
            n_vis += 1

    R = np.array(records)
    cls, p_ok, p_no = R[:, 0].astype(int), R[:, 1], R[:, 2]
    n_no, n_ok = int((cls == 1).sum()), int((cls == 0).sum())

    def at(t):
        det = (np.maximum(p_ok, p_no) >= a.thresh)
        rej = (p_no >= t) & (p_no >= p_ok) & det | ((p_no >= t) & ~det)
        return {"reject_thresh": t,
                "nocap_recall": round(float(rej[cls == 1].mean()), 4),
                "false_reject_rate": round(float(rej[cls == 0].mean()), 4),
                "missed_detections": int((~det & (p_no < t)).sum())}

    report = {"model": a.model, "split": a.split_dir, "images": len(coco["images"]),
              "bottles": {"ok": n_ok, "nocap": n_no}, "det_thresh": a.thresh,
              "at_det_thresh": at(a.thresh),
              "sweep": [at(t) for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)]}
    print(json.dumps(report, indent=1))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
