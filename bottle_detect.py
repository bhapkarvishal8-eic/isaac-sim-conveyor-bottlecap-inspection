#!/usr/bin/env python3
"""Bottle cap / no-cap RT-DETR inference server (runs in phase4_perception_training/rtdetr_venv).

Protocol (one request per stdin line):  <image.png> <out.json> [<min_score>]
Reply: 'OK <n>' or 'ERR <msg>'. JSON: {"bottles": [{"xyxy": [...], "p_ok": s, "p_nocap": s}]}
RT-DETR often returns an ok and a no-cap query on the same bottle; queries are grouped per bottle
(IoU >= 0.5) and the per-class maximum score is kept, so the caller sees one entry per bottle.
Preprocessing matches training: plain resize to 640x640, RGB, float32 in [0,255].
"""
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def detect(sess, bgr, min_score=0.2):
    h, w = bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (640, 640)).astype(np.float32).transpose(2, 0, 1)[None]
    labels, boxes, scores = sess.run(None, {"images": x, "orig_target_sizes": np.array([[w, h]], dtype=np.int64)})
    dets = sorted(((float(s), int(l), [float(v) for v in b]) for l, b, s in zip(labels[0], boxes[0], scores[0])
                   if s >= min_score), key=lambda d: -d[0])
    bottles = []
    for s, l, b in dets:
        for g in bottles:
            if iou(g["xyxy"], b) >= 0.5:
                g["p_ok" if l == 0 else "p_nocap"] = max(g["p_ok" if l == 0 else "p_nocap"], s)
                break
        else:
            bottles.append({"xyxy": b, "p_ok": s if l == 0 else 0.0, "p_nocap": s if l == 1 else 0.0})
    return bottles


def main():
    model = sys.argv[1]
    sess = ort.InferenceSession(model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("READY " + ",".join(sess.get_providers()), flush=True)
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        try:
            bottles = detect(sess, cv2.imread(parts[0]), float(parts[2]) if len(parts) > 2 else 0.2)
            json.dump({"bottles": bottles}, open(parts[1], "w"))
            print(f"OK {len(bottles)}", flush=True)
        except Exception as e:
            print(f"ERR {e!r}", flush=True)


if __name__ == "__main__":
    main()
