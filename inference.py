"""ONNX inference for solar panel inspection models (YOLO-family detectors).

Pre-processing, NMS and drawing are ported from Solar/onnx_inference_test.ipynb
so the Streamlit app produces the same output as the notebook.

Supported ONNX output layouts:
  - (1, 4 + nc, N)  raw YOLOv8/YOLO11/YOLO26 head (end2end=False), e.g. best.onnx
  - (1, N, 4 + nc)  the same, transposed
  - (1, K, 6)       end-to-end export: x1, y1, x2, y2, score, class_id
"""
import ast
import colorsys
import time

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_CLASS_NAMES = {
    0: "Healthy",
    1: "Dust or dirt accumulation",
    2: "Bird droppings/environmental contamination",
    3: "Cracks",
    4: "Physical damage",
    5: "Other visible abnormalities",
    6: "Snow",
    7: "Leaf/feather debris",
}

# Fixed color per class (BGR for OpenCV)
CLASS_COLORS = {
    0: (80, 200, 80),    # Healthy - green
    1: (40, 160, 230),   # Dust - orange
    2: (200, 200, 60),   # Bird droppings - teal
    3: (40, 40, 230),    # Cracks - red
    4: (180, 60, 200),   # Physical damage - purple
    5: (60, 220, 230),   # Other abnormalities - yellow
    6: (230, 180, 120),  # Snow - light blue
    7: (60, 120, 160),   # Leaf/feather - brown
}

HEALTHY_NAMES = {"healthy", "clean", "normal", "good"}


def class_color(class_id):
    """Color for a class id; ids outside CLASS_COLORS get a stable generated color."""
    if class_id in CLASS_COLORS:
        return CLASS_COLORS[class_id]
    r, g, b = colorsys.hsv_to_rgb((class_id * 0.618034) % 1.0, 0.75, 0.9)
    return int(b * 255), int(g * 255), int(r * 255)


def is_healthy(class_name):
    return class_name.strip().lower() in HEALTHY_NAMES


def nms(xyxy, scores, class_ids, conf_threshold, iou_threshold):
    """Class-aware NMS. Returns indices to keep, sorted by descending score."""
    keep = []
    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        b = xyxy[idx]
        xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1).tolist()
        kept = cv2.dnn.NMSBoxes(xywh, scores[idx].tolist(), conf_threshold, iou_threshold)
        keep.extend(idx[k] for k in np.array(kept).flatten())
    return sorted(keep, key=lambda i: -scores[i])


def tile_windows(w, h, grid, overlap):
    """Top-left/bottom-right windows of a grid x grid tiling with fractional overlap."""
    if grid <= 1:
        return []
    tw = int(np.ceil(w / (grid - (grid - 1) * overlap)))
    th = int(np.ceil(h / (grid - (grid - 1) * overlap)))
    xs = np.linspace(0, w - tw, grid).astype(int)
    ys = np.linspace(0, h - th, grid).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


class SolarPanelDetector:
    def __init__(self, onnx_path, name=None):
        self.path = onnx_path
        self.name = name or onnx_path
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.metadata = meta
        try:
            self.class_names = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
        except (KeyError, ValueError, SyntaxError):
            self.class_names = DEFAULT_CLASS_NAMES
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.img_size = inp.shape[2] if isinstance(inp.shape[2], int) else 640
        self.end2end = meta.get("end2end", "").lower() == "true"

    @staticmethod
    def letterbox(img, size=640, color=(114, 114, 114)):
        """Resize keeping aspect ratio and pad to size x size (same as Ultralytics)."""
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = (size - new_w) / 2, (size - new_h) / 2
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return out, r, (left, top)

    def preprocess(self, img_bgr):
        lb, ratio, pad = self.letterbox(img_bgr, self.img_size)
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]  # 1x3xHxW
        return np.ascontiguousarray(x), ratio, pad

    def _decode(self, output):
        """Raw model output -> (xyxy, scores, class_ids) in letterbox space."""
        out = output[0]
        if self.end2end or (out.shape[-1] == 6 and out.shape[0] > out.shape[-1]):
            return out[:, :4].copy(), out[:, 4], out[:, 5].astype(int)
        preds = out.T if out.shape[0] < out.shape[1] else out  # (N, 4 + num_classes)
        boxes_cxcywh, scores_all = preds[:, :4], preds[:, 4:]
        xyxy = np.empty_like(boxes_cxcywh)
        xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        return xyxy, scores_all.max(1), scores_all.argmax(1)

    def _infer(self, img_bgr, conf_threshold, iou_threshold):
        """One forward pass -> NMS'd (xyxy, scores, class_ids) in img_bgr coords."""
        x, ratio, pad = self.preprocess(img_bgr)
        output = self.session.run(None, {self.input_name: x})[0]
        xyxy, scores, class_ids = self._decode(output)

        keep = scores >= conf_threshold
        xyxy, scores, class_ids = xyxy[keep], scores[keep], class_ids[keep]
        if len(scores) == 0:
            return xyxy.reshape(0, 4), scores, class_ids

        # undo letterbox -> original image coords
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / ratio
        h, w = img_bgr.shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)

        keep = nms(xyxy, scores, class_ids, conf_threshold, iou_threshold)
        return xyxy[keep], scores[keep], class_ids[keep]

    def detect(self, img_bgr, conf_threshold=0.25, iou_threshold=0.45, review_below=0.50,
               tiles=1, tile_overlap=0.2):
        """Returns (detections, inference_ms).

        tiles > 1 adds a tiles x tiles grid of overlapping crops on top of the full-frame
        pass, so small defects (cracks) keep enough pixels after resizing to the model input.
        """
        t0 = time.perf_counter()
        parts = [self._infer(img_bgr, conf_threshold, iou_threshold)]
        h, w = img_bgr.shape[:2]
        for x1, y1, x2, y2 in tile_windows(w, h, tiles, tile_overlap):
            b, s, c = self._infer(img_bgr[y1:y2, x1:x2], conf_threshold, iou_threshold)
            b[:, [0, 2]] += x1
            b[:, [1, 3]] += y1
            parts.append((b, s, c))
        xyxy = np.concatenate([p[0] for p in parts])
        scores = np.concatenate([p[1] for p in parts])
        class_ids = np.concatenate([p[2] for p in parts]).astype(int)
        order = nms(xyxy, scores, class_ids, conf_threshold, iou_threshold) if len(parts) > 1 \
            else range(len(scores))
        ms = (time.perf_counter() - t0) * 1000

        detections = [{
            "class_id": int(class_ids[i]),
            "class_name": self.class_names.get(int(class_ids[i]), str(class_ids[i])),
            "confidence": float(scores[i]),
            "box": [int(v) for v in xyxy[i]],
            "needs_review": float(scores[i]) < review_below,
        } for i in order]
        return detections, ms

    @staticmethod
    def draw(img_bgr, detections, labels=None):
        out = img_bgr.copy()
        thick = max(2, int(round(sum(out.shape[:2]) / 2 * 0.003)))
        font_scale = max(0.5, thick / 3)
        for n, d in enumerate(detections):
            x1, y1, x2, y2 = d["box"]
            color = class_color(d["class_id"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            label = labels[n] if labels else \
                f'{d["class_name"]} {d["confidence"]:.2f}' + (" [REVIEW]" if d["needs_review"] else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thick // 2))
            y_text = y1 - 4 if y1 - th - 8 > 0 else y1 + th + 4
            cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 4), color, -1)
            cv2.putText(out, label, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)
        return out

    def predict(self, img_bgr, conf_threshold=0.25, iou_threshold=0.45, review_below=0.50,
                tiles=1, tile_overlap=0.2):
        """Returns (detections, annotated_bgr)."""
        detections, _ = self.detect(img_bgr, conf_threshold, iou_threshold, review_below,
                                    tiles, tile_overlap)
        return detections, self.draw(img_bgr, detections)
