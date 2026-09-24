"""ONNX inference for the solar panel inspection model (YOLO26n, best.onnx).

Pre-processing, NMS and drawing are ported from Solar/onnx_inference_test.ipynb
so the Streamlit app produces the same output as the notebook.
"""
import ast

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


class SolarPanelDetector:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        try:
            self.class_names = {int(k): v for k, v in ast.literal_eval(meta["names"]).items()}
        except (KeyError, ValueError, SyntaxError):
            self.class_names = DEFAULT_CLASS_NAMES
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.img_size = inp.shape[2] if isinstance(inp.shape[2], int) else 640

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

    def postprocess(self, output, ratio, pad, orig_shape, conf_threshold, iou_threshold, review_below):
        preds = output[0].T  # (8400, 4 + num_classes)
        boxes_cxcywh, scores_all = preds[:, :4], preds[:, 4:]
        class_ids = scores_all.argmax(1)
        scores = scores_all.max(1)

        keep = scores >= conf_threshold
        boxes_cxcywh, scores, class_ids = boxes_cxcywh[keep], scores[keep], class_ids[keep]
        if len(scores) == 0:
            return []

        # cx,cy,w,h -> x1,y1,x2,y2 (in letterbox space)
        xyxy = np.empty_like(boxes_cxcywh)
        xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # undo letterbox -> original image coords
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / ratio
        h, w = orig_shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)

        # class-aware NMS
        results = []
        for c in np.unique(class_ids):
            idx = np.where(class_ids == c)[0]
            b = xyxy[idx]
            xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1).tolist()
            kept = cv2.dnn.NMSBoxes(xywh, scores[idx].tolist(), conf_threshold, iou_threshold)
            for k in np.array(kept).flatten():
                i = idx[k]
                results.append({
                    "class_id": int(c),
                    "class_name": self.class_names.get(int(c), str(c)),
                    "confidence": float(scores[i]),
                    "box": [int(v) for v in xyxy[i]],
                    "needs_review": float(scores[i]) < review_below,
                })
        return sorted(results, key=lambda d: -d["confidence"])

    @staticmethod
    def draw(img_bgr, detections):
        out = img_bgr.copy()
        thick = max(2, int(round(sum(out.shape[:2]) / 2 * 0.003)))
        font_scale = max(0.5, thick / 3)
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            color = CLASS_COLORS.get(d["class_id"], (255, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            label = f'{d["class_name"]} {d["confidence"]:.2f}' + (" [REVIEW]" if d["needs_review"] else "")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thick // 2))
            y_text = y1 - 4 if y1 - th - 8 > 0 else y1 + th + 4
            cv2.rectangle(out, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 4), color, -1)
            cv2.putText(out, label, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), max(1, thick // 2), cv2.LINE_AA)
        return out

    def predict(self, img_bgr, conf_threshold=0.25, iou_threshold=0.45, review_below=0.50):
        """Returns (detections, annotated_bgr)."""
        x, ratio, pad = self.preprocess(img_bgr)
        output = self.session.run(None, {self.input_name: x})[0]
        detections = self.postprocess(output, ratio, pad, img_bgr.shape,
                                      conf_threshold, iou_threshold, review_below)
        return detections, self.draw(img_bgr, detections)
