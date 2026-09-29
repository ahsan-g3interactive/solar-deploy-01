"""Incident-level evaluation of video results against labelled physical incidents.

Ground-truth CSV (one or more rows per physical incident, "Healthy" rows are ignored):

    incident_id,class_name,frame,x1,y1,x2,y2          spatial: box on one or more frames
    incident_id,class_name,start_frame,end_frame      temporal: when the incident is visible
    incident_id,class_name,start_s,end_s              temporal, in seconds

Box coordinates are pixels in the original video frame. The incidents CSV exported by the
app uses the same columns, so it can be corrected by a reviewer and uploaded back as ground truth.
"""
import pandas as pd

from inference import is_healthy
from video import iou


def load_ground_truth(csv_file, fps):
    df = pd.read_csv(csv_file)
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"incident_id", "class_name"} <= set(df.columns):
        raise ValueError("Ground truth CSV needs 'incident_id' and 'class_name' columns.")
    if "start_s" in df.columns and "start_frame" not in df.columns:
        df["start_frame"] = (df["start_s"] * fps).round()
        df["end_frame"] = (df["end_s"] * fps).round()
    has_boxes = {"frame", "x1", "y1", "x2", "y2"} <= set(df.columns)
    has_span = {"start_frame", "end_frame"} <= set(df.columns)
    if not (has_boxes or has_span):
        raise ValueError("Ground truth CSV needs either frame,x1,y1,x2,y2 or "
                         "start_frame,end_frame (or start_s,end_s) columns.")

    incidents = []
    for inc_id, g in df.groupby("incident_id", sort=False):
        cls = str(g["class_name"].iloc[0]).strip()
        if is_healthy(cls):
            continue
        boxes = []
        if has_boxes:
            for _, r in g.dropna(subset=["frame", "x1", "y1", "x2", "y2"]).iterrows():
                boxes.append((int(r["frame"]), [r["x1"], r["y1"], r["x2"], r["y2"]]))
        frames = [b[0] for b in boxes]
        if has_span and g["start_frame"].notna().any():
            frames += [int(g["start_frame"].min()), int(g["end_frame"].max())]
        if not frames:
            continue
        incidents.append({"incident_id": inc_id, "class_name": cls, "boxes": boxes,
                          "start_frame": min(frames), "end_frame": max(frames)})
    return incidents


def _match_score(gt, pred, frame_tol, iou_threshold):
    if gt["boxes"]:
        best = 0.0
        for f, gbox in gt["boxes"]:
            for o in pred["observations"]:
                if abs(o["frame"] - f) <= frame_tol:
                    best = max(best, iou(gbox, o["box"]))
        return best if best >= iou_threshold else 0.0
    # temporal only: overlap of visible spans (widened by the sampling tolerance)
    s = max(gt["start_frame"] - frame_tol, pred["start_frame"])
    e = min(gt["end_frame"] + frame_tol, pred["end_frame"])
    if e < s:
        return 0.0
    span = max(gt["end_frame"], pred["end_frame"]) - min(gt["start_frame"], pred["start_frame"])
    return (e - s + 1) / (span + 1)


def _greedy_match(gts, preds, frame_tol, iou_threshold, class_aware):
    pairs = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            if class_aware and g["class_name"] != p["class_name"]:
                continue
            s = _match_score(g, p, frame_tol, iou_threshold)
            if s > 0:
                pairs.append((s, gi, pi))
    matches, used_g, used_p = {}, set(), set()
    for _, gi, pi in sorted(pairs, reverse=True):
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            matches[gi] = pi
    return matches


def _ratio(a, b):
    return a / b if b else None


def evaluate(gts, preds, panels, frame_tol, iou_threshold=0.3):
    """Returns (summary dict, per-class DataFrame, missed list, false-alert list)."""
    strict = _greedy_match(gts, preds, frame_tol, iou_threshold, class_aware=True)
    loose = _greedy_match(gts, preds, frame_tol, iou_threshold, class_aware=False)
    tp = len(strict)
    fp_loose = len(preds) - len(loose)  # detections that match no real incident of any type
    recall, precision = _ratio(tp, len(gts)), _ratio(tp, len(preds))
    summary = {
        "gt_incidents": len(gts),
        "pred_incidents": len(preds),
        "true_positives": tp,
        "missed": len(gts) - tp,
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / (recall + precision) if recall and precision else None,
        "defect_recall_any_type": _ratio(len(loose), len(gts)),
        "defect_precision_any_type": _ratio(len(loose), len(preds)),
        "false_alerts": fp_loose,
        "false_alerts_per_100_panels": _ratio(fp_loose * 100, panels),
        "panels": panels,
    }

    classes = sorted({g["class_name"] for g in gts} | {p["class_name"] for p in preds})
    matched_p = set(strict.values())
    rows = []
    for c in classes:
        g_idx = [i for i, g in enumerate(gts) if g["class_name"] == c]
        p_idx = [i for i, p in enumerate(preds) if p["class_name"] == c]
        found = sum(i in strict for i in g_idx)
        correct = sum(i in matched_p for i in p_idx)
        rows.append({"class_name": c, "gt": len(g_idx), "found": found,
                     "recall": _ratio(found, len(g_idx)), "predicted": len(p_idx),
                     "correct": correct, "precision": _ratio(correct, len(p_idx)),
                     "false_alerts": len(p_idx) - correct})
    per_class = pd.DataFrame(rows)

    missed = [{"incident_id": g["incident_id"], "class_name": g["class_name"],
               "start_frame": g["start_frame"], "end_frame": g["end_frame"],
               "found_as_other_type": gi in loose}
              for gi, g in enumerate(gts) if gi not in strict]
    loose_p = set(loose.values())
    false_alerts = [{"incident_id": p["track_id"], "class_name": p["class_name"],
                     "start_frame": p["start_frame"], "end_frame": p["end_frame"],
                     "max_confidence": round(p["max_confidence"], 3)}
                    for pi, p in enumerate(preds) if pi not in loose_p]
    return summary, per_class, missed, false_alerts
