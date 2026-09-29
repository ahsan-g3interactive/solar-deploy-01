"""Drone video pipeline: sample frames, detect, track detections into physical incidents.

A track is one physical object (a panel region) followed across sampled frames. Tracks whose
voted class is not "Healthy" are incidents; every confirmed track counts as an inspected
panel region, which is used as the default for "false alerts per 100 panels".
"""
import io
import zipfile
from collections import defaultdict

import cv2
import numpy as np

from inference import SolarPanelDetector, is_healthy

try:
    import imageio_ffmpeg
except ImportError:  # annotated video falls back to OpenCV mp4v (download only, no browser playback)
    imageio_ffmpeg = None

MAX_VIDEO_WIDTH = 1280


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_distance(a, b):
    """Center distance normalised by the mean box diagonal (drone motion can kill IoU)."""
    ca = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    diag = (np.hypot(a[2] - a[0], a[3] - a[1]) + np.hypot(b[2] - b[0], b[3] - b[1])) / 2
    return float(np.linalg.norm(ca - cb) / max(diag, 1.0))


class IncidentTracker:
    """Greedy class-agnostic IoU / center-distance tracker; class is decided by weighted vote."""

    def __init__(self, iou_threshold=0.3, max_distance=0.6, max_age=3, min_hits=2):
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = []
        self._next_id = 1

    def update(self, frame_idx, detections):
        """Assigns a track id to each detection (in place, key 'track_id')."""
        live = [t for t in self.tracks if t["misses"] <= self.max_age]
        pairs = []
        for ti, t in enumerate(live):
            for di, d in enumerate(detections):
                ov = iou(t["last_box"], d["box"])
                dist = center_distance(t["last_box"], d["box"])
                if ov >= self.iou_threshold or dist <= self.max_distance:
                    pairs.append((ov - dist, ti, di))
        used_t, used_d = set(), set()
        for _, ti, di in sorted(pairs, reverse=True):
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self._add(live[ti], frame_idx, detections[di])
        for ti, t in enumerate(live):
            if ti not in used_t:
                t["misses"] += 1
        for di, d in enumerate(detections):
            if di not in used_d:
                t = {"track_id": self._next_id, "observations": [], "votes": defaultdict(float),
                     "misses": 0}
                self._next_id += 1
                self.tracks.append(t)
                self._add(t, frame_idx, d)

    @staticmethod
    def _add(track, frame_idx, det):
        track["observations"].append({"frame": frame_idx, "box": det["box"],
                                      "class_name": det["class_name"],
                                      "confidence": det["confidence"]})
        track["votes"][det["class_name"]] += det["confidence"]
        track["last_box"] = det["box"]
        track["misses"] = 0
        det["track_id"] = track["track_id"]

    def confirmed(self):
        out = []
        for t in self.tracks:
            if len(t["observations"]) < self.min_hits:
                continue
            obs = t["observations"]
            cls = max(t["votes"], key=t["votes"].get)
            confs = [o["confidence"] for o in obs if o["class_name"] == cls]
            best = max((o for o in obs if o["class_name"] == cls), key=lambda o: o["confidence"])
            out.append({
                "track_id": t["track_id"],
                "class_name": cls,
                "healthy": is_healthy(cls),
                "start_frame": obs[0]["frame"],
                "end_frame": obs[-1]["frame"],
                "hits": len(obs),
                "max_confidence": max(confs),
                "mean_confidence": float(np.mean(confs)),
                "best_frame": best["frame"],
                "best_box": best["box"],
                "observations": obs,
            })
        return out


class VideoWriter:
    def __init__(self, path, size, fps):
        self.path = path
        self.size = size
        if imageio_ffmpeg is not None:
            self.h264 = True
            self._gen = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264",
                                                   pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                                   macro_block_size=2, quality=6)
            self._gen.send(None)
        else:
            self.h264 = False
            self._cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame_bgr):
        frame_bgr = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA)
        if self.h264:
            self._gen.send(np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        else:
            self._cv.write(frame_bgr)

    def close(self):
        if self.h264:
            self._gen.close()
        else:
            self._cv.release()


def video_info(path):
    cap = cv2.VideoCapture(path)
    info = {"fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    info["duration"] = info["frames"] / info["fps"] if info["fps"] else 0
    return info


def process_video(path, detector: SolarPanelDetector, out_path, sample_fps=2.0, max_seconds=None,
                  conf=0.25, iou_thr=0.45, review=0.5, tiles=1, tile_overlap=0.2,
                  tracker_kwargs=None, keep_frames=0, progress=None):
    """Runs the detector on sampled frames and tracks detections into incidents.

    Returns a dict with per-frame detections, confirmed tracks, timing, and up to
    `keep_frames` evenly spread raw sampled frames (for the labelling export).
    """
    info = video_info(path)
    step = max(1, int(round(info["fps"] / sample_fps)))
    last = info["frames"] if not max_seconds else min(info["frames"], int(max_seconds * info["fps"]))
    n_samples = max(1, (last + step - 1) // step)
    keep_every = max(1, n_samples // keep_frames) if keep_frames else 0

    scale = min(1.0, MAX_VIDEO_WIDTH / max(info["width"], 1))
    out_size = (int(info["width"] * scale) // 2 * 2, int(info["height"] * scale) // 2 * 2)
    writer = VideoWriter(out_path, out_size, info["fps"] / step)
    tracker = IncidentTracker(**(tracker_kwargs or {}))

    cap = cv2.VideoCapture(path)
    frames, kept, infer_ms = [], {}, []
    idx = sample_no = 0
    try:
        while idx < last:
            if idx % step:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            dets, ms = detector.detect(frame, conf, iou_thr, review, tiles, tile_overlap)
            infer_ms.append(ms)
            tracker.update(idx, dets)
            frames.append({"frame": idx, "time_s": idx / info["fps"], "detections": dets})
            if keep_every and sample_no % keep_every == 0 and len(kept) < keep_frames:
                kept[idx] = frame
            labels = [f'#{d["track_id"]} {d["class_name"]} {d["confidence"]:.2f}' for d in dets]
            annotated = detector.draw(frame, dets, labels)
            cv2.putText(annotated, f"t={idx / info['fps']:.1f}s  frame {idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(annotated)
            sample_no += 1
            idx += 1
            if progress:
                progress(min(1.0, sample_no / n_samples))
    finally:
        cap.release()
        writer.close()

    tracks = tracker.confirmed()
    return {
        "info": info,
        "step": step,
        "sampled_frames": len(frames),
        "frames": frames,
        "tracks": tracks,
        "incidents": [t for t in tracks if not t["healthy"]],
        "panels_estimate": len(tracks),
        "mean_infer_ms": float(np.mean(infer_ms)) if infer_ms else 0.0,
        "total_infer_s": float(np.sum(infer_ms)) / 1000,
        "video_path": out_path,
        "h264": writer.h264,
        "kept_frames": kept,
    }


def incidents_table(result):
    """Rows in the same layout as the ground-truth CSV (one row per incident, best frame box)."""
    fps = result["info"]["fps"]
    return [{
        "incident_id": t["track_id"],
        "class_name": t["class_name"],
        "start_frame": t["start_frame"],
        "end_frame": t["end_frame"],
        "start_s": round(t["start_frame"] / fps, 2),
        "end_s": round(t["end_frame"] / fps, 2),
        "frame": t["best_frame"],
        "x1": t["best_box"][0], "y1": t["best_box"][1],
        "x2": t["best_box"][2], "y2": t["best_box"][3],
        "hits": t["hits"],
        "max_confidence": round(t["max_confidence"], 3),
        "mean_confidence": round(t["mean_confidence"], 3),
    } for t in result["incidents"]]


def labelling_zip(result, class_names, prefix):
    """Sampled frames + model pre-labels in YOLO format, ready for CVAT / Label Studio / Roboflow.

    File names start with `prefix` (use the flight id) so the flight-based split tool can
    keep every frame of one flight on the same side of the train/test split.
    """
    by_frame = {f["frame"]: f["detections"] for f in result["frames"]}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for frame_idx, img in sorted(result["kept_frames"].items()):
            stem = f"{prefix}_f{frame_idx:06d}"
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                continue
            zf.writestr(f"images/{stem}.jpg", jpg.tobytes())
            h, w = img.shape[:2]
            lines = []
            for d in by_frame.get(frame_idx, []):
                x1, y1, x2, y2 = d["box"]
                lines.append(f'{d["class_id"]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} '
                             f'{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}')
            zf.writestr(f"labels/{stem}.txt", "\n".join(lines))
        zf.writestr("classes.txt", "\n".join(class_names[k] for k in sorted(class_names)))
        zf.writestr("data.yaml", "names:\n" + "".join(f"  {k}: {class_names[k]}\n"
                                                      for k in sorted(class_names)))
        zf.writestr("README.txt",
                    "Labels are MODEL PRE-LABELS, not ground truth. Review and correct every box\n"
                    "(including missed defects) before using them for evaluation or training.\n"
                    "Frame numbers in file names match the ground-truth CSV 'frame' column.\n")
    return buf.getvalue()
