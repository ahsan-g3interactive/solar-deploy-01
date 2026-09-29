import hashlib
import json
import re
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

from evaluation import evaluate, load_ground_truth
from inference import SolarPanelDetector, class_color, is_healthy
from video import incidents_table, labelling_zip, process_video, video_info

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
MAX_URL_BYTES = 20 * 1024 * 1024
VIDEO_ENABLED = False  # drone video + evaluation log are built but not released yet
COMPARE_ALL = "Compare all models"
TILE_OPTIONS = {"Off": 1, "2 x 2": 2, "3 x 3": 3, "4 x 4": 4}
THERMAL_NOTE = ("RGB model: finds visible surface defects only. It cannot detect hot spots, bypass-diode "
                "or other electrical faults, so **never use it to declare a panel electrically healthy**. "
                "Thermal faults need a separate radiometric-thermal model compared against neighbouring "
                "panels and operating data.")

st.set_page_config(page_title="Solar Panel Inspection", page_icon="☀️", layout="wide")

ss = st.session_state
ss.setdefault("video_runs", {})        # run key -> {model name -> result}
ss.setdefault("eval_log", {})          # (run key, model) -> summary row
ss.setdefault("eval_class_log", {})    # (run key, model) -> per-class rows
ss.setdefault("tmpdir", tempfile.mkdtemp(prefix="solar_"))


@st.cache_resource
def load_detector(path, name):
    return SolarPanelDetector(path, name=name)


def is_lfs_pointer(path):
    """True when the .onnx is a Git LFS pointer (the weights were never downloaded)."""
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def available_models():
    """Returns (models, info, problems).

    models: display name -> path, in models.json order, then any other .onnx in models/.
    info: display name -> {description, default}. problems: display name -> why it can't be used.
    """
    registry = json.loads((MODELS_DIR / "models.json").read_text(encoding="utf-8"))         if (MODELS_DIR / "models.json").exists() else []
    listed = {e["file"] for e in registry}
    entries = registry + [{"file": p.name, "name": p.stem}
                          for p in sorted(MODELS_DIR.glob("*.onnx")) if p.name not in listed]
    models, info, problems = {}, {}, {}
    for e in entries:
        path = MODELS_DIR / e["file"]
        name = e.get("name", path.stem)
        if not path.exists():
            problems[name] = f"not installed: add models/{e['file']}"
        elif is_lfs_pointer(path):
            problems[name] = "Git LFS pointer only: run `git lfs pull` to download the weights"
        else:
            models[name] = str(path)
            info[name] = {"description": e.get("description", ""), "default": e.get("default", False),
                          "size_mb": path.stat().st_size / 1e6}
    return models, info, problems


@st.cache_resource
def single_pass_ms(path, name):
    """Median CPU time of one 640x640 forward pass, for the video time estimate."""
    det = load_detector(path, name)
    img = np.full((det.img_size, det.img_size, 3), 114, np.uint8)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        det.detect(img)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def slug(name):
    """'YOLO26x (large)' -> 'yolo26x_large', for download file names."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def decode_image(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the image. Use a .jpg, .png, .bmp or .webp file.")
    return img


def fetch_url(url):
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20, stream=True)
    resp.raise_for_status()
    data = resp.raw.read(MAX_URL_BYTES + 1, decode_content=True)
    if len(data) > MAX_URL_BYTES:
        raise ValueError("Image at URL is larger than 20 MB.")
    return data


def bgr_to_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def pct(v):
    return "–" if v is None or pd.isna(v) else f"{v:.0%}"


def detections_df(detections):
    return pd.DataFrame([{
        "#": i,
        "Class": d["class_name"],
        "Confidence": round(d["confidence"], 2),
        "Box (x1,y1,x2,y2)": str(d["box"]),
        "Status": "Human Review Required" if d["needs_review"] else "OK",
    } for i, d in enumerate(detections, 1)])


def show_detections_table(detections):
    df = detections_df(detections)

    def highlight(row):
        color = "background-color: rgba(255, 170, 0, 0.25)" if row["Status"] != "OK" else ""
        return [color] * len(row)

    st.dataframe(df.style.apply(highlight, axis=1), hide_index=True, width="stretch")
    counts = df["Class"].value_counts()
    st.markdown("**Summary:** " + ", ".join(f"{k}: {v}" for k, v in counts.items()))


def download_image(img_bgr, file_name, key):
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if ok:
        st.download_button("Download annotated image", buf.tobytes(), file_name=file_name,
                           mime="image/jpeg", key=key)


def show_result(name, img_bgr, detector, settings):
    with st.spinner(f"Running {detector.name} on {name}..."):
        detections, ms = detector.detect(img_bgr, **settings)
    annotated = detector.draw(img_bgr, detections)

    n_review = sum(d["needs_review"] for d in detections)
    n_defects = sum(not is_healthy(d["class_name"]) for d in detections)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detections", len(detections))
    c2.metric("Defects (non-healthy)", n_defects)
    c3.metric("Human review required", n_review)
    c4.metric("Inference time", f"{ms:.0f} ms")

    left, right = st.columns(2)
    left.image(rgb(img_bgr), caption="Input", width="stretch")
    right.image(rgb(annotated), caption=f"{detector.name}: {len(detections)} detections", width="stretch")

    if not detections:
        st.info(f"No detections above confidence {settings['conf_threshold']:.2f}.")
    else:
        show_detections_table(detections)
    download_image(annotated, f"result_{Path(name).stem}_{slug(detector.name)}.jpg", f"dl_{name}_{detector.name}")


def compare_result(name, img_bgr, detectors, settings):
    results = {}
    for det in detectors:
        with st.spinner(f"Running {det.name} on {name}..."):
            results[det.name] = det.detect(img_bgr, **settings)

    rows = []
    for model, (dets, ms) in results.items():
        confs = [d["confidence"] for d in dets]
        rows.append({"Model": model, "Detections": len(dets),
                     "Defects": sum(not is_healthy(d["class_name"]) for d in dets),
                     "Human review": sum(d["needs_review"] for d in dets),
                     "Mean confidence": round(float(np.mean(confs)), 3) if confs else None,
                     "Inference ms": round(ms, 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    per_class = pd.DataFrame([{"Model": m, "Class": d["class_name"]}
                              for m, (dets, _) in results.items() for d in dets])
    if not per_class.empty:
        st.markdown("**Detections per class**")
        st.dataframe(pd.crosstab(per_class["Class"], per_class["Model"]), width="stretch")

    cols = st.columns(min(3, len(detectors) + 1))
    cols[0].image(rgb(img_bgr), caption="Input", width="stretch")
    for i, det in enumerate(detectors, 1):
        dets, ms = results[det.name]
        annotated = det.draw(img_bgr, dets)
        with cols[i % len(cols)]:
            st.image(rgb(annotated), caption=f"{det.name}: {len(dets)} detections, {ms:.0f} ms",
                     width="stretch")
            download_image(annotated, f"result_{Path(name).stem}_{slug(det.name)}.jpg",
                           f"dl_{name}_{det.name}")
    for det in detectors:
        dets, _ = results[det.name]
        with st.expander(f"{det.name}: detection details ({len(dets)})"):
            if dets:
                show_detections_table(dets)
            else:
                st.info("No detections.")


def run_photo(name, img_bgr, detectors, settings):
    if len(detectors) == 1:
        show_result(name, img_bgr, detectors[0], settings)
    else:
        compare_result(name, img_bgr, detectors, settings)


# ---------------- Sidebar ----------------
st.title("☀️ Solar Panel Inspection")
st.caption("YOLO ONNX models: detect dust, bird droppings, cracks, physical damage, snow, debris and "
           "other visible abnormalities on solar panels.")

with st.sidebar:
    st.header("Model")
    models, model_info, _ = available_models()
    if not models:
        st.error(f"No models found. Put .onnx files in {MODELS_DIR.name}/.")
        st.stop()
    options = list(models) + ([COMPARE_ALL] if len(models) > 1 else [])
    default = next((i for i, m in enumerate(models) if model_info[m]["default"]), 0)
    choice = st.selectbox("Model", options, index=default, label_visibility="collapsed")
    selected = list(models) if choice == COMPARE_ALL else [choice]
    detectors = [load_detector(models[m], m) for m in selected]
    by_name = {d.name: d for d in detectors}

    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Detections below this are dropped.")
    iou = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05,
                    help="Overlap threshold for removing duplicate boxes.")
    review = st.slider("Human review below", 0.05, 1.00, 0.50, 0.05,
                       help="Detections between the confidence threshold and this value "
                            "are flagged 'Human Review Required'.")
    tiling = st.selectbox("High-resolution tiling", list(TILE_OPTIONS),
                          help="Also runs the model on overlapping crops, so small cracks keep enough "
                               "pixels. Slower: 2x2 = 5 passes, 3x3 = 10, 4x4 = 17.")
    overlap = st.slider("Tile overlap", 0.0, 0.5, 0.2, 0.05, disabled=tiling == "Off")
    settings = {"conf_threshold": conf, "iou_threshold": iou, "review_below": review,
                "tiles": TILE_OPTIONS[tiling], "tile_overlap": overlap}

    st.divider()
    st.subheader("Classes")
    for cid, cname in detectors[0].class_names.items():
        st.markdown(f"<span style='display:inline-block;width:12px;height:12px;"
                    f"background:{bgr_to_hex(class_color(cid))};border-radius:2px;margin-right:6px'>"
                    f"</span>{cname}", unsafe_allow_html=True)
    st.divider()
    st.caption("⚠️ " + THERMAL_NOTE)

soon = "" if VIDEO_ENABLED else " (coming soon)"
tab_photo, tab_video, tab_log = st.tabs(["📷 Photo", f"🎥 Drone video{soon}", f"📊 Evaluation log{soon}"])

# ---------------- Photo ----------------
with tab_photo:
    source = st.radio("Source", ["Upload image", "Image URL"], horizontal=True,
                      label_visibility="collapsed")
    if source == "Upload image":
        files = st.file_uploader("Upload one or more solar panel images",
                                 type=["jpg", "jpeg", "png", "bmp", "webp"],
                                 accept_multiple_files=True)
        for f in files or []:
            st.subheader(f.name)
            try:
                run_photo(f.name, decode_image(f.getvalue()), detectors, settings)
            except ValueError as e:
                st.error(str(e))
            st.divider()
    else:
        st.caption("In Google Images: right-click the image → **Copy image address**, then paste it here.")
        url = st.text_input("Image URL", placeholder="https://.../panel.jpg")
        if url:
            try:
                img = decode_image(fetch_url(url.strip()))
                run_photo(url.rsplit("/", 1)[-1].split("?")[0] or "url_image", img, detectors, settings)
            except requests.RequestException as e:
                st.error(f"Could not download the image: {e}")
            except ValueError as e:
                st.error(f"{e} Make sure the URL points directly to an image file.")

# ---------------- Video ----------------
if not VIDEO_ENABLED:
    tab_video.info("🚧 Drone video inspection is coming soon.")
    tab_log.info("🚧 Incident-level evaluation of drone videos is coming soon.")
    st.stop()

with tab_video:
    st.caption("Baseline testing framework: sample frames, detect, and link detections across frames "
               "into physical incidents. Upload labelled incidents to measure recall, precision and "
               "false alerts per 100 panels.")
    vfile = st.file_uploader("Drone video", type=["mp4", "mov", "avi", "mkv", "m4v"])

    with st.expander("Flight metadata (used to break down results by site, altitude and angle)",
                     expanded=vfile is not None):
        m1, m2, m3, m4 = st.columns(4)
        site = m1.text_input("Site", "site-1")
        flight = m2.text_input("Flight ID", Path(vfile.name).stem if vfile else "flight-1",
                               help="Used to name exported frames, so train/test can be split by flight.")
        altitude = m3.number_input("Altitude (m AGL)", 0.0, 500.0, 30.0, 5.0)
        angle = m4.number_input("Camera pitch (° below horizon)", 0, 90, 90,
                                help="90 = straight down (nadir), smaller = oblique.")
        panels_in = st.number_input("Panels inspected in this video (0 = estimate from tracks)",
                                    0, 1_000_000, 0,
                                    help="Denominator of 'false alerts per 100 panels'. Count panels from "
                                         "the site layout for exact numbers.")

    with st.expander("Processing settings"):
        p1, p2, p3 = st.columns(3)
        sample_fps = p1.slider("Frames analysed per second", 0.5, 10.0, 2.0, 0.5)
        max_seconds = p2.number_input("Max seconds to process (0 = whole video)", 0, 36000, 0)
        keep_frames = p3.number_input("Frames to export for labelling", 0, 500, 50)
        t1, t2, t3 = st.columns(3)
        min_hits = t1.slider("Min sampled frames per incident", 1, 10, 2,
                             help="Detections seen in fewer frames are treated as noise.")
        max_age = t2.slider("Frames an incident can be lost", 0, 10, 3)
        track_iou = t3.slider("Tracking IoU", 0.05, 0.9, 0.3, 0.05)

    if vfile is not None:
        vhash = hashlib.md5(vfile.getvalue()).hexdigest()[:12]
        vpath = Path(ss.tmpdir) / f"{vhash}{Path(vfile.name).suffix}"
        if not vpath.exists():
            vpath.write_bytes(vfile.getvalue())
        info = video_info(str(vpath))
        st.caption(f"{info['width']}x{info['height']}, {info['fps']:.1f} fps, "
                   f"{info['duration']:.1f} s, {info['frames']} frames")

        tracker_kwargs = {"iou_threshold": track_iou, "max_age": max_age, "min_hits": min_hits}
        run_key = (vhash, sample_fps, max_seconds, keep_frames, tuple(sorted(tracker_kwargs.items())),
                   tuple(sorted(settings.items())))
        runs = ss.video_runs.setdefault(run_key, {})
        todo = [d for d in detectors if d.name not in runs]

        if todo:
            step = max(1, int(round(info["fps"] / sample_fps)))
            n_frames = min(info["frames"], int(max_seconds * info["fps"])) if max_seconds else info["frames"]
            passes = 1 + (settings["tiles"] ** 2 if settings["tiles"] > 1 else 0)
            est = {d.name: (n_frames + step - 1) // step * passes * single_pass_ms(models[d.name], d.name) / 1000
                   for d in todo}
            st.caption("Estimated processing time: " +
                       ", ".join(f"{m} ≈ {s / 60:.1f} min" if s >= 90 else f"{m} ≈ {s:.0f} s"
                                 for m, s in est.items()))
            if max(est.values()) > 600:
                st.warning("This will take a while. Lower 'Frames analysed per second', set "
                           "'Max seconds to process', or turn tiling off.")

        if todo and st.button(f"▶ Run video inspection ({', '.join(d.name for d in todo)})",
                              type="primary"):
            for det in todo:
                bar = st.progress(0.0, text=f"{det.name}: processing...")
                out_path = str(Path(ss.tmpdir) / f"{vhash}_{slug(det.name)}_{abs(hash(run_key))}.mp4")
                runs[det.name] = process_video(
                    str(vpath), det, out_path, sample_fps, max_seconds or None,
                    settings["conf_threshold"], settings["iou_threshold"], settings["review_below"],
                    settings["tiles"], settings["tile_overlap"], tracker_kwargs, keep_frames,
                    progress=lambda p, n=det.name: bar.progress(p, text=f"{n}: {p:.0%}"))
                bar.empty()
            st.rerun()

        done = {d.name: runs[d.name] for d in detectors if d.name in runs}
        if done:
            # ---- model comparison ----
            st.subheader("Results")
            rows = []
            for model, r in done.items():
                row = {"Model": model, "Sampled frames": r["sampled_frames"],
                       "ms / frame": round(r["mean_infer_ms"], 1),
                       "Analysis fps": round(1000 / r["mean_infer_ms"], 1) if r["mean_infer_ms"] else None,
                       "Panel regions tracked": r["panels_estimate"],
                       "Incidents": len(r["incidents"])}
                for cls, n in pd.Series([i["class_name"] for i in r["incidents"]],
                                        dtype=object).value_counts().items():
                    row[cls] = n
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).fillna(0), hide_index=True, width="stretch")

            model_tabs = st.tabs(list(done))
            for mtab, (model, r) in zip(model_tabs, done.items()):
                with mtab:
                    vid_col, inc_col = st.columns([3, 2])
                    with vid_col:
                        video_bytes = Path(r["video_path"]).read_bytes()
                        if r["h264"]:
                            st.video(video_bytes)
                        else:
                            st.info("Install imageio-ffmpeg for in-browser playback.")
                        st.download_button("Download annotated video", video_bytes,
                                           file_name=f"{flight}_{slug(model)}.mp4", mime="video/mp4",
                                           key=f"dlv_{model}")
                    with inc_col:
                        inc = pd.DataFrame(incidents_table(r))
                        st.markdown(f"**{len(inc)} incidents** (confirmed in ≥ {min_hits} sampled frames)")
                        if inc.empty:
                            st.info("No defect incidents found.")
                        else:
                            st.dataframe(inc[["incident_id", "class_name", "start_s", "end_s", "hits",
                                              "max_confidence"]], hide_index=True, width="stretch")
                            st.download_button("Download incidents CSV (ground-truth template)",
                                               inc.to_csv(index=False), f"{flight}_{slug(model)}_incidents.csv",
                                               "text/csv", key=f"dlc_{model}")
                        if r["kept_frames"]:
                            st.download_button(
                                f"Download {len(r['kept_frames'])} frames + pre-labels for labelling",
                                labelling_zip(r, by_name[model].class_names, flight),
                                f"{flight}_{slug(model)}_frames.zip", "application/zip", key=f"dlz_{model}")

            # ---- evaluation against labelled incidents ----
            st.subheader("Evaluate against labelled incidents")
            st.caption("CSV columns: `incident_id,class_name,frame,x1,y1,x2,y2` (boxes on one or more "
                       "frames) or `incident_id,class_name,start_frame,end_frame` (or `start_s,end_s`). "
                       "Tip: correct the exported incidents CSV and add the missed ones.")
            gt_file = st.file_uploader("Ground-truth incidents CSV", type=["csv"], key=f"gt_{vhash}")
            if gt_file is not None:
                try:
                    gts = load_ground_truth(gt_file, info["fps"])
                except (ValueError, KeyError, pd.errors.ParserError) as e:
                    st.error(f"Could not read ground truth: {e}")
                    gts = None
                if gts is not None:
                    summaries, details = [], {}
                    for model, r in done.items():
                        panels = panels_in or r["panels_estimate"]
                        summary, per_class, missed, fas = evaluate(gts, r["incidents"], panels, r["step"])
                        summaries.append({"model": model, **summary})
                        details[model] = (per_class, missed, fas)
                        meta = {"video": vfile.name, "site": site, "flight": flight,
                                "altitude_m": altitude, "camera_pitch_deg": angle, "model": model,
                                "tiling": tiling, "sample_fps": sample_fps}
                        ss.eval_log[(run_key, model)] = {**meta, **summary}
                        ss.eval_class_log[(run_key, model)] = [
                            {**meta, **row} for row in per_class.to_dict("records")]

                    sdf = pd.DataFrame(summaries)
                    show = sdf[["model", "gt_incidents", "pred_incidents", "true_positives", "missed",
                                "false_alerts", "panels"]].copy()
                    for c in ["recall", "precision", "f1", "defect_recall_any_type"]:
                        show[c] = sdf[c].map(pct)
                    show["false alerts / 100 panels"] = sdf["false_alerts_per_100_panels"].map(
                        lambda v: "–" if v is None else f"{v:.1f}")
                    st.dataframe(show, hide_index=True, width="stretch")
                    st.caption("recall/precision require the correct defect type; "
                               "'defect_recall_any_type' counts an incident as found even if the type is "
                               "wrong. False alerts are detections matching no real incident of any type.")
                    for model, (per_class, missed, fas) in details.items():
                        with st.expander(f"{model}: by defect type, missed incidents, false alerts"):
                            pc = per_class.copy()
                            pc["recall"] = pc["recall"].map(pct)
                            pc["precision"] = pc["precision"].map(pct)
                            st.dataframe(pc, hide_index=True, width="stretch")
                            a, b = st.columns(2)
                            a.markdown(f"**Missed ({len(missed)})**")
                            a.dataframe(pd.DataFrame(missed), hide_index=True, width="stretch")
                            b.markdown(f"**False alerts ({len(fas)})**")
                            b.dataframe(pd.DataFrame(fas), hide_index=True, width="stretch")
                    st.success("Added to the Evaluation log tab.")

# ---------------- Evaluation log ----------------
with tab_log:
    st.caption("Every evaluated video/model run in this session. Download the log to keep it and "
               "upload it again later to combine results from several flights and sites.")
    prev = st.file_uploader("Load a previously downloaded log (summary CSV)", type=["csv"], key="prev_log")
    frames = [pd.DataFrame(list(ss.eval_log.values()))]
    if prev is not None:
        frames.append(pd.read_csv(prev))
    log = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()

    if log.empty:
        st.info("No evaluations yet. Run a video and upload its ground-truth CSV in the Drone video tab.")
    else:
        dims = [c for c in ["model", "site", "flight", "altitude_m", "camera_pitch_deg", "tiling",
                            "video"] if c in log.columns]
        group = st.multiselect("Break down by", dims, default=["model"])
        if group:
            agg = log.groupby(group, dropna=False)[
                ["gt_incidents", "pred_incidents", "true_positives", "false_alerts", "panels"]].sum()
            agg["recall"] = agg["true_positives"] / agg["gt_incidents"].replace(0, np.nan)
            agg["precision"] = agg["true_positives"] / agg["pred_incidents"].replace(0, np.nan)
            agg["false alerts / 100 panels"] = (agg["false_alerts"] * 100 /
                                               agg["panels"].replace(0, np.nan)).round(1)
            agg["recall"] = agg["recall"].map(pct)
            agg["precision"] = agg["precision"].map(pct)
            st.dataframe(agg.reset_index(), hide_index=True, width="stretch")

        class_log = pd.DataFrame([r for rows in ss.eval_class_log.values() for r in rows])
        if not class_log.empty:
            st.markdown("**By defect type** (this session)")
            by = [c for c in group if c in class_log.columns] + ["class_name"]
            cagg = class_log.groupby(by, dropna=False)[["gt", "found", "predicted", "correct",
                                                        "false_alerts"]].sum()
            cagg["recall"] = (cagg["found"] / cagg["gt"].replace(0, np.nan)).map(pct)
            cagg["precision"] = (cagg["correct"] / cagg["predicted"].replace(0, np.nan)).map(pct)
            st.dataframe(cagg.reset_index(), hide_index=True, width="stretch")
            st.download_button("Download per-class log CSV", class_log.to_csv(index=False),
                               "evaluation_log_by_class.csv", "text/csv")
        st.download_button("Download summary log CSV", log.to_csv(index=False),
                           "evaluation_log.csv", "text/csv")
