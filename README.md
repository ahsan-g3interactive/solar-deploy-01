# Solar Panel Inspection (Streamlit + ONNX)

Streamlit app that runs YOLO ONNX models on solar panel photos (upload or image URL) and drone
videos, draws the detected defects with class name and confidence, and lists every detection.
Detections below the "Human review below" threshold are flagged **Human Review Required**.

Models: **YOLO26n (nano)**, the fast baseline, **YOLO12s (small)**, a middle ground between speed
and accuracy, and **YOLO26x (large)**, all trained on the same data. All use the classes: Healthy, Dust or dirt accumulation, Bird droppings/environmental contamination,
Cracks, Physical damage, Other visible abnormalities, Snow, Leaf/feather debris.

> The RGB model finds visible surface defects only. It must **not** be used to declare a panel
> electrically healthy: hot spots and electrical faults need a separate radiometric-thermal model
> compared against neighbouring panels and operating data.

## Features
- **Model dropdown**: the models in `models/models.json` (plus any other `.onnx` in `models/`) are
  listed in the sidebar by display name. "Compare all models" runs every model on the same photo
  side by side (detections, per-class counts, confidence, inference time).
- **Drone video pipeline and evaluation log: coming soon.** Built but hidden in the app until
  released; set `VIDEO_ENABLED = True` in `app.py` to turn them on.
- **Drone video pipeline** (baseline testing framework):
  1. Frames are sampled (default 2 per second) and run through each selected model.
  2. Detections are tracked across frames into **incidents** (one per physical defect), so a
     defect seen in 20 frames is counted once. Incidents seen in fewer than *N* frames are dropped.
  3. Outputs: annotated H.264 video, incidents CSV, and a ZIP of sampled frames with model
     pre-labels in YOLO format for labelling in CVAT / Label Studio / Roboflow.
- **Incident-level evaluation**: upload a ground-truth incidents CSV to get, per model:
  defect recall, precision, F1, false alerts per 100 inspected panels, results by defect type,
  and lists of missed incidents and false alerts. The **Evaluation log** tab accumulates runs and
  breaks results down by model, site, flight, altitude, camera angle and tiling; download it and
  load it again later to combine flights.
- **High-resolution tiling** for small cracks: optional 2x2 / 3x3 / 4x4 overlapping crops on top
  of the full-frame pass. Wide drone video often lacks the pixels to show cracks; use it with
  high-resolution frames or closer detail passes.

## Ground-truth CSV
One or more rows per physical incident (Healthy rows are ignored). Either boxes on frames:
```
incident_id,class_name,frame,x1,y1,x2,y2
1,Cracks,240,812,400,905,470
```
or the time the incident is visible:
```
incident_id,class_name,start_frame,end_frame      (or start_s,end_s)
1,Cracks,230,300
```
The incidents CSV exported by the app has these columns, so a reviewer can correct it (fix types,
delete false alerts, add missed incidents) and upload it back as ground truth.

## Recommended workflow
1. Run the current model on representative drone videos from your operating conditions (baseline).
2. Export the sampled frames, correct the pre-labels, and label the real physical incidents.
3. Evaluate per incident, and read the Evaluation log by defect type, altitude, angle and site.
4. Retrain on your drone frames with a flight- or site-based split (frames of one flight are near
   duplicates, so a random split overstates accuracy):
   ```bash
   python tools/split_by_flight.py labelled_flight1/ labelled_flight2/ ... -o dataset/ --test 0.2 --val 0.1
   yolo detect train data=dataset/data.yaml model=yolo26n.pt imgsz=640
   ```
5. Export the new model to ONNX, drop it in `models/`, and compare it against the baseline.

## Files
- `app.py`: Streamlit frontend
- `inference.py`: ONNX pre-processing, inference, tiling, NMS and drawing
- `video.py`: frame sampling, incident tracking, annotated video, labelling export
- `evaluation.py`: incident-level metrics against ground truth
- `tools/split_by_flight.py`: flight/site-based train/val/test split for retraining
- `models/models.json`: model registry (file, display name, description, default)
- `models/yolo26n.onnx`: YOLO26n (nano) baseline; `models/yolo12s.onnx`: YOLO12s (small);
  `models/yolo26x.onnx`: YOLO26x (large)
- `tools/check_models.py`: pre-push check that every model loads, runs, and is tracked by Git LFS
- `requirements.txt`: Python dependencies

Supported ONNX outputs: Ultralytics YOLOv8/YOLO11/YOLO12/YOLO26 detection heads, `(1, 4+nc, N)` or the
end-to-end `(1, K, 6)` export. Class names are read from the model metadata.

## Adding a model
Model files are stored with **Git LFS** (see `.gitattributes`): GitHub rejects files over 100 MB and
a YOLO26x ONNX is around 200 MB. Streamlit Community Cloud downloads LFS files when it clones.

```bash
git lfs install                          # once per machine
# 1. copy the exported ONNX to models/, e.g. models/yolo26x.onnx
# 2. register it in models/models.json (file, name, description) if it is not listed yet
python tools/check_models.py             # loads + runs every model, checks LFS tracking
git add models/ && git lfs ls-files      # the .onnx must appear in this list
git commit -m "Add YOLO26x model" && git push
```
Unregistered `.onnx` files in `models/` still appear, under their file name. A listed model whose
file is missing (or is only an LFS pointer) is shown greyed out in the sidebar instead of crashing.

Large models are much slower on CPU; the Drone video tab estimates the processing time per model
before you start. On Streamlit Community Cloud, memory is limited: keeping all models loaded
uses noticeably more RAM than the files themselves, so check the app's resource usage after deploying.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Streamlit's default upload limit is 200 MB; for longer videos set `server.maxUploadSize` in
`.streamlit/config.toml`.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo, with `app.py` at the repo root and `models/` included
   (model files through Git LFS, see above).
2. On https://share.streamlit.io click **Create app**, then pick the repo, branch `main`, and main file `app.py`.
3. Under **Advanced settings**, pick Python 3.11 or 3.12.
4. Click **Deploy**.
