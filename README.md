# Solar Panel Inspection (Streamlit + ONNX)

Streamlit app that runs the YOLO26n `best.onnx` model on a solar panel photo (upload or image URL),
draws the detected defects with class name and confidence, and lists every detection.
Detections below the "Human review below" threshold are flagged **Human Review Required**.

Classes: Healthy, Dust or dirt accumulation, Bird droppings/environmental contamination, Cracks,
Physical damage, Other visible abnormalities, Snow, Leaf/feather debris.

## Files
- `app.py`: Streamlit frontend
- `inference.py`: ONNX pre-processing, inference, NMS and drawing
- `best.onnx`: the model
- `requirements.txt`: Python dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo, with `app.py` at the repo root and `best.onnx` included.
2. On https://share.streamlit.io click **Create app**, then pick the repo, branch `main`, and main file `app.py`.
3. Under **Advanced settings**, pick Python 3.11 or 3.12.
4. Click **Deploy**.
