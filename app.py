from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

from inference import CLASS_COLORS, SolarPanelDetector

MODEL_PATH = Path(__file__).parent / "best.onnx"
MAX_URL_BYTES = 20 * 1024 * 1024

st.set_page_config(page_title="Solar Panel Inspection", page_icon="☀️", layout="wide")


@st.cache_resource
def load_detector():
    return SolarPanelDetector(str(MODEL_PATH))


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


def show_result(name, img_bgr, detector, conf, iou, review):
    with st.spinner(f"Running inspection on {name}..."):
        detections, annotated = detector.predict(img_bgr, conf, iou, review)

    n_review = sum(d["needs_review"] for d in detections)
    n_defects = sum(d["class_name"] != "Healthy" for d in detections)
    c1, c2, c3 = st.columns(3)
    c1.metric("Detections", len(detections))
    c2.metric("Defects (non-healthy)", n_defects)
    c3.metric("Human review required", n_review)

    left, right = st.columns(2)
    left.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
    right.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption=f"Detections: {len(detections)}", use_container_width=True)

    if not detections:
        st.info(f"No detections above confidence {conf:.2f}.")
    else:
        df = pd.DataFrame([{
            "#": i,
            "Class": d["class_name"],
            "Confidence": round(d["confidence"], 2),
            "Box (x1,y1,x2,y2)": str(d["box"]),
            "Status": "Human Review Required" if d["needs_review"] else "OK",
        } for i, d in enumerate(detections, 1)])

        def highlight(row):
            color = "background-color: rgba(255, 170, 0, 0.25)" if row["Status"] != "OK" else ""
            return [color] * len(row)

        st.dataframe(df.style.apply(highlight, axis=1), hide_index=True, use_container_width=True)

        counts = df["Class"].value_counts()
        st.markdown("**Summary:** " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

    ok, buf = cv2.imencode(".jpg", annotated)
    if ok:
        st.download_button("Download annotated image", buf.tobytes(),
                           file_name=f"result_{Path(name).stem}.jpg", mime="image/jpeg",
                           key=f"dl_{name}")


# ---------------- UI ----------------
st.title("☀️ Solar Panel Inspection")
st.caption("YOLO26n ONNX model: detects dust, bird droppings, cracks, physical damage, snow, "
           "debris and other abnormalities on solar panels.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Put best.onnx next to app.py.")
    st.stop()

detector = load_detector()

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05,
                     help="Detections below this are dropped.")
    iou = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05,
                    help="Overlap threshold for removing duplicate boxes.")
    review = st.slider("Human review below", 0.05, 1.00, 0.50, 0.05,
                       help="Detections between the confidence threshold and this value "
                            "are flagged 'Human Review Required'.")
    st.divider()
    st.subheader("Classes")
    for cid, cname in detector.class_names.items():
        color = bgr_to_hex(CLASS_COLORS.get(cid, (255, 255, 255)))
        st.markdown(f"<span style='display:inline-block;width:12px;height:12px;"
                    f"background:{color};border-radius:2px;margin-right:6px'></span>{cname}",
                    unsafe_allow_html=True)

tab_upload, tab_url = st.tabs(["📁 Upload image", "🔗 Image URL"])

with tab_upload:
    files = st.file_uploader("Upload one or more solar panel images",
                             type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True)
    for f in files or []:
        st.subheader(f.name)
        try:
            show_result(f.name, decode_image(f.getvalue()), detector, conf, iou, review)
        except ValueError as e:
            st.error(str(e))
        st.divider()

with tab_url:
    st.caption("In Google Images: right-click the image → **Copy image address**, then paste it here.")
    url = st.text_input("Image URL", placeholder="https://.../panel.jpg")
    if url:
        try:
            img = decode_image(fetch_url(url.strip()))
            show_result(url.rsplit("/", 1)[-1].split("?")[0] or "url_image", img,
                        detector, conf, iou, review)
        except requests.RequestException as e:
            st.error(f"Could not download the image: {e}")
        except ValueError as e:
            st.error(f"{e} Make sure the URL points directly to an image file.")
