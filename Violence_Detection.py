import streamlit as st
import torch
import torchvision.models.video as models_video
import torch.nn as nn
import cv2
import numpy as np
from torch.nn import Softmax
from pathlib import Path
import tempfile
import os
from utils import video_to_clips
import gdown

# ────────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS (Match Your Training)
# ────────────────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH      = 16
IMG_HEIGHT           = 112
IMG_WIDTH            = 112
CONFIDENCE_THRESHOLD = 0.5

# ────────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI SETUP
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Violence Detection (3D ResNet-18)", layout="wide")


st.title("🎥 Violence Detection in Surveillance Videos")
st.markdown("""
    This tool uses a **3D Convolutional Neural Network** trained on fight detection datasets
    to classify uploaded videos as either **Violent** or **Non-Violent**. Upload a short surveillance video
    clip and click the button below to run inference.
""")
st.markdown("""
 Upload a short surveillance video clip and click the button below to run inference.
""")

example_videos = ["sample_videos\download1.mp4",
                  "sample_videos\NV_1.mp4",
                  "sample_videos\NV_15.mp4",
                  "sample_videos\V_17.mp4",
                  "sample_videos\V115.mp4","sample_videos\V_139.mp4",
                  "sample_videos\V_222.mp4"]

st.sidebar.header("📊 Settings")
st.sidebar.markdown("Adjust model settings below:")
threshold_slider = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05)

uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov"])

example_selection = st.selectbox("Choose an example video:", ["None"] + example_videos)

# ────────────────────────────────────────────────────────────────────────────────
# MODEL LOADING FUNCTION
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource()
def load_3dcnn_model(model_path: str):
    class MyResNet3D(nn.Module):
        def __init__(self):
            super(MyResNet3D, self).__init__()
            backbone = models_video.r3d_18(pretrained=False)
            num_features = backbone.fc.in_features
            backbone.fc = nn.Linear(num_features, 2)
            self.net = backbone

        def forward(self, x):
            return self.net(x)

    model = MyResNet3D()
    checkpoint = torch.load(model_path, map_location="cpu")
    model.net.load_state_dict(checkpoint)
    model.eval()
    return model


MODEL_PATH = "model/best_3dcnn_model.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        url = "https://drive.google.com/uc?id=1NZPyfqtrcTY_mOSiaiOc6tPYwEY_j7vN"  # Use file ID
        gdown.download(url, MODEL_PATH, quiet=False)



download_model()
model = load_3dcnn_model(MODEL_PATH)
softmax = Softmax(dim=1)

# ────────────────────────────────────────────────────────────────────────────────
# VIDEO PROCESSING & INFERENCE
# ────────────────────────────────────────────────────────────────────────────────
def classify_video(video_path: str, sequence_length: int, img_h: int, img_w: int, threshold: float):
    clips = video_to_clips(video_path)
    if not clips:
        raise RuntimeError(f"Video has fewer than {sequence_length} frames.")

    clip_probs = []
    for clip_tensor in clips:
        with torch.no_grad():
            logits = model(clip_tensor)
            probs = softmax(logits)
            p_fight = probs[0, 1].item()
        clip_probs.append(p_fight)

    avg_prob = sum(clip_probs) / len(clip_probs)
    fight_count = sum(p > threshold for p in clip_probs)
    nonfight_count = len(clip_probs) - fight_count
    overall_label = "Violence" if avg_prob > threshold else "Non-Violence"

    return overall_label, avg_prob, clip_probs

# ────────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP LOGIC
# ────────────────────────────────────────────────────────────────────────────────

video_path = None

# Use example video if no file uploaded
if uploaded_file is None and example_selection != "None":
    video_path = example_selection
elif uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

if video_path:
    st.video(video_path)

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🚀 Run Violence Detection"):
            with st.spinner("Analyzing video clips..."):
                try:
                    overall_label, avg_prob, clip_probs = classify_video(
                        video_path=video_path,
                        sequence_length=SEQUENCE_LENGTH,
                        img_h=IMG_HEIGHT,
                        img_w=IMG_WIDTH,
                        threshold=threshold_slider
                    )
                except RuntimeError as e:
                    st.error(str(e))
                    overall_label, avg_prob, clip_probs = None, None, None

            if overall_label is not None:
                st.success("✅ Prediction Complete")

                colA, colB, colC = st.columns(3)
                colA.metric("📹 Clips Analyzed", len(clip_probs))
                colB.metric("🔥 Avg. Fight Probability", f"{avg_prob*100:.2f}%")
                colC.metric("🧠 Prediction", overall_label)
            else:
                st.error("❌ Video could not be processed. Check the input.")
    with col2:
        st.markdown("### 🔎 Details")
        st.markdown("Use the slider on the left to adjust the decision threshold.")
else:
    st.info("📤 Please upload a video file or select an example.")
