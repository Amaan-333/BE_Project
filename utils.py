# utils.py

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Hyperparameters: must match what you used during training
SEQUENCE_LENGTH = 16     # number of frames per clip
IMG_HEIGHT = 112         # resize height
IMG_WIDTH = 112          # resize width

# Normalization used by 3D‐ResNet pretrained on Kinetics
RESNET3D_MEAN = [0.43216, 0.394666, 0.37645]
RESNET3D_STD  = [0.22803, 0.22145, 0.216989]

def video_to_clips(video_path: str):
    """
    Reads all frames from a video file and splits them into non-overlapping
    clips of length SEQUENCE_LENGTH. Returns a list of PyTorch tensors,
    each of shape [1, 3, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read all frames (BGR)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to (IMG_HEIGHT, IMG_WIDTH)
        frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        all_frames.append(frame_rgb)

    cap.release()

    # If fewer than SEQUENCE_LENGTH frames, no valid clip can be formed.
    if len(all_frames) < SEQUENCE_LENGTH:
        return []

    # Build non-overlapping clips:
    clips = []
    num_full_clips = len(all_frames) // SEQUENCE_LENGTH
    for i in range(num_full_clips):
        # Clip frames i*SEQUENCE_LENGTH : (i+1)*SEQUENCE_LENGTH
        clip_frames = all_frames[i * SEQUENCE_LENGTH : (i + 1) * SEQUENCE_LENGTH]

        # Convert list of H×W×3 uint8 → PIL, apply transforms, stack
        transform = transforms.Compose([
            transforms.ToTensor(),  # gives [3, H, W] in [0,1]
            transforms.Normalize(mean=RESNET3D_MEAN, std=RESNET3D_STD)
        ])

        # For each of the 16 frames, apply transform → gives a list of 16 tensors of shape [3,H,W]
        per_frame_tensors = []
        for fr in clip_frames:
            pil_img = Image.fromarray(fr)         # H×W×3 RGB
            tf_img  = transform(pil_img)          # [3, H, W]
            per_frame_tensors.append(tf_img)

        # Stack them along the frame dimension → shape [3, 16, H, W]
        clip_tensor = torch.stack(per_frame_tensors, dim=1)  # [3, D, H, W]
        # Add batch dimension → [1, 3, D, H, W]
        clip_tensor = clip_tensor.unsqueeze(0)
        clips.append(clip_tensor)

    return clips
