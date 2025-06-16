import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=3,
        classes=3
    ).to(device)
    checkpoint = torch.load("deeplabv3+_SMDG+ORIGA+G1020_withweights.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image):
    image = image.resize((512, 512))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np.transpose(2, 0, 1)  # HWC -> CHW
    image_tensor = torch.tensor(image_np).unsqueeze(0)
    return image_tensor

def postprocess_mask(mask):
    mask = torch.argmax(mask.squeeze(), dim=0).cpu().numpy()
    return mask

def overlay_mask(image_pil, mask):
    image_np = np.array(image_pil).astype(np.uint8)
    overlay = image_np.copy()
    colors = {
        1: (0, 255, 0),  # disc - green
        2: (0, 0, 255)   # cup - red
    }
    for cls_id, color in colors.items():
        overlay[mask == cls_id] = (overlay[mask == cls_id] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    return Image.fromarray(overlay)

def calculate_vcdr(mask):
    # Cup = 2, Disc = 1
    cup_coords = np.where(mask == 2)
    disc_coords = np.where(mask == 1)
    if len(cup_coords[0]) == 0 or len(disc_coords[0]) == 0:
        return 0.0
    cup_height = cup_coords[0].max() - cup_coords[0].min() + 1
    disc_height = disc_coords[0].max() - disc_coords[0].min() + 1
    if disc_height == 0:
        return 0.0
    return cup_height / disc_height

def classify_vcdr(vcdr):
    # Example thresholds (adjust as needed)
    if vcdr < 0.3:
        return "Normal"
    elif 0.3 <= vcdr < 0.5:
        return "Mild"
    elif 0.5 <= vcdr < 0.7:
        return "Moderate"
    else:
        return "Severe"

# Streamlit app
st.title("Glaucoma Cup-Disc Segmentation Demo")

uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    model = load_model()
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)

    mask = postprocess_mask(output)
    overlay = overlay_mask(image.resize((512, 512)), mask)
    st.image(overlay, caption="Segmentation Overlay", use_column_width=True)

    # Calculate VCDR and classification
    vcdr = calculate_vcdr(mask)
    classification = classify_vcdr(vcdr)
    st.markdown(f"**VCDR:** {vcdr:.4f}")
    st.markdown(f"**Glaucoma Severity:** {classification}")