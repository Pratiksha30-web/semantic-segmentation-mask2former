import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

st.title("Image Segmentation using Mask2Former")

@st.cache_resource
def load_model():

    processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-ade-semantic"
    )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-ade-semantic"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return processor, model, device


processor, model, device = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Original Image")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    seg_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0].cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(seg_map, cmap="nipy_spectral")
    ax.axis("off")

    st.pyplot(fig)