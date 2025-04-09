import streamlit as st
import os
import gdown
from PIL import Image
import torch
from generator import UNetGenerator
from utils import generate_image_from_sketch

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
file_id = "1cyI-A-D1PLcTW3nKH6RmkNLPUNefHJk0"
model_url = f"https://drive.google.com/uc?id={file_id}"
model_path = "pix2pix_generator.pth"

@st.cache_resource(show_spinner="Loading Generator...")
def load_generator():
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False, fuzzy=True)
        print("Model Downloaded Successfully!")
    else:
        print("Model already exists, skipping download.")
    model = UNetGenerator()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

generator = load_generator()

st.title("🎨 Sketch to Image")

uploaded = st.file_uploader("Upload a Sketch", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    fixed_size = (784, 784)
    image = image.resize(fixed_size)

    generated_image = generate_image_from_sketch(image, generator)
    generated_image = generated_image.resize(fixed_size)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sketch Input")
        st.image(image, width=fixed_size[0])

    with col2:
        st.subheader("Generated Image")
        st.image(generated_image, width=fixed_size[0])
