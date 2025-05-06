import streamlit as st
from PIL import Image
import os
import pickle
import numpy as np
import os
from PIL import Image
import torch
from download import download_flickr8k_dataset
import matplotlib.pyplot as plt
from image_search import query_custom_model  # your custom model logic
from image_search import query_by_image, load_custom_model, query_by_text
import shutil

st.set_page_config(page_title="Image Search with CLIP + Custom Model", layout="wide")

from image_search import (
    train_rag_on_images,
    query_similar_images,
    query_custom_model
)

@st.cache_resource(show_spinner="Preparing dataset...")
def prepare_dataset():
    download_flickr8k_dataset()
prepare_dataset()
# Check if ChromaDB is already trained
def is_trained():
    return os.path.exists("chroma_db/") and len(os.listdir("chroma_db/")) > 0

# # Load custom-trained model files
# @st.cache_resource
# def load_custom_model():
#     with open("nn_model.pkl", "rb") as f:
#         custom_nn_model = pickle.load(f)
#     custom_image_embeddings = np.load("image_embeddings.npy")
#     with open("image_paths.pkl", "rb") as f:
#         custom_image_paths = pickle.load(f)

#     # Patch paths so they point to correct dataset dir
#     fixed_paths = []
#     for path in custom_image_paths:
#         filename = os.path.basename(path)
#         fixed_path = os.path.join("flickr8k/Flicker8k_Dataset", filename)
#         fixed_paths.append(fixed_path)

#     return custom_nn_model, custom_image_embeddings, fixed_paths


custom_nn_model, _, custom_image_paths = load_custom_model()

st.title("🔍 Image Search using CLIP + Custom Model")
st.markdown("Upload an image, enter a prompt, or both to find similar images!")

st.header("🔧 Search Settings")
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_input("Enter a text prompt (optional)")

with col2:
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])
    query_image = Image.open(uploaded_file).convert("RGB") if uploaded_file else None

search_button = st.button("🔎 Search")

# Ensure ChromaDB is ready
if not is_trained():
    st.warning("Setting up search database... Please wait a moment.")
    train_rag_on_images("flickr8k/Flicker8k_Dataset")

# Perform search
if search_button:
    if not prompt and not query_image:
        st.error("Please enter a prompt, upload an image, or both.")
    else:
        with st.spinner("Searching for similar images..."):
            clip_results, _ = query_similar_images(prompt=prompt, image=query_image, k=5)
            # custom_results = query_custom_model(prompt=prompt, image=query_image, k=5)

        if clip_results:
            st.subheader("📎 CLIP (ChromaDB) Results")
            cols = st.columns(5)
            for i, col in enumerate(cols):
                try:
                    img = Image.open(clip_results[i])
                    col.image(img.resize((200, 200)), caption=os.path.basename(clip_results[i]))
                except:
                    col.write("No image")

        # Custom model results (if query image provided)
        if query_image:
            st.subheader("🎯 Custom Model Results")
            # Save uploaded query image temporarily
            temp_path = "temp_query.jpg"
            query_image.save(temp_path)

            # Query the custom model
            custom_results = query_by_image(temp_path, custom_nn_model, custom_image_paths)

            # Create static dir for custom model results
            STATIC_DIR = "static"
            os.makedirs(STATIC_DIR, exist_ok=True)

            cols = st.columns(5)
            for i, (col, path) in enumerate(zip(cols, custom_results[:5])):
                filename = os.path.basename(path)
                static_path = os.path.join(STATIC_DIR, filename)
                if not os.path.exists(static_path):
                    shutil.copy(path, static_path)
                with col:
                    col.image(static_path, width=200, caption=f"Match {i+1}")

            if os.path.exists(temp_path):
                os.remove(temp_path)

        elif prompt:
            st.subheader("🎯 Custom Model (Text) Results")
            results = query_by_text(prompt, custom_nn_model, custom_image_paths)

            STATIC_DIR = "static"
            os.makedirs(STATIC_DIR, exist_ok=True)

            cols = st.columns(5)
            for i, (col, path) in enumerate(zip(cols, results[:5])):
                filename = os.path.basename(path)
                static_path = os.path.join(STATIC_DIR, filename)
                if not os.path.exists(static_path):
                    shutil.copy(path, static_path)
                col.image(static_path, width=200, caption=f"Match {i+1}")
