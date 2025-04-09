import streamlit as st
from PIL import Image
import os

from image_search import (
    train_rag_on_images,
    query_similar_images,
    clip_embedder
)

# Check if ChromaDB is already trained (basic file-based check)
def is_trained():
    return os.path.exists("chroma_db/") and len(os.listdir("chroma_db/")) > 0

st.set_page_config(page_title="Image Search with CLIP + ChromaDB", layout="wide")
st.title("🔍 CLIP-based Image Search Engine")
st.markdown("Upload an image, enter a prompt, or both to find similar images!")

# Input section (moved from sidebar)
st.header("🔧 Search Settings")
col1, col2 = st.columns([2, 1])

with col1:
    search_text = st.text_input("Enter a text prompt (optional)")

with col2:
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "png", "jpeg"])
    query_image = Image.open(uploaded_file).convert("RGB") if uploaded_file else None

search_button = st.button("🔎 Search")

# Ensure RAG is trained
if not is_trained():
    st.warning("Training database... this might take a minute.")
    train_rag_on_images("flickr8k/Flicker8k_Dataset")

# Perform search
if search_button:
    if not search_text and not query_image:
        st.error("Please enter a prompt, upload an image, or both.")
    else:
        with st.spinner("Searching for similar images..."):
            results = query_similar_images(prompt=search_text, image=query_image, k=5)

        st.subheader("🔁 Top 5 Similar Images")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            try:
                result_img_path = results[i]
                result_image = Image.open(result_img_path)
                col.image(result_image.resize((200, 200)), caption=os.path.basename(result_img_path))
            except Exception as e:
                col.write("No image found.")
