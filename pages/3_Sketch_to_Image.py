import os
from PIL import Image
import torch
from generator import UNetGenerator
from utils import generate_image_from_sketch
from download import download_flickr8k_dataset
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from image_search import query_custom_model  # your custom model logic
from image_search import query_by_image, load_custom_model
import shutil
import streamlit as st
st.set_page_config(
    page_title="Sketch to Image or Search 🎨",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="auto"
)

# 📍 Path settings
DATASET_DIR = "flickr8k/Flickr8k_Dataset"
MODEL_PATH = "pix2pix_generator.pth"
FILE_ID = "1cyI-A-D1PLcTW3nKH6RmkNLPUNefHJk0"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Face detection threshold
FACE_AREA_THRESHOLD = 0.25

# Setup MTCNN for face detection
device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)

# Download and extract dataset
@st.cache_resource(show_spinner="Preparing dataset...")
def prepare_dataset():
    download_flickr8k_dataset()

# Download model if not already present
@st.cache_resource(show_spinner="Loading Pix2Pix Generator...")
def load_generator():
    import gdown
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    model = UNetGenerator()
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Face detection logic
def is_single_face_within_threshold(img: Image.Image, threshold=FACE_AREA_THRESHOLD, visualize=False):
    width, height = img.size
    image_area = width * height
    boxes, _ = mtcnn.detect(img)

    if boxes is None or len(boxes) != 1:
        return False, 0

    # Calculate face area ratio
    x1, y1, x2, y2 = boxes[0]
    face_area = (x2 - x1) * (y2 - y1)
    face_ratio = face_area / image_area

    if visualize:
        plt.imshow(img)
        ax = plt.gca()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        plt.axis('off')
        st.pyplot(plt)

    return face_ratio >= threshold, face_ratio

# 🚀 Streamlit UI
st.title("🎨 Sketch to Image")

uploaded = st.file_uploader("Upload a Sketch", type=["jpg", "jpeg", "png"])
if uploaded:
    prepare_dataset()
    generator = load_generator()
    sketch = Image.open(uploaded).convert("RGB")
    sketch = sketch.resize((784, 784))

    st.subheader("Sketch Preview")
    st.image(sketch, width=392)

    use_pix2pix, face_ratio = is_single_face_within_threshold(sketch, visualize=True)

    if use_pix2pix:
        st.success(f"✅ Detected one clear face (covering {face_ratio*100:.2f}%) — using Pix2Pix!")
        generated_image = generate_image_from_sketch(sketch, generator)
        st.subheader("Generated Image")
        st.image(generated_image.resize((784, 784)))
    else:
        st.warning(f"⚠️ Pix2Pix skipped (face count ≠ 1 or face < {FACE_AREA_THRESHOLD*100:.0f}%). Showing similar images.")
        
        # Save the uploaded sketch temporarily
        temp_path = "temp_sketch.jpg"
        sketch.save(temp_path)
        
        # Load model and query
        nn_model, _, image_paths = load_custom_model()
        results = query_by_image(temp_path, nn_model, image_paths)
        
        # Create static/ dir to store publicly accessible files
        STATIC_DIR = "static"
        os.makedirs(STATIC_DIR, exist_ok=True)
        
        st.subheader("🖼️ Top 5 Similar Images Using Custom Model")
    
        # Show top 5 results in a single row
        cols = st.columns(5)
        for i, (col, path) in enumerate(zip(cols, results[:5])):
            filename = os.path.basename(path)
            static_path = os.path.join(STATIC_DIR, filename)
    
            # Copy file to static/ so it can be served
            if not os.path.exists(static_path):
                shutil.copy(path, static_path)
    
            # Link to the file relative to the server
            public_url = f"/static/{filename}"
    
            # Display image and link inside each column
            with col:
                col.image(static_path, width=150, caption=f"Match {i+1}")
    
        # Clean up temporary sketch file
        if os.path.exists(temp_path):
            os.remove(temp_path)
