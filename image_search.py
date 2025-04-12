import sys
import platform
import os
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import clip  # ✅ Required for custom model queries using OpenAI's CLIP
from pathlib import Path

# SQLite patch for Streamlit deployment (non-Windows)
if platform.system() != "Windows":
    try:
        import pysqlite3
        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        raise ImportError("pysqlite3-binary is not installed. Add it to requirements.txt for deployment.")

# ✅ HuggingFace CLIP model for ChromaDB
from transformers import CLIPModel, CLIPProcessor

# ✅ ChromaDB and embedding functions
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction

# Optional dataset downloader
from download import download_flickr8k_dataset

TOP_K = 5 

# -------------------------
# HuggingFace CLIP Wrapper
# -------------------------
class CLIPEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features.cpu().numpy()[0].tolist()

    def embed_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()[0].tolist()

    def embed_combined(self, text, image):
        text_emb = torch.tensor(self.embed_text(text))
        image_emb = torch.tensor(self.embed_image(image))
        avg_emb = (text_emb + image_emb) / 2
        return avg_emb.numpy().tolist()


# -------------------------
# CLIP Setup
# -------------------------
clip_embedder = CLIPEmbeddingFunction()

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


# -------------------------
# Load Custom Model Files
# -------------------------
with open("nn_model.pkl", "rb") as f:
    custom_nn_model = pickle.load(f)
with open("image_paths.pkl", "rb") as f:
    custom_image_paths = pickle.load(f)
custom_image_embeddings = np.load("image_embeddings.npy")


# -------------------------
# ChromaDB Training
# -------------------------
def train_rag_on_images(image_folder):
    if not os.path.exists(image_folder):
        print(f"{image_folder} not found. Downloading dataset...")
        download_flickr8k_dataset()

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="image_search")

    image_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file))

    image_paths = image_paths[:4000]
    print(f"Embedding {len(image_paths)} images...")

    for idx, path in enumerate(tqdm(image_paths)):
        embedding = clip_embedder.embed_image(path)
        collection.add(
            documents=[path],
            embeddings=[embedding],
            ids=[f"img_{idx}"]
        )

    print("Training complete. Embeddings stored in ChromaDB.")


# -------------------------
# Unified Custom Query
# -------------------------
def query_custom_model(prompt=None, image=None, k=4):
    if prompt is None and image is None:
        raise ValueError("You must provide at least a prompt or an image.")

    query_embed = None

    if prompt:
        tokenized = clip.tokenize(prompt).to(device)
        with torch.no_grad():
            text_embed = clip_model.encode_text(tokenized).cpu().numpy()
        query_embed = text_embed

    if image:
        tensor = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embed = clip_model.encode_image(tensor).cpu().numpy()
        if query_embed is not None:
            query_embed = (query_embed + image_embed) / 2.0
        else:
            query_embed = image_embed

    dists, indices = custom_nn_model.kneighbors(query_embed, n_neighbors=k)
    # return [custom_image_paths[i] for i in indices[0]]
    return [Path(custom_image_paths[i]).as_posix() for i in indices[0]]


# -------------------------
# ChromaDB Query
# -------------------------
def query_similar_images(prompt=None, image=None, k=4, show_custom=True):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("image_search")

    if prompt and image:
        query_vector = clip_embedder.embed_combined(prompt, image)
    elif image:
        query_vector = clip_embedder.embed_image(image)
    elif prompt:
        query_vector = clip_embedder.embed_text(prompt)
    else:
        raise ValueError("Please provide at least a prompt or an image.")

    results = collection.query(query_embeddings=[query_vector], n_results=k)
    clip_image_paths = results['documents'][0]

    if any(not os.path.exists(path) for path in clip_image_paths):
        print("Some image files are missing. Re-downloading dataset...")
        download_flickr8k_dataset()

    custom_results = query_custom_model(prompt, image, k=k) if show_custom else []
    return clip_image_paths, custom_results


# -------------------------
# Optional Fallback Loader
# -------------------------
def load_custom_model(model_dir=".", base_image_dir="flickr8k/Flicker8k_Dataset"):
    with open(f"{model_dir}/nn_model.pkl", "rb") as f:
        nn_model = pickle.load(f)
    with open(f"{model_dir}/image_paths.pkl", "rb") as f:
        image_paths = pickle.load(f)
    image_embeddings = np.load(f"{model_dir}/image_embeddings.npy")

    # 🔧 Normalize paths for Linux environments
    image_paths = [p.replace("\\", "/") for p in image_paths]
    fixed_paths = []
    for path in image_paths:
        if not os.path.exists(path):
            filename = os.path.basename(path)
            fixed_paths.append(os.path.join(base_image_dir, filename))
        else:
            fixed_paths.append(path)

    return nn_model, image_embeddings, fixed_paths

# Inside image_search.py

def update_custom_image_paths(folder_path):
    global custom_image_paths
    custom_image_paths = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if name.lower().endswith((".jpg", ".jpeg", ".png"))]

def query_by_image(image_path, nn_model, image_paths):
    image = Image.open(image_path).convert("RGB")
    tensor = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(tensor).cpu().numpy()
    dists, indices = nn_model.kneighbors(image_features, n_neighbors=TOP_K)
    return [image_paths[i] for i in indices[0]]

def query_by_text(prompt, nn_model, image_paths):
    # Tokenize and encode the text
    with torch.no_grad():
        text_tokens = clip.tokenize([prompt]).to(device)
        text_features = clip_model.encode_text(text_tokens).cpu().numpy()

    # Get top matches
    dists, indices = nn_model.kneighbors(text_features, n_neighbors=TOP_K)
    return [image_paths[i] for i in indices[0]]
