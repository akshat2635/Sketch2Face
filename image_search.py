import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    raise ImportError("pysqlite3-binary is not installed. Please add it to requirements.txt.")


import torch
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from tqdm import tqdm
import subprocess
from download import download_flickr8k_dataset

class CLIPEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed_image(self, image):
        # If `image` is a string, treat it as a file path and open it
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"Could not open image at {image}. Error: {e}")

        # Process the image through CLIP
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

# Initialize CLIP embedder
clip_embedder = CLIPEmbeddingFunction()

def train_rag_on_images(image_folder):
    
    # Check if image folder exists; if not, run downloader
    if not os.path.exists(image_folder):
        print(f"{image_folder} not found. Downloading dataset...")
        download_flickr8k_dataset()
    
    """Embeds and stores image features in ChromaDB."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="image_search")

    image_paths = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(root, file))

    print(f"Embedding {len(image_paths)} images...")

    for idx, path in enumerate(tqdm(image_paths)):
        embedding = clip_embedder.embed_image(path)
        collection.add(
            documents=[path],
            embeddings=[embedding],
            ids=[f"img_{idx}"]
        )
    print("Training complete. Image embeddings stored in ChromaDB.")

def query_similar_images(prompt=None, image=None, k=5):
    """Returns k similar images using text, image, or both."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("image_search")

    if prompt and image:
        query_vector = clip_embedder.embed_combined(prompt, image)
    elif image:
        query_vector = clip_embedder.embed_image(image)
    elif prompt:
        query_vector = clip_embedder.embed_text(prompt)
    else:
        raise ValueError("Please provide at least a prompt or an image path.")

    results = collection.query(query_embeddings=[query_vector], n_results=k)
    image_paths = results['documents'][0]

    # Check if any result path is missing
    if any(not os.path.exists(path) for path in image_paths):
        print("Some image files are missing. Re-downloading dataset...")
        download_flickr8k_dataset()

    return image_paths

