import torch
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms


from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def get_image_caption(image: Image.Image) -> str:
    image = image.convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption


def generate_image_from_sketch(sketch_img: Image.Image, generator_model: torch.nn.Module, device='cpu') -> Image.Image:
    # Resize and convert to RGB to ensure 3 channels
    sketch_img = sketch_img.resize((256, 256)).convert("RGB")

    # Convert to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize([0.5]*3, [0.5]*3)  # → [-1, 1]
    ])
    input_tensor = transform(sketch_img).unsqueeze(0).to(device)

    # Generate output
    generator_model.to(device)
    generator_model.eval()
    with torch.no_grad():
        output_tensor = generator_model(input_tensor)

    # Denormalize output back to [0, 1]
    output_tensor = output_tensor.squeeze(0).cpu() * 0.5 + 0.5
    output_img = transforms.ToPILImage()(output_tensor.clamp(0, 1))

    return output_img

def dodge_sketch(image: Image.Image) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    inverted = 255 - gray
    blur = cv2.GaussianBlur(inverted, (21, 21), sigmaX=0, sigmaY=0)
    dodge = cv2.divide(gray, 255 - blur, scale=256)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    sketch = cv2.subtract(dodge, edges // 3)
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)
    sketch = cv2.fastNlMeansDenoising(sketch, h=15, templateWindowSize=7, searchWindowSize=21)
    return Image.fromarray(sketch).convert("RGB")

def sobel_sketch(image: Image.Image) -> Image.Image:
    # Convert PIL Image to NumPy array (RGB) and then to grayscale
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    
    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    inverted_edges = 255 - sobel_combined
    
    sketch = cv2.fastNlMeansDenoising(inverted_edges, h=15, templateWindowSize=7, searchWindowSize=21)
    
    return Image.fromarray(sketch).convert("RGB")

def lattice_sketch(image: Image.Image) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    inverted = 255 - sharpened
    blur = cv2.GaussianBlur(inverted, (21, 21), 0)
    dodge = cv2.divide(sharpened, 255 - blur, scale=256)
    sketch = cv2.equalizeHist(dodge)
    sketch = cv2.fastNlMeansDenoising(sketch, h=75, templateWindowSize=7, searchWindowSize=21)
    return Image.fromarray(sketch).convert("RGB")

def canny_sketch(image: Image.Image) -> Image.Image:
    def gaussian_blur(image, kernel_size=5, sigma=1.4):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def sobel_filters(img):
        Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.hypot(Ix, Iy)
        angle = np.arctan2(Iy, Ix)
        magnitude = magnitude / magnitude.max() * 255
        return magnitude.astype(np.uint8), angle

    def non_max_suppression(mag, angle):
        M, N = mag.shape
        output = np.zeros((M, N), dtype=np.uint8)
        angle = angle * 180 / np.pi
        angle[angle < 0] += 180
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                q, r = 255, 255
                a = angle[i, j]
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    q = mag[i, j + 1]
                    r = mag[i, j - 1]
                elif (22.5 <= a < 67.5):
                    q = mag[i + 1, j - 1]
                    r = mag[i - 1, j + 1]
                elif (67.5 <= a < 112.5):
                    q = mag[i + 1, j]
                    r = mag[i - 1, j]
                elif (112.5 <= a < 157.5):
                    q = mag[i - 1, j - 1]
                    r = mag[i + 1, j + 1]
                output[i, j] = mag[i, j] if mag[i, j] >= q and mag[i, j] >= r else 0
        return output

    def threshold(image, low_ratio=0.05, high_ratio=0.15):
        high_threshold = image.max() * high_ratio
        low_threshold = high_threshold * low_ratio
        M, N = image.shape
        res = np.zeros((M, N), dtype=np.uint8)
        strong, weak = np.uint8(255), np.uint8(75)
        strong_i, strong_j = np.where(image >= high_threshold)
        weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak
        return res, weak, strong

    def hysteresis(img, weak=75, strong=255):
        M, N = img.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if img[i, j] == weak:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
        return img

    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = gaussian_blur(gray)
    mag, angle = sobel_filters(blurred)
    nms = non_max_suppression(mag, angle)
    thresh, weak, strong = threshold(nms)
    result = hysteresis(thresh, weak, strong)
    inverted = 255 - result
    return Image.fromarray(inverted).convert("RGB")
