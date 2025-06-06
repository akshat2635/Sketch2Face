# Sketch ↔ Image Translation

A Streamlit web application that demonstrates bidirectional image-sketch translation using a **pix2pix** GAN model and traditional computer vision techniques.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.44.1-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.6.0-orange.svg)
![OpenCV](https://img.shields.io/badge/opencv-v4.11.0-green.svg)

## 🎯 Features

### 🖼️ Image to Sketch

Transform any image into artistic sketches using four different algorithms:

- **Dodge Sketch**: Creates smooth, pencil-like sketches using dodge blending
- **Sobel Sketch**: Edge-based sketches using Sobel gradient operators
- **Lattice Sketch**: Sharpened sketches with histogram equalization
- **Canny Sketch**: Clean edge detection using the Canny algorithm

### 🎨 Sketch to Image

Convert sketches into realistic images using a trained **pix2pix GAN** model:

- Upload hand-drawn sketches or generated sketches
- AI-powered image generation using U-Net architecture
- High-quality 784x784 output resolution

## 🏗️ Project Structure

```
cv project/
├── Home.py                     # Main Streamlit app entry point
├── pages/
│   ├── 1_Image_to_Sketch.py   # Image-to-sketch conversion page
│   └── 2_Sketch_to_Image.py   # Sketch-to-image conversion page
├── generator.py               # U-Net GAN generator architecture
├── utils.py                   # Image processing utilities
├── requirements.txt           # Python dependencies
├── packages.txt              # System packages for deployment
├── pix2pix_generator.pth     # Pre-trained model weights
├── test_images/              # Sample test images
├── Sketch-2-Image/           # Training notebook and metrics
│   ├── sketch-to-image.ipynb # Model training notebook
│   ├── eval_metrics.csv      # Training evaluation metrics
│   └── training_progress.gif # Training visualization
└── __pycache__/              # Python cache files
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd "cv project"
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run Home.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - Use the sidebar to navigate between features

## 🔧 Technical Details

### Model Architecture

- **Generator**: U-Net architecture with encoder-decoder structure
- **Training**: pix2pix GAN framework with adversarial loss
- **Input Resolution**: 256x256 for processing, 784x784 for display
- **Model Size**: Pre-trained weights (~45MB)

### Image Processing Algorithms

#### Dodge Sketch

- Grayscale conversion and inversion
- Gaussian blur with dodge blending
- Laplacian edge enhancement
- Noise reduction with Non-local Means

#### Sobel Sketch

- Sobel X and Y gradient computation
- Edge magnitude calculation
- Weighted edge combination
- Denoising for clean output

#### Lattice Sketch

- Sharpening kernel application
- Dodge blending technique
- Histogram equalization
- Advanced noise reduction

#### Canny Sketch

- Custom Canny edge detection implementation
- Gaussian blur preprocessing
- Non-maximum suppression
- Hysteresis thresholding

## 🎮 Usage

### Image to Sketch

1. Navigate to "🔄 Image to Sketch" in the sidebar
2. Upload an image (JPG, JPEG, or PNG)
3. View all four sketch styles generated automatically
4. Compare different artistic interpretations

### Sketch to Image

1. Navigate to "🎨 Sketch to Image" in the sidebar
2. Upload a sketch or line drawing
3. Wait for the AI model to process (first load downloads the model)
4. View the generated realistic image

## 📊 Model Performance

The pix2pix model was trained on sketch-image pairs and evaluated using:

- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual quality metric
- **Training Progress**: Visualized in `Sketch-2-Image/training_progress.gif`
- **Evaluation Metrics**: Detailed results in `Sketch-2-Image/eval_metrics.csv`

## 🛠️ Development

### Adding New Sketch Algorithms

1. Implement your function in `utils.py`:

   ```python
   def custom_sketch(image: Image.Image) -> Image.Image:
       # Your implementation here
       return processed_image
   ```

2. Import and use in `pages/1_Image_to_Sketch.py`

### Model Training

- Training notebook: `Sketch-2-Image/sketch-to-image.ipynb`
- Includes data preprocessing, model training, and evaluation
- Uses PyTorch and custom U-Net implementation

## 📦 Dependencies

### Core Libraries

- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **PIL (Pillow)**: Image processing
- **NumPy**: Numerical computations

### Additional Tools

- **gdown**: Google Drive file downloads
- **tqdm**: Progress bars
- **pandas**: Data manipulation

## 🌐 Deployment

### Streamlit Cloud

The app is deployment-ready with:

- `requirements.txt`: Python dependencies
- `packages.txt`: System-level packages (libgl1-mesa-glx)
- Automatic model download from Google Drive

### Local Development

```bash
# Install in development mode
pip install -e .

# Run with debug mode
streamlit run Home.py --logger.level=debug
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **pix2pix**: Isola et al. "Image-to-Image Translation with Conditional Adversarial Networks"
- **U-Net**: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **Streamlit**: Amazing framework for rapid ML app development

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](../../issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Made with ❤️ using Streamlit and PyTorch**
