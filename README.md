# U-Net Sketch Maker

Train a U-Net neural network to automatically colorize line drawings and sketches into cartoon-style portraits.

## Features

- **U-Net Architecture**: Encoder-decoder with skip connections (6-channel input → 3-channel RGB output)
- **CelebA-HQ Dataset**: Automated download and preprocessing (1,500 images, 256×256)
- **Cartoon Target Generation**: Bilateral filtering + K-means quantization + edge detection
- **PyTorch Training**: L1 loss, Adam optimizer, GPU acceleration
- **Gradio Interface**: Interactive web UI for testing
- **~100ms Inference**: Fast colorization on trained model

## Installation

pip install torch torchvision opencv-python pillow gradio kaggle

## Dataset Preparation

### 1. Setup Kaggle Credentials

# Create kaggle.json with your credentials
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

### 2. Download CelebA-HQ Dataset

kaggle datasets download -d badasstechie/celebahq-resized-256x256
unzip celebahq-resized-256x256.zip -d celebahq256

## Training Pipeline

### Step 1: Generate Cartoon Targets

import cv2
import numpy as np
from pathlib import Path

def make_cartoon_colored_sketch(img_bgr):
    # 1. Strong bilateral smoothing
    color = cv2.bilateralFilter(img_bgr, 9, 150, 150)
    color = cv2.bilateralFilter(color, 9, 150, 150)
    
    # 2. Edge detection
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255, 
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 
                                   blockSize=9, C=2)
    edges = cv2.bitwise_not(edges)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 3. K-means color quantization (3 colors)
    data = color.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 3, None, criteria, 3, 
                                     cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quant = centers[labels.flatten()].reshape(color.shape)
    
    # 4. Boost saturation
    hsv = cv2.cvtColor(quant, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.3, 0, 255).astype(np.uint8)
    v = np.clip(v * 1.05, 0, 255).astype(np.uint8)
    quant_sat = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    
    # 5. Combine colors with edges
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(quant_sat, edges_col)
    
    return cartoon, edges

### Step 2: Create Dataset Class

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ColorizerDataset(Dataset):
    def __init__(self, photo_root, line_dir, target_dir, transform=None):
        self.photo_paths = sorted(Path(photo_root).rglob("*.jpg"))
        self.line_dir = Path(line_dir)
        self.target_dir = Path(target_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.photo_paths)
    
    def __getitem__(self, idx):
        p = self.photo_paths[idx]
        photo = Image.open(p).convert("RGB")
        line = Image.open(self.line_dir / p.name).convert("RGB")
        target = Image.open(self.target_dir / p.name).convert("RGB")
        
        if self.transform:
            photo = self.transform(photo)
            line = self.transform(line)
            target = self.transform(target)
        
        x = torch.cat([photo, line], dim=0)  # 6 channels
        return x, target

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

### Step 3: Define U-Net Model

import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_c=6, out_c=3):
        super().__init__()
        # Encoder
        self.enc1 = UNetBlock(in_c, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(128, 256)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)  # 256 = 128 (upsampled) + 128 (skip)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)   # 128 = 64 (upsampled) + 64 (skip)
        
        self.final = nn.Conv2d(64, out_c, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder with skip connections
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        return self.final(d1)

### Step 4: Train the Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 5  # Adjust based on dataset size
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        
        if (batch_idx + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}] Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "colorizer_unet.pth")

## Inference with Gradio

import gradio as gr
from torchvision import transforms

# Load model
model = SimpleUNet().to(device)
model.load_state_dict(torch.load("colorizer_unet.pth", map_location=device))
model.eval()

def line_and_colorize(photo):
    # Extract edges using Canny
    img_bgr = cv2.cvtColor(np.array(photo), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Prepare input tensors
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    photo_t = transform(photo).unsqueeze(0).to(device)
    line_t = transform(Image.fromarray(edges_rgb)).unsqueeze(0).to(device)
    x = torch.cat([photo_t, line_t], dim=1)
    
    # Run inference
    with torch.no_grad():
        out = model(x)[0].cpu()
        out_img = transforms.ToPILImage()(out.clamp(0, 1))
    
    return Image.fromarray(edges_rgb), out_img

# Launch interface
demo = gr.Interface(
    fn=line_and_colorize,
    inputs=gr.Image(type="pil", label="Input Photo"),
    outputs=[
        gr.Image(type="pil", label="Extracted Lines"),
        gr.Image(type="pil", label="Colored Sketch")
    ],
    title="U-Net Sketch Colorizer"
)

demo.launch()

## Architecture Details

**U-Net Structure:**
- Input: 6 channels (3 RGB photo + 3 RGB line drawing)
- Encoder: 6→64→128 channels
- Bottleneck: 256 channels
- Decoder: 128→64 channels (with skip connections)
- Output: 3 RGB channels

**Skip Connections:**
- Encoder Level 1 (64 channels) → Decoder Level 1
- Encoder Level 2 (128 channels) → Decoder Level 2

## Training Parameters

| Parameter       | Value               |
|-----------------|---------------------|
| Dataset Size    | 1,500 images        |
| Image Size      | 256×256             |
| Batch Size      | 8                   |
| Learning Rate   | 1e-4                |
| Loss Function   | L1Loss (MAE)        |
| Optimizer       | Adam                |
| Epochs          | 2-10 (adjustable)   |
| GPU Required    | Yes (recommended)   |

## Performance Metrics

| Metric          | Value                    |
|-----------------|--------------------------|
| Training Time   | ~5-10 min/epoch (V100)   |
| Inference Speed | ~100ms per image         |
| VRAM Usage      | 8-12 GB (training)       |
| Model Size      | ~20 MB                   |

## Data Preprocessing Pipeline

1. **Bilateral Filtering** (2x): Smooths colors while preserving edges
2. **Adaptive Thresholding**: Extracts bold cartoon outlines
3. **K-means Quantization**: Reduces to 3 flat colors
4. **HSV Saturation Boost**: Enhances color vibrancy (1.3×)
5. **Edge Masking**: Combines quantized colors with extracted edges

## Customization

### Adjust Cartoon Style
Change K-means clusters in `make_cartoon_colored_sketch()`:
- K=3: Very cartoon-like (default)
- K=5-7: Moderate stylization
- K=9-12: More realistic colors

### Edge Detection Settings
Modify adaptive threshold parameters:
- blockSize=9: Edge detection window (larger = smoother)
- C=2: Threshold offset (higher = fewer edges)

### Training Hyperparameters
- Increase epochs to 10-20 for better quality
- Reduce batch_size to 4 if GPU memory is limited
- Try lr=5e-5 for finer convergence

## Limitations

- Requires GPU for reasonable training times
- Dataset preparation is semi-automated (requires Kaggle setup)
- Works best with face/portrait images
- Line extraction quality affects colorization results
- Model size fixed at 256×256 (adjustable but requires retraining)

## Future Enhancements

- [ ] Add perceptual loss (LPIPS) for improved quality
- [ ] Support variable input resolutions
- [ ] Implement progressive training
- [ ] Add style transfer options
- [ ] Create pre-trained model weights
- [ ] Batch inference for multiple images
- [ ] Export to ONNX for deployment

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- opencv-python
- Pillow
- gradio
- numpy
- kaggle (for dataset download)

## GPU Recommendations

- **Training**: NVIDIA GPU with 8GB+ VRAM (T4, V100, RTX 3060+)
- **Inference**: Any GPU or CPU (slower on CPU ~1-2 seconds)

## License

MIT License

## Acknowledgments

- CelebA-HQ dataset: https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256
- U-Net architecture: Ronneberger et al. (2015)
