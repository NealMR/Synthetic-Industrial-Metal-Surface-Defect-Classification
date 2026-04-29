import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import json
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MODEL_PATH = "model_best.pth"
CLASS_NAMES_PATH = "class_names.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Class Names
try:
    with open(CLASS_NAMES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
except Exception:
    # Fallback to default alphabetical order if file missing
    CLASS_NAMES = ['crack', 'hole', 'normal', 'rust', 'scratch']

NUM_CLASSES = len(CLASS_NAMES)

# --- MODEL ARCHITECTURE ---
def build_model(num_classes: int):
    # Use the same architecture as training
    m = models.efficientnet_b2()
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return m

# Load Model
model = build_model(NUM_CLASSES)
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

model = model.to(DEVICE)
model.eval()

# --- PREPROCESSING ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

inference_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# --- GRAD-CAM UTILITY ---
def get_gradcam(model, img_tensor, target_class):
    model.eval()
    target_layer = model.features[-1]
    act, grad = {}, {}
    def fw_hook(m, i, o): act['f'] = o.detach()
    def bw_hook(m, gi, go): grad['b'] = go[0].detach()
    h1 = target_layer.register_forward_hook(fw_hook)
    h2 = target_layer.register_full_backward_hook(bw_hook)
    
    output = model(img_tensor)
    model.zero_grad()
    output[0, target_class].backward()
    h1.remove(); h2.remove()
    
    w = grad['b'].mean((2, 3), keepdim=True)
    cam = torch.relu(torch.sum(w * act['f'], dim=1, keepdim=True))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam.cpu().numpy()[0, 0]

def predict(img):
    if img is None: return None, None
    
    # Preprocess
    img_t = inference_transforms(img).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        top_idx = outputs.argmax(1).item()
    
    # Generate Heatmap
    model.zero_grad()
    cam = get_gradcam(model, img_t, top_idx)
    
    # Resize cam to 224x224
    from scipy.ndimage import zoom
    cam_resized = zoom(cam, 224/cam.shape[0])
    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() + 1e-8)
    
    # Overlay heatmap on original image
    img_resized = img.resize((224, 224)).convert("RGB")
    img_np = np.array(img_resized) / 255.0
    heatmap = plt.get_cmap('jet')(cam_resized)[:, :, :3]
    overlay = (0.6 * img_np + 0.4 * heatmap)
    overlay = (overlay * 255).astype(np.uint8)
    
    return confidences, Image.fromarray(overlay)

# --- GRADIO INTERFACE ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Industrial AI Quality Inspection")
    gr.Markdown("Upload a photo of a metal surface to detect defects with Explainable AI heatmaps.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Surface Image")
            btn = gr.Button("🔍 Analyze Surface", variant="primary")
        
        with gr.Column():
            output_heatmap = gr.Image(type="pil", label="AI Focus (Heatmap)")
            output_label = gr.Label(num_top_classes=5, label="Detection Results")
            
    btn.click(fn=predict, inputs=input_img, outputs=[output_label, output_heatmap])

if __name__ == "__main__":
    demo.launch(share=False)
