import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import json

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

# --- PREDICTION FUNCTION ---
def predict(image):
    if image is None:
        return "Please upload an image."
    
    # Preprocess
    img_t = inference_transforms(image).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
    
    # Format results
    results = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    return results

# --- GRADIO INTERFACE ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🛡️ Metal Surface Defect Classifier
        ### AI-Powered Industrial Quality Inspection
        Upload a photo of a metal surface to detect defects like cracks, holes, rust, or scratches.
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Surface Image")
            btn = gr.Button("Analyze Surface", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(num_top_classes=NUM_CLASSES, label="Detection Results")
            
    gr.Examples(
        examples=[], # Will be populated if sample images exist
        inputs=input_img
    )
    
    btn.click(fn=predict, inputs=input_img, outputs=output_label)
    
    gr.Markdown(
        """
        ---
        *Developed for KaggleHacX Hackathon*
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
