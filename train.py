import os, sys, zipfile, time, random, copy, warnings, json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import seaborn as sns

# --- CONFIGURATION ---
CLASS_NAMES = ['crack', 'hole', 'normal', 'rust', 'scratch']
BATCH_SIZE = 16
LR_INIT = 1e-3
IMG_SIZE = 224
TOTAL_EPOCHS = 10
FREEZE_EPOCHS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = 'synthetic_industrial_metal_surface_defects/industrial_defect_dataset'

# --- 1. DATA PREPARATION ---
def setup_data():
    if not os.path.exists(DATA_ROOT):
        zip_path = 'synthetic_industrial_metal_surface_defects.zip'
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall('.')
    
    train_dir = os.path.join(DATA_ROOT, 'train')
    val_dir = os.path.join(DATA_ROOT, 'val')
    
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds = datasets.ImageFolder(val_dir, val_tf)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, val_ds

# --- 2. MODEL ARCHITECTURE ---
def build_model():
    m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, len(CLASS_NAMES))
    )
    return m

# --- 3. GRAD-CAM EXPLAINABILITY ---
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

# --- 4. MAIN PIPELINE ---
if __name__ == '__main__':
    train_loader, val_loader, val_ds = setup_data()
    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=1e-4)
    
    history = {'train_loss':[], 'val_acc':[], 'val_loss':[]}
    best_acc = 0
    
    print(f"Starting Training on {DEVICE}...")
    for epoch in range(1, TOTAL_EPOCHS + 1):
        if epoch == 1:
            for name, p in model.named_parameters():
                if 'classifier' not in name: p.requires_grad = False
            print("[Phase 1] Training Classifier Head Only")
        
        if epoch == FREEZE_EPOCHS + 1:
            for p in model.parameters(): p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LR_INIT*0.1, weight_decay=1e-4)
            print("[Phase 2] Fine-tuning Entire Model")
            
        model.train()
        running_loss = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        v_loss, correct = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                v_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = correct / len(val_ds)
        avg_v_loss = v_loss / len(val_loader)
        history['train_loss'].append(running_loss / len(train_loader))
        history['val_loss'].append(avg_v_loss)
        history['val_acc'].append(acc)
        
        print(f"Epoch {epoch}/{TOTAL_EPOCHS} | Train Loss: {history['train_loss'][-1]:.4f} | Val Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'model_best.pth')
            print("Saved New Best Model")

    # --- 5. GENERATE FINAL ARTIFACTS ---
    print("Generating Final Artifacts...")
    
    # Training Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy Curve'); plt.legend()
    plt.savefig('training_curves.png'); plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix'); plt.savefig('confusion_matrix.png'); plt.close()
    
    # Grad-CAM Heatmaps
    plt.figure(figsize=(15, 4))
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    for i in range(5):
        idx = random.randint(0, len(val_ds)-1)
        img, label = val_ds[idx]
        cam = get_gradcam(model, img.unsqueeze(0).to(DEVICE), label)
        plt.subplot(1, 5, i+1)
        plt.imshow(np.clip(img.permute(1,2,0).numpy()*std+mean, 0, 1))
        plt.imshow(transforms.ToPILImage()(torch.from_numpy(cam)).resize((224,224)), alpha=0.4, cmap='jet')
        plt.title(f'Label: {CLASS_NAMES[label]}'); plt.axis('off')
    plt.savefig('explainability_audit.png'); plt.close()
    
    # History
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
        
    print("All files generated successfully.")
