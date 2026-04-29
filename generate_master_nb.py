import json
import os

# Define the Master Cells
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# 🛡️ Industrial Defect Detection - Master Training Pipeline\n", "### Optimized for Google Colab & Local RTX 3050"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, sys, zipfile, time, random, copy, warnings\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "import torchvision\n",
            "import torchvision.transforms as transforms\n",
            "from torchvision import datasets, models\n",
            "from torch.utils.data import DataLoader\n",
            "from sklearn.metrics import f1_score, classification_report, confusion_matrix\n",
            "import seaborn as sns\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# --- ENVIRONMENT DETECTION ---\n",
            "try:\n",
            "    import google.colab\n",
            "    IN_COLAB = True\n",
            "    ROOT_DIR = '/content'\n",
            "except:\n",
            "    IN_COLAB = False\n",
            "    ROOT_DIR = '.'\n",
            "\n",
            "print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')\n",
            "print(f'Running on: {\"Google Colab\" if IN_COLAB else \"Local Machine\"}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 📦 Dataset Setup & Auto-Discovery"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Find and Unzip any dataset zip files\n",
            "def auto_unzip(search_path):\n",
            "    for f in os.listdir(search_path):\n",
            "        if f.endswith('.zip'):\n",
            "            print(f'📦 Extracting {f}...')\n",
            "            with zipfile.ZipFile(os.path.join(search_path, f), 'r') as zf:\n",
            "                zf.extractall(search_path)\n",
            "\n",
            "if IN_COLAB: auto_unzip(ROOT_DIR)\n",
            "\n",
            "# 2. Recursively find train/val folders\n",
            "def find_data_dirs(search_path):\n",
            "    for root, dirs, _ in os.walk(search_path):\n",
            "        if 'train' in [d.lower() for d in dirs] and 'val' in [d.lower() for d in dirs]:\n",
            "            return os.path.join(root, 'train'), os.path.join(root, 'val')\n",
            "    return None, None\n",
            "\n",
            "TRAIN_DIR, VAL_DIR = find_data_dirs(ROOT_DIR)\n",
            "\n",
            "if not TRAIN_DIR:\n",
            "    print(\"❌ WARNING: Could not find 'train' and 'val' folders. Ensure you upload your dataset.\")\n",
            "else:\n",
            "    print(f'✅ Found Train Dir: {TRAIN_DIR}')\n",
            "    print(f'✅ Found Val Dir: {VAL_DIR}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### ⚙️ Configuration & Transforms"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "CLASS_NAMES = ['crack', 'hole', 'normal', 'rust', 'scratch']\n",
            "BATCH_SIZE = 16\n",
            "LR_INIT = 1e-3\n",
            "IMG_SIZE = 224\n",
            "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "\n",
            "mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]\n",
            "train_tf = transforms.Compose([\n",
            "    transforms.Grayscale(3),\n",
            "    transforms.Resize((IMG_SIZE, IMG_SIZE)),\n",
            "    transforms.RandomHorizontalFlip(),\n",
            "    transforms.RandomVerticalFlip(),\n",
            "    transforms.RandomRotation(10),\n",
            "    transforms.ToTensor(),\n",
            "    transforms.Normalize(mean, std)\n",
            "])\n",
            "val_tf = transforms.Compose([\n",
            "    transforms.Grayscale(3),\n",
            "    transforms.Resize((IMG_SIZE, IMG_SIZE)),\n",
            "    transforms.ToTensor(),\n",
            "    transforms.Normalize(mean, std)\n",
            "])\n",
            "\n",
            "if TRAIN_DIR:\n",
            "    train_ds = datasets.ImageFolder(TRAIN_DIR, train_tf)\n",
            "    val_ds = datasets.ImageFolder(VAL_DIR, val_tf)\n",
            "    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)\n",
            "    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)\n",
            "    print(f'Classes: {train_ds.classes}')\n",
            "else:\n",
            "    print(\"Please upload data and run the discovery cell again.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 🧠 Model Building (EfficientNet-B2)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def build_model():\n",
            "    m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)\n",
            "    in_features = m.classifier[1].in_features\n",
            "    m.classifier = nn.Sequential(\n",
            "        nn.Dropout(0.4),\n",
            "        nn.Linear(in_features, len(CLASS_NAMES))\n",
            "    )\n",
            "    return m\n",
            "\n",
            "model = build_model().to(DEVICE)\n",
            "criterion = nn.CrossEntropyLoss()\n",
            "optimizer = optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=1e-4)\n",
            "print(\"Model ready for training.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 🚀 Training Loop (2-Phase)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "epochs = 10\n",
            "best_acc = 0\n",
            "history = {'train_loss':[], 'val_acc':[]}\n",
            "\n",
            "for epoch in range(epochs):\n",
            "    if epoch == 0: # Freeze\n",
            "        for name, p in model.named_parameters():\n",
            "            if 'classifier' not in name: p.requires_grad = False\n",
            "    if epoch == 2: # Unfreeze\n",
            "        for p in model.parameters(): p.requires_grad = True\n",
            "        optimizer = optim.Adam(model.parameters(), lr=LR_INIT*0.1, weight_decay=1e-4)\n",
            "    \n",
            "    model.train()\n",
            "    tl = 0\n",
            "    for i, (imgs, labels) in enumerate(train_loader):\n",
            "        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)\n",
            "        optimizer.zero_grad()\n",
            "        loss = criterion(model(imgs), labels)\n",
            "        loss.backward()\n",
            "        optimizer.step()\n",
            "        tl += loss.item()\n",
            "    \n",
            "    model.eval()\n",
            "    c = 0\n",
            "    with torch.no_grad():\n",
            "        for imgs, labels in val_loader:\n",
            "            c += (model(imgs.to(DEVICE)).argmax(1) == labels.to(DEVICE)).sum().item()\n",
            "    \n",
            "    acc = c / len(val_ds)\n",
            "    print(f'Epoch {epoch+1} | Acc: {acc:.4f}')\n",
            "    if acc > best_acc:\n",
            "        best_acc = acc\n",
            "        torch.save(model.state_dict(), 'model_best.pth')\n",
            "        print(\"Saved best weights.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 🔍 Explainable AI (Heatmaps)"]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def get_cam(model, img_t, target):\n",
            "    model.eval(); layer = model.features[-1]\n",
            "    act, grad = {}, {}\n",
            "    h1 = layer.register_forward_hook(lambda m,i,o: act.update({'f':o.detach()}))\n",
            "    h2 = layer.register_full_backward_hook(lambda m,gi,go: grad.update({'b':go[0].detach()}))\n",
            "    model(img_t)[0, target].backward()\n",
            "    h1.remove(); h2.remove()\n",
            "    w = grad['b'].mean((2,3), keepdim=True)\n",
            "    cam = (w * act['f']).sum(1, keepdim=True).clamp(min=0)\n",
            "    return (cam / (cam.max() + 1e-8)).cpu().numpy()[0,0]\n",
            "\n",
            "plt.figure(figsize=(15, 5))\n",
            "for i in range(5):\n",
            "    img, label = val_ds[random.randint(0, len(val_ds)-1)]\n",
            "    cam = get_cam(model, img.unsqueeze(0).to(DEVICE), label)\n",
            "    plt.subplot(1, 5, i+1)\n",
            "    plt.imshow(np.clip(img.permute(1,2,0).numpy()*std+mean,0,1))\n",
            "    plt.imshow(transforms.ToPILImage()(torch.from_numpy(cam)).resize((224,224)), alpha=0.4, cmap='jet')\n",
            "    plt.axis('off')\n",
            "plt.show()"
        ]
    }
]

nb = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
with open('main/train_master.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Successfully created main/train_master.ipynb")
