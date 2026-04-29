import os
import shutil

# Team Name Placeholder (Change this to your actual team name)
TEAM_NAME = "Industrial_Defect_Experts"
ROOT_DIR = "."
SUBMISSION_ROOT = f"{TEAM_NAME}_Submission"

folders = [
    f"{SUBMISSION_ROOT}/main",
    f"{SUBMISSION_ROOT}/submission"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Files to copy into main/
main_files = [
    "main/app.py",
    "main/train_master.ipynb",
    "model_best.pth",
    "class_names.json",
    "training.log"
]

# Files to copy into submission/
submission_files = [
    "submission/methodology.pdf",
    "training_curves.png",
    "confusion_matrix.png",
    "explainability_audit.png"
]

print("Organizing final submission...")

for f in main_files:
    if os.path.exists(f):
        shutil.copy(f, f"{SUBMISSION_ROOT}/main/")
    else:
        print(f"Warning: {f} not found!")

for f in submission_files:
    if os.path.exists(f):
        shutil.copy(f, f"{SUBMISSION_ROOT}/submission/")
    else:
        print(f"Warning: {f} not found!")

print(f"Final submission folder created: {SUBMISSION_ROOT}")
print("Next step: Zip this folder and submit!")
