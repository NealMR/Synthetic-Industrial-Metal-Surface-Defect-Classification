from fpdf import FPDF
import os

class MethodologyPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Industrial Defect Detection Methodology', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 8, body)
        self.ln()

    def add_image_section(self, image_path, title):
        if os.path.exists(image_path):
            self.chapter_title(title)
            # Add image with fixed width to fit page
            self.image(image_path, x=10, w=180)
            self.ln(5)

def generate_full_pdf():
    pdf = MethodologyPDF()
    pdf.add_page()
    
    # Sections based on methodology.md content
    sections = [
        ("1. Problem Statement", "Quality control in industrial manufacturing is critical. This project automates the detection of defects: Crack, Hole, Rust, and Scratch."),
        ("2. Dataset & Preprocessing", "Images resized to 224x224. Data augmentation includes flips, rotations, and jitter to ensure robustness."),
        ("3. Model Architecture", "Architecture: EfficientNet-B2. Rationale: Optimal accuracy-to-efficiency ratio for industrial edge deployment."),
        ("4. Training Strategy", "Two-phase training: Head-warmup (2 epochs) followed by Full Fine-tuning (8 epochs). Adam optimizer with weight decay used."),
        ("5. Final Metrics", "Validation Accuracy: 100.00%. Macro F1-Score: 1.0000. Model Size: 34.2 MB."),
        ("6. Innovation (Grad-CAM)", "Implemented Explainable AI (Grad-CAM) to visualize defect locations and ensure model transparency.")
    ]
    
    for title, body in sections:
        pdf.chapter_title(title)
        pdf.chapter_body(body)
    
    # Add Images
    pdf.add_image_section('training_curves.png', 'Training Loss & Accuracy Curves')
    pdf.add_image_section('confusion_matrix.png', 'Final Confusion Matrix')
    pdf.add_image_section('explainability_audit.png', 'Grad-CAM Explainability (AI Heatmaps)')
    
    os.makedirs('submission', exist_ok=True)
    pdf.output('submission/methodology.pdf')
    print("✅ High-quality methodology.pdf with images generated in submission/ folder.")

if __name__ == '__main__':
    generate_full_pdf()
