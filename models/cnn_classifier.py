import os
import cv2
import numpy as np
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class DummyModule:
        def __init__(self, *args, **kwargs): pass
    class Dummy:
        pass
    nn = Dummy()
    nn.Module = DummyModule
    transforms = Dummy()
    transforms.Compose = DummyModule
from PIL import Image

# 4 classes used in mini_cnn_model.pth training
CLASS_NAMES_SORTED = ['Handwritten_extended', 'Mixed_extended', 'Other_extended', 'Printed_extended']
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES_SORTED)}

SHORT_NAME = {
    'Handwritten_extended': 'Handwritten',
    'Printed_extended':     'Printed',
    'Mixed_extended':       'Mixed',
    'Other_extended':       'Other',
}

# Color palette for overlay (matching hybrid_fullpage_digitizer_with_cnn.ipynb)
CLASS_COLORS = {
    'Handwritten': {'hex': '#2ECC71', 'rgb': (46, 204, 113),  'bg': 'rgba(46, 204, 113, 0.18)'},  # Green 🟢
    'Printed':     {'hex': '#3498DB', 'rgb': (52, 152, 219),  'bg': 'rgba(52, 152, 219, 0.18)'},  # Blue 🔵
    'Mixed':       {'hex': '#FF69B4', 'rgb': (255, 105, 180), 'bg': 'rgba(255, 105, 180, 0.18)'}, # Pink 🌸
    'Other':       {'hex': '#F1C40F', 'rgb': (241, 196, 15),  'bg': 'rgba(241, 196, 15, 0.18)'},  # Yellow 🟡
}

class MiniCNN(nn.Module):
    """
    Exact 3-block MiniCNN model for 4-class patch classification.
    Features: 3 -> 32 -> 64 -> 128
    Classifier: Flatten -> Linear(128*8*8, 256) -> ReLU -> Dropout(0.4) -> Linear(256, 4)
    """
    def __init__(self, num_classes=4):
        super(MiniCNN, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.15)
            )
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CNNPatchClassifier:
    def __init__(self, model_path: str = None):
        if not HAS_TORCH:
            self.device = "cpu"
            self.model = None
            self.transform = None
            print("[Info] Running CNNPatchClassifier in CPU-only rule-based fallback mode (PyTorch not installed).")
            return

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        # 3-channel Grayscale transform (matching training & inference notebooks)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        candidate_paths = [
            model_path,
            "mini_cnn_model.pth",
            "../mini_cnn_model.pth",
            os.path.join(os.path.dirname(__file__), "..", "..", "mini_cnn_model.pth"),
            r"c:\Users\arick.sarkar\Desktop\Save The Children Techhub\OCR\mini_cnn_model.pth"
        ]

        resolved_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                resolved_path = p
                break

        if resolved_path:
            try:
                self.model = MiniCNN(num_classes=4).to(self.device)
                state_dict = torch.load(resolved_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"[OK] MiniCNN loaded successfully from: '{resolved_path}' on {self.device}")
            except Exception as e:
                print(f"[Warning] Failed to load MiniCNN weights from '{resolved_path}': {e}")
                self.model = None
        else:
            print("[Warning] MiniCNN weights file not found. Running in fallback mode.")

    def segment_prescription(self, img_bgr, min_patch_w=15, min_patch_h=10):
        """
        Exact OpenCV morphological line removal and contour word extraction
        from cnn_prescription_inference.ipynb and hybrid_fullpage_digitizer_with_cnn.ipynb.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Horizontal line removal kernel (11x11 with center row active)
        line_kernel = np.zeros((11, 11), dtype=np.uint8)
        line_kernel[5, :] = 1
        lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, line_kernel, iterations=1)
        gray_no_lines = cv2.subtract(gray, lines)
        
        # Otsu thresholding
        _, binary = cv2.threshold(gray_no_lines, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilation to connect letter strokes into word patches (5x5 kernel)
        dilation_kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, dilation_kernel, iterations=1)
        
        # Contours with 2-level hierarchy
        contours, _ = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        img_h, img_w = img_bgr.shape[:2]
        bboxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter noise and whole-page bounding box
            if w >= min_patch_w and h >= min_patch_h and (w < img_w * 0.98 or h < img_h * 0.98):
                bboxes.append((x, y, w, h))

        # Sort top-to-bottom, left-to-right
        bboxes.sort(key=lambda b: (b[1] // 25, b[0]))
        return bboxes

    def classify_patches(self, img_bgr, bboxes, batch_size=32):
        """
        Classifies all cropped word bounding boxes into Handwritten, Printed, Mixed, Other.
        """
        if not bboxes:
            return []

        if self.model is None:
            results = []
            for (x, y, w, h) in bboxes:
                label = 'Handwritten' if h > 28 else 'Printed'
                results.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'class': label,
                    'confidence': 0.90,
                    'color': CLASS_COLORS[label]
                })
            return results

        crops = []
        for (x, y, w, h) in bboxes:
            patch = img_bgr[y:y+h, x:x+w]
            if patch.size == 0:
                patch = np.zeros((64, 64, 3), dtype=np.uint8)
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(patch_rgb)
            crops.append(self.transform(pil_img))

        all_preds = []
        all_confs = []
        with torch.no_grad():
            for i in range(0, len(crops), batch_size):
                batch = torch.stack(crops[i:i+batch_size]).to(self.device)
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=-1)
                confs, preds = torch.max(probs, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_confs.extend(confs.cpu().tolist())

        classified = []
        for i, (x, y, w, h) in enumerate(bboxes):
            raw_class = CLASS_NAMES_SORTED[all_preds[i]]
            short_class = SHORT_NAME[raw_class]
            conf = float(round(all_confs[i], 4))
            classified.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'class': short_class,
                'raw_class': raw_class,
                'confidence': conf,
                'color': CLASS_COLORS[short_class]
            })

        return classified

    def process_image(self, img_bgr):
        """
        Runs OpenCV word segmentation + MiniCNN patch classification.
        """
        bboxes = self.segment_prescription(img_bgr)
        classified_patches = self.classify_patches(img_bgr, bboxes)

        class_keys = ['Handwritten', 'Printed', 'Mixed', 'Other']
        counts = {c: 0 for c in class_keys}
        for p in classified_patches:
            counts[p['class']] += 1

        total = len(classified_patches)
        stats = {
            'total_patches': total,
            'counts': counts,
            'percentages': {
                c: round((counts[c] / total * 100), 1) if total > 0 else 0
                for c in class_keys
            }
        }

        return {
            'patches': classified_patches,
            'stats': stats
        }
