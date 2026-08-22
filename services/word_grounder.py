import json
import re
import cv2
import io
import base64
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional
import requests

try:
    from services.ngrok_client import NgrokJobClient
except ImportError:
    from WebView_2.services.ngrok_client import NgrokJobClient

ENTITY_COLOR_MAP = {
    # Facility & Header
    "facility_name": "#6366f1",        # Indigo
    "facility_address": "#818cf8",     # Indigo Light
    "facility_phone": "#a855f7",       # Purple Light

    # Patient Details
    "patient_name": "#8b5cf6",         # Purple
    "patient_address": "#3b82f6",      # Blue
    "age": "#ec4899",                  # Pink
    "gender": "#f43f5e",               # Rose
    "date": "#f59e0b",                 # Amber

    # Prescriber Details
    "doctor_name": "#06b6d4",          # Cyan
    "doctor_specialty": "#0284c7",     # Sky Blue
    "license_no": "#64748b",           # Slate
    "ptr_no": "#475569",               # Dark Slate
    "s2_no": "#334155",                # Deep Slate

    # Clinical Conditions
    "diagnosis_indication": "#ef4444", # Red

    # Medication Main & Sub-Categories
    "medications": "#10b981",          # Emerald Green
    "medication_drug_name": "#10b981", # Emerald Green
    "medication_dosage": "#059669",    # Dark Emerald
    "medication_quantity": "#34d399",  # Mint Green
    "medication_frequency": "#84cc16", # Lime Green
    "medication_duration": "#22c55e",  # Green
    "medication_route": "#06b6d4"      # Cyan
}

DEFAULT_ENTITY_COLOR = "#0ea5e9"    # Sky Blue default


class WordGroundingService:
    def __init__(self, ngrok_client: Optional[NgrokJobClient] = None, ollama_url: str = "http://localhost:11434"):
        self.client = ngrok_client
        self.ollama_url = ollama_url.rstrip("/")

    def detect_word_boxes(
        self,
        image_pil: Image.Image,
        ocr_transcript: str,
        model_name: str = "qwen2.5vl:7b"
    ) -> List[Dict[str, Any]]:
        """
        Executes Paragraph-Guided Word Grounding on normalized 1000x1000 image.
        Returns word-level bounding boxes mapped to the original image dimensions [x, y, w, h].
        """
        if not ocr_transcript or not ocr_transcript.strip():
            return []

        img_w, img_h = image_pil.size
        
        # 1. Resize image to 1000x1000 for 1:1 Qwen coordinate precision
        img_1000 = image_pil.resize((1000, 1000), Image.LANCZOS)
        
        buf = io.BytesIO()
        img_1000.convert('RGB').save(buf, format="JPEG", quality=90)
        image_b64_1000 = base64.b64encode(buf.getvalue()).decode("utf-8")

        prompt_text = f"""Below is the OCR transcript for this medical prescription image:
"{ocr_transcript.strip()}"

Task: Locate every word or phrase in the image.
You MUST output a valid JSON list where EVERY item contains BOTH "text" and "bbox_2d" coordinates as [xmin, ymin, xmax, ymax] scaled 0 to 1000.

Example Format:
[
  {{"text": "Name:", "bbox_2d": [20, 31, 87, 56]}},
  {{"text": "Armando", "bbox_2d": [149, 28, 320, 62]}}
]"""

        detections = []

        # 1. Try Ngrok Colab Job Queue
        if self.client and self.client.base_url:
            try:
                raw_result, elapsed = self.client.run_inference(
                    model=model_name,
                    prompt=prompt_text,
                    image_b64=image_b64_1000,
                    options={"temperature": 0.0},
                    timeout_sec=600
                )
                detections = self._parse_detections_json(raw_result, img_w, img_h)
            except Exception as e:
                print(f"[Ngrok Grounding Warning]: {e} -> Fallback to local Ollama", flush=True)

        # 2. Try Local Ollama Fallback
        if not detections:
            try:
                payload = {
                    "model": model_name,
                    "messages": [{
                        "role": "user",
                        "content": prompt_text,
                        "images": [image_b64_1000]
                    }],
                    "stream": False,
                    "options": {"temperature": 0.0}
                }
                resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=40)
                if resp.status_code == 200:
                    raw_result = resp.json().get("message", {}).get("content", "")
                    detections = self._parse_detections_json(raw_result, img_w, img_h)
            except Exception as e:
                print(f"[Local Grounding Error]: {e}", flush=True)

        return detections

    def _parse_detections_json(self, raw_result: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
        detections = []
        # Extract JSON block
        json_match = re.search(r'\[\s*\{.*\}\s*\]', raw_result, re.DOTALL)
        if json_match:
            try:
                items = json.loads(json_match.group(0))
                for item in items:
                    word_text = str(item.get("text", "")).strip()
                    box = item.get("bbox_2d") or item.get("box_2d") or []
                    if word_text and len(box) == 4:
                        xmin, ymin, xmax, ymax = map(int, box)
                        # Convert 0-1000 coordinates to original image pixel scale
                        x1 = int((xmin / 1000.0) * img_w)
                        y1 = int((ymin / 1000.0) * img_h)
                        x2 = int((xmax / 1000.0) * img_w)
                        y2 = int((ymax / 1000.0) * img_h)
                        w = max(1, x2 - x1)
                        h = max(1, y2 - y1)
                        detections.append({
                            "word": word_text,
                            "text": word_text,
                            "bbox": [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)],
                            "box": [max(0, x1), max(0, y1), w, h],
                            "norm_bbox": [xmin, ymin, xmax, ymax]
                        })
            except Exception:
                pass

        # Fallback regex parser
        if not detections:
            for line in raw_result.splitlines():
                m = re.search(r'["\']?text["\']?:\s*["\'](.*?)["\'].*?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', line)
                if m:
                    w, xmin, ymin, xmax, ymax = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
                    x1 = int((xmin / 1000.0) * img_w)
                    y1 = int((ymin / 1000.0) * img_h)
                    x2 = int((xmax / 1000.0) * img_w)
                    y2 = int((ymax / 1000.0) * img_h)
                    detections.append({
                        "word": w.strip(),
                        "text": w.strip(),
                        "bbox": [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)],
                        "box": [max(0, x1), max(0, y1), max(1, x2 - x1), max(1, y2 - y1)],
                        "norm_bbox": [xmin, ymin, xmax, ymax]
                    })

        return detections

    def fuse_ner_entities(
        self,
        word_detections: List[Dict[str, Any]],
        ner_entities: Dict[str, Any],
        img_w: int,
        img_h: int
    ) -> List[Dict[str, Any]]:
        """
        Matches extracted NER entities (e.g. patient_name="Armando Ceguia") against word-level detections,
        merges their bounding boxes into single entity boxes, and generates underline coordinates.
        """
        fused_entity_boxes = []
        if not word_detections or not ner_entities:
            return fused_entity_boxes

        def normalize_string(s: str) -> str:
            return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

        word_tokens = []
        for det in word_detections:
            norm_w = normalize_string(det.get("word", det.get("text", "")))
            if norm_w:
                word_tokens.append({
                    "raw_word": det.get("word", det.get("text", "")),
                    "norm_word": norm_w,
                    "bbox": det["bbox"]
                })

        def is_token_match(target: str, candidate: str) -> bool:
            if target == candidate:
                return True
            if len(target) >= 3 and target in candidate:
                return True
            return False

        # Process top-level entity fields
        for key, val in ner_entities.items():
            if key == "medications":
                continue
            if not isinstance(val, str) or not val.strip():
                continue

            entity_text = val.strip()
            entity_words = [normalize_string(w) for w in entity_text.split() if normalize_string(w)]
            if not entity_words:
                continue

            matched_boxes = []
            matched_words = []

            for i in range(len(word_tokens)):
                if i + len(entity_words) <= len(word_tokens):
                    seq_match = True
                    for k in range(len(entity_words)):
                        if not is_token_match(entity_words[k], word_tokens[i + k]["norm_word"]):
                            seq_match = False
                            break
                    if seq_match:
                        for k in range(len(entity_words)):
                            matched_boxes.append(word_tokens[i + k]["bbox"])
                            matched_words.append(word_tokens[i + k]["raw_word"])
                        break

            if not matched_boxes:
                for wt in word_tokens:
                    for ew in entity_words:
                        if is_token_match(ew, wt["norm_word"]):
                            matched_boxes.append(wt["bbox"])
                            matched_words.append(wt["raw_word"])

            if matched_boxes:
                x1 = min(b[0] for b in matched_boxes)
                y1 = min(b[1] for b in matched_boxes)
                x2 = max(b[2] for b in matched_boxes)
                y2 = max(b[3] for b in matched_boxes)
                color = ENTITY_COLOR_MAP.get(key, DEFAULT_ENTITY_COLOR)

                fused_entity_boxes.append({
                    "entity_key": key,
                    "label": f"{key.replace('_', ' ').title()}: {entity_text}",
                    "entity_text": entity_text,
                    "bbox": [x1, y1, x2, y2],
                    "box": [x1, y1, max(1, x2 - x1), max(1, y2 - y1)],
                    "underline": {"x1": x1, "y1": y2 + 2, "x2": x2, "y2": y2 + 2},
                    "color": color,
                    "matched_words": matched_words
                })

        # Process nested medication entities
        meds = ner_entities.get("medications", [])
        if isinstance(meds, list):
            med_subfields = [
                ("drug_name", "Drug Name", "#10b981"),
                ("dosage", "Dosage", "#059669"),
                ("quantity", "Quantity", "#34d399"),
                ("frequency", "Frequency", "#84cc16"),
                ("duration", "Duration", "#22c55e"),
                ("route", "Route", "#06b6d4")
            ]
            for idx, med in enumerate(meds):
                if not isinstance(med, dict):
                    continue
                for sub_key, sub_label, sub_color in med_subfields:
                    val = med.get(sub_key, "")
                    if not isinstance(val, str) or not val.strip():
                        continue
                    val_text = val.strip()
                    sub_words = [normalize_string(w) for w in val_text.split() if normalize_string(w)]
                    if not sub_words:
                        continue

                    matched_boxes = []
                    matched_words = []
                    for wt in word_tokens:
                        for sw in sub_words:
                            if is_token_match(sw, wt["norm_word"]):
                                matched_boxes.append(wt["bbox"])
                                matched_words.append(wt["raw_word"])

                    if matched_boxes:
                        x1 = min(b[0] for b in matched_boxes)
                        y1 = min(b[1] for b in matched_boxes)
                        x2 = max(b[2] for b in matched_boxes)
                        y2 = max(b[3] for b in matched_boxes)

                        fused_entity_boxes.append({
                            "entity_key": f"medication_{sub_key}",
                            "label": f"Rx {sub_label}: {val_text}",
                            "entity_text": val_text,
                            "bbox": [x1, y1, x2, y2],
                            "box": [x1, y1, max(1, x2 - x1), max(1, y2 - y1)],
                            "underline": {"x1": x1, "y1": y2 + 2, "x2": x2, "y2": y2 + 2},
                            "color": sub_color,
                            "matched_words": matched_words
                        })

        return fused_entity_boxes
