import os
import io
import re
import base64
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
import torch

from .ngrok_client import NgrokJobClient, DEFAULT_NGROK_URL

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class MultiEngineOCREnsemble:
    """
    Orchestrates:
    1. Colab Ngrok Job Queue (qwen2.5vl:7b, frob/unlimited-ocr)
    2. Local Ollama Server (Qwen2.5-VL)
    3. Fine-Tuned Tesseract LSTM (eng_stc_finetune)
    4. Fine-Tuned EasyOCR (easyocr_stc_finetune)
    """

    def __init__(
        self,
        ngrok_client: Optional[NgrokJobClient] = None,
        ollama_url: str = "http://localhost:11434",
        vlm_model: str = "qwen2.5vl:7b"
    ):
        self.ngrok_client = ngrok_client or NgrokJobClient()
        self.ollama_url = ollama_url.rstrip("/")
        self.vlm_model = vlm_model
        self.base_dir = Path(__file__).resolve().parent.parent.parent

        # 1. Setup Tesseract Path & Prefix
        self._init_tesseract()

        # 2. Setup EasyOCR Reader
        self.easyocr_reader = None
        self._init_easyocr()

    def _init_tesseract(self):
        tess_candidates = [
            Path(r"C:\Users\arick.sarkar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path("/usr/bin/tesseract"),
        ]
        for cand in tess_candidates:
            if cand.exists():
                pytesseract.pytesseract.tesseract_cmd = str(cand)
                break

        custom_tessdata = self.base_dir / "Tessaract Training" / "tessdata_custom"
        if custom_tessdata.exists():
            os.environ["TESSDATA_PREFIX"] = str(custom_tessdata).replace("\\", "/")
            self.tesseract_lang = "eng_stc_finetune"
        else:
            self.tesseract_lang = "eng"

    def _init_easyocr(self):
        if not EASYOCR_AVAILABLE:
            return
        try:
            user_net_dir = Path.home() / ".EasyOCR" / "user_network"
            if (user_net_dir / "easyocr_stc_finetune.yaml").exists():
                self.easyocr_reader = easyocr.Reader(
                    ['custom'],
                    gpu=torch.cuda.is_available(),
                    recog_network='easyocr_stc_finetune',
                    download_enabled=True
                )
            else:
                self.easyocr_reader = easyocr.Reader(
                    ['en'],
                    gpu=torch.cuda.is_available(),
                    download_enabled=True
                )
        except Exception as e:
            print(f"Warning: EasyOCR custom network init: {e}", flush=True)
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), download_enabled=True)
            except Exception:
                self.easyocr_reader = None

    def run_tesseract(self, image_path: str, min_conf: int = 20) -> Dict[str, Any]:
        """Runs fine-tuned Tesseract and extracts word bounding boxes."""
        start_t = time.time()
        img = cv2.imread(image_path)
        if img is None:
            return {"text": "", "boxes": [], "time_sec": 0.0}
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            data = pytesseract.image_to_data(
                img_rgb,
                lang=self.tesseract_lang,
                config="--psm 3",
                output_type=Output.DICT
            )
            raw_text = pytesseract.image_to_string(img_rgb, lang=self.tesseract_lang, config="--psm 3")
            
            boxes = []
            for i in range(len(data['text'])):
                txt = data['text'][i].strip()
                conf = int(float(data['conf'][i]))
                if txt and conf >= min_conf:
                    bx, by, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    boxes.append({
                        "text": txt,
                        "conf": conf,
                        "bbox": [bx, by, bw, bh],
                        "source": "tesseract"
                    })

            return {
                "text": raw_text.strip(),
                "boxes": boxes,
                "time_sec": round(time.time() - start_t, 3)
            }
        except Exception as e:
            return {"text": "", "boxes": [], "error": str(e), "time_sec": round(time.time() - start_t, 3)}

    def run_easyocr(self, image_path: str, min_conf: float = 0.15) -> Dict[str, Any]:
        """Runs fine-tuned EasyOCR with scale normalization."""
        start_t = time.time()
        if not self.easyocr_reader:
            return {"text": "", "boxes": [], "time_sec": 0.0}

        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"text": "", "boxes": [], "time_sec": 0.0}
            
            orig_h, orig_w = img.shape[:2]
            scale = 1.0
            max_dim = max(orig_h, orig_w)
            if max_dim > 1000:
                scale = 1000.0 / max_dim
                resized_img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
            else:
                resized_img = img

            results = self.easyocr_reader.readtext(resized_img, batch_size=8)
            boxes = []
            full_lines = []
            for bbox, text, conf in results:
                text_clean = text.strip()
                if text_clean and conf >= min_conf:
                    pts = np.array(bbox, dtype=np.float32)
                    if scale != 1.0:
                        pts /= scale
                    
                    x1 = int(pts[:, 0].min())
                    y1 = int(pts[:, 1].min())
                    x2 = int(pts[:, 0].max())
                    y2 = int(pts[:, 1].max())
                    w = max(1, x2 - x1)
                    h = max(1, y2 - y1)
                    conf_pct = int(round(conf * 100))
                    
                    boxes.append({
                        "text": text_clean,
                        "conf": conf_pct,
                        "bbox": [x1, y1, w, h],
                        "source": "easyocr"
                    })
                    full_lines.append(text_clean)

            return {
                "text": "\n".join(full_lines),
                "boxes": boxes,
                "time_sec": round(time.time() - start_t, 3)
            }
        except Exception as e:
            return {"text": "", "boxes": [], "error": str(e), "time_sec": round(time.time() - start_t, 3)}

    def run_vlm(self, image_path: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Runs Vision-Language transcription via Colab Ngrok job queue, falling back to local Ollama."""
        start_t = time.time()
        
        with Image.open(image_path) as img:
            if max(img.width, img.height) > 1024:
                img.thumbnail((1024, 1024))
            buffered = io.BytesIO()
            img.convert('RGB').save(buffered, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        default_prompt = (
            "You are an expert medical transcriptionist. Transcribe all text visible in this prescription scan with extreme precision. "
            "Transcribe both printed header information and handwritten medications, dosages, frequency, and patient details. "
            "Preserve layout and line breaks cleanly."
        )
        prompt_text = custom_prompt or default_prompt

        # 1. Try Colab Ngrok Job Queue first
        if self.ngrok_client and self.ngrok_client.base_url:
            try:
                print(f"--> Sending VLM job to Ngrok Colab Queue: {self.ngrok_client.base_url}", flush=True)
                raw_result, elapsed = self.ngrok_client.run_inference(
                    model=self.vlm_model,
                    prompt=prompt_text,
                    image_b64=img_b64,
                    options={"temperature": 0.0, "num_predict": 512},
                    timeout_sec=600
                )
                if raw_result and raw_result.strip():
                    return {
                        "text": raw_result.strip(),
                        "time_sec": elapsed,
                        "model": self.vlm_model,
                        "backend": "colab_ngrok"
                    }
            except Exception as ngrok_err:
                print(f"[Ngrok Job Queue Warning]: {ngrok_err} -> Falling back to local Ollama", flush=True)

        # 2. Local Ollama Fallback
        import requests
        payload = {
            "model": self.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [img_b64]
                }
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 512}
        }

        try:
            resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=25)
            if resp.status_code == 200:
                out_text = resp.json().get("message", {}).get("content", "").strip()
                return {
                    "text": out_text,
                    "time_sec": round(time.time() - start_t, 3),
                    "model": "local_ollama",
                    "backend": "local_ollama"
                }
        except Exception as e:
            return {
                "text": "",
                "error": f"VLM Error: {str(e)}",
                "time_sec": round(time.time() - start_t, 3)
            }

        return {"text": "", "time_sec": round(time.time() - start_t, 3)}

    def zoom_and_reocr_patch(self, image_path: str, bbox: List[int], context_label: str = "") -> str:
        """Crops an uncertain patch, scales it up, and runs high-resolution VLM re-OCR."""
        img = Image.open(image_path)
        x, y, w, h = bbox
        
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.width, x + w + pad_x)
        y2 = min(img.height, y + h + pad_y)
        
        crop = img.crop((x1, y1, x2, y2))
        
        buffered = io.BytesIO()
        crop.save(buffered, format="PNG")
        crop_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        prompt = (
            f"Carefully read the handwritten or printed characters in this zoomed medical prescription crop (Expected entity: '{context_label}'). "
            "Output only the exact transcribed word/number with nothing else."
        )
        
        if self.ngrok_client and self.ngrok_client.base_url:
            try:
                res, _ = self.ngrok_client.run_inference(
                    model=self.vlm_model,
                    prompt=prompt,
                    image_b64=crop_b64,
                    timeout_sec=40
                )
                if res and res.strip():
                    return res.strip()
            except Exception:
                pass

        import requests
        payload = {
            "model": "Qwen2.5-VL-3B-Instruct-UD-Q4_K_XL:latest",
            "messages": [{"role": "user", "content": prompt, "images": [crop_b64]}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 64}
        }
        try:
            resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "").strip()
        except Exception:
            pass
        return ""
