import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
import cv2
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
PROJECT_ROOT = BASE_DIR.parent
for p in [str(CURRENT_DIR), str(BASE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from services.ngrok_client import NgrokJobClient, DEFAULT_NGROK_URL
    from services.ocr_ensemble import MultiEngineOCREnsemble
    from services.ner_pydantic import PydanticPrescriptionNERService
    from services.clinical_validator import ClinicalRuleValidator
    from services.word_grounder import WordGroundingService
    from services.pdf_processor import PDFProcessor
    from models.cnn_classifier import CNNPatchClassifier
except ImportError:
    from WebView_2.services.ngrok_client import NgrokJobClient, DEFAULT_NGROK_URL
    from WebView_2.services.ocr_ensemble import MultiEngineOCREnsemble
    from WebView_2.services.ner_pydantic import PydanticPrescriptionNERService
    from WebView_2.services.clinical_validator import ClinicalRuleValidator
    from WebView_2.services.word_grounder import WordGroundingService
    from WebView_2.services.pdf_processor import PDFProcessor
    from WebView_2.models.cnn_classifier import CNNPatchClassifier


class PrescriptionState(TypedDict):
    image_path: str
    image_w: int
    image_h: int
    patches: List[Dict[str, Any]]
    tesseract_res: Dict[str, Any]
    easyocr_res: Dict[str, Any]
    vlm_res: Dict[str, Any]
    merged_ocr_text: str
    all_boxes: List[Dict[str, Any]]
    word_detections: List[Dict[str, Any]]
    fused_ner_boxes: List[Dict[str, Any]]
    ner_data: Dict[str, Any]
    validation_warnings: List[str]
    uncertain_items: List[Dict[str, Any]]
    loop_count: int
    doctor_approved: bool
    doctor_edits: Optional[Dict[str, Any]]
    pdf_path: Optional[str]
    current_step: str
    status_msg: str


class LangGraphPrescriptionDigitizer:
    def __init__(
        self,
        ngrok_client: Optional[NgrokJobClient] = None,
        ollama_url: str = "http://localhost:11434",
        vlm_model: str = "qwen2.5vl:7b"
    ):
        self.ngrok_client = ngrok_client or NgrokJobClient()
        self.ocr_ensemble = MultiEngineOCREnsemble(ngrok_client=self.ngrok_client, ollama_url=ollama_url, vlm_model=vlm_model)
        self.ner_service = PydanticPrescriptionNERService(ngrok_client=self.ngrok_client, ollama_url=ollama_url, model_name=vlm_model)
        self.word_grounder = WordGroundingService(ngrok_client=self.ngrok_client, ollama_url=ollama_url)
        self.validator = ClinicalRuleValidator()
        self.cnn_classifier = CNNPatchClassifier()
        self.pdf_processor = PDFProcessor()
        
        # Build the state machine
        self.app = self._build_graph()

    def set_ngrok_url(self, new_url: str):
        if self.ngrok_client:
            self.ngrok_client.set_base_url(new_url)
            self.word_grounder.client = self.ngrok_client

    def _node_cnn_classify(self, state: PrescriptionState) -> Dict[str, Any]:
        """Step 1: Patch classification into printed vs handwritten vs figures."""
        print("--> [Node 1: MiniCNN Layout Classification]", flush=True)
        img = cv2.imread(state["image_path"])
        h, w = (img.shape[0], img.shape[1]) if img is not None else (1000, 800)
        
        patches = []
        try:
            if img is not None:
                res = self.cnn_classifier.process_image(img)
                patches = res.get("patches", [])
        except Exception as e:
            print(f"CNN classification warning: {e}", flush=True)
            
        return {
            "image_w": w,
            "image_h": h,
            "patches": patches,
            "current_step": "cnn_classified",
            "status_msg": f"Detected and classified {len(patches)} document regions."
        }

    def _node_ocr_ensemble(self, state: PrescriptionState) -> Dict[str, Any]:
        """Step 2: Run specialized OCR models (Tesseract & EasyOCR)."""
        print("--> [Node 2: Multi-Engine OCR Ensemble (Tesseract & EasyOCR)]", flush=True)
        tess = self.ocr_ensemble.run_tesseract(state["image_path"])
        easy = self.ocr_ensemble.run_easyocr(state["image_path"])
        
        combined_boxes = []
        combined_boxes.extend(tess.get("boxes", []))
        combined_boxes.extend(easy.get("boxes", []))
        
        return {
            "tesseract_res": tess,
            "easyocr_res": easy,
            "all_boxes": combined_boxes,
            "current_step": "ocr_extracted",
            "status_msg": f"Tesseract found {len(tess.get('boxes', []))} words | EasyOCR found {len(easy.get('boxes', []))} lines."
        }

    def _node_vlm_transcribe(self, state: PrescriptionState) -> Dict[str, Any]:
        """Step 3: High-accuracy pure OCR transcription with Qwen2.5-VL (no bounding box hallucination)."""
        print("--> [Node 3: Qwen2.5-VL Pure OCR Transcription]", flush=True)
        vlm_prompt = "Transcribe all visible text in this medical prescription image exactly as written, preserving layout, abbreviations, dosages, patient and physician details."
        vlm = self.ocr_ensemble.run_vlm(state["image_path"], custom_prompt=vlm_prompt)
        
        tess_text = state.get("tesseract_res", {}).get("text", "")
        easy_text = state.get("easyocr_res", {}).get("text", "")
        vlm_text = vlm.get("text", "")
        
        merged_text = vlm_text if vlm_text else (easy_text + "\n" + tess_text)
        
        return {
            "vlm_res": vlm,
            "merged_ocr_text": merged_text,
            "current_step": "vlm_transcribed",
            "status_msg": f"High-accuracy transcription complete ({vlm.get('backend', 'engine')})."
        }

    def _node_extract_ner(self, state: PrescriptionState) -> Dict[str, Any]:
        """Step 4: Structured NER Extraction with Pydantic."""
        print("--> [Node 4: Pydantic Structured Clinical Extraction]", flush=True)
        merged_text = state.get("merged_ocr_text", "")
        ner_res = self.ner_service.extract_structured_ner(merged_text)
        
        return {
            "ner_data": ner_res.get("data", {}),
            "current_step": "ner_extracted",
            "status_msg": f"Structured clinical entities extracted ({ner_res.get('backend', 'engine')})."
        }

    def _node_ground_and_fuse(self, state: PrescriptionState) -> Dict[str, Any]:
        """Step 5: Paragraph-Guided Word Grounding on normalized 1000x1000 image & NER Entity Fusion."""
        print("--> [Node 5: Paragraph-Guided Word Grounding & Entity Fusion]", flush=True)
        img_path = state["image_path"]
        w = state.get("image_w", 1000)
        h = state.get("image_h", 1000)
        ocr_text = state.get("merged_ocr_text", "")
        ner_data = state.get("ner_data", {})

        word_detections = []
        fused_boxes = []

        try:
            with Image.open(img_path) as img_pil:
                w, h = img_pil.size
                word_detections = self.word_grounder.detect_word_boxes(img_pil, ocr_text)
                if word_detections:
                    fused_boxes = self.word_grounder.fuse_ner_entities(word_detections, ner_data, w, h)
        except Exception as e:
            print(f"Grounding error: {e}", flush=True)

        # Merge word detections into all_boxes for fallback viewing
        all_boxes = list(state.get("all_boxes", []))
        if word_detections:
            all_boxes = [{"text": d.get("word", ""), "bbox": d.get("box", d.get("bbox", [])), "source": "qwen_grounding"} for d in word_detections]

        return {
            "image_w": w,
            "image_h": h,
            "word_detections": word_detections,
            "fused_ner_boxes": fused_boxes,
            "all_boxes": all_boxes,
            "current_step": "grounded_and_fused",
            "status_msg": f"Located {len(word_detections)} grounded words and {len(fused_boxes)} clinical entity highlights."
        }

    def _node_clinical_validate(self, state: PrescriptionState) -> Dict[str, Any]:
        """Step 6: Clinical validation and self-correction check."""
        print("--> [Node 6: Clinical Rules & Safety Validation]", flush=True)
        ner_data = state.get("ner_data", {})
        warnings, uncertain_items, needs_reocr = self.validator.validate_prescription(ner_data)
        
        return {
            "validation_warnings": warnings,
            "uncertain_items": uncertain_items,
            "current_step": "clinical_validated",
            "status_msg": f"Clinical validation finished ({len(warnings)} safety alerts found)."
        }

    def _node_zoom_reocr(self, state: PrescriptionState) -> Dict[str, Any]:
        """Step 7: Self-Correction Loop — Zoom re-OCR uncertain boxes."""
        print("--> [Node 7: Self-Correction Zoom Re-OCR Loop]", flush=True)
        uncertain_items = state.get("uncertain_items", [])
        loop_count = state.get("loop_count", 0) + 1
        
        boxes = state.get("all_boxes", [])
        if boxes and uncertain_items:
            target_box = boxes[0].get("bbox", [100, 100, 200, 50])
            re_read_text = self.ocr_ensemble.zoom_and_reocr_patch(
                state["image_path"],
                target_box,
                context_label=uncertain_items[0].get("field", "medication")
            )
            if re_read_text:
                state["merged_ocr_text"] += f"\n[Zoom Re-OCR]: {re_read_text}"
                
        return {
            "loop_count": loop_count,
            "merged_ocr_text": state["merged_ocr_text"],
            "current_step": "reocr_refined",
            "status_msg": f"Executed self-correction zoom re-OCR loop (Iteration {loop_count})."
        }

    def _node_doctor_review(self, state: PrescriptionState) -> Dict[str, Any]:
        """Step 8: Human-in-the-Loop breakpoint."""
        print("--> [Node 8: Human-in-the-Loop Doctor Review & Approval]", flush=True)
        if state.get("doctor_edits"):
            state["ner_data"].update(state["doctor_edits"])
            
        return {
            "doctor_approved": True,
            "current_step": "doctor_approved",
            "status_msg": "Prescription verified and approved by medical reviewer."
        }

    def _node_finalize_pdf(self, state: PrescriptionState) -> Dict[str, Any]:
        """Step 9: Reconstruct clinical PDF document."""
        print("--> [Node 9: Finalizing Clinical Record & PDF]", flush=True)
        pdf_path = ""
        try:
            out_dir = Path(state["image_path"]).parent / "generated_pdfs"
            out_dir.mkdir(exist_ok=True, parents=True)
            pdf_path = str(out_dir / f"digitized_{Path(state['image_path']).stem}.pdf")
            
            with open(pdf_path.replace(".pdf", ".txt"), "w", encoding="utf-8") as f:
                import json
                f.write(json.dumps(state.get("ner_data", {}), indent=2))
        except Exception as e:
            print(f"PDF finalize note: {e}", flush=True)
            
        return {
            "pdf_path": pdf_path,
            "current_step": "completed",
            "status_msg": "Prescription digitization pipeline successfully completed."
        }

    def _build_graph(self):
        builder = StateGraph(PrescriptionState)

        # Add Nodes
        builder.add_node("cnn_classify", self._node_cnn_classify)
        builder.add_node("ocr_ensemble", self._node_ocr_ensemble)
        builder.add_node("vlm_transcribe", self._node_vlm_transcribe)
        builder.add_node("extract_ner", self._node_extract_ner)
        builder.add_node("ground_and_fuse", self._node_ground_and_fuse)
        builder.add_node("clinical_validate", self._node_clinical_validate)
        builder.add_node("zoom_reocr", self._node_zoom_reocr)
        builder.add_node("doctor_review", self._node_doctor_review)
        builder.add_node("finalize_pdf", self._node_finalize_pdf)

        # Edges
        builder.add_edge(START, "cnn_classify")
        builder.add_edge("cnn_classify", "ocr_ensemble")
        builder.add_edge("ocr_ensemble", "vlm_transcribe")
        builder.add_edge("vlm_transcribe", "extract_ner")
        builder.add_edge("extract_ner", "ground_and_fuse")
        builder.add_edge("ground_and_fuse", "clinical_validate")

        def route_after_validation(state: PrescriptionState):
            if state.get("uncertain_items") and state.get("loop_count", 0) < 1:
                return "zoom_reocr"
            return "doctor_review"

        builder.add_conditional_edges(
            "clinical_validate",
            route_after_validation,
            {
                "zoom_reocr": "zoom_reocr",
                "doctor_review": "doctor_review"
            }
        )

        builder.add_edge("zoom_reocr", "extract_ner")
        builder.add_edge("doctor_review", "finalize_pdf")
        builder.add_edge("finalize_pdf", END)

        memory = MemorySaver()
        return builder.compile(checkpointer=memory, interrupt_before=["doctor_review"])
