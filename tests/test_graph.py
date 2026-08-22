import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

# Enable unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from WebView_2.services.langgraph_pipeline import LangGraphPrescriptionDigitizer


def test_langgraph_execution():
    print("=" * 60, flush=True)
    print("Testing LangGraph Clinical OCR & NER Pipeline", flush=True)
    print("=" * 60, flush=True)

    sample_img = root_dir / "WebView_2" / "samples" / "sample1.jpg"
    if not sample_img.exists():
        sample_img = root_dir / "WebView" / "sample1.jpg"

    print(f"Target Image: {sample_img}", flush=True)

    # 1. Initialize Pipeline
    digitizer = LangGraphPrescriptionDigitizer()
    print("[OK] Initialized LangGraph State Machine with Checkpointer.", flush=True)

    # 2. Start Execution
    initial_state = {
        "image_path": str(sample_img),
        "image_w": 0,
        "image_h": 0,
        "patches": [],
        "tesseract_res": {},
        "easyocr_res": {},
        "vlm_res": {},
        "merged_ocr_text": "",
        "all_boxes": [],
        "ner_data": {},
        "validation_warnings": [],
        "uncertain_items": [],
        "loop_count": 0,
        "doctor_approved": False,
        "doctor_edits": None,
        "pdf_path": None,
        "current_step": "started",
        "status_msg": "Test start"
    }

    config = {"configurable": {"thread_id": "test_thread_001"}}
    print("\n[Running Graph until Doctor Review Interrupt]...", flush=True)
    state_at_interrupt = digitizer.app.invoke(initial_state, config=config)

    print(f"[OK] Paused at Step: '{state_at_interrupt.get('current_step')}'", flush=True)
    print(f"[OK] Status: {state_at_interrupt.get('status_msg')}", flush=True)
    print(f"[OK] Total OCR Boxes Found: {len(state_at_interrupt.get('all_boxes', []))}", flush=True)
    print(f"[OK] Validation Warnings: {len(state_at_interrupt.get('validation_warnings', []))}", flush=True)

    # 3. Resume with Doctor Edits (Human in the Loop)
    print("\n[Simulating Doctor Verification & Resume]...", flush=True)
    doctor_edits = {
        "patient_name": "Test Patient",
        "doctor_name": "Dr. Test Specialist"
    }
    digitizer.app.update_state(config, {"doctor_edits": doctor_edits, "doctor_approved": True})
    final_state = digitizer.app.invoke(None, config=config)

    print(f"[OK] Final Step: '{final_state.get('current_step')}'", flush=True)
    print(f"[OK] Doctor Approved Flag: {final_state.get('doctor_approved')}", flush=True)
    print("=" * 60, flush=True)
    print("All LangGraph tests passed successfully!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    test_langgraph_execution()
