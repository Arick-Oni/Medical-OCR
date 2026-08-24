import os
import io
import cv2
import json
import uuid
import base64
import shutil
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure both WebView_2 and project root are in sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Safe imports for both package and direct module execution
try:
    from services.ngrok_client import NgrokJobClient, DEFAULT_NGROK_URL
    from services.langgraph_pipeline import LangGraphPrescriptionDigitizer, PrescriptionState
    from services.chat_assistant import ClinicalChatAssistant
    from services.rag_archive_service import ArchiveRAGService
    from services.pdf_processor import PDFProcessor
    from services.db_history import DatabaseHistoryManager
    from models.cnn_classifier import CLASS_COLORS
except ImportError:
    from WebView_2.services.ngrok_client import NgrokJobClient, DEFAULT_NGROK_URL
    from WebView_2.services.langgraph_pipeline import LangGraphPrescriptionDigitizer, PrescriptionState
    from WebView_2.services.chat_assistant import ClinicalChatAssistant
    from WebView_2.services.rag_archive_service import ArchiveRAGService
    from WebView_2.services.pdf_processor import PDFProcessor
    from WebView_2.services.db_history import DatabaseHistoryManager
    from WebView_2.models.cnn_classifier import CLASS_COLORS

STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

SAMPLE_DIRS = [
    str((BASE_DIR / "samples").resolve()),
    str((BASE_DIR.parent / "Tessaract Training" / "archive").resolve()),
    str((BASE_DIR.parent / "Colab_Prescription_Digitizer").resolve()),
    str(BASE_DIR.parent.resolve())
]

app = FastAPI(
    title="MediDigitizer AI 2.0 (LangGraph-Powered)",
    description="Stateful Clinical Prescription OCR, Self-Correcting Medical NER & Human-in-the-Loop Review Platform"
)

# Mount Static Files & Templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize Ngrok Client & Services
ngrok_client = NgrokJobClient(base_url=DEFAULT_NGROK_URL)
digitizer = LangGraphPrescriptionDigitizer(ngrok_client=ngrok_client)
chat_assistant = ClinicalChatAssistant(ollama_url=DEFAULT_NGROK_URL)
rag_service = ArchiveRAGService()
db_history = DatabaseHistoryManager()

# App State Settings
APP_SETTINGS = {
    "ngrok_url": DEFAULT_NGROK_URL,
    "ocr_unlimited_model": "frob/unlimited-ocr:q8_0",
    "ocr_qwen_model": "qwen2.5vl:7b",
    "ner_model": "qwen2.5vl:7b",
    "enable_cnn": True,
    "enable_ocr": True,
    "enable_ner": True,
}

# In-memory thread storage for active LangGraph runs
ACTIVE_THREADS: Dict[str, Dict[str, Any]] = {}


class SettingsUpdate(BaseModel):
    ngrok_url: Optional[str] = None
    ocr_unlimited_model: Optional[str] = None
    ocr_qwen_model: Optional[str] = None
    ner_model: Optional[str] = None


class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None
    ner_data: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = None
    model: Optional[str] = None


class ApprovalRequest(BaseModel):
    thread_id: str
    doctor_edits: Dict[str, Any]


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "settings": APP_SETTINGS,
            "class_colors": CLASS_COLORS,
            "version": "2.0 (LangGraph + Grounded OCR)"
        }
    )


@app.get("/api/samples")
async def list_sample_images():
    """Returns ONLY sample1.jpg for the quick test section as requested."""
    for s_dir in SAMPLE_DIRS:
        f_path = os.path.join(s_dir, "sample1.jpg")
        if os.path.exists(f_path):
            return [{"name": "sample1.jpg", "path": f_path, "size": os.path.getsize(f_path)}]
    
    # Fallback to creating sample1.jpg in samples directory if missing
    samples_dir = BASE_DIR / "samples"
    samples_dir.mkdir(exist_ok=True, parents=True)
    target_sample = samples_dir / "sample1.jpg"
    if target_sample.exists():
        return [{"name": "sample1.jpg", "path": str(target_sample.resolve()), "size": os.path.getsize(target_sample)}]
    
    return []


@app.get("/api/samples/{filename}")
async def get_sample_file(filename: str):
    for s_dir in SAMPLE_DIRS:
        f_path = os.path.join(s_dir, filename)
        if os.path.exists(f_path) and os.path.isfile(f_path):
            with open(f_path, "rb") as f:
                return StreamingResponse(io.BytesIO(f.read()), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail=f"Sample '{filename}' not found")


@app.post("/api/preview")
async def preview_document(
    file: Optional[UploadFile] = File(None),
    sample_name: Optional[str] = Form(None)
):
    """
    Instantly extracts and returns page preview images (Base64) for images and PDFs
    without running heavy inference models.
    """
    file_bytes = None
    filename = "upload"

    if file:
        file_bytes = await file.read()
        filename = file.filename
    elif sample_name:
        for s_dir in SAMPLE_DIRS:
            f_path = os.path.join(s_dir, sample_name)
            if os.path.exists(f_path) and os.path.isfile(f_path):
                with open(f_path, "rb") as f:
                    file_bytes = f.read()
                filename = sample_name
                break

    if not file_bytes:
        # Fallback to default sample1.jpg
        for s_dir in SAMPLE_DIRS:
            f_path = os.path.join(s_dir, "sample1.jpg")
            if os.path.exists(f_path):
                with open(f_path, "rb") as f:
                    file_bytes = f.read()
                filename = "sample1.jpg"
                break

    if not file_bytes:
        raise HTTPException(status_code=400, detail="No file or sample provided")

    is_pdf = PDFProcessor.is_pdf(file_bytes, filename)
    if is_pdf:
        pages_raw = PDFProcessor.render_pdf_to_images(file_bytes, target_max_dim=800)
    else:
        pages_raw = [PDFProcessor.load_single_image(file_bytes)]

    preview_pages = []
    for p in pages_raw:
        buf = io.BytesIO()
        p["image_pil"].save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        preview_pages.append({
            "page_number": p["page_number"],
            "total_pages": p["total_pages"],
            "width": p["width"],
            "height": p["height"],
            "image_b64": f"data:image/jpeg;base64,{img_b64}"
        })

    return {
        "status": "success",
        "filename": filename,
        "is_pdf": is_pdf,
        "total_pages": len(preview_pages),
        "pages": preview_pages
    }


@app.get("/api/health/ngrok")
async def check_ngrok_health(url: Optional[str] = Query(None)):
    health = ngrok_client.check_health(test_url=url)
    return health


@app.get("/api/settings")
async def get_settings():
    return APP_SETTINGS


@app.post("/api/settings")
async def update_settings(payload: SettingsUpdate):
    if payload.ngrok_url:
        APP_SETTINGS["ngrok_url"] = payload.ngrok_url
        ngrok_client.set_base_url(payload.ngrok_url)
        digitizer.set_ngrok_url(payload.ngrok_url)
        chat_assistant.ollama_url = payload.ngrok_url.rstrip("/")
    if payload.ocr_unlimited_model:
        APP_SETTINGS["ocr_unlimited_model"] = payload.ocr_unlimited_model
    if payload.ocr_qwen_model:
        APP_SETTINGS["ocr_qwen_model"] = payload.ocr_qwen_model
    if payload.ner_model:
        APP_SETTINGS["ner_model"] = payload.ner_model
        chat_assistant.model_name = payload.ner_model
    return {"status": "success", "settings": APP_SETTINGS}


@app.post("/api/pipeline/start")
async def start_pipeline(
    file: UploadFile = File(None),
    sample_name: Optional[str] = Form(None),
    sample_path: Optional[str] = Form(None)
):
    """Initializes a new LangGraph prescription processing run."""
    thread_id = str(uuid.uuid4())[:8]
    
    if file and file.filename:
        file_path = UPLOADS_DIR / f"{thread_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        target_img_path = str(file_path.resolve())
    elif sample_name:
        target_img_path = None
        for s_dir in SAMPLE_DIRS:
            f_path = os.path.join(s_dir, sample_name)
            if os.path.exists(f_path) and os.path.isfile(f_path):
                target_img_path = str(Path(f_path).resolve())
                break
        if not target_img_path:
            target_img_path = str((BASE_DIR / "samples" / "sample1.jpg").resolve())
    elif sample_path and os.path.exists(sample_path):
        target_img_path = str(Path(sample_path).resolve())
    else:
        default_sample = BASE_DIR / "samples" / "sample1.jpg"
        target_img_path = str(default_sample.resolve())

    initial_state = {
        "image_path": target_img_path,
        "image_w": 0,
        "image_h": 0,
        "patches": [],
        "tesseract_res": {},
        "easyocr_res": {},
        "vlm_res": {},
        "merged_ocr_text": "",
        "all_boxes": [],
        "word_detections": [],
        "fused_ner_boxes": [],
        "ner_data": {},
        "validation_warnings": [],
        "uncertain_items": [],
        "loop_count": 0,
        "doctor_approved": False,
        "doctor_edits": None,
        "pdf_path": None,
        "current_step": "started",
        "status_msg": "Initializing LangGraph state machine..."
    }

    config = {"configurable": {"thread_id": thread_id}}
    
    final_state = digitizer.app.invoke(initial_state, config=config)
    ACTIVE_THREADS[thread_id] = {
        "state": final_state,
        "config": config,
        "image_path": target_img_path
    }

    # Save to history database
    try:
        db_history.save_run(
            thread_id=thread_id,
            image_name=file.filename if file else (sample_name or os.path.basename(target_img_path)),
            image_path=target_img_path,
            merged_ocr_text=final_state.get("merged_ocr_text", ""),
            ner_data=final_state,
            validation_warnings=final_state.get("validation_warnings", []),
            status="pending"
        )
    except Exception as db_err:
        print(f"[Database History Warning]: Failed to save starting run log: {db_err}", flush=True)

    return {
        "thread_id": thread_id,
        "current_step": final_state.get("current_step"),
        "status_msg": final_state.get("status_msg"),
        "state": final_state
    }


@app.get("/api/pipeline/state/{thread_id}")
async def get_pipeline_state(thread_id: str):
    """Retrieves current LangGraph checkpoint state."""
    if thread_id not in ACTIVE_THREADS:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ACTIVE_THREADS[thread_id]["state"]


@app.post("/api/pipeline/approve")
async def approve_and_finalize(payload: ApprovalRequest):
    """Resumes the paused LangGraph workflow past the Human-in-the-Loop review breakpoint."""
    thread_id = payload.thread_id
    if thread_id not in ACTIVE_THREADS:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread_info = ACTIVE_THREADS[thread_id]
    config = thread_info["config"]

    digitizer.app.update_state(config, {"doctor_edits": payload.doctor_edits, "doctor_approved": True})
    resumed_state = digitizer.app.invoke(None, config=config)
    ACTIVE_THREADS[thread_id]["state"] = resumed_state

    # Update history database with final approved data
    try:
        db_history.save_run(
            thread_id=thread_id,
            image_name=os.path.basename(thread_info["image_path"]),
            image_path=thread_info["image_path"],
            merged_ocr_text=resumed_state.get("merged_ocr_text", ""),
            ner_data=resumed_state,
            validation_warnings=resumed_state.get("validation_warnings", []),
            status="approved"
        )
    except Exception as db_err:
        print(f"[Database History Warning]: Failed to update approved run log: {db_err}", flush=True)

    return {
        "thread_id": thread_id,
        "status": "approved",
        "current_step": resumed_state.get("current_step"),
        "state": resumed_state
    }


@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest):
    """Conversational clinical assistant for Q&A and drug interaction checking."""
    ner_data = payload.ner_data
    if not ner_data and payload.thread_id and payload.thread_id in ACTIVE_THREADS:
        ner_data = ACTIVE_THREADS[payload.thread_id]["state"].get("ner_data", {})

    ner_data = ner_data or {}
    
    # Save user query to history database
    if payload.thread_id:
        try:
            db_history.save_chat_message(payload.thread_id, "user", payload.question)
        except Exception as db_err:
            print(f"[Database History Warning]: Failed to save user chat log: {db_err}", flush=True)

    use_model = payload.model or chat_assistant.model_name
    
    response_text = chat_assistant.ask(
        question=payload.question,
        prescription_ner=ner_data,
        history=payload.history,
        model_name=use_model
    )

    # Save assistant response to history database
    if payload.thread_id:
        try:
            db_history.save_chat_message(payload.thread_id, "assistant", response_text)
        except Exception as db_err:
            print(f"[Database History Warning]: Failed to save assistant chat log: {db_err}", flush=True)

    return {"answer": response_text}


@app.get("/api/archive/search")
async def search_archive(q: str = Query("", description="Search term")):
    """Searches indexed prescription scans."""
    results = rag_service.search(query=q, top_k=8)
    return {"results": results}


@app.get("/api/history")
async def get_history_runs(limit: int = Query(50, description="Max history logs to fetch")):
    """Retrieves all previous prescription digitization run logs from history database."""
    try:
        runs = db_history.list_runs(limit=limit)
        return {"runs": runs, "backend": db_history.backend}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database list failed: {str(e)}")


@app.get("/api/history/{thread_id}")
async def get_history_run_details(thread_id: str):
    """Retrieves details of a specific historical run log."""
    try:
        run = db_history.get_run(thread_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found in history")
        return {"run": run}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database get failed: {str(e)}")


@app.delete("/api/history/{thread_id}")
async def delete_history_run(thread_id: str):
    """Removes a run log from the history database."""
    try:
        success = db_history.delete_run(thread_id)
        if not success:
            raise HTTPException(status_code=404, detail="Run not found in history")
        return {"status": "success", "message": f"Run {thread_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database delete failed: {str(e)}")


@app.get("/api/image/view")
async def view_image_by_path(path: str = Query(..., description="Absolute path to the image")):
    """Returns the image file from the local file system securely."""
    clean_path = Path(path).resolve()
    if not clean_path.exists() or not clean_path.is_file():
        # Safely extract filename by normalizing backslashes on Linux hosts
        filename = path.replace("\\", "/").split("/")[-1]
        possible_paths = [
            BASE_DIR / "uploads" / filename,
            BASE_DIR / "samples" / filename,
            BASE_DIR / filename
        ]
        found = False
        for p in possible_paths:
            if p.exists() and p.is_file():
                clean_path = p
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Image path not found")
    
    # Check that it's an image
    if clean_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
        raise HTTPException(status_code=400, detail="File is not a supported image format")
        
    return FileResponse(str(clean_path))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

