import os
import glob
from pathlib import Path
from typing import List, Dict, Any


class ArchiveRAGService:
    """Provides semantic and keyword search across all indexed sample prescriptions and archive scans."""

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.archive_dir = self.base_dir / "Tessaract Training" / "archive"
        self.indexed_docs = []
        self._index_archive()

    def _index_archive(self):
        """Indexes archive image paths and any pre-computed OCR transcripts."""
        if not self.archive_dir.exists():
            return

        annotated_dir = self.base_dir / "Tessaract Training" / "annotated_archive_outputs"
        
        image_exts = {".png", ".jpg", ".jpeg", ".webp"}
        for img_p in sorted(self.archive_dir.iterdir()):
            if img_p.suffix.lower() in image_exts:
                txt_p = annotated_dir / f"{img_p.stem}_ocr.txt"
                content = ""
                if txt_p.exists():
                    try:
                        content = txt_p.read_text(encoding="utf-8")
                    except Exception:
                        pass
                
                self.indexed_docs.append({
                    "filename": img_p.name,
                    "image_path": str(img_p),
                    "text": content,
                    "stem": img_p.stem
                })

    def search(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """Searches indexed prescriptions for query terms."""
        if not query.strip():
            return self.indexed_docs[:top_k]

        q_terms = query.lower().split()
        scored_results = []

        for doc in self.indexed_docs:
            score = 0
            doc_text = (doc["filename"] + " " + doc["text"]).lower()
            
            for term in q_terms:
                if term in doc_text:
                    score += 10
                    # Boost if in filename
                    if term in doc["filename"].lower():
                        score += 20

            if score > 0:
                scored_results.append({**doc, "score": score})

        # Sort by relevance
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k] if scored_results else self.indexed_docs[:top_k]
