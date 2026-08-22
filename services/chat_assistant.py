import json
from typing import List, Dict, Any, Optional
import requests

from .ngrok_client import NgrokJobClient


class ClinicalChatAssistant:
    """Conversational Assistant for prescription Q&A, dosage inquiries, and drug interaction analysis."""

    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "qwen2.5vl:7b"):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.ngrok_client = NgrokJobClient(base_url=self.ollama_url)

    def ask(self, question: str, prescription_ner: Dict[str, Any], history: Optional[List[Dict[str, str]]] = None, model_name: Optional[str] = None) -> str:
        """Answers clinical questions grounded in the extracted prescription data with strict guardrails."""
        history = history or []
        prescription_summary = json.dumps(prescription_ner, indent=2)
        active_model = model_name or self.model_name

        system_prompt = (
            "You are an expert Clinical Pharmacist and Medical Verification AI Assistant.\n"
            "You have direct access to the extracted and verified prescription data:\n"
            f"```json\n{prescription_summary}\n```\n\n"
            "Strict System Guardrails:\n"
            "1. You only answer questions related to medicine, health, pharmacology, medical conditions, clinical safety, prescriptions, patient diagnostics, or schedules.\n"
            "2. If the user asks about unrelated topics (e.g. programming, creative writing, cooking, general knowledge, or logic puzzles), politely decline to answer, stating that your expertise is strictly limited to clinical and pharmaceutical assistance.\n"
            "3. Format all responses using structured Markdown. Use headers (###), bold text (**bold**), clean bullet points, or markdown tables for medication schedules or drug details where appropriate.\n"
            "4. If a potential interaction or safety warning is detected, highlight it using a prominent '⚠️ SAFETY WARNING:' section.\n"
            "5. Always append the following medical disclaimer at the very end of your response, separated by a horizontal rule (---):\n"
            "   *_Disclaimer: This AI assessment is for clinical decision support only and does not replace professional medical judgment._*"
        )

        # Format chat history for prompt template
        prompt_parts = []
        for turn in history:
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt_parts.append(f"USER: {question}")
        combined_prompt = "\n\n".join(prompt_parts)

        # 1. Try Ngrok Colab Job Queue first if ngrok url configured
        if self.ngrok_client and "ngrok-free.dev" in self.ngrok_client.base_url:
            try:
                print(f"--> Sending Chat job to Ngrok Colab Queue (Model: {active_model}): {self.ngrok_client.base_url}", flush=True)
                raw_out, elapsed = self.ngrok_client.run_inference(
                    model=active_model,
                    prompt=combined_prompt,
                    system=system_prompt,
                    options={"temperature": 0.1},
                    timeout_sec=600
                )
                if raw_out:
                    return raw_out.strip()
            except Exception as e:
                print(f"[Ngrok Chat Warning]: {e} -> Falling back to local Ollama", flush=True)

        # 2. Local Fallback via Ollama
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": question})

        payload = {
            "model": active_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1}
        }

        try:
            # When calling locally, we target the standard Ollama chat API endpoint on localhost
            resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "").strip()
            else:
                return f"Assistant error (HTTP {resp.status_code}): {resp.text}"
        except Exception as e:
            return f"Clinical Assistant connection error (both Ngrok and Local Offline): {str(e)}"
