import json
import re
import time
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import requests

from .ngrok_client import NgrokJobClient


class MedicationItem(BaseModel):
    drug_name: str = Field(default="", description="Brand or generic medication name")
    dosage: str = Field(default="", description="Dosage strength e.g., 500mg, 250mg/5ml, 1 cap")
    quantity: Optional[str] = Field(default="", description="Quantity dispensed e.g. #21, 1 bottle")
    frequency: str = Field(default="", description="Administration frequency e.g., 3x a day, BID, QID, after meals")
    route: Optional[str] = Field(default="PO", description="Route of administration e.g. PO (Oral), Topical, IV")
    duration: Optional[str] = Field(default="", description="Duration of treatment e.g., for 7 days")


class PrescriptionSchema(BaseModel):
    facility_name: Optional[str] = Field(default="", description="Hospital, clinic, or medical center name")
    facility_address: Optional[str] = Field(default="", description="Clinic street address and city")
    facility_phone: Optional[str] = Field(default="", description="Clinic telephone / contact number")
    patient_name: Optional[str] = Field(default="", description="Patient's full name")
    age: Optional[str] = Field(default="", description="Patient age")
    gender: Optional[str] = Field(default="", description="Patient sex/gender")
    date: Optional[str] = Field(default="", description="Prescription issue date")
    doctor_name: Optional[str] = Field(default="", description="Prescribing physician name")
    doctor_specialty: Optional[str] = Field(default="", description="Physician specialization or department")
    license_no: Optional[str] = Field(default="", description="Doctor Medical License number")
    ptr_no: Optional[str] = Field(default="", description="Professional Tax Receipt (PTR) number")
    s2_no: Optional[str] = Field(default="", description="Special S2 dangerous drug license number")
    diagnosis_indication: Optional[str] = Field(default="", description="Clinical indication, diagnosis, or Rx notes")
    medications: List[MedicationItem] = Field(default_factory=list, description="List of prescribed medications")


EXPANDED_PRESCRIPTION_NER_SCHEMA = {
    "type": "object",
    "properties": {
        "facility_name": {"type": "string"},
        "facility_address": {"type": "string"},
        "facility_phone": {"type": "string"},
        "patient_name": {"type": "string"},
        "patient_address": {"type": "string"},
        "age": {"type": "string"},
        "gender": {"type": "string"},
        "date": {"type": "string"},
        "doctor_name": {"type": "string"},
        "doctor_specialty": {"type": "string"},
        "license_no": {"type": "string"},
        "ptr_no": {"type": "string"},
        "s2_no": {"type": "string"},
        "dea_no": {"type": "string"},
        "npi_no": {"type": "string"},
        "prescription_no": {"type": "string"},
        "refills": {"type": "string"},
        "dispense_as_written": {"type": "string"},
        "diagnosis_indication": {"type": "string"},
        "chief_complaint": {"type": "string"},
        "diagnostic_tests": {"type": "string"},
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"},
                    "dosage": {"type": "string"},
                    "quantity": {"type": "string"},
                    "frequency": {"type": "string"},
                    "route": {"type": "string"},
                    "duration": {"type": "string"}
                },
                "required": ["drug_name", "dosage", "quantity", "frequency", "route", "duration"]
            }
        }
    },
    "required": [
        "facility_name", "facility_address", "facility_phone",
        "patient_name", "patient_address", "age", "gender", "date",
        "doctor_name", "doctor_specialty", "license_no", "ptr_no", "s2_no", "dea_no", "npi_no",
        "prescription_no", "refills", "dispense_as_written", "diagnosis_indication",
        "chief_complaint", "diagnostic_tests", "medications"
    ]
}


class PydanticPrescriptionNERService:
    def __init__(
        self,
        ngrok_client: Optional[NgrokJobClient] = None,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5vl:7b"
    ):
        self.ngrok_client = ngrok_client or NgrokJobClient()
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name

    def extract_structured_ner(self, ocr_text: str) -> Dict[str, Any]:
        """Extracts structured clinical data using high-capability WebView 1 schema and instructions."""
        start_t = time.time()
        text_clean = ocr_text.strip()
        if not text_clean:
            return {"data": PrescriptionSchema().model_dump(), "time_sec": 0.0, "status": "empty_input"}

        system_prompt = (
            "You are an expert Clinical Medical Information Extraction & Named Entity Recognition (NER) AI assistant.\n"
            "Carefully analyze the entire medical prescription transcription text from top to bottom and extract all administrative and medical entities into the structured JSON schema.\n\n"
            "CRITICAL EXTRACTION GUIDELINES:\n"
            "1. FACILITY & HEADER: Extract hospital, clinic, or medical center names, street addresses, postal codes, and telephone numbers at the top into `facility_name`, `facility_address`, and `facility_phone`.\n"
            "2. PATIENT DETAILS: Extract the complete full name of the patient into `patient_name` (e.g. 'Mme. M. John Hubbard'). Infer gender if titles like 'Mme.', 'Mr.', 'Mrs.', 'M.' or sex indicators are present.\n"
            "3. PRESCRIBER: Extract the doctor/physician name into `doctor_name` and any license/PTR numbers into their respective fields.\n"
            "4. DIAGNOSIS: Extract any clinical indication or condition (e.g. 'coronary artery disease') into `diagnosis_indication`.\n"
            "5. MEDICATIONS: For each prescribed drug, cleanly separate `drug_name`, `dosage` (e.g. '650mg', '81mg'), `route` (e.g. 'PO'), and `frequency` (e.g. 'q4h PRN', 'daily').\n"
            "6. If a field is not present in the text, use an empty string '' (or empty array [] for medications)."
        )

        user_prompt = f"Extract all structured clinical entities from this prescription text:\n\n{text_clean}"

        raw_out = None
        elapsed = 0.0
        backend = "fallback"

        # 1. Try Colab Ngrok Job Queue
        if self.ngrok_client and self.ngrok_client.base_url:
            try:
                raw_out, elapsed = self.ngrok_client.run_inference(
                    model=self.model_name,
                    prompt=user_prompt,
                    system=system_prompt,
                    json_schema=EXPANDED_PRESCRIPTION_NER_SCHEMA,
                    options={"temperature": 0.0},
                    timeout_sec=600
                )
                if raw_out:
                    backend = "colab_ngrok"
            except Exception as e:
                print(f"[Ngrok NER Warning]: {e} -> Falling back to local Ollama", flush=True)

        # 2. Local Ollama Fallback
        if not raw_out:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt + f"\n\nJSON Schema:\n{json.dumps(EXPANDED_PRESCRIPTION_NER_SCHEMA, indent=2)}"},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.0}
            }
            try:
                resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=30)
                if resp.status_code == 200:
                    raw_out = resp.json().get("message", {}).get("content", "").strip()
                    elapsed = round(time.time() - start_t, 3)
                    backend = "local_ollama"
            except Exception as e:
                print(f"[Local NER Error]: {e}", flush=True)

        if raw_out:
            try:
                clean_json_str = self._clean_json(raw_out)
                data_dict = json.loads(clean_json_str)

                # Safe validation into PrescriptionSchema
                medications_list = []
                for med in data_dict.get("medications", []):
                    medications_list.append(MedicationItem(
                        drug_name=med.get("drug_name") or "",
                        dosage=med.get("dosage") or "",
                        quantity=med.get("quantity") or "",
                        frequency=med.get("frequency") or "",
                        route=med.get("route") or "PO",
                        duration=med.get("duration") or ""
                    ))

                validated_obj = PrescriptionSchema(
                    facility_name=data_dict.get("facility_name") or "",
                    facility_address=data_dict.get("facility_address") or "",
                    facility_phone=data_dict.get("facility_phone") or "",
                    patient_name=data_dict.get("patient_name") or "",
                    age=data_dict.get("age") or "",
                    gender=data_dict.get("gender") or "",
                    date=data_dict.get("date") or "",
                    doctor_name=data_dict.get("doctor_name") or "",
                    doctor_specialty=data_dict.get("doctor_specialty") or "",
                    license_no=data_dict.get("license_no") or "",
                    ptr_no=data_dict.get("ptr_no") or "",
                    s2_no=data_dict.get("s2_no") or "",
                    diagnosis_indication=data_dict.get("diagnosis_indication") or "",
                    medications=medications_list
                )

                return {
                    "data": validated_obj.model_dump(),
                    "time_sec": elapsed,
                    "status": "success",
                    "backend": backend
                }
            except Exception as parse_err:
                print(f"[NER Parsing/Validation Error]: {parse_err}", flush=True)

        # 3. Rule-based fallback
        validated_obj = self._fallback_regex_extract(text_clean)
        return {
            "data": validated_obj.model_dump(),
            "time_sec": round(time.time() - start_t, 3),
            "status": "fallback"
        }

    @staticmethod
    def _clean_json(raw_str: str) -> str:
        clean = raw_str.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return clean.strip()

    def _fallback_regex_extract(self, text: str) -> PrescriptionSchema:
        """Lightweight regex extractor for license numbers, dates, and obvious drugs."""
        ptr_matches = re.findall(r"(?i)PTR\s*(?:No\.?|#)?\s*[:.]?\s*(\d+)", text)
        lic_matches = re.findall(r"(?i)Lic\.?\s*(?:No\.?|#)?\s*[:.]?\s*\(?(\d+)\)?", text)
        s2_matches = re.findall(r"(?i)S2\s*(?:No\.?|#)?\s*[:.]?\s*([A-Za-z0-9-]+)", text)
        dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
        
        return PrescriptionSchema(
            license_no=lic_matches[0] if lic_matches else "",
            ptr_no=ptr_matches[0] if ptr_matches else "",
            s2_no=s2_matches[0] if s2_matches else "",
            date=dates[0] if dates else ""
        )
