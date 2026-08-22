import re
from typing import Dict, List, Any, Tuple


KNOWN_COMMON_DRUGS = {
    "amoxicillin", "ampicillin", "paracetamol", "acetaminophen", "ibuprofen",
    "mefenamic", "azithromycin", "ciprofloxacin", "clarithromycin", "doxycycline",
    "metformin", "amlodipine", "losartan", "atorvastatin", "omeprazole",
    "pantoprazole", "cetirizine", "loratadine", "salbutamol", "montelukast",
    "prednisone", "dexamethasone", "co-amoxiclav", "clindamycin", "cephalexin",
    "aspirin", "clopidogrel", "tramadol", "diclofenac", "fluconazole"
}

STANDARD_DOSAGE_UNITS = [r"mg", r"g", r"mcg", r"ml", r"cap", r"tab", r"tablets", r"capsules", r"drops", r"iu", r"puff"]
DOSAGE_REGEX = re.compile(r"\b\d+(?:\.\d+)?\s*(?:" + "|".join(STANDARD_DOSAGE_UNITS) + r")\b", re.IGNORECASE)


class ClinicalRuleValidator:
    """Validates extracted medical entities against clinical dosage and nomenclature rules."""

    @staticmethod
    def validate_prescription(ner_data: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]], bool]:
        """
        Returns:
            warnings: List of clinical alert messages
            uncertain_items: List of items that need visual re-OCR zoom
            needs_reocr: Boolean flag indicating if self-correction loop should trigger
        """
        warnings = []
        uncertain_items = []
        medications = ner_data.get("medications", [])

        if not medications:
            warnings.append("No medications detected in prescription body (Rx section may need zoom review).")

        for idx, med in enumerate(medications, 1):
            drug_name = (med.get("drug_name") or "").strip()
            dosage = (med.get("dosage") or "").strip()
            frequency = (med.get("frequency") or "").strip()

            if not drug_name:
                warnings.append(f"Medication #{idx}: Missing drug name.")
                uncertain_items.append({"field": f"medication_{idx}_name", "value": ""})
                continue

            # Check dosage presence and unit validity
            if not dosage:
                warnings.append(f"Medication #{idx} ('{drug_name}'): Missing dosage strength.")
                uncertain_items.append({"field": f"medication_{idx}_dosage", "value": drug_name})
            elif not DOSAGE_REGEX.search(dosage):
                warnings.append(f"Medication #{idx} ('{drug_name}'): Unusual dosage format '{dosage}' (missing standard unit like mg, ml, tab).")
                uncertain_items.append({"field": f"medication_{idx}_dosage", "value": dosage})

            # Check frequency
            if not frequency:
                warnings.append(f"Medication #{idx} ('{drug_name}'): Missing administration frequency or instructions.")

            # Drug name sanity check
            clean_name = drug_name.lower().split()[0]
            if len(clean_name) > 3 and clean_name not in KNOWN_COMMON_DRUGS:
                # Check for near matches
                # If name looks strange or contains non-alphanumeric noise
                if re.search(r"[^a-zA-Z0-9\s-]", drug_name):
                    warnings.append(f"Medication #{idx} ('{drug_name}'): Contains non-standard characters, may be OCR artifact.")
                    uncertain_items.append({"field": f"medication_{idx}_name", "value": drug_name})

        needs_reocr = len(uncertain_items) > 0
        return warnings, uncertain_items, needs_reocr
