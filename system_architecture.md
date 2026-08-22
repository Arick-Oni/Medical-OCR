# MediDigitizer AI 2.0 - System Architecture & Pipeline Flowchart

This document details the system architecture and stateful processing flowchart of the LangGraph-powered **MediDigitizer AI 2.0** application.

---

## 1. High-Level Architecture

The application is structured into a stateful, modular pipeline utilizing local classification, hybrid multi-engine OCR, remote vision-language processing via a job queue, and safety-guided clinical entity extraction:

```
[ Frontend: Interactive Canvas + Review Drawer ]
                      │   ▲
                      ▼   │  JSON State Events
          [ Backend: FastAPI App Controller ]
                      │   ▲
                      ▼   │  Invoke / Resume
             [ LangGraph State Machine ]
              ├── Local MiniCNN (PyTorch)
              ├── Local Tesseract & EasyOCR
              └── Remote Colab GPU (via Ngrok Tunnel)
                     ├── Qwen2.5-VL (OCR & Word Grounding)
                     └── frob/unlimited-ocr (Fine-tuned Transcription)
```

---

## 2. Step-by-Step Processing Flowchart

Below is the stateful execution flow managed by the **LangGraph** orchestration engine. The machine executes nodes sequentially, supports self-correcting loops for low-confidence data, and holds a state breakpoint for manual review:

```mermaid
flowchart TD
    %% Define styles
    classDef startEnd fill:#1F2937,stroke:#374151,stroke-width:2px,color:#F3F4F6;
    classDef nodeStyle fill:#1E293B,stroke:#475569,stroke-width:1.5px,color:#F8FAFC;
    classDef loopStyle fill:#3B0764,stroke:#6B21A8,stroke-width:1.5px,color:#F3E8FF;
    classDef hitlStyle fill:#78350F,stroke:#D97706,stroke-width:2px,color:#FEF3C7;
    classDef decStyle fill:#0F766E,stroke:#0D9488,stroke-width:2px,color:#F0FDFA;

    %% Workflow Nodes
    Start([Upload Image or Select Sample]) --> Node1[Node 1: MiniCNN Layout Classification]
    Node1 --> Node2[Node 2: Multi-Engine OCR Ensemble <br/><i>Tesseract & EasyOCR</i>]
    Node2 --> Node3[Node 3: VLM Pure OCR Transcription <br/><i>High-Accuracy Qwen Text</i>]
    Node3 --> Node4[Node 4: Structured Clinical Extraction <br/><i>Plain-Schema NER & Safe Validation</i>]
    Node4 --> Node5[Node 5: Word Grounding & Entity Fusion <br/><i>1000x1000 Coordinate Projection</i>]
    Node5 --> Node6[Node 6: Clinical Rules & Safety Validation <br/><i>Dose Limits & Warnings check</i>]

    %% Validation Check & Loops
    Node6 --> Dec1{Uncertain Fields & <br/>Loop Count < 1?}
    
    Dec1 -- Yes --> Node7[Node 7: Self-Correction Zoom Re-OCR Loop <br/><i>Patch Zooming & Text Refinement</i>]
    Node7 --> Node4
    
    Dec1 -- No --> Node8[Node 8: Doctor Review & HITL Breakpoint <br/><i>Interactive Form & Canvas Underlines</i>]
    
    %% Approval & Finalization
    Node8 --> Dec2{Doctor Edits & Approved?}
    Dec2 -- No / Paused --> Node8
    Dec2 -- Yes / Approved --> Node9[Node 9: Finalize PDF & Clinical Records]
    Node9 --> End([Completed Digitized Record])

    %% Apply Styles
    class Start,End startEnd;
    class Node1,Node2,Node3,Node4,Node5,Node6 nodeStyle;
    class Node7 loopStyle;
    class Node8 hitlStyle;
    class Dec1,Dec2 decStyle;
```

---

## 3. Core Pipeline Components

### Node 1: MiniCNN Layout Classification
* **Input**: Native image array (`OpenCV`).
* **Processing**: Performs sliding-window patch segmentation and feeds regions into `mini_cnn_model.pth`.
* **Output**: Classifies boxes into **Handwritten**, **Printed**, or **Mixed** text blocks to direct OCR strategies.

### Node 2: Multi-Engine OCR Ensemble
* **Input**: Image paths.
* **Processing**: Executes Tesseract LSTM and EasyOCR locally in parallel.
* **Output**: Generates initial raw line and word bounding boxes.

### Node 3: VLM Pure OCR Transcription
* **Input**: Prescription scan.
* **Processing**: Sends image + transcription request to remote Qwen2.5-VL via Colab Ngrok queue.
* **Output**: Returns accurate text transcription without coordinate hallucination.

### Node 4: Structured Clinical Extraction
* **Input**: Combined OCR text.
* **Processing**: Extracts information using the expanded prescription JSON schema.
* **Output**: Safely parses Patient name, Doctor details, Date, and a list of Medications into structured fields.

### Node 5: Word Grounding & Entity Fusion
* **Input**: Original image + Node 3 OCR text.
* **Processing**: Normalizes image to $1000 \times 1000$ and asks Qwen for raw `[xmin, ymin, xmax, ymax]` word locations. Projects coordinates back to native image dimensions.
* **Output**: Links extracted NER entities (e.g. Drug names) to coordinates, generating highlights and underlines.

### Node 6 & 7: Validation & Self-Correction
* **Processing**: Evaluates extraction confidence. If crucial numbers are low-contrast or illegible, Node 7 crops the image box, runs a local high-contrast zoom pass, and merges refined characters back into the main transcription.

### Node 8: Human-in-the-Loop Review
* **Processing**: Pauses the LangGraph machine state. Serves findings to the interactive front-end. Doctors can review overlays on the document canvas, edit any field, and click **Approve & Finalize** to resume the workflow.
