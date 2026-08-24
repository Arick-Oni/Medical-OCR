// MediDigitizer AI 2.0 - LangGraph & Grounded OCR Controller
document.addEventListener("DOMContentLoaded", () => {
    let currentThreadId = null;
    let currentNERData = {};
    let selectedFile = null;
    let selectedSampleName = null;
    let currentImageDataUrl = null;
    let fullState = null;
    let isLiveUnlocked = false; // Render Cloud Safe Mode (Locked by default)

    // JSON Tab UI Elements
    const selectJsonSection = document.getElementById("selectJsonSection");
    const jsonCodeView = document.getElementById("jsonCodeView");
    const btnCopyJSON = document.getElementById("btnCopyJSON");

    // History Modal UI Elements
    const btnOpenHistory = document.getElementById("btnOpenHistory");
    const historyModal = document.getElementById("historyModal");
    const btnCloseHistory = document.getElementById("btnCloseHistory");
    const btnCloseHistoryFooter = document.getElementById("btnCloseHistoryFooter");
    const historyLoading = document.getElementById("historyLoading");
    const historyEmpty = document.getElementById("historyEmpty");
    const historyTable = document.getElementById("historyTable");
    const historyTableBody = document.getElementById("historyTableBody");
    const historyDbBackend = document.getElementById("historyDbBackend");

    // Lock & Demo Mode UI Elements
    const modeBadge = document.getElementById("modeBadge");
    const modeBadgeText = document.getElementById("modeBadgeText");
    const btnToggleLock = document.getElementById("btnToggleLock");
    const btnToggleLockIcon = document.getElementById("btnToggleLockIcon");
    const btnToggleLockText = document.getElementById("btnToggleLockText");
    const btnProcessIcon = document.getElementById("btnProcessIcon");
    const btnProcessText = document.getElementById("btnProcessText");
    const chatLockBanner = document.getElementById("chatLockBanner");
    const btnUnlockFromChat = document.getElementById("btnUnlockFromChat");
    const unlockModal = document.getElementById("unlockModal");
    const btnCloseUnlockModal = document.getElementById("btnCloseUnlockModal");
    const btnCancelUnlock = document.getElementById("btnCancelUnlock");
    const btnConfirmUnlock = document.getElementById("btnConfirmUnlock");

    // Initialize Canvas Overlay Controller
    const canvas = new PrescriptionCanvas('documentCanvas', 'canvasTooltip', 'canvasViewport');

    // UI Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const btnProcess = document.getElementById("btnProcess");
    const samplesList = document.getElementById("samplesList");
    const canvasPlaceholder = document.getElementById("canvasPlaceholder");
    const hitlBanner = document.getElementById("hitlBanner");
    const btnApproveState = document.getElementById("btnApproveState");
    const emptyNERState = document.getElementById("emptyNERState");
    const nerDetailsContainer = document.getElementById("nerDetailsContainer");
    const transcriptionCodeBlock = document.getElementById("transcriptionCodeBlock");
    const btnCopyTranscription = document.getElementById("btnCopyTranscription");
    const chatContainer = document.getElementById("chatContainer");
    const chatInput = document.getElementById("chatInput");
    const btnSendChat = document.getElementById("btnSendChat");
    const btnThemeToggle = document.getElementById("btnThemeToggle");
    const selectChatModel = document.getElementById("selectChatModel");
    const inputChatModelCustom = document.getElementById("inputChatModelCustom");

    // Toolbar Zoom Buttons
    document.getElementById("btnZoomIn").addEventListener("click", () => canvas.zoomIn());
    document.getElementById("btnZoomOut").addEventListener("click", () => canvas.zoomOut());
    document.getElementById("btnZoomFit").addEventListener("click", () => canvas.fitToScreen());

    // Layer Filter Chips
    const layerBindings = [
        { id: "layerOCR", layer: "ocr" },
        { id: "layerNER", layer: "ner" },
        { id: "layerWordBoxes", layer: "word_boxes" },
        { id: "layerHandwritten", layer: "handwritten" },
        { id: "layerPrinted", layer: "printed" },
        { id: "layerMixed", layer: "mixed" }
    ];

    layerBindings.forEach(({ id, layer }) => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener("change", (e) => {
                canvas.setLayer(layer, e.target.checked);
                const parent = el.closest(".layer-chip");
                if (parent) parent.classList.toggle("active", e.target.checked);
            });
        }
    });

    // Tab Switching
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetEl = document.getElementById(targetId);
            if (targetEl) targetEl.classList.add("active");
        });
    });

    // Theme Toggle
    btnThemeToggle.addEventListener("click", () => {
        document.body.classList.toggle("dark-theme");
        document.body.classList.toggle("light-theme");
    });

    // Copy Transcription
    if (btnCopyTranscription) {
        btnCopyTranscription.addEventListener("click", () => {
            navigator.clipboard.writeText(transcriptionCodeBlock.textContent);
            btnCopyTranscription.textContent = "Copied!";
            setTimeout(() => {
                btnCopyTranscription.innerHTML = '<i class="fa-solid fa-copy"></i> Copy';
            }, 2000);
        });
    }

    // Ngrok & Settings Elements
    const ngrokStatusPill = document.getElementById("ngrokStatusPill");
    const statusDot = document.getElementById("statusDot");
    const statusLabel = document.getElementById("statusLabel");
    const btnOpenSettings = document.getElementById("btnOpenSettings");
    const btnCloseSettings = document.getElementById("btnCloseSettings");
    const btnCancelSettings = document.getElementById("btnCancelSettings");
    const btnSaveSettings = document.getElementById("btnSaveSettings");
    const btnTestNgrok = document.getElementById("btnTestNgrok");
    const settingsModal = document.getElementById("settingsModal");
    const inputNgrokUrl = document.getElementById("inputNgrokUrl");
    const ngrokTestResult = document.getElementById("ngrokTestResult");

    checkNgrokStatus();

    async function checkNgrokStatus(testUrl = null) {
        statusLabel.textContent = "Checking Ngrok...";
        statusDot.className = "status-dot pulsing";
        try {
            const url = testUrl ? `/api/health/ngrok?url=${encodeURIComponent(testUrl)}` : "/api/health/ngrok";
            const resp = await fetch(url);
            const data = await resp.json();

            if (data.status === "online") {
                statusDot.className = "status-dot online";
                statusLabel.textContent = `Ngrok Online (${data.latency_ms}ms)`;
                if (ngrokTestResult) ngrokTestResult.innerHTML = `<span style="color:#10b981;">✓ Connected! Latency: ${data.latency_ms}ms</span>`;
            } else if (data.status === "warning") {
                statusDot.className = "status-dot warning";
                statusLabel.textContent = "Ngrok Warning";
                if (ngrokTestResult) ngrokTestResult.innerHTML = `<span style="color:#f59e0b;">⚠ ${data.message}</span>`;
            } else {
                statusDot.className = "status-dot offline";
                statusLabel.textContent = "Ngrok Offline (Local Fallback)";
                if (ngrokTestResult) ngrokTestResult.innerHTML = `<span style="color:#ef4444;">✗ Offline: ${data.message}</span>`;
            }
        } catch (err) {
            statusDot.className = "status-dot offline";
            statusLabel.textContent = "Ngrok Offline";
        }
    }

    btnOpenSettings.addEventListener("click", () => { settingsModal.style.display = "flex"; });
    btnCloseSettings.addEventListener("click", () => { settingsModal.style.display = "none"; });
    btnCancelSettings.addEventListener("click", () => { settingsModal.style.display = "none"; });
    ngrokStatusPill.addEventListener("click", () => { settingsModal.style.display = "flex"; });

    btnTestNgrok.addEventListener("click", () => {
        checkNgrokStatus(inputNgrokUrl.value.trim());
    });

    btnSaveSettings.addEventListener("click", async () => {
        const payload = {
            ngrok_url: inputNgrokUrl.value.trim(),
            ocr_qwen_model: document.getElementById("selectOCRQwen").value.trim(),
            ner_model: document.getElementById("selectNERModel").value.trim()
        };
        try {
            await fetch("/api/settings", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload)
            });
            settingsModal.style.display = "none";
            checkNgrokStatus();
            alert("Settings updated successfully!");
        } catch (e) {
            alert("Failed to save settings: " + e.message);
        }
    });

    // ----------------- DEMO / LIVE LOCK STATE MANAGER -----------------
    function setLiveLockState(unlocked) {
        isLiveUnlocked = unlocked;
        if (unlocked) {
            if (modeBadge) {
                modeBadge.className = "mode-badge live";
                if (modeBadgeText) modeBadgeText.textContent = "Live Mode Active";
                modeBadge.title = "Live pipeline inference and AI chat are unlocked.";
            }
            if (btnToggleLock) {
                btnToggleLock.className = "btn-unlock-toggle live-active";
                btnToggleLock.innerHTML = '<i class="fa-solid fa-lock-open" id="btnToggleLockIcon"></i> <span id="btnToggleLockText">Lock Safe Mode</span>';
                btnToggleLock.title = "Click to return to safe demo mode";
            }
            if (btnProcess) {
                btnProcess.classList.remove("btn-locked");
                if (selectedFile || selectedSampleName) {
                    btnProcess.disabled = false;
                }
                btnProcess.innerHTML = '<i class="fa-solid fa-play" id="btnProcessIcon"></i> <span id="btnProcessText">Run LangGraph Pipeline</span>';
                btnProcess.title = "Execute LangGraph state machine on selected prescription scan";
            }
            if (chatLockBanner) {
                chatLockBanner.style.display = "none";
            }
            if (chatInput) {
                chatInput.disabled = false;
                chatInput.placeholder = "Ask a clinical question...";
            }
            if (btnSendChat) {
                btnSendChat.disabled = false;
            }
            const dropSpan = dropzone.querySelector('.drop-text span');
            if (dropSpan && (selectedFile || selectedSampleName)) {
                dropSpan.textContent = "Ready to process — Click Run LangGraph Pipeline";
            }
        } else {
            if (modeBadge) {
                modeBadge.className = "mode-badge demo";
                if (modeBadgeText) modeBadgeText.textContent = "Safe Demo Mode";
                modeBadge.title = "Render Cloud Safe Mode: Viewing pre-computed run fe6126fa. Live pipeline & AI chat locked to preserve cloud resources.";
            }
            if (btnToggleLock) {
                btnToggleLock.className = "btn-unlock-toggle";
                btnToggleLock.innerHTML = '<i class="fa-solid fa-lock" id="btnToggleLockIcon"></i> <span id="btnToggleLockText">Unlock Live Mode</span>';
                btnToggleLock.title = "Click to unlock live pipeline execution and AI Chat";
            }
            if (btnProcess) {
                btnProcess.classList.add("btn-locked");
                btnProcess.disabled = true;
                btnProcess.innerHTML = '<i class="fa-solid fa-lock" id="btnProcessIcon"></i> <span id="btnProcessText">Pipeline Locked (Demo Mode)</span>';
                btnProcess.title = "Live pipeline execution is locked in Safe Demo Mode to prevent cloud server freezing. Click 'Unlock Live Mode' to enable.";
            }
            if (chatLockBanner) {
                chatLockBanner.style.display = "flex";
            }
            if (chatInput) {
                chatInput.disabled = true;
                chatInput.placeholder = "🔒 Live AI Chat locked in Demo Mode. Unlock to query...";
            }
            if (btnSendChat) {
                btnSendChat.disabled = true;
            }
            const dropSpan = dropzone.querySelector('.drop-text span');
            if (dropSpan && (selectedFile || selectedSampleName)) {
                dropSpan.textContent = "Sample loaded (Safe Demo Mode)";
            }
        }
    }

    // Unlock Modal Bindings
    if (btnToggleLock) {
        btnToggleLock.addEventListener("click", () => {
            if (isLiveUnlocked) {
                setLiveLockState(false);
            } else {
                if (unlockModal) unlockModal.style.display = "flex";
            }
        });
    }

    if (btnUnlockFromChat) {
        btnUnlockFromChat.addEventListener("click", () => {
            if (unlockModal) unlockModal.style.display = "flex";
        });
    }

    if (btnCloseUnlockModal) {
        btnCloseUnlockModal.addEventListener("click", () => {
            if (unlockModal) unlockModal.style.display = "none";
        });
    }

    if (btnCancelUnlock) {
        btnCancelUnlock.addEventListener("click", () => {
            if (unlockModal) unlockModal.style.display = "none";
        });
    }

    if (btnConfirmUnlock) {
        btnConfirmUnlock.addEventListener("click", () => {
            setLiveLockState(true);
            if (unlockModal) unlockModal.style.display = "none";
        });
    }

    // ----------------- LOAD SAMPLES -----------------
    loadSamples();

    function loadSamples() {
        fetch("/api/samples")
            .then(res => res.json())
            .then(samples => {
                samplesList.innerHTML = "";
                if (!samples || samples.length === 0) {
                    samplesList.innerHTML = "<span style='color:var(--text-muted); font-size:11px;'>No sample found</span>";
                    return;
                }
                samples.forEach((s, idx) => {
                    const chip = document.createElement("div");
                    chip.className = "sample-chip" + (idx === 0 ? " active" : "");
                    chip.textContent = s.name;
                    chip.addEventListener("click", () => selectSample(s.name, chip));
                    samplesList.appendChild(chip);
                });
            })
            .catch(() => {
                samplesList.innerHTML = "<span style='color:red; font-size:11px;'>Could not load samples</span>";
            });
    }

    function selectSample(sampleName, chipElem) {
        document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
        if (chipElem) chipElem.classList.add("active");
        selectedSampleName = sampleName;
        selectedFile = null;

        dropzone.querySelector('.drop-text strong').textContent = sampleName;
        dropzone.querySelector('.drop-text span').textContent = isLiveUnlocked
            ? "Sample selected — Click Run LangGraph Pipeline"
            : "Sample loaded (Safe Demo Mode)";

        if (isLiveUnlocked) {
            btnProcess.disabled = false;
        } else {
            btnProcess.disabled = true;
        }

        // If sample1.jpg is selected, load the full pre-computed historical run fe6126fa with all markings
        if (sampleName === "sample1.jpg") {
            loadPastRun("fe6126fa", false);
            return;
        }

        // Fetch instant Base64 preview via /api/preview for other samples
        const formData = new FormData();
        formData.append("sample_name", sampleName);

        fetch("/api/preview", { method: "POST", body: formData })
            .then(res => res.json())
            .then(data => {
                if (data.pages && data.pages.length > 0) {
                    currentImageDataUrl = data.pages[0].image_b64;
                    if (canvasPlaceholder) canvasPlaceholder.style.display = "none";
                    canvas.loadImage(currentImageDataUrl, [], [], [], []);
                }
            })
            .catch(err => {
                console.error("Preview load error:", err);
                currentImageDataUrl = `/api/samples/${sampleName}`;
                if (canvasPlaceholder) canvasPlaceholder.style.display = "none";
                canvas.loadImage(currentImageDataUrl, [], [], [], []);
            });
    }

    // ----------------- DROPZONE & FILE INPUT -----------------
    dropzone.addEventListener("click", () => {
        if (!isLiveUnlocked) {
            if (unlockModal) unlockModal.style.display = "flex";
            return;
        }
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (!isLiveUnlocked) {
            if (unlockModal) unlockModal.style.display = "flex";
            return;
        }
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    function handleFileUpload(file) {
        selectedFile = file;
        selectedSampleName = null;
        document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
        
        if (isLiveUnlocked) {
            btnProcess.disabled = false;
        }

        dropzone.querySelector('.drop-text strong').textContent = file.name;
        dropzone.querySelector('.drop-text span').textContent = `${(file.size / 1024).toFixed(1)} KB — Ready to process`;

        // Fetch instant Base64 preview via /api/preview
        const formData = new FormData();
        formData.append("file", file);

        fetch("/api/preview", { method: "POST", body: formData })
            .then(res => res.json())
            .then(data => {
                if (data.pages && data.pages.length > 0) {
                    currentImageDataUrl = data.pages[0].image_b64;
                    if (canvasPlaceholder) canvasPlaceholder.style.display = "none";
                    canvas.loadImage(currentImageDataUrl, [], [], [], []);
                }
            })
            .catch(err => {
                console.error("File preview error:", err);
                const reader = new FileReader();
                reader.onload = (evt) => {
                    currentImageDataUrl = evt.target.result;
                    if (canvasPlaceholder) canvasPlaceholder.style.display = "none";
                    canvas.loadImage(currentImageDataUrl, [], [], [], []);
                };
                reader.readAsDataURL(file);
            });
    }

    // ----------------- RUN LANGGRAPH PIPELINE -----------------
    btnProcess.addEventListener("click", async () => {
        if (!isLiveUnlocked) {
            if (unlockModal) unlockModal.style.display = "flex";
            return;
        }

        btnProcess.disabled = true;
        btnProcess.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Executing LangGraph Machine...`;
        updateStepper("stepCNN");

        // Clear and reset chat assistant to default greeting
        if (chatContainer) {
            chatContainer.innerHTML = "";
            appendChatMsg("assistant", "Hello! I am your clinical assistant. Ask about drug interactions, dosage warnings, or patient instructions based on this prescription.");
        }

        const formData = new FormData();
        if (selectedFile) {
            formData.append("file", selectedFile);
        } else if (selectedSampleName) {
            formData.append("sample_name", selectedSampleName);
        }

        try {
            const resp = await fetch("/api/pipeline/start", {
                method: "POST",
                body: formData
            });
            const result = await resp.json();

            if (result.thread_id) {
                currentThreadId = result.thread_id;
                renderGraphState(result.state);
            }
        } catch (err) {
            alert("Pipeline execution error: " + err.message);
        } finally {
            if (isLiveUnlocked) {
                btnProcess.disabled = false;
                btnProcess.innerHTML = `<i class="fa-solid fa-play"></i> Run LangGraph Pipeline`;
            } else {
                setLiveLockState(false);
            }
        }
    });

    function renderGraphState(state) {
        fullState = state;
        currentNERData = state.ner_data || {};
        const cnnPatches = state.patches || [];
        const allBoxes = state.all_boxes || [];
        const fusedNerBoxes = state.fused_ner_boxes || [];
        const wordDetections = state.word_detections || [];
        const ocrText = state.merged_ocr_text || "";

        updateStepper("stepHITL");

        // Render Canvas Overlays
        if (currentImageDataUrl) {
            canvas.loadImage(
                currentImageDataUrl,
                cnnPatches,
                allBoxes,
                fusedNerBoxes,
                wordDetections
            );
        }

        // Show HITL Banner & NER Form
        emptyNERState.style.display = "none";
        nerDetailsContainer.style.display = "block";

        if (state.doctor_approved) {
            hitlBanner.className = "hitl-alert-box approved-state";
            hitlBanner.innerHTML = `
                <div class="hitl-alert-text" style="color: #2ECC71;">
                    <i class="fa-solid fa-circle-check"></i> <strong>Prescription Approved & Finalized:</strong> The clinical records have been signed off by the doctor.
                </div>
            `;
            hitlBanner.style.display = "flex";
        } else {
            hitlBanner.className = "hitl-alert-box";
            hitlBanner.innerHTML = `
                <div class="hitl-alert-text">
                    <i class="fa-solid fa-triangle-exclamation"></i> <strong>Doctor Verification Required:</strong> Review extracted entities, edit values if needed, and finalize.
                </div>
                <button class="btn-primary btn-sm" id="btnApproveState">
                    <i class="fa-solid fa-check"></i> Approve & Finalize
                </button>
            `;
            hitlBanner.style.display = "flex";

            // Bind click listener back to the newly created button
            const btnApprove = document.getElementById("btnApproveState");
            if (btnApprove) {
                btnApprove.addEventListener("click", handleApproveState);
            }
        }

        // Populate NER Form
        document.getElementById("fieldDoctor").value = currentNERData.doctor_name || "";
        document.getElementById("fieldLicense").value = currentNERData.license_no || currentNERData.ptr_no || "";
        document.getElementById("fieldFacility").value = currentNERData.facility_name || "";
        document.getElementById("fieldDate").value = currentNERData.date || "";
        document.getElementById("fieldPatient").value = currentNERData.patient_name || "";
        document.getElementById("fieldAgeGender").value = [currentNERData.age, currentNERData.gender].filter(Boolean).join(" / ");

        // Populate Medications
        renderMedicationsTable(currentNERData.medications || []);

        // Populate OCR Transcription
        transcriptionCodeBlock.textContent = ocrText || "(No transcription output)";

        // Populate JSON code view
        updateJsonView();
    }

    if (selectJsonSection) {
        selectJsonSection.addEventListener("change", updateJsonView);
    }

    if (btnCopyJSON) {
        btnCopyJSON.addEventListener("click", () => {
            if (jsonCodeView) {
                navigator.clipboard.writeText(jsonCodeView.textContent);
                btnCopyJSON.textContent = "Copied!";
                setTimeout(() => {
                    btnCopyJSON.innerHTML = '<i class="fa-solid fa-copy"></i> Copy';
                }, 2000);
            }
        });
    }

    function updateJsonView() {
        if (!fullState) return;
        const section = selectJsonSection ? selectJsonSection.value : "full";
        let displayData = fullState;

        if (section === "cnn") {
            displayData = fullState.patches || [];
        } else if (section === "tesseract") {
            displayData = fullState.tesseract_res || {};
        } else if (section === "easyocr") {
            displayData = fullState.easyocr_res || {};
        } else if (section === "vlm_ocr") {
            displayData = fullState.vlm_res || {};
        } else if (section === "ner_data") {
            displayData = fullState.ner_data || {};
        } else if (section === "word_detections") {
            displayData = fullState.word_detections || [];
        } else if (section === "fused_ner_boxes") {
            displayData = fullState.fused_ner_boxes || [];
        }

        if (jsonCodeView) {
            jsonCodeView.textContent = JSON.stringify(displayData, null, 2);
        }
    }

    function renderMedicationsTable(meds) {
        const container = document.getElementById("medicationsTableContainer");
        if (!meds || meds.length === 0) {
            container.innerHTML = "<p style='color:#8b949e; font-size:12px; padding:10px 0;'>No medications detected in prescription.</p>";
            return;
        }

        let html = `
            <table class="meds-table">
                <thead>
                    <tr>
                        <th>Drug Name</th>
                        <th>Dosage</th>
                        <th>Frequency</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody>
        `;

        meds.forEach((m) => {
            html += `
                <tr>
                    <td><span class="med-pill">${m.drug_name || 'N/A'}</span></td>
                    <td><strong>${m.dosage || 'N/A'}</strong></td>
                    <td>${m.frequency || 'N/A'}</td>
                    <td>${m.duration || 'N/A'}</td>
                </tr>
            `;
        });

        html += `</tbody></table>`;
        container.innerHTML = html;
    }

    // Approve & Finalize
    async function handleApproveState() {
        if (!currentThreadId) return;

        const btnApprove = document.getElementById("btnApproveState");
        if (btnApprove) {
            btnApprove.disabled = true;
            btnApprove.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Finalizing PDF...`;
        }

        const edits = {
            doctor_name: document.getElementById("fieldDoctor").value,
            license_no: document.getElementById("fieldLicense").value,
            facility_name: document.getElementById("fieldFacility").value,
            date: document.getElementById("fieldDate").value,
            patient_name: document.getElementById("fieldPatient").value
        };

        try {
            const resp = await fetch("/api/pipeline/approve", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    thread_id: currentThreadId,
                    doctor_edits: edits
                })
            });
            const data = await resp.json();
            
            // Re-render state to update the approval banner to green
            renderGraphState(data.state);
            
            alert("Prescription verified and finalized by doctor!");
        } catch (err) {
            alert("Approval error: " + err.message);
            if (btnApprove) {
                btnApprove.disabled = false;
                btnApprove.innerHTML = `<i class="fa-solid fa-check"></i> Approve & Finalize`;
            }
        }
    }

    if (btnApproveState) {
        btnApproveState.addEventListener("click", handleApproveState);
    }

    // Chat Assistant
    if (selectChatModel) {
        selectChatModel.addEventListener("change", () => {
            if (selectChatModel.value === "custom") {
                inputChatModelCustom.style.display = "block";
            } else {
                inputChatModelCustom.style.display = "none";
            }
        });
    }

    btnSendChat.addEventListener("click", sendChatMessage);
    chatInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendChatMessage();
    });

    document.querySelectorAll(".chip-btn").forEach(chip => {
        chip.addEventListener("click", () => {
            if (!isLiveUnlocked) {
                if (unlockModal) unlockModal.style.display = "flex";
                return;
            }
            chatInput.value = chip.getAttribute("data-q");
            sendChatMessage();
        });
    });

    async function sendChatMessage() {
        if (!isLiveUnlocked) {
            if (unlockModal) unlockModal.style.display = "flex";
            return;
        }

        const text = chatInput.value.trim();
        if (!text) return;

        let activeModel = selectChatModel ? selectChatModel.value : "qwen2.5vl:7b";
        if (activeModel === "custom" && inputChatModelCustom) {
            activeModel = inputChatModelCustom.value.trim() || "qwen2.5vl:7b";
        }

        appendChatMsg("user", text);
        chatInput.value = "";

        const loadingMsg = appendChatMsg("assistant", "Thinking...");

        try {
            const resp = await fetch("/api/chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    question: text,
                    thread_id: currentThreadId,
                    ner_data: currentNERData,
                    model: activeModel
                })
            });
            const data = await resp.json();
            const ans = data.answer || "No response received.";
            if (typeof marked !== "undefined") {
                loadingMsg.innerHTML = marked.parse(ans);
            } else {
                loadingMsg.textContent = ans;
            }
        } catch (err) {
            loadingMsg.textContent = "Error: " + err.message;
        }
    }

    function appendChatMsg(role, text) {
        const msg = document.createElement("div");
        msg.className = `chat-msg ${role}`;
        if (role === "assistant" && typeof marked !== "undefined") {
            msg.innerHTML = marked.parse(text);
        } else {
            msg.textContent = text;
        }
        chatContainer.appendChild(msg);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return msg;
    }

    function updateStepper(activeId) {
        const nodes = ["stepUpload", "stepCNN", "stepOCR", "stepNER", "stepGround", "stepHITL"];
        let passed = true;
        nodes.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                if (id === activeId) {
                    el.className = "step-node active";
                    passed = false;
                } else if (passed) {
                    el.className = "step-node completed";
                } else {
                    el.className = "step-node";
                }
            }
        });
    }

    // History Modal Interactions
    if (btnOpenHistory) {
        btnOpenHistory.addEventListener("click", () => {
            historyModal.style.display = "flex";
            loadHistoryRuns();
        });
    }

    [btnCloseHistory, btnCloseHistoryFooter].forEach(btn => {
        if (btn) {
            btn.addEventListener("click", () => {
                historyModal.style.display = "none";
            });
        }
    });

    async function loadHistoryRuns() {
        if (!historyLoading || !historyEmpty || !historyTable || !historyTableBody) return;
        historyLoading.style.display = "block";
        historyEmpty.style.display = "none";
        historyTable.style.display = "none";
        historyTableBody.innerHTML = "";

        try {
            const resp = await fetch("/api/history");
            const data = await resp.json();
            const runs = data.runs || [];
            
            if (historyDbBackend) {
                historyDbBackend.textContent = `Backend: ${data.backend || 'sqlite'}`;
            }

            historyLoading.style.display = "none";

            if (runs.length === 0) {
                historyEmpty.style.display = "block";
                return;
            }

            historyTable.style.display = "table";
            runs.forEach(run => {
                const tr = document.createElement("tr");
                tr.style.borderBottom = "1px solid var(--border-color)";
                
                // Extract parsed patient/doctor from nested full state if it was saved that way, or default
                const stateObj = run.ner_data || {};
                const nerPart = stateObj.ner_data || {};
                
                const patientName = nerPart.patient_name || "(Unknown)";
                const doctorName = nerPart.doctor_name || "(Unknown)";
                
                const dateStr = run.created_at ? new Date(run.created_at).toLocaleString() : "N/A";
                const badgeClass = run.status === "approved" ? "med-pill" : "status-pill";
                const badgeStyle = run.status === "approved" ? "" : "background:rgba(241,196,15,0.15); color:#F1C40F; padding:2px 6px; border-radius:4px; font-weight:600;";

                tr.innerHTML = `
                    <td style="padding: 10px 8px;"><code>${run.thread_id}</code></td>
                    <td style="padding: 10px 8px;"><strong>${patientName}</strong></td>
                    <td style="padding: 10px 8px;">${doctorName}</td>
                    <td style="padding: 10px 8px; color: var(--text-secondary); font-size: 11px;">${dateStr}</td>
                    <td style="padding: 10px 8px;"><span class="${badgeClass}" style="${badgeStyle}">${run.status.toUpperCase()}</span></td>
                    <td style="padding: 10px 8px; text-align: center; display: flex; gap: 6px; justify-content: center; align-items: center;">
                        <button class="btn-primary btn-sm btn-load-run" data-thread="${run.thread_id}" style="padding: 3px 8px; font-size: 11px;"><i class="fa-solid fa-eye"></i> Load</button>
                        <button class="btn-secondary btn-sm btn-delete-run" data-thread="${run.thread_id}" style="padding: 3px 8px; font-size: 11px; color: #E74C3C; border-color: rgba(231,76,60,0.2);"><i class="fa-solid fa-trash"></i></button>
                    </td>
                `;

                // Bind Load
                tr.querySelector(".btn-load-run").addEventListener("click", async () => {
                    historyModal.style.display = "none";
                    await loadPastRun(run.thread_id, true);
                });

                // Bind Delete
                tr.querySelector(".btn-delete-run").addEventListener("click", async (e) => {
                    e.stopPropagation();
                    if (confirm(`Are you sure you want to delete run ${run.thread_id}?`)) {
                        try {
                            const delResp = await fetch(`/api/history/${run.thread_id}`, { method: "DELETE" });
                            if (delResp.ok) {
                                tr.remove();
                                if (historyTableBody.children.length === 0) {
                                    historyTable.style.display = "none";
                                    historyEmpty.style.display = "block";
                                }
                            }
                        } catch (err) {
                            alert("Delete error: " + err.message);
                        }
                    }
                });

                historyTableBody.appendChild(tr);
            });
        } catch (err) {
            historyLoading.style.display = "none";
            historyEmpty.style.display = "block";
            historyEmpty.innerHTML = `<span style="color:#e74c3c;"><i class="fa-solid fa-triangle-exclamation"></i> Error loading history: ${err.message}</span>`;
        }
    }

    async function loadPastRun(threadId, showAlert = false) {
        try {
            const resp = await fetch(`/api/history/${threadId}`);
            if (!resp.ok) throw new Error("Failed to fetch historical run details");
            const data = await resp.json();
            const run = data.run;
            if (!run) throw new Error("No run log returned");

            currentThreadId = run.thread_id;
            selectedSampleName = run.image_name || "sample1.jpg";
            selectedFile = null;

            // Highlight corresponding sample chip if present
            document.querySelectorAll(".sample-chip").forEach(c => {
                c.classList.toggle("active", c.textContent === run.image_name || c.textContent === "sample1.jpg");
            });

            // Update dropzone display
            const dropName = dropzone.querySelector('.drop-text strong');
            const dropSub = dropzone.querySelector('.drop-text span');
            if (dropName) dropName.textContent = run.image_name || "sample1.jpg";
            if (dropSub) {
                dropSub.textContent = isLiveUnlocked
                    ? "Sample selected — Click Run LangGraph Pipeline"
                    : `Historical run ${run.thread_id} loaded (Safe Demo Mode)`;
            }
            
            // The saved ner_data column contains the ENTIRE state object!
            const state = run.ner_data || {};
            
            // Make sure the state knows it was approved so the banner renders green
            state.doctor_approved = (run.status === "approved");

            // Set current image view using securely proxied local path
            currentImageDataUrl = `/api/image/view?path=${encodeURIComponent(run.image_path)}`;
            if (canvasPlaceholder) canvasPlaceholder.style.display = "none";
            
            // Render the graph state to Canvas overlay, NER details, and step-by-step JSON
            renderGraphState(state);
            
            // Clear and reload chat history for this specific run
            if (chatContainer) {
                chatContainer.innerHTML = "";
                const historyMsgs = run.chat_history || [];
                if (historyMsgs.length === 0) {
                    appendChatMsg("assistant", "Hello! I am your clinical assistant. Ask about drug interactions, dosage warnings, or patient instructions based on this prescription.");
                } else {
                    historyMsgs.forEach(msg => {
                        appendChatMsg(msg.role, msg.content);
                    });
                }
            }

            if (showAlert) {
                alert(`Historical run ${threadId} loaded successfully onto workspace.`);
            }
        } catch (err) {
            console.error("Error loading historical run:", err);
            if (showAlert) {
                alert("Error loading historical run: " + err.message);
            }
        }
    }

    // Initialize Safe Demo Mode (locked by default)
    setLiveLockState(false);

    // Auto-load default historical run fe6126fa on homepage load
    loadPastRun("fe6126fa", false);
});
