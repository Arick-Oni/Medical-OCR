// Distinct OCR Bounding Box Line Colors
const OCR_LINE_COLORS = [
    '#E74C3C', '#9B59B6', '#E67E22', '#1ABC9C', '#6C5CE7',
    '#E91E63', '#009688', '#FF5722', '#A55EEA', '#00BCD4',
    '#8854D0', '#D35400', '#C0392B', '#8E44AD', '#16A085',
    '#D63031', '#6C5CE7', '#E17055', '#0984E3', '#A55EEA'
];

class PrescriptionCanvas {
    constructor(canvasId, tooltipId, viewportId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.tooltip = document.getElementById(tooltipId);
        this.viewport = document.getElementById(viewportId);

        this.img = new Image();
        this.isLoaded = false;

        // Viewport Transform State
        this.scale = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.isDragging = false;
        this.startX = 0;
        this.startY = 0;

        // Data Overlays
        this.cnnPatches = [];
        this.ocrDetections = [];
        this.fusedNerBoxes = [];
        this.wordDetections = [];

        // Layer Filter Flags
        this.layers = {
            ocr: true,
            ner: true,
            word_boxes: false,
            handwritten: true,
            printed: true,
            mixed: true
        };

        this.initEvents();
    }

    initEvents() {
        // Zoom on Wheel
        this.viewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
            const newScale = Math.min(Math.max(0.2, this.scale * zoomFactor), 5.0);

            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            this.panX -= (mouseX - this.panX) * (newScale / this.scale - 1);
            this.panY -= (mouseY - this.panY) * (newScale / this.scale - 1);
            this.scale = newScale;
            this.render();
        }, { passive: false });

        // Pan on Drag
        this.viewport.addEventListener('mousedown', (e) => {
            if (e.button === 0) {
                this.isDragging = true;
                this.startX = e.clientX - this.panX;
                this.startY = e.clientY - this.panY;
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                this.panX = e.clientX - this.startX;
                this.panY = e.clientY - this.startY;
                this.render();
            } else {
                this.handleHover(e);
            }
        });

        window.addEventListener('mouseup', () => {
            this.isDragging = false;
        });

        window.addEventListener('resize', () => {
            if (this.isLoaded) this.fitToScreen();
        });
    }

    loadImage(dataUrl, cnnPatches = [], ocrDetections = [], fusedNerBoxes = [], wordDetections = []) {
        this.cnnPatches = cnnPatches || [];
        this.ocrDetections = ocrDetections || [];
        this.fusedNerBoxes = fusedNerBoxes || [];
        this.wordDetections = wordDetections || [];

        this.img = new Image();
        if (dataUrl && !dataUrl.startsWith("data:")) {
            this.img.crossOrigin = "anonymous";
        }
        this.img.onload = () => {
            this.isLoaded = true;
            this.fitToScreen();
        };
        this.img.onerror = (err) => {
            console.error("Canvas Image Load Error:", err, "URL:", dataUrl);
        };
        this.img.src = dataUrl;
    }

    fitToScreen() {
        if (!this.isLoaded) return;
        const vWidth = this.viewport.clientWidth || 700;
        const vHeight = this.viewport.clientHeight || 550;

        const natW = this.img.naturalWidth || this.img.width || 800;
        const natH = this.img.naturalHeight || this.img.height || 1000;

        const scaleX = (vWidth - 32) / natW;
        const scaleY = (vHeight - 32) / natH;
        this.scale = Math.min(scaleX, scaleY, 1.0);

        this.panX = (vWidth - natW * this.scale) / 2;
        this.panY = (vHeight - natH * this.scale) / 2;

        this.canvas.width = vWidth;
        this.canvas.height = vHeight;
        this.render();
    }

    zoomIn() {
        this.scale = Math.min(5.0, this.scale * 1.25);
        this.render();
    }

    zoomOut() {
        this.scale = Math.max(0.2, this.scale / 1.25);
        this.render();
    }

    setLayer(layerName, isEnabled) {
        if (this.layers.hasOwnProperty(layerName)) {
            this.layers[layerName] = isEnabled;
            this.render();
        }
    }

    render() {
        if (!this.isLoaded) return;

        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.scale, this.scale);

        // 1. Draw Base Image
        ctx.drawImage(this.img, 0, 0);

        // 2. Draw MiniCNN Classification Patches
        for (const p of this.cnnPatches) {
            const cls = p.class;
            if (cls === 'Handwritten' && !this.layers.handwritten) continue;
            if (cls === 'Printed' && !this.layers.printed) continue;
            if (cls === 'Mixed' && !this.layers.mixed) continue;

            const [x, y, w, h] = p.bbox;
            const colorHex = p.color ? p.color.hex : '#2ECC71';

            ctx.fillStyle = (p.color && p.color.bg) ? p.color.bg : 'rgba(46, 204, 113, 0.18)';
            ctx.fillRect(x, y, w, h);

            ctx.strokeStyle = colorHex;
            ctx.lineWidth = 1.5 / this.scale;
            ctx.setLineDash([4 / this.scale, 2 / this.scale]);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);
        }

        // 3. Draw OCR Text Grounding Boxes
        if (this.layers.ocr && this.ocrDetections) {
            this.ocrDetections.forEach((o, idx) => {
                const box = o.bbox || o.box;
                if (!box) return;
                const [x, y, w, h] = box;

                const colorHex = OCR_LINE_COLORS[idx % OCR_LINE_COLORS.length];

                ctx.strokeStyle = colorHex;
                ctx.lineWidth = 2 / this.scale;
                ctx.strokeRect(x, y, w, h);

                const labelText = (o.text || o.label || o.tag || '').trim();
                if (labelText) {
                    const fontSize = Math.max(9, 11 / this.scale);
                    ctx.font = `bold ${fontSize}px sans-serif`;
                    const paddingX = 4 / this.scale;
                    const paddingY = 2 / this.scale;
                    const textMetrics = ctx.measureText(labelText);
                    const badgeW = textMetrics.width + (paddingX * 2);
                    const badgeH = fontSize + (paddingY * 2);

                    const badgeX = x;
                    const badgeY = Math.max(0, y - badgeH - 2);

                    ctx.fillStyle = colorHex;
                    if (ctx.roundRect) {
                        ctx.beginPath();
                        ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 3 / this.scale);
                        ctx.fill();
                    } else {
                        ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
                    }

                    ctx.fillStyle = '#FFFFFF';
                    ctx.textBaseline = 'top';
                    ctx.fillText(labelText, badgeX + paddingX, badgeY + paddingY);
                }
            });
        }

        // 4. Draw Fused NER Entity Underlines & Badges
        if (this.layers.ner && this.fusedNerBoxes) {
            this.fusedNerBoxes.forEach((fn) => {
                const bbox = fn.bbox;
                if (!bbox) return;
                const [x1, y1, x2, y2] = bbox;
                const w = Math.max(1, x2 - x1);
                const h = Math.max(1, y2 - y1);
                const colorHex = fn.color || '#8b5cf6';

                ctx.fillStyle = fn.color ? `${fn.color}26` : 'rgba(139, 92, 246, 0.15)';
                ctx.fillRect(x1, y1, w, h);

                ctx.strokeStyle = colorHex;
                ctx.lineWidth = 3 / this.scale;
                ctx.beginPath();
                ctx.moveTo(x1, y2 + (3 / this.scale));
                ctx.lineTo(x2, y2 + (3 / this.scale));
                ctx.stroke();

                const labelText = (fn.label || fn.entity_text || '').trim();
                if (labelText) {
                    const fontSize = Math.max(10, 12 / this.scale);
                    ctx.font = `bold ${fontSize}px sans-serif`;
                    const paddingX = 5 / this.scale;
                    const paddingY = 3 / this.scale;
                    const textMetrics = ctx.measureText(labelText);
                    const badgeW = textMetrics.width + (paddingX * 2);
                    const badgeH = fontSize + (paddingY * 2);

                    const badgeX = x1;
                    const badgeY = y2 + (8 / this.scale);

                    ctx.fillStyle = colorHex;
                    if (ctx.roundRect) {
                        ctx.beginPath();
                        ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 4 / this.scale);
                        ctx.fill();
                    } else {
                        ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
                    }

                    ctx.fillStyle = '#FFFFFF';
                    ctx.textBaseline = 'top';
                    ctx.fillText(labelText, badgeX + paddingX, badgeY + paddingY);
                }
            });
        }

        // 5. Draw Raw Qwen Word Bounding Boxes
        if (this.layers.word_boxes && this.wordDetections) {
            this.wordDetections.forEach((w) => {
                const bbox = w.bbox;
                if (!bbox) return;
                const [x1, y1, x2, y2] = bbox;
                const width = Math.max(1, x2 - x1);
                const height = Math.max(1, y2 - y1);
                const colorHex = '#00BCD4';

                ctx.strokeStyle = colorHex;
                ctx.lineWidth = 1.5 / this.scale;
                ctx.strokeRect(x1, y1, width, height);

                const labelText = (w.word || w.text || '').trim();
                if (labelText) {
                    const fontSize = Math.max(8, 10 / this.scale);
                    ctx.font = `${fontSize}px sans-serif`;
                    ctx.fillStyle = colorHex;
                    ctx.fillText(labelText, x1, Math.max(0, y1 - 2));
                }
            });
        }

        ctx.restore();
    }

    handleHover(e) {
        if (!this.isLoaded || this.isDragging) {
            this.tooltip.style.display = 'none';
            return;
        }

        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const imgX = (mouseX - this.panX) / this.scale;
        const imgY = (mouseY - this.panY) / this.scale;

        let hit = null;

        // 1. Check Fused NER Entity Hit
        if (this.layers.ner && this.fusedNerBoxes) {
            for (const fn of this.fusedNerBoxes) {
                const bbox = fn.bbox;
                if (!bbox) continue;
                const [x1, y1, x2, y2] = bbox;
                if (imgX >= x1 && imgX <= x2 && imgY >= y1 && imgY <= y2 + 20) {
                    hit = {
                        type: 'NER Entity',
                        title: `[${fn.entity_key.replace('_', ' ').toUpperCase()}]`,
                        desc: fn.label,
                        color: fn.color || '#8b5cf6'
                    };
                    break;
                }
            }
        }

        // 2. Check OCR Detections hit
        if (!hit && this.layers.ocr && this.ocrDetections) {
            for (let i = 0; i < this.ocrDetections.length; i++) {
                const o = this.ocrDetections[i];
                const box = o.bbox || o.box;
                if (!box) continue;
                const [x, y, w, h] = box;
                if (imgX >= x && imgX <= x + w && imgY >= y && imgY <= y + h) {
                    const colorHex = OCR_LINE_COLORS[i % OCR_LINE_COLORS.length];
                    hit = {
                        type: `OCR Line ${i + 1}`,
                        title: o.tag ? `[${o.tag.toUpperCase()}]` : `[LINE ${i + 1}]`,
                        desc: o.text || o.label || '(No transcription)',
                        color: colorHex
                    };
                    break;
                }
            }
        }

        // 3. Check CNN Patches hit
        if (!hit && this.cnnPatches) {
            for (const p of this.cnnPatches) {
                const [x, y, w, h] = p.bbox;
                if (imgX >= x && imgX <= x + w && imgY >= y && imgY <= y + h) {
                    hit = {
                        type: 'MiniCNN Patch',
                        title: `${p.class}`,
                        desc: `Confidence: ${(p.confidence * 100).toFixed(1)}%`,
                        color: p.color ? p.color.hex : '#3498DB'
                    };
                    break;
                }
            }
        }

        if (hit) {
            this.tooltip.style.display = 'block';
            this.tooltip.style.left = `${e.clientX - this.viewport.getBoundingClientRect().left + 15}px`;
            this.tooltip.style.top = `${e.clientY - this.viewport.getBoundingClientRect().top + 15}px`;
            this.tooltip.innerHTML = `
                <div style="font-weight: 700; color: ${hit.color}; margin-bottom: 2px;">${hit.type} - ${hit.title}</div>
                <div style="color: #F0F6FC;">${hit.desc}</div>
            `;
        } else {
            this.tooltip.style.display = 'none';
        }
    }
}
