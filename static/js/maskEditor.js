const MAX_WORKING_DIM = 1200;
const MAX_HISTORY = 30;

export class MaskEditor {
  constructor() {
    this.modal = document.getElementById("maskEditorModal");
    this.wrap = document.getElementById("editorCanvasWrap");
    this.baseCanvas = document.getElementById("editorBaseCanvas");
    this.maskCanvas = document.getElementById("editorMaskCanvas");
    this.baseCtx = this.baseCanvas.getContext("2d");
    this.maskCtx = this.maskCanvas.getContext("2d", { willReadFrequently: true });

    this.tool = "brush";
    this.brushSize = 30;
    this.scale = 1;
    this.pan = { x: 0, y: 0 };
    this.history = [];
    this.historyIndex = -1;
    this.drawing = false;
    this.panning = false;
    this.spaceHeld = false;

    this._bindToolbar();
    this._bindCanvas();
  }

  _bindToolbar() {
    document.getElementById("toolBrush").addEventListener("click", () => this._setTool("brush"));
    document.getElementById("toolEraser").addEventListener("click", () => this._setTool("eraser"));
    document.getElementById("brushSize").addEventListener("input", (e) => {
      this.brushSize = Number(e.target.value);
    });
    document.getElementById("toolZoomIn").addEventListener("click", () => this._zoom(1.25));
    document.getElementById("toolZoomOut").addEventListener("click", () => this._zoom(0.8));
    document.getElementById("toolUndo").addEventListener("click", () => this._undo());
    document.getElementById("toolRedo").addEventListener("click", () => this._redo());
    document.getElementById("toolReset").addEventListener("click", () => this._reset());
    document.getElementById("editorCancelBtn").addEventListener("click", () => this._close(false));
    document.getElementById("editorApplyBtn").addEventListener("click", () => this._close(true));

    window.addEventListener("keydown", (e) => {
      if (this.modal.classList.contains("hidden")) return;
      if (e.code === "Space") { this.spaceHeld = true; this.wrap.style.cursor = "grab"; }
      if ((e.ctrlKey || e.metaKey) && e.key === "z") { e.preventDefault(); this._undo(); }
      if ((e.ctrlKey || e.metaKey) && (e.key === "y" || (e.shiftKey && e.key === "Z"))) { e.preventDefault(); this._redo(); }
    });
    window.addEventListener("keyup", (e) => {
      if (e.code === "Space") { this.spaceHeld = false; this.wrap.style.cursor = "crosshair"; }
    });
  }

  _setTool(tool) {
    this.tool = tool;
    document.getElementById("toolBrush").classList.toggle("active", tool === "brush");
    document.getElementById("toolEraser").classList.toggle("active", tool === "eraser");
  }

  _zoom(factor) {
    this.scale = Math.min(4, Math.max(1, this.scale * factor));
    this._applyTransform();
  }

  _applyTransform() {
    const t = `scale(${this.scale}) translate(${this.pan.x}px, ${this.pan.y}px)`;
    this.baseCanvas.style.transform = t;
    this.maskCanvas.style.transform = t;
  }

  _bindCanvas() {
    const getPos = (e) => {
      const rect = this.maskCanvas.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * this.maskCanvas.width;
      const y = ((e.clientY - rect.top) / rect.height) * this.maskCanvas.height;
      return { x, y };
    };

    const pointerDown = (e) => {
      if (this.spaceHeld || e.button === 1) {
        this.panning = true;
        this._panStart = { x: e.clientX, y: e.clientY, origin: { ...this.pan } };
        return;
      }
      this.drawing = true;
      this._lastPos = getPos(e);
      this._paintDot(this._lastPos);
    };

    const pointerMove = (e) => {
      if (this.panning) {
        const dx = (e.clientX - this._panStart.x) / this.scale;
        const dy = (e.clientY - this._panStart.y) / this.scale;
        this.pan = { x: this._panStart.origin.x + dx, y: this._panStart.origin.y + dy };
        this._applyTransform();
        return;
      }
      if (!this.drawing) return;
      const pos = getPos(e);
      this._paintLine(this._lastPos, pos);
      this._lastPos = pos;
    };

    const pointerUp = () => {
      if (this.drawing) this._commitHistory();
      this.drawing = false;
      this.panning = false;
    };

    this.wrap.addEventListener("pointerdown", pointerDown);
    window.addEventListener("pointermove", pointerMove);
    window.addEventListener("pointerup", pointerUp);
    this.wrap.addEventListener("wheel", (e) => {
      e.preventDefault();
      this._zoom(e.deltaY < 0 ? 1.1 : 0.9);
    }, { passive: false });
  }

  _paintDot(pos) {
    this.maskCtx.globalCompositeOperation = this.tool === "eraser" ? "destination-out" : "source-over";
    this.maskCtx.fillStyle = "rgba(0, 87, 255, 1)";
    this.maskCtx.beginPath();
    this.maskCtx.arc(pos.x, pos.y, this.brushSize / 2, 0, Math.PI * 2);
    this.maskCtx.fill();
  }

  _paintLine(from, to) {
    this.maskCtx.globalCompositeOperation = this.tool === "eraser" ? "destination-out" : "source-over";
    this.maskCtx.strokeStyle = "rgba(0, 87, 255, 1)";
    this.maskCtx.lineWidth = this.brushSize;
    this.maskCtx.lineCap = "round";
    this.maskCtx.lineJoin = "round";
    this.maskCtx.beginPath();
    this.maskCtx.moveTo(from.x, from.y);
    this.maskCtx.lineTo(to.x, to.y);
    this.maskCtx.stroke();
    this._paintDot(to);
  }

  _commitHistory() {
    const snapshot = this.maskCtx.getImageData(0, 0, this.maskCanvas.width, this.maskCanvas.height);
    this.history = this.history.slice(0, this.historyIndex + 1);
    this.history.push(snapshot);
    if (this.history.length > MAX_HISTORY) this.history.shift();
    this.historyIndex = this.history.length - 1;
  }

  _undo() {
    if (this.historyIndex <= 0) return;
    this.historyIndex -= 1;
    this.maskCtx.putImageData(this.history[this.historyIndex], 0, 0);
  }

  _redo() {
    if (this.historyIndex >= this.history.length - 1) return;
    this.historyIndex += 1;
    this.maskCtx.putImageData(this.history[this.historyIndex], 0, 0);
  }

  _reset() {
    this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
    this._commitHistory();
  }

  /**
   * Open the editor. baseImage: HTMLImageElement (natural size = original campaign image).
   * initialMaskDataUrl: optional data URL of a grayscale mask (white=person) at any resolution.
   * Returns a Promise resolving to a Blob (black/white PNG mask, image natural resolution) or null if cancelled.
   */
  open(baseImage, initialMaskDataUrl) {
    return new Promise((resolve) => {
      this._resolve = resolve;

      const naturalW = baseImage.naturalWidth;
      const naturalH = baseImage.naturalHeight;
      const scaleDown = Math.min(1, MAX_WORKING_DIM / Math.max(naturalW, naturalH));
      const workW = Math.round(naturalW * scaleDown);
      const workH = Math.round(naturalH * scaleDown);

      [this.baseCanvas, this.maskCanvas].forEach((c) => {
        c.width = workW;
        c.height = workH;
      });

      const wrapRect = this.wrap.getBoundingClientRect();
      const fit = Math.min(wrapRect.width / workW, wrapRect.height / workH);
      this.baseCanvas.style.width = this.maskCanvas.style.width = `${workW * fit}px`;
      this.baseCanvas.style.height = this.maskCanvas.style.height = `${workH * fit}px`;
      this.baseCanvas.style.left = this.maskCanvas.style.left = `${(wrapRect.width - workW * fit) / 2}px`;
      this.baseCanvas.style.top = this.maskCanvas.style.top = `${(wrapRect.height - workH * fit) / 2}px`;

      this.baseCtx.clearRect(0, 0, workW, workH);
      this.baseCtx.drawImage(baseImage, 0, 0, workW, workH);

      this.scale = 1;
      this.pan = { x: 0, y: 0 };
      this._applyTransform();
      this.history = [];
      this.historyIndex = -1;

      const finishSetup = () => {
        this._commitHistory();
        this.modal.classList.remove("hidden");
      };

      this.maskCtx.clearRect(0, 0, workW, workH);
      if (initialMaskDataUrl) {
        const maskImg = new Image();
        maskImg.onload = () => {
          // Draw as a solid-blue overlay wherever the source mask is bright.
          const off = document.createElement("canvas");
          off.width = workW; off.height = workH;
          const offCtx = off.getContext("2d");
          offCtx.drawImage(maskImg, 0, 0, workW, workH);
          const data = offCtx.getImageData(0, 0, workW, workH);
          const out = this.maskCtx.createImageData(workW, workH);
          for (let i = 0; i < data.data.length; i += 4) {
            const v = data.data[i];
            out.data[i] = 0; out.data[i + 1] = 87; out.data[i + 2] = 255;
            out.data[i + 3] = v > 100 ? 255 : 0;
          }
          this.maskCtx.putImageData(out, 0, 0);
          finishSetup();
        };
        maskImg.onerror = finishSetup;
        maskImg.src = initialMaskDataUrl;
      } else {
        finishSetup();
      }
    });
  }

  _close(apply) {
    this.modal.classList.add("hidden");
    if (!apply) {
      this._resolve(null);
      return;
    }

    // Flatten the colored overlay into a pure white-on-black mask.
    const src = this.maskCtx.getImageData(0, 0, this.maskCanvas.width, this.maskCanvas.height);
    const flat = document.createElement("canvas");
    flat.width = this.maskCanvas.width;
    flat.height = this.maskCanvas.height;
    const flatCtx = flat.getContext("2d");
    const outData = flatCtx.createImageData(flat.width, flat.height);
    for (let i = 0; i < src.data.length; i += 4) {
      const on = src.data[i + 3] > 10;
      outData.data[i] = on ? 255 : 0;
      outData.data[i + 1] = on ? 255 : 0;
      outData.data[i + 2] = on ? 255 : 0;
      outData.data[i + 3] = 255;
    }
    flatCtx.putImageData(outData, 0, 0);

    flat.toBlob((blob) => {
      this._resolve({ blob, previewDataUrl: flat.toDataURL("image/png") });
    }, "image/png");
  }
}
