import { createCanvas } from "/file=javascript/layerforge/js/utils/CommonUtils.js?v=patch26";

export class LayerForgeMaskEditor {
    constructor() {
        this.overlay = null;
        this.container = null;
        this.bgCanvas = null;
        this.maskCanvas = null;
        this.cursorCanvas = null;
        this.ctx = null;
        this.maskCtx = null;
        this.cursorCtx = null;
        
        this.scale = 1;
        this.panX = 0;
        this.panY = 0;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        
        this.isDrawing = false;
        this.brushSize = 20;
        this.brushHardness = 1.0;
        this.brushOpacity = 1.0;
        this.isEraser = false;
        this.resolvePromise = null;
        this.lastAnchorPos = null;
        
        this.history = [];
        this.historyIndex = -1;
        
        this.injectCSS();
    }

    injectCSS() {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.type = 'text/css';
        link.href = new URL('./css/layerforge_mask_editor.css?v=patch1', import.meta.url).href;
        document.head.appendChild(link);
    }

    createUI() {
        if (this.overlay) return;

        this.overlay = document.createElement('div');
        this.overlay.className = 'layerforge-mask-editor-overlay';
        
        // Toolbar
        const toolbar = document.createElement('div');
        toolbar.className = 'layerforge-mask-editor-toolbar';
        
        const tools = document.createElement('div');
        tools.className = 'layerforge-mask-editor-tools';
        
        // Brush Button
        this.brushBtn = document.createElement('button');
        this.brushBtn.className = 'layerforge-mask-editor-button active';
        this.brushBtn.textContent = 'Brush';
        this.brushBtn.onclick = () => this.setTool('brush');
        
        // Eraser Button
        this.eraserBtn = document.createElement('button');
        this.eraserBtn.className = 'layerforge-mask-editor-button';
        this.eraserBtn.textContent = 'Eraser';
        this.eraserBtn.onclick = () => this.setTool('eraser');
        
        // Size Slider
        const sliderContainer = document.createElement('div');
        sliderContainer.className = 'layerforge-mask-editor-slider-container';
        sliderContainer.textContent = 'Size: ';
        
        this.sizeSlider = document.createElement('input');
        this.sizeSlider.type = 'range';
        this.sizeSlider.min = '1';
        this.sizeSlider.max = '200';
        this.sizeSlider.value = this.brushSize;
        this.sizeSlider.className = 'layerforge-mask-editor-slider';
        this.sizeSlider.oninput = (e) => this.brushSize = parseInt(e.target.value);
        
        sliderContainer.appendChild(this.sizeSlider);

        // Opacity Slider
        const opacityContainer = document.createElement('div');
        opacityContainer.className = 'layerforge-mask-editor-slider-container';
        opacityContainer.textContent = 'Opacity: ';
        
        this.opacitySlider = document.createElement('input');
        this.opacitySlider.type = 'range';
        this.opacitySlider.min = '0.1';
        this.opacitySlider.max = '1.0';
        this.opacitySlider.step = '0.1';
        this.opacitySlider.value = this.brushOpacity;
        this.opacitySlider.className = 'layerforge-mask-editor-slider';
        this.opacitySlider.oninput = (e) => this.brushOpacity = parseFloat(e.target.value);
        
        opacityContainer.appendChild(this.opacitySlider);

        // Hardness Slider
        const hardnessContainer = document.createElement('div');
        hardnessContainer.className = 'layerforge-mask-editor-slider-container';
        hardnessContainer.textContent = 'Hardness: ';
        
        this.hardnessSlider = document.createElement('input');
        this.hardnessSlider.type = 'range';
        this.hardnessSlider.min = '0';
        this.hardnessSlider.max = '1.0';
        this.hardnessSlider.step = '0.1';
        this.hardnessSlider.value = this.brushHardness;
        this.hardnessSlider.className = 'layerforge-mask-editor-slider';
        this.hardnessSlider.oninput = (e) => this.brushHardness = parseFloat(e.target.value);
        
        hardnessContainer.appendChild(this.hardnessSlider);
        
        // Clear Button
        const clearBtn = document.createElement('button');
        clearBtn.className = 'layerforge-mask-editor-button';
        clearBtn.textContent = 'Clear';
        clearBtn.onclick = () => this.clearMask();

        tools.append(this.brushBtn, this.eraserBtn, sliderContainer, opacityContainer, hardnessContainer, clearBtn);
        
        const actions = document.createElement('div');
        actions.className = 'layerforge-mask-editor-tools';
        
        // Cancel Button
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'layerforge-mask-editor-button danger';
        cancelBtn.textContent = 'Cancel';
        cancelBtn.onclick = () => this.close(null);
        
        // Save Button
        const saveBtn = document.createElement('button');
        saveBtn.className = 'layerforge-mask-editor-button primary';
        saveBtn.textContent = 'Save';
        saveBtn.onclick = () => this.save();
        
        actions.append(cancelBtn, saveBtn);
        
        toolbar.append(tools, actions);
        this.overlay.appendChild(toolbar);
        
        // Canvas Container
        this.container = document.createElement('div');
        this.container.className = 'layerforge-mask-editor-container';
        // Position absolute for centering via transform
        this.container.style.position = 'absolute';
        this.container.style.top = '50%';
        this.container.style.left = '50%';
        
        // We will create canvases when opening
        
        this.overlay.appendChild(this.container);
        document.body.appendChild(this.overlay);
        
        // Event Listeners for Overlay (for global events)
        window.addEventListener('keydown', this.handleKeyDown.bind(this));
        this.overlay.addEventListener('wheel', this.handleWheel.bind(this), { passive: false });
    }

    async open(imageUrl, maskImage = null, maskContext = null) {
        this.createUI();
        this.overlay.style.display = 'flex';
        
        this.scale = 1;
        this.panX = 0;
        this.panY = 0;
        this.history = [];
        this.historyIndex = -1;
        
        const img = await this.loadImage(imageUrl);
        
        let mImg = null;
        if (maskImage) {
            if (!(maskImage instanceof HTMLImageElement) && !(maskImage instanceof HTMLCanvasElement)) {
                 mImg = await this.loadImage(maskImage);
            } else {
                 mImg = maskImage;
            }
        }
        
        let finalWidth = img.width;
        let finalHeight = img.height;
        
        let maskX = 0;
        let maskY = 0;
        let imgX = 0;
        let imgY = 0;
        
        if (mImg) {
            if (mImg.width !== finalWidth || mImg.height !== finalHeight) {
                const outputBounds = maskContext?.outputAreaBounds;
                const maskDrawingAreaBounds = maskContext?.maskDrawingAreaBounds;

                const hasBounds = outputBounds && maskDrawingAreaBounds &&
                    Number.isFinite(outputBounds.x) &&
                    Number.isFinite(outputBounds.y) &&
                    Number.isFinite(outputBounds.width) &&
                    Number.isFinite(outputBounds.height) &&
                    Number.isFinite(maskDrawingAreaBounds.x) &&
                    Number.isFinite(maskDrawingAreaBounds.y) &&
                    Number.isFinite(maskDrawingAreaBounds.width) &&
                    Number.isFinite(maskDrawingAreaBounds.height);

                const sizeMatches = hasBounds &&
                    outputBounds.width === finalWidth &&
                    outputBounds.height === finalHeight &&
                    maskDrawingAreaBounds.width === mImg.width &&
                    maskDrawingAreaBounds.height === mImg.height;

                if (sizeMatches) {
                    maskX = Math.round(maskDrawingAreaBounds.x - outputBounds.x);
                    maskY = Math.round(maskDrawingAreaBounds.y - outputBounds.y);
                } else {
                    maskX = 0;
                    maskY = 0;
                }
            } else {
            }
        }
        
        this.width = finalWidth;
        this.height = finalHeight;
        
        const maxW = window.innerWidth * 0.95;
        const maxH = window.innerHeight * 0.9;
        const scaleW = maxW / this.width;
        const scaleH = maxH / this.height;
        
        const minScale = Math.min(scaleW, scaleH);
        
        if (minScale >= 0.8) {
            this.scale = 1;
        } else {
            this.scale = minScale;
        }
        
        this.updateTransform();
        
        this.container.innerHTML = '';
        
        const bg = createCanvas(this.width, this.height);
        this.bgCanvas = bg.canvas;
        this.bgCanvas.className = 'layerforge-mask-editor-canvas-layer';
        this.bgCtx = bg.ctx;
        this.bgCtx.drawImage(img, imgX, imgY);
        this.container.appendChild(this.bgCanvas);
        
        const mask = createCanvas(this.width, this.height);
        this.maskCanvas = mask.canvas;
        this.maskCanvas.className = 'layerforge-mask-editor-canvas-layer';
        this.maskCtx = mask.ctx;
        this.container.appendChild(this.maskCanvas);
        
        if (mImg) {
            this.maskCtx.drawImage(mImg, maskX, maskY);
            
            this.saveState();
        }

        const cursor = createCanvas(this.width, this.height);
        this.cursorCanvas = cursor.canvas;
        this.cursorCanvas.className = 'layerforge-mask-editor-canvas-layer';
        this.cursorCtx = cursor.ctx;
        this.container.appendChild(this.cursorCanvas);
        
        this.cursorCanvas.addEventListener('pointerdown', this.handlePointerDown.bind(this));
        window.addEventListener('pointermove', this.handlePointerMove.bind(this));
        window.addEventListener('pointerup', this.handlePointerUp.bind(this));
        
        return new Promise((resolve) => {
            this.resolvePromise = resolve;
        });
    }
    
    loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            if (typeof src === 'string') img.src = src;
            else resolve(src); // already an element or blob?
        });
    }
    
    close(result) {
        if (this.overlay) {
            this.overlay.style.display = 'none';
        }
        
        // Remove Listeners
        if (this.boundHandlePointerDown) {
            if (this.cursorCanvas) {
                this.cursorCanvas.removeEventListener('pointerdown', this.boundHandlePointerDown);
            }
            window.removeEventListener('pointermove', this.boundHandlePointerMove);
            window.removeEventListener('pointerup', this.boundHandlePointerUp);
            window.removeEventListener('keydown', this.boundHandleKeyDown);
            window.removeEventListener('keyup', this.boundHandleKeyUp);
            if (this.overlay) {
                this.overlay.removeEventListener('wheel', this.boundHandleWheel);
            }
        }

        if (this.resolvePromise) {
            this.resolvePromise(result);
            this.resolvePromise = null;
        }
    }
    
    save() {
        // Return the mask canvas
        // We can return a Blob or the canvas itself. 
        // MaskEditorIntegration expects an Image or Canvas probably.
        // Let's return the canvas, the integration can convert it.
        this.close(this.maskCanvas);
    }
    
    setTool(tool) {
        this.isEraser = (tool === 'eraser');
        this.brushBtn.classList.toggle('active', !this.isEraser);
        this.eraserBtn.classList.toggle('active', this.isEraser);
    }
    
    clearMask() {
        this.maskCtx.clearRect(0, 0, this.width, this.height);
        this.saveState();
    }
    
    updateTransform() {
        if (!this.container) return;
        this.container.style.width = `${this.width}px`;
        this.container.style.height = `${this.height}px`;
        
        // Center + Pan + Scale
        // We use transform to handle everything
        this.container.style.transform = `translate(calc(-50% + ${this.panX}px), calc(-50% + ${this.panY}px)) scale(${this.scale})`;
        this.container.style.transformOrigin = 'center center';
        
        // Ensure canvases fill the container and stack correctly
        [this.bgCanvas, this.maskCanvas, this.cursorCanvas].forEach(c => {
            if (c) {
                c.style.width = '100%';
                c.style.height = '100%';
                c.style.position = 'absolute';
                c.style.top = '0';
                c.style.left = '0';
            }
        });
    }
    
    // Interaction Handlers
    
    handleWheel(e) {
        if (!this.overlay || this.overlay.style.display === 'none') return;
        
        e.preventDefault();
        e.stopPropagation();

        // Adjust Brush Size: Ctrl + Wheel
        if (e.ctrlKey || e.metaKey) {
            const delta = -Math.sign(e.deltaY) * 5;
            const newSize = Math.max(1, Math.min(200, this.brushSize + delta));
            this.brushSize = newSize;
            if (this.sizeSlider) {
                this.sizeSlider.value = newSize;
            }
            // Trigger cursor update
            const pos = this.getPointerPos(e);
            this.updateCursor(pos.x, pos.y);
        } 
        // Zoom: Wheel (No Modifier)
        else {
            const delta = -Math.sign(e.deltaY) * 0.1;
            const newScale = Math.max(0.1, Math.min(10, this.scale + delta));
            this.scale = newScale;
            this.updateTransform();
        }
    }
    
    getPointerPos(e) {
        const rect = this.container.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) / this.scale,
            y: (e.clientY - rect.top) / this.scale
        };
    }
    
    handlePointerDown(e) {
        if (e.button === 1 || e.altKey) { // Middle click or Alt+Left
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            e.preventDefault();
            return;
        }
        
        if (e.button === 0) {
            const pos = this.getPointerPos(e);
            if (e.shiftKey && this.lastAnchorPos) {
                this.isDrawing = false;
                this.lastDrawPos = this.lastAnchorPos;
                this.draw(pos.x, pos.y);
                this.lastDrawPos = pos;
                this.lastAnchorPos = pos;
                this.saveState();
                return;
            }
            this.isDrawing = true;
            this.lastDrawPos = pos;
            this.lastAnchorPos = pos;
            this.drawDot(pos.x, pos.y);
        }
    }

    drawDot(x, y) {
        // Safety check
        if (typeof this.brushOpacity === 'undefined') this.brushOpacity = 1.0;
        if (typeof this.brushHardness === 'undefined') this.brushHardness = 1.0;

        // Note: isEraser handling
        // If eraser, we usually use destination-out. 
        // ShadowBlur with destination-out is tricky in some browsers.
        // Standard approach for soft eraser:
        // 1. Set globalCompositeOperation = 'destination-out'
        // 2. Set fillStyle = 'rgba(0,0,0,1)' (color doesn't matter for dest-out, alpha does)
        // 3. Set shadowColor = 'rgba(0,0,0,1)'
        // 4. Set shadowBlur.
        
        // For Brush (source-over):
        // We use radial gradient for smoother hardness falloff
        
        this.maskCtx.save(); // Save context state
        
        const radius = this.brushSize / 2;
        
        if (this.isEraser) {
            this.maskCtx.globalCompositeOperation = 'destination-out';
            
            if (this.brushHardness >= 0.99) {
                this.maskCtx.fillStyle = 'white';
                this.maskCtx.beginPath();
                this.maskCtx.arc(x, y, radius, 0, Math.PI * 2);
                this.maskCtx.fill();
            } else {
                const gradient = this.maskCtx.createRadialGradient(x, y, 0, x, y, radius);
                const solidRadiusRatio = this.brushHardness;
                
                gradient.addColorStop(0, 'rgba(0,0,0,1)');
                gradient.addColorStop(Math.max(0, solidRadiusRatio), 'rgba(0,0,0,1)');
                gradient.addColorStop(1, 'rgba(0,0,0,0)');
                
                this.maskCtx.fillStyle = gradient;
                this.maskCtx.beginPath();
                this.maskCtx.arc(x, y, radius, 0, Math.PI * 2);
                this.maskCtx.fill();
            }
        } else {
            this.maskCtx.globalCompositeOperation = 'source-over';
            
            if (this.brushHardness >= 0.99) {
                // Hard brush
                this.maskCtx.fillStyle = `rgba(255,255,255,${this.brushOpacity})`;
                this.maskCtx.beginPath();
                this.maskCtx.arc(x, y, radius, 0, Math.PI * 2);
                this.maskCtx.fill();
            } else {
                // Soft brush using gradient
                const gradient = this.maskCtx.createRadialGradient(x, y, 0, x, y, radius);
                const solidRadiusRatio = this.brushHardness;
                const opacity = this.brushOpacity;
                
                gradient.addColorStop(0, `rgba(255,255,255,${opacity})`);
                gradient.addColorStop(Math.max(0, solidRadiusRatio), `rgba(255,255,255,${opacity})`);
                gradient.addColorStop(1, `rgba(255,255,255,0)`);
                
                this.maskCtx.fillStyle = gradient;
                this.maskCtx.beginPath();
                this.maskCtx.arc(x, y, radius, 0, Math.PI * 2);
                this.maskCtx.fill();
            }
        }
        
        this.maskCtx.restore();
    }
    
    handlePointerMove(e) {
        if (this.isDragging) {
            const dx = e.clientX - this.lastMouseX;
            const dy = e.clientY - this.lastMouseY;
            this.panX += dx;
            this.panY += dy;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.updateTransform();
            e.preventDefault();
            e.stopPropagation();
            return; 
        }
        
        const pos = this.getPointerPos(e);
        this.updateCursor(pos.x, pos.y);

        if (this.isDrawing) {
            this.draw(pos.x, pos.y);
            this.lastDrawPos = pos;
        }
    }
    
    handlePointerUp(e) {
        if (this.isDrawing) {
            this.isDrawing = false;
            this.saveState();
            if (this.lastDrawPos) {
                this.lastAnchorPos = this.lastDrawPos;
            }
        }
        this.isDragging = false;
    }
    
    handleKeyDown(e) {
        if (this.overlay && this.overlay.style.display !== 'none') {
            if (e.key === 'z' && e.ctrlKey) {
                e.preventDefault();
                this.undo();
            }
        }
    }
    
    updateCursor(x, y) {
        if (!this.cursorCtx) return;
        this.cursorCtx.clearRect(0, 0, this.width, this.height);
        
        this.cursorCtx.beginPath();
        this.cursorCtx.arc(x, y, this.brushSize / 2, 0, Math.PI * 2);
        this.cursorCtx.strokeStyle = 'white';
        this.cursorCtx.lineWidth = 2 / this.scale;
        this.cursorCtx.stroke();
        this.cursorCtx.beginPath();
        this.cursorCtx.arc(x, y, this.brushSize / 2, 0, Math.PI * 2);
        this.cursorCtx.strokeStyle = 'black';
        this.cursorCtx.lineWidth = 1 / this.scale;
        this.cursorCtx.stroke();
    }
    
    draw(x, y) {
        if (!this.lastDrawPos) return;
        
        const dx = x - this.lastDrawPos.x;
        const dy = y - this.lastDrawPos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        const step = Math.max(1, this.brushSize * 0.15); 
        const steps = Math.ceil(distance / step);
        
        for (let i = 1; i <= steps; i++) {
            const t = i / steps;
            const cx = this.lastDrawPos.x + dx * t;
            const cy = this.lastDrawPos.y + dy * t;
            this.drawDot(cx, cy);
        }
    }
    
    saveState() {
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }
        this.history.push(this.maskCanvas.toDataURL());
        this.historyIndex++;
    }
    
    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            const img = new Image();
            img.src = this.history[this.historyIndex];
            img.onload = () => {
                this.maskCtx.clearRect(0, 0, this.width, this.height);
                this.maskCtx.drawImage(img, 0, 0);
            };
        }
    }
}

export const layerForgeMaskEditor = new LayerForgeMaskEditor();
