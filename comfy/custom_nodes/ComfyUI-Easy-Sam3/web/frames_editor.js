import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const getRealURL = obj => {
    return api.apiURL(`/view?filename=${encodeURIComponent(obj.filename)}&type=${obj.type}&subfolder=${obj.subfolder}&rand=${Math.random()}`)
}
const makeUUID = _ =>{
  let dt = new Date().getTime()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0
    dt = Math.floor(dt / 16)
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
  return uuid
}
const chainCallback = (object, property, callback) => {
  if (object == undefined) {
    //This should not happen.
    console.error("Tried to add callback to non-existant object")
    return;
  }
  if (property in object) {
    const callback_orig = object[property]
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r
    };
  } else {
    object[property] = callback;
  }
}
const getUserSettingsValue = id => id ? app?.ui?.settings?.getSettingValue(id) : null
/**
 * Get setting value by id
 * @param {string} id - setting id
 * @param {string} storge_key - key to get value from local storage
 * @returns {string|object|null} - setting value
 */
const getSetting = (id, storage_key=null) => {
    try{
        let setting = id ? getUserSettingsValue(id) : null
        if(setting === null || setting === undefined) setting = storage_key ? localStorage[storage_key] : (localStorage[id] || null)
        return setting
    }
    catch(e){
        console.error(e)
        return null
    }
}
export const $t = (word) => {
    let _locale = getSetting('Comfy.Locale') || 'en'
    switch (_locale){
        case 'zh-CN':
        case 'zh':
            return word['zh'] || word['en'] || word
        default:
            return word['en'] || word
    }
}


app.registerExtension({
    name: "Comfy.EasyUse.FramesEditor",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const nodeName = nodeData.name;
        if (nodeName === "easy framesEditor"){
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; height: 100%; background: #0f1011; overflow: hidden; box-sizing: border-box;border-radius:4px; margin: 0; padding: 0; display: flex; flex-direction: column;";

                // Toolbar
                const toolbar = document.createElement("div");
                toolbar.style.cssText = "flex: 0 0 32px; width: 100%; background: #222; display: flex; align-items: center; justify-content: space-between; padding: 0 4px; box-sizing: border-box; border-bottom: 1px solid #333; z-index: 10;";

                // Left side (Undo/Redo)
                const leftGroup = document.createElement("div");
                leftGroup.style.display = "flex";
                leftGroup.style.gap = "4px";

                const createBtn = (iconSvg, title, onClick, isActive = false) => {
                    const btn = document.createElement("div");
                    btn.style.cssText = `width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; cursor: pointer; border-radius: 4px; color: ${isActive ? '#fff' : '#ccc'}; background-color: ${isActive ? '#444' : 'transparent'};`;
                    btn.innerHTML = iconSvg;
                    btn.title = title;
                    btn.onmouseover = () => { if (!btn.classList.contains("active")) btn.style.backgroundColor = "#333"; };
                    btn.onmouseout = () => { if (!btn.classList.contains("active")) btn.style.backgroundColor = "transparent"; };
                    btn.onclick = onClick;
                    return btn;
                };

                const undoIcon = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z"/></svg>`;
                const redoIcon = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M18.4 10.6C16.55 9 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z"/></svg>`;
                const resetIcon = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>`;
                const pointIcon = `<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>`;
                const boxIcon = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2" stroke-dasharray="4 4"/></svg>`;

                
                const undoBtn = createBtn(undoIcon, "Undo", () => this.undo());
                const redoBtn = createBtn(redoIcon, "Redo", () => this.redo());
                const resetBtn = createBtn(resetIcon, "Clear All", () => {
                    const { positivePoints, negativePoints, bboxes } = this.canvasWidget;
                    if (positivePoints.length === 0 && negativePoints.length === 0 && bboxes.length === 0) return;
                    this.canvasWidget.positivePoints = [];
                    this.canvasWidget.negativePoints = [];
                    this.canvasWidget.bboxes = [];
                    this.canvasWidget.history = [];
                    this.canvasWidget.historyIndex = -1;
                    this.redrawCanvas();
                    this.updateUndoRedoUI();
                    this.updateWidgetValue();
                });

                leftGroup.appendChild(undoBtn);
                leftGroup.appendChild(redoBtn);
                leftGroup.appendChild(resetBtn);

                // Right side (Modes)
                const rightGroup = document.createElement("div");
                rightGroup.style.display = "flex";
                rightGroup.style.gap = "4px";

                let pointBtn, boxBtn;

                const setMode = (mode) => {
                    this.canvasWidget.mode = mode;
                    pointBtn.style.backgroundColor = mode === 'point' ? '#444' : 'transparent';
                    pointBtn.classList.toggle("active", mode === 'point');
                    pointBtn.style.color = mode === 'point' ? '#fff' : '#ccc';
                    
                    boxBtn.style.backgroundColor = mode === 'box' ? '#444' : 'transparent';
                    boxBtn.classList.toggle("active", mode === 'box');
                    boxBtn.style.color = mode === 'box' ? '#fff' : '#ccc';
                };

                pointBtn = createBtn(pointIcon, "Point Mode", () => setMode('point'), true);
                pointBtn.classList.add("active");
                boxBtn = createBtn(boxIcon, "Box Mode", () => setMode('box'), false);

                rightGroup.appendChild(pointBtn);
                rightGroup.appendChild(boxBtn);

                toolbar.appendChild(leftGroup);
                toolbar.appendChild(rightGroup);
                container.appendChild(toolbar);

                // Canvas Wrapper
                const canvasWrapper = document.createElement("div");
                canvasWrapper.style.cssText = "flex: 1; width: 100%; position: relative; overflow: hidden; display: flex; align-items: center; justify-content: center; background: #0f1011;";
                container.appendChild(canvasWrapper);

                // Create canvas for image and points
                const canvas = document.createElement("canvas");
                canvas.width = 512;
                canvas.height = 512;
                // Use max-width and max-height instead of width/height 100% to prevent overflow
                canvas.style.cssText = "display: block; max-width: 100%; max-height: 100%; object-fit: contain; cursor: crosshair; margin: 0 auto;";
                canvasWrapper.appendChild(canvas);
                
                const ctx = canvas.getContext("2d");
                // console.log("Canvas created:", canvas);

                // Tracker UI
                const tracker = document.createElement("div");
                tracker.style.cssText = "flex: 0 0 32px; width: 100%; background: #222; display: none; align-items: center; justify-content: space-between; padding: 0 8px; box-sizing: border-box; border-top: 1px solid #333; gap: 2px;";

                const frameInfo = document.createElement("div");
                frameInfo.style.cssText = "color: #ccc; font-family: monospace; font-size: 12px; min-width: 40px; text-align: center; user-select: none;";
                frameInfo.innerText = "0/0";

                const slider = document.createElement("input");
                slider.type = "range";
                slider.min = "0";
                slider.max = "0";
                slider.value = "0";
                slider.step = "1";
                slider.style.cssText = "flex: 1; height: 4px; cursor: pointer;";

                tracker.appendChild(frameInfo);
                tracker.appendChild(slider);
                container.appendChild(tracker);

                slider.addEventListener("input", (e) => {
                    const frameIndex = parseInt(e.target.value);
                    this.canvasWidget.frameIndex = frameIndex;
                    this.canvasWidget.frameInfo.innerText = `${frameIndex + 1}/${this.canvasWidget.previewFrames.length}`;
                    this.updateWidgetValue();
                    
                    const img = new Image();
                    img.onload = () => {
                        this.canvasWidget.image = img;
                        canvas.width = img.width;
                        canvas.height = img.height;
                        this.redrawCanvas();
                    };
                    img.src = getRealURL(this.canvasWidget.previewFrames[frameIndex]);
                });

                // Store state
                this.canvasWidget = {
                    canvas: canvas,
                    ctx: ctx,
                    container: container,
                    tracker: tracker,
                    slider: slider,
                    frameInfo: frameInfo,
                    image: null,
                    positivePoints: [],
                    negativePoints: [],
                    bboxes: [],
                    hoverBBox: null,
                    hoveredPoint: null,
                    mode: 'point', // 'point' | 'box'
                    history: [],
                    historyIndex: -1,
                    isDrawingBox: false,
                    currentBox: null,
                    frameIndex:0,
                    previewFrames: [],
                };

                // Add as DOM widget
                const widget = this.addDOMWidget("canvas", "points_editor", container, );
                this.uuid = makeUUID();
                // Store widget reference for updates
                this.canvasWidget.domWidget = widget;

                // Get info widget
                const infoWidget = this.widgets.find(w=> w.name == 'info')
                setTimeout(_=>{
                    if(infoWidget && infoWidget.value) {
                        try {
                            const info = JSON.parse(infoWidget.value);
                            // Set positive/negative points
                            if (Array.isArray(info.positive_coords)) {
                                this.canvasWidget.positivePoints = info.positive_coords;
                            }
                            if (Array.isArray(info.negative_coords)) {
                                this.canvasWidget.negativePoints = info.negative_coords;
                            }
                            // Set bboxes
                            if (Array.isArray(info.bbox)) {
                                this.canvasWidget.bboxes = info.bbox;
                            }
                            // Set slider/frameIndex
                            if (typeof info.frame_index === 'number' && this.canvasWidget.slider) {
                                this.canvasWidget.frameIndex = info.frame_index;
                                this.canvasWidget.slider.value = info.frame_index;
                                this.canvasWidget.frameInfo.innerText = `${info.frame_index + 1}/${this.canvasWidget.previewFrames.length}`;
                            }
                            this.redrawCanvas();
                        } catch (e) {
                            // ignore parse errors
                        }
                    }
                },1)
                
                // Make widget dynamically sized - override computeSize
                widget.computeSize = (width) => {
                    // Return a fixed minimum height to prevent infinite growth loops
                    // The actual height will be handled by onResize
                    const nodeHeight = this.size ? this.size[1] : 500;
                    const widgetHeight = Math.max(245, nodeHeight - 150);
                    return [width, widgetHeight];
                };

                
                // Update container height dynamically when node size changes
                chainCallback(this, "onResize", function(size) {
                    // Update container to match widget size
                    // Using a larger offset (80) to account for header and padding safely
                    const containerHeight = Math.max(245, size[1] - 150);
                    container.style.height = containerHeight + "px";
                });

                // Also update on draw to handle any size changes
                chainCallback(this, "onDrawForeground", function(ctx) {
                    // Handle any additional drawing if needed
                    const containerHeight = Math.max(245, this.size[1] - 150);
                    if (container.style.height !== containerHeight + "px") {
                        container.style.height = containerHeight + "px";
                    }
                });

                // Handle image input changes
                chainCallback(this, "onExecuted", function(message) {
                    if (message.preview && message.preview[0]) {
                        const {preview_str, is_init} = message.preview[0];
                        const previewData = JSON.parse(preview_str);
                        this.canvasWidget.previewFrames = previewData;
                        // Update tracker
                        if(is_init){
                            this.canvasWidget.frameIndex = 0;
                            this.restoreState({ positivePoints: [], negativePoints: [], bboxes: [] });
                        }
                        if (previewData.length > 1) {
                            this.canvasWidget.tracker.style.display = "flex";
                            slider.max = previewData.length - 1;
                            slider.value = this.canvasWidget.frameIndex;
                            this.canvasWidget.slider = slider;
                            this.canvasWidget.frameInfo.innerText = `${this.canvasWidget.frameIndex + 1}/${previewData.length}`;
                        } else {
                            this.canvasWidget.tracker.style.display = "none";
                        }

                        const img = new Image();
                        img.onload = () => {
                            // console.log(`Image loaded: ${img.width}x${img.height}`);
                            this.canvasWidget.image = img;
                            canvas.width = img.width;
                            canvas.height = img.height;
                            // console.log(`[Canvas resized to: ${canvas.width}x${canvas.height}`);
                            this.redrawCanvas();
                        };
                        
                        if(previewData?.length>0){
                            if (this.canvasWidget.frameIndex >= previewData.length) {
                                this.canvasWidget.frameIndex = 0;
                                slider.value = 0;
                                this.canvasWidget.slider = slider;
                            }
                            img.src = getRealURL(previewData[this.canvasWidget.frameIndex]);
                        }
                    
                    }
                });

                // History Management
                this.addToHistory = () => {
                    const { positivePoints, negativePoints, bboxes, history, historyIndex } = this.canvasWidget;
                    // Remove future history if we are in the middle
                    if (historyIndex < history.length - 1) {
                        this.canvasWidget.history = history.slice(0, historyIndex + 1);
                    }
                    // Push new state
                    const state = {
                        positivePoints: JSON.parse(JSON.stringify(positivePoints)),
                        negativePoints: JSON.parse(JSON.stringify(negativePoints)),
                        bboxes: JSON.parse(JSON.stringify(bboxes))
                    };
                    this.canvasWidget.history.push(state);
                    this.canvasWidget.historyIndex++;
                    this.updateUndoRedoUI();
                    this.updateWidgetValue();
                };

                this.undo = () => {
                    const { history, historyIndex } = this.canvasWidget;
                    if (historyIndex > 0) {
                        this.canvasWidget.historyIndex--;
                        const state = history[this.canvasWidget.historyIndex];
                        this.restoreState(state);
                    } else if (historyIndex === 0) {
                         // Initial empty state or first action? 
                         // If we want to undo the first action to "empty", we need an initial empty state in history.
                         // Let's assume index -1 is empty.
                         this.canvasWidget.historyIndex--;
                         this.restoreState({ positivePoints: [], negativePoints: [], bboxes: [] });
                    }
                    this.updateUndoRedoUI();
                };

                this.redo = () => {
                    const { history, historyIndex } = this.canvasWidget;
                    if (historyIndex < history.length - 1) {
                        this.canvasWidget.historyIndex++;
                        const state = history[this.canvasWidget.historyIndex];
                        this.restoreState(state);
                    }
                    this.updateUndoRedoUI();
                };

                this.restoreState = (state) => {
                    this.canvasWidget.positivePoints = JSON.parse(JSON.stringify(state.positivePoints));
                    this.canvasWidget.negativePoints = JSON.parse(JSON.stringify(state.negativePoints));
                    this.canvasWidget.bboxes = JSON.parse(JSON.stringify(state.bboxes));
                    this.redrawCanvas();
                    this.updateWidgetValue();
                };

                this.updateUndoRedoUI = () => {
                    const { historyIndex, history, positivePoints, negativePoints, bboxes } = this.canvasWidget;
                    undoBtn.style.color = historyIndex >= 0 ? '#ccc' : '#555';
                    undoBtn.style.cursor = historyIndex >= 0 ? 'pointer' : 'default';
                    redoBtn.style.color = historyIndex < history.length - 1 ? '#ccc' : '#555';
                    redoBtn.style.cursor = historyIndex < history.length - 1 ? 'pointer' : 'default';

                    const hasContent = positivePoints.length > 0 || negativePoints.length > 0 || bboxes.length > 0;
                    resetBtn.style.color = hasContent ? '#ccc' : '#555';
                    resetBtn.style.cursor = hasContent ? 'pointer' : 'default';
                };

                this.updateWidgetValue = () => {
                    const { positivePoints, negativePoints, bboxes, image, frameIndex } = this.canvasWidget;
                    const info_widget = this.widgets.find(w=> w.name == 'info')
                    info_widget.value = image ? JSON.stringify({
                        positive_coords: positivePoints,
                        negative_coords: negativePoints,
                        bbox: bboxes,
                        frame_index: frameIndex
                    }) : '';
                }

                // Event Listeners
                const getCoords = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    return {
                        x: (e.clientX - rect.left) * scaleX,
                        y: (e.clientY - rect.top) * scaleY
                    };
                };

                canvas.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    const coords = getCoords(e);
                    const { mode, image } = this.canvasWidget;
                    if(!image) return
                    if (mode === 'point') {
                        if (e.button === 0) { // Left click: Positive
                            this.canvasWidget.positivePoints.push(coords);
                        } else if (e.button === 2) { // Right click: Negative
                            this.canvasWidget.negativePoints.push(coords);
                        }
                        this.addToHistory();
                        this.redrawCanvas();
                    } else if (mode === 'box') {
                        if (e.button === 0) {
                            this.canvasWidget.isDrawingBox = true;
                            this.canvasWidget.currentBox = { x: coords.x, y: coords.y, w: 0, h: 0 };
                        }
                    }
                });

                canvas.addEventListener('mousemove', (e) => {
                    const coords = getCoords(e);
                    const { mode, isDrawingBox, currentBox, image } = this.canvasWidget;
                    if(!image) return
                    if (mode === 'box' && isDrawingBox && currentBox) {
                        currentBox.w = coords.x - currentBox.x;
                        currentBox.h = coords.y - currentBox.y;
                        this.redrawCanvas();
                    } else {
                        // Hover effects (optional, for points)
                        // ...
                    }
                });

                canvas.addEventListener('mouseup', (e) => {
                    const { mode, isDrawingBox, currentBox, image } = this.canvasWidget;
                    if(!image) return
                    if (mode === 'box' && isDrawingBox && currentBox) {
                        // Normalize box (w/h can be negative)
                        const box = {
                            x: Math.min(currentBox.x, currentBox.x + currentBox.w),
                            y: Math.min(currentBox.y, currentBox.y + currentBox.h),
                            w: Math.abs(currentBox.w),
                            h: Math.abs(currentBox.h)
                        };
                        // Only add if it has some size
                        if (box.w > 5 && box.h > 5) {
                            this.canvasWidget.bboxes.push(box);
                            this.addToHistory();
                        }
                        this.canvasWidget.isDrawingBox = false;
                        this.canvasWidget.currentBox = null;
                        this.redrawCanvas();
                    }
                });

                canvas.addEventListener('contextmenu', (e) => {
                    e.preventDefault();
                });

                // Draw initial placeholder
                this.redrawCanvas();

                // Set initial node size
                const nodeWidth = Math.max(400, this.size[0] || 400);
                const nodeHeight = 530; // Initial height: canvas (400) + space (80)
                this.setSize([nodeWidth, nodeHeight]);

                // Set initial container height
                container.style.height = "400px";

                this.updateUndoRedoUI();
            });

            // Helper: Redraw canvas
            nodeType.prototype.redrawCanvas = function() {
                const {canvas, ctx, image, positivePoints, negativePoints, bboxes, currentBox, hoveredPoint, mode} = this.canvasWidget;

                // Clear
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // pointSize: scale with canvas display size (considering zoom/scale)
                let pointSize = Math.max(2, Math.min(canvas.width, canvas.height) * 0.008);

                // Draw image if available
                if (image) {
                    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
                } else {
                    const desc = [
                        {"en": "Start with your own image or video", "zh": "从您自己的图像或视频开始"},
                        {"en": "Left Click: Add Positive Point", "zh": "左键点击：添加正面点"},
                        {"en": "Right Click: Add Negative Point", "zh": "右键点击：添加负面点"},
                        {"en": "Drag Box: Add Bounding Box", "zh": "拖动框：添加边界框"},
                    ]
                    // Placeholder
                    ctx.fillStyle = "transparent";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = "#ddd";
                    ctx.font = "34px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("🖼️", canvas.width / 2, canvas.height / 2 - 50);
                    ctx.font = "24px sans-serif";
                    desc.map((cate,index)=>{
                        ctx.fillText($t(cate), canvas.width / 2, canvas.height / 2 + (index * 40));
                    })
                }

                // Draw bboxes
                ctx.strokeStyle = "#00f";
                ctx.lineWidth = 2;
                for (const box of bboxes) {
                    ctx.strokeRect(box.x, box.y, box.w, box.h);
                    // Optional: fill slightly
                    ctx.fillStyle = "rgba(0, 0, 255, 0.1)";
                    ctx.fillRect(box.x, box.y, box.w, box.h);
                }

                // Draw current box
                if (currentBox) {
                    ctx.strokeStyle = "#0ff";
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    ctx.strokeRect(currentBox.x, currentBox.y, currentBox.w, currentBox.h);
                    ctx.setLineDash([]);
                }

                // Draw positive points (green)
                ctx.strokeStyle = "#139613";
                ctx.fillStyle = "#139613";
                for (const point of positivePoints) {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, pointSize, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }

                // Draw negative points (red)
                ctx.strokeStyle = "#8A1616";
                ctx.fillStyle = "#8A1616";
                for (const point of negativePoints) {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, pointSize, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            };

            
        }
    }
})