import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "Painter.VideoCombine",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "PainterVideoCombine") return;

        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            onDrawForeground?.apply(this, arguments);
            if (this.progress > 0 && this.progress < 1) {
                ctx.save();
                ctx.fillStyle = "#FFD700";
                ctx.fillRect(0, -2, this.size[0] * this.progress, 4);
                ctx.restore();
            }
        };

        function getHeaderAndWidgetHeight(node) {
            let height = 24;
            if (node.widgets) {
                for (const w of node.widgets) {
                    if (w.name !== "painter_preview" && w.type !== "hidden") {
                        height += (w.computeSize ? w.computeSize(node.size[0])[1] : 20) + 18;
                    }
                }
            }
            return height;
        }

        function findVideoElement(node) {
            if (!node.widgets) return null;
            for (const w of node.widgets) {
                if (w.element?.tagName === "VIDEO") return w.element;
                const vid = w.element?.querySelector("video");
                if (vid) return vid;
            }
            return null;
        }

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            getExtraMenuOptions?.apply(this, arguments);
            const video = findVideoElement(this);
            const newOptions = [];

            newOptions.push({
                content: "Save preview",
                callback: () => {
                    const params = this.properties["painter_output_cache"];
                    if (params) {
                        const url = api.apiURL(`/view?filename=${params.filename}&subfolder=${params.subfolder}&type=${params.type}`);
                        const a = document.createElement("a");
                        a.href = url; a.download = params.filename;
                        document.body.appendChild(a); a.click(); document.body.removeChild(a);
                    }
                }
            });

            newOptions.push({
                content: (video && video.paused) ? "Resume preview" : "Pause preview",
                callback: () => { if (video) video.paused ? video.play() : video.pause(); }
            });

            newOptions.push({
                content: "Sync preview",
                callback: () => {
                    if (video) {
                        video.pause();
                        video.currentTime = 0;
                        video.load(); 
                        video.play();
                    }
                }
            });

            if (options.length > 0) newOptions.push(null);
            options.unshift(...newOptions);
        };

        nodeType.prototype.onResize = function (size) {
            if (this.painter_aspect) {
                const headHeight = getHeaderAndWidgetHeight(this);
                const targetVideoHeight = size[0] / this.painter_aspect;
                const totalHeight = Math.ceil(headHeight + targetVideoHeight);

                if (Math.abs(size[1] - totalHeight) > 0.5) {
                    size[1] = totalHeight;
                }
            }
            
            const widget = this.widgets?.find(w => w.name === "painter_preview");
            if (widget?.element) {
                // Fix: Use 100% width to fit parent container instead of fixed pixel size
                widget.element.style.width = "100%";
                widget.element.style.left = "0px";
                const contentH = size[1] - getHeaderAndWidgetHeight(this);
                widget.element.style.height = `${contentH}px`;
            }
        };

        nodeType.prototype.onExecuted = function (message) {
            if (message?.painter_output) {
                this.properties["painter_output_cache"] = message.painter_output[0];
                updateVideoPreview(this, message.painter_output[0]);
            }
        };

        nodeType.prototype.onConfigure = function () {
            if (this.properties?.["painter_output_cache"]) {
                updateVideoPreview(this, this.properties["painter_output_cache"]);
            }
        };

        function updateVideoPreview(node, data) {
            let widget = node.widgets?.find(w => w.name === "painter_preview");
            
            if (!widget) {
                const element = document.createElement("div");
                element.style.display = "flex";
                element.style.justifyContent = "center";
                element.style.alignItems = "center";
                element.style.padding = "0px";
                element.style.margin = "0px";
                element.style.overflow = "hidden";
                element.style.boxSizing = "border-box";
                
                widget = node.addDOMWidget("painter_preview", "preview", element, {
                    serialize: false, hideOnZoom: false
                });
            }

            const url = api.apiURL(`/view?filename=${data.filename}&subfolder=${data.subfolder}&type=${data.type}`);
            widget.element.innerHTML = "";

            const video = document.createElement("video");
            video.src = url;
            video.controls = false; 
            video.loop = true; 
            video.autoplay = true; 
            video.muted = true;
            
            video.style.width = "100%";
            video.style.height = "100%";
            video.style.objectFit = "cover"; 
            video.style.display = "block";

            const triggerCtx = (e) => {
                e.preventDefault(); e.stopPropagation();
                if (app.canvas.processContextMenu) app.canvas.processContextMenu(node, e);
                else app.canvas._mousedown_callback(e);
                return false;
            };
            video.addEventListener('contextmenu', triggerCtx, true);
            video.addEventListener('pointerdown', (e) => { if (e.button === 2) triggerCtx(e); }, true);

            video.addEventListener('mouseenter', () => { video.muted = false; });
            video.addEventListener('mouseleave', () => { video.muted = true; });

            video.onloadedmetadata = () => {
                if (video.videoWidth && video.videoHeight) {
                    node.painter_aspect = video.videoWidth / video.videoHeight;
                    node.onResize(node.size);
                    node.setDirtyCanvas(true, true);
                }
            };

            widget.element.appendChild(video);
        }
    }
});