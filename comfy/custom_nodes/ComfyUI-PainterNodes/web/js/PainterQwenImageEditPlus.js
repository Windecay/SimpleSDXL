import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Painter.QwenImageEditPlus.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "PainterQwenImageEditPlus") return;
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            onNodeCreated?.apply(this, arguments);
            
            if (!this.size || this.size[0] < 320) {
                this.setSize([320, 400]);
            }
            
            const modeWidget = this.widgets.find(w => w.name === "mode");
            if (!modeWidget) return;
            
            // Function to check if input is a dynamic image input (image1-image10, not mask)
            const isDynamicImageInput = (name) => {
                return /^image\d+$/.test(name);  // Matches image1, image2, etc., but not image1_mask
            };
            
            const updateInputs = () => {
                const count = parseInt(modeWidget.value.split("_")[0]);
                
                const connections = {};
                this.inputs.forEach(input => {
                    if (isDynamicImageInput(input.name) && input.link) {
                        const link = this.graph.links[input.link];
                        if (link) {
                            connections[input.name] = {
                                origin_id: link.origin_id,
                                origin_slot: link.origin_slot
                            };
                        }
                    }
                });
                
                const toRemove = this.inputs.filter(input => {
                    if (!isDynamicImageInput(input.name)) return false;
                    const num = parseInt(input.name.replace("image", ""));
                    return num && num > count;
                });
                
                toRemove.forEach(input => {
                    if (input.link) {
                        this.disconnectInput(input.name);
                    }
                    const idx = this.inputs.indexOf(input);
                    if (idx > -1) {
                        this.inputs.splice(idx, 1);
                    }
                });
                
                for (let i = 1; i <= count; i++) {
                    const inputName = `image${i}`;
                    const exists = this.inputs.find(inp => inp.name === inputName);
                    if (!exists) {
                        this.addInput(inputName, "IMAGE");
                        if (connections[inputName]) {
                            const { origin_id, origin_slot } = connections[inputName];
                            const originNode = this.graph.getNodeById(origin_id);
                            if (originNode) {
                                originNode.connect(origin_slot, this, inputName);
                            }
                        }
                    }
                }
                
                app.canvas.draw(true, true);
            };
            
            const originalCallback = modeWidget.callback;
            modeWidget.callback = (v) => {
                if (originalCallback) originalCallback(v);
                updateInputs();
            };
            
            setTimeout(updateInputs, 10);
        };
        
        const originalComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function(out) {
            const size = originalComputeSize?.apply(this, arguments) || [320, 200];
            size[0] = Math.max(size[0], 320);
            size[1] = Math.max(size[1], 200);
            return size;
        };
    }
});