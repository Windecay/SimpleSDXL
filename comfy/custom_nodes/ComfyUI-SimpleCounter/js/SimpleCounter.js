import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "SimpleCounter",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Simple Counter") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = origOnNodeCreated ? origOnNodeCreated.apply(this) : undefined;
                const start = this.widgets.find((w) => w.name === "start");
                const count = this.widgets.find((w) => w.name === "count");
                let counter = start.value;
                count.type = "converted-widget"; // hidden
                count.serializeValue = () => { return counter++; }
                api.addEventListener("promptQueued", () => { counter = start.value; }); // reset
                return r;
            }
        }
    }
})
