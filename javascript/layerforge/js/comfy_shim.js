
// Shim for ComfyUI's ui.js $el
export function $el(tag, propsOrContent, content) {
    const split = tag.split(".");
    const tagName = split.shift();
    const idSplit = tagName.split("#");
    const realTagName = idSplit[0] || "div";
    const id = idSplit[1];
    const classes = split;

    const el = document.createElement(realTagName);
    if (id) el.id = id;
    if (classes.length > 0) el.className = classes.join(" ");
    
    if (propsOrContent) {
        if (typeof propsOrContent === 'string') {
            el.innerHTML = propsOrContent;
        } else if (Array.isArray(propsOrContent)) {
            propsOrContent.forEach(child => {
                if (typeof child === 'string') {
                    el.appendChild(document.createTextNode(child));
                } else {
                    el.appendChild(child);
                }
            });
        } else {
            // Properties
            const props = propsOrContent;
            for (const key in props) {
                if (key === 'parent') {
                    props.parent.appendChild(el);
                } else if (key === 'style') {
                    Object.assign(el.style, props.style);
                } else if (key === 'dataset') {
                    Object.assign(el.dataset, props.dataset);
                } else if (key === 'on' || key === 'onclick' || key.startsWith('on')) {
                     // Event listeners
                     const eventName = key.startsWith('on') ? key.substring(2).toLowerCase() : key;
                     el.addEventListener(eventName, props[key]);
                } else if (key === 'for') {
                    el.setAttribute('for', props[key]);
                } else {
                    el[key] = props[key];
                    // Also set attribute for standard HTML attributes
                    if (['href', 'src', 'type', 'rel', 'class', 'id', 'name', 'value'].includes(key)) {
                         el.setAttribute(key, props[key]);
                    }
                }
            }
        }
    }

    if (content) {
         if (typeof content === 'string') {
            el.innerHTML = content;
        } else if (Array.isArray(content)) {
            content.forEach(child => {
                if (typeof child === 'string') {
                    el.appendChild(document.createTextNode(child));
                } else if (child instanceof Node) {
                    el.appendChild(child);
                }
            });
        }
    }

    return el;
}

export const ComfyApp = {
    clipspace: {
        imgs: []
    },
    copyToClipspace: (node) => {
    }
};

export const app = {
    // Mock app object
    registerExtension: (ext) => {
    }
};

export const api = {
    // Shim api object if needed
};
