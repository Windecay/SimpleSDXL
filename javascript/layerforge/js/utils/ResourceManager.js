// @ts-ignore
import { $el } from "/file=javascript/layerforge/js/comfy_shim.js?v=patch26";
import { createModuleLogger } from "/file=javascript/layerforge/js/utils/LoggerUtils.js?v=patch26";
import { withErrorHandling, createValidationError, createNetworkError } from "/file=javascript/layerforge/js/ErrorHandler.js?v=patch26";
const log = createModuleLogger('ResourceManager');
export const addStylesheet = withErrorHandling(function (url) {
    if (!url) {
        throw createValidationError("URL is required", { url });
    }
    log.debug('Adding stylesheet:', { url });
    if (url.endsWith(".js")) {
        url = url.substr(0, url.length - 2) + "css";
    }
    $el("link", {
        parent: document.head,
        rel: "stylesheet",
        type: "text/css",
        href: url.startsWith("http") ? url : getUrl(url),
    });
    log.debug('Stylesheet added successfully:', { finalUrl: url });
}, 'addStylesheet');
export function getUrl(path, baseUrl) {
    if (!path) {
        throw createValidationError("Path is required", { path });
    }
    
    // Hardcoded prefix for Gradio environment to ensure absolute paths
    const GRADIO_PREFIX = "/file=javascript/layerforge/js/";

    if (baseUrl) {
        return new URL(path, baseUrl).toString();
    }
    
    // Check if path is already absolute or a full URL
    if (path.startsWith("http") || path.startsWith("/")) {
        return path;
    }
    
    // Handle relative paths starting with ./ or ../
    if (path.startsWith("./")) {
        path = path.substring(2);
    } else if (path.startsWith("../")) {
        // This is tricky without real path resolution, but for our structure:
        // utils/../foo -> foo
        // We assume we are in js root for relative paths usually?
        // Actually, let's just strip it and append to prefix which points to js root
        path = path.replace(/^\.\.\//, "");
    }

    // Force absolute path
    return GRADIO_PREFIX + path;
}
export const loadTemplate = withErrorHandling(async function (path, baseUrl) {
    if (!path) {
        throw createValidationError("Path is required", { path });
    }
    const url = getUrl(path, baseUrl);
    log.debug('Loading template:', { path, url });
    const response = await fetch(url);
    if (!response.ok) {
        throw createNetworkError(`Failed to load template: ${url}`, {
            url,
            status: response.status,
            statusText: response.statusText
        });
    }
    const content = await response.text();
    log.debug('Template loaded successfully:', { path, contentLength: content.length });
    return content;
}, 'loadTemplate');
