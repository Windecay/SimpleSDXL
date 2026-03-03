// based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/script.js
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

/**
 * Get the currently selected top-level UI tab button (e.g. the button that says "Extras").
 */
function get_uiCurrentTab() {
    return gradioApp().querySelector('#tabs > .tab-nav > button.selected');
}

/**
 * Get the first currently visible top-level UI tab content (e.g. the div hosting the "txt2img" UI).
 */
function get_uiCurrentTabContent() {
    return gradioApp().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])');
}

var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];
var uiLoadedCallbacks = [];
var uiTabChangeCallbacks = [];
var optionsChangedCallbacks = [];
var uiAfterUpdateTimeout = null;
var uiCurrentTab = null;

/**
 * Register callback to be called at each UI update.
 * The callback receives an array of MutationRecords as an argument.
 */
function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called soon after UI updates.
 * The callback receives no arguments.
 *
 * This is preferred over `onUiUpdate` if you don't need
 * access to the MutationRecords, as your function will
 * not be called quite as often.
 */
function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI is loaded.
 * The callback receives no arguments.
 */
function onUiLoaded(callback) {
    uiLoadedCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI tab is changed.
 * The callback receives no arguments.
 */
function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}

/**
 * Register callback to be called when the options are changed.
 * The callback receives no arguments.
 * @param callback
 */
function onOptionsChanged(callback) {
    optionsChangedCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * The callbacks are executed after a short while, unless another call to this function
 * is made before that time. IOW, the callbacks are executed only once, even
 * when there are multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 200);
}

var executedOnLoaded = false;

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m) {
        if (!executedOnLoaded && gradioApp().querySelector('#generate_button')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, m);
        scheduleAfterUiUpdateCallbacks();
        const newTab = get_uiCurrentTab();
        if (newTab && (newTab !== uiCurrentTab)) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
    initStylePreviewOverlay();
});

var onAppend = function(elem, f) {
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(m) {
            if (m.addedNodes.length) {
                f(m.addedNodes);
            }
        });
    });
    observer.observe(elem, {childList: true});
}

function addObserverIfDesiredNodeAvailable(querySelector, callback) {
    var elem = document.querySelector(querySelector);
    if (!elem) {
        window.setTimeout(() => addObserverIfDesiredNodeAvailable(querySelector, callback), 1000);
        return;
    }

    onAppend(elem, callback);
}

/**
 * Show reset button on toast "Connection errored out."
 */
addObserverIfDesiredNodeAvailable(".toast-wrap", function(added) {
    added.forEach(function(element) {
         if (element.innerText.includes("Connection errored out.")) {
             window.setTimeout(function() {
                const buttons = {
                    "reset_button": "remove",
                    "generate_button": "add",
                    "skip_button": "add",
                    "stop_button": "add"
                };
                for (const [id, action] of Object.entries(buttons)) {
                    const btn = document.getElementById(id);
                    if (btn) {
                        btn.classList[action]("hidden");
                    }
                }
            });
         }
    });
});

/**
 * Add a ctrl+enter as a shortcut to start a generation
 */
document.addEventListener('keydown', function(e) {
    const isModifierKey = (e.metaKey || e.ctrlKey || e.altKey);
    const isEnterKey = (e.key == "Enter" || e.keyCode == 13);

    if(isModifierKey && isEnterKey) {
        const generateButton = gradioApp().querySelector('button:not(.hidden)[id=generate_button]');
        if (generateButton) {
            generateButton.click();
            e.preventDefault();
            return;
        }

        const stopButton = gradioApp().querySelector('button:not(.hidden)[id=stop_button]')
        if(stopButton) {
            stopButton.click();
            e.preventDefault();
            return;
        }
    }
});

function initStylePreviewOverlay() {
    let overlayVisible = false;
    const samplesPath = document.querySelector("meta[name='samples-path']").getAttribute("content")
    const overlay = document.createElement('div');
    const tooltip = document.createElement('div');
    tooltip.className = 'preview-tooltip';
    overlay.appendChild(tooltip);
    overlay.id = 'stylePreviewOverlay';
    document.body.appendChild(overlay);
    document.addEventListener('mouseover', function (e) {
        const label = e.target.closest('.style_selections label');
        if (!label) return;
        label.removeEventListener("mouseout", onMouseLeave);
        label.addEventListener("mouseout", onMouseLeave);
        overlayVisible = true;
        overlay.style.opacity = "1";
        const originalText = label.querySelector("span").getAttribute("data-original-text");
        const name = originalText || label.querySelector("span").textContent;
        overlay.style.backgroundImage = `url("${samplesPath.replace(
            "fooocus_v2",
            name.toLowerCase().replaceAll(" ", "_")
        ).replaceAll("\\", "\\\\")}")`;

        tooltip.textContent = label.querySelector("span").textContent || name;

        function onMouseLeave() {
            overlayVisible = false;
            overlay.style.opacity = "0";
            overlay.style.backgroundImage = "";
            label.removeEventListener("mouseout", onMouseLeave);
        }
    });
    document.addEventListener('mousemove', function (e) {
        if (!overlayVisible) return;
        overlay.style.left = `${e.clientX}px`;
        overlay.style.top = `${e.clientY}px`;
        overlay.className = e.clientY > window.innerHeight / 2 ? "lower-half" : "upper-half";
    });
    // 新增文本悬浮层
    const textOverlay = document.createElement('div');
    textOverlay.id = 'styleTextOverlay';
    Object.assign(textOverlay.style, {
        position: 'fixed',
        left: '0',
        top: '0',
        background: 'rgba(0,0,0,0.8)',
        color: 'white',
        padding: '8px',
        borderRadius: '4px',
        pointerEvents: 'none',
        display: 'none',
        zIndex: 9999,
        maxWidth: '320px',
        backdropFilter: 'blur(3px)'
    });
    document.body.appendChild(textOverlay);

    document.addEventListener('mouseover', function(e) {
        const container = e.target.closest('.style_item');
        if (!container) {
            textOverlay.style.display = 'none';
            return;
        }

        const styleDataRaw = container.getAttribute('data-style-data');
        if (!styleDataRaw) {
            return;
        }

        try {
            const styleData = JSON.parse(styleDataRaw || '{}');
            textOverlay.innerHTML = `
                <div style="font-weight:bold; margin-bottom: 6px; font-size: 14px">${styleData.name || ''}</div>
                ${styleData.prompt ? `<div style="color:#ddd;font-size:12px;margin:4px 0">Prompt: ${styleData.prompt}</div>` : ''}
                ${styleData.negative_prompt ? `<div style="color:#888;font-size:12px">Negative: ${styleData.negative_prompt}</div>` : ''}
            `;
            textOverlay.style.display = 'block';
        } catch (e) {
            console.error('Error parsing style data:', e);
            textOverlay.style.display = 'none';
        }
    });
    document.addEventListener('mousemove', function(e) {
        if (textOverlay.style.display === 'block') {
            textOverlay.style.left = `${e.clientX + 15}px`;
            textOverlay.style.top = `${e.clientY + 15}px`;

            const rect = textOverlay.getBoundingClientRect();
            if (rect.right > window.innerWidth) {
                textOverlay.style.left = `${window.innerWidth - rect.width - 5}px`;
            }
            if (rect.bottom > window.innerHeight) {
                textOverlay.style.top = `${window.innerHeight - rect.height - 5}px`;
            }
        }
    });
    document.addEventListener('mouseout', function(e) {
        if (!e.relatedTarget || !e.relatedTarget.closest('.style_item')) {
            textOverlay.style.display = 'none';
        }
    });
    document.addEventListener('click', function(e) {
        const btn = e.target.closest('.style-button');
        if (!btn) return;

        const styleName = btn.getAttribute('data-style-name') || btn.textContent.trim();
        const checkboxes = document.querySelectorAll('.style_selections input[type="checkbox"]');
        for (const cb of checkboxes) {
            const label = cb.nextElementSibling;
            if (label && label.textContent.trim() === styleName) {
                cb.click();
                break;
            }
        }
    });
    document.addEventListener('contextmenu', function(e) {
        const modelDropdown = e.target.closest(
            '#model_dropdown_base, #model_dropdown_refiner, [id^="lora_dropdown"]'
        );
        if (modelDropdown) {
            e.preventDefault();
            e.stopPropagation();

            let buttonId;
            switch(modelDropdown.id) {
                case 'model_dropdown_base':
                    buttonId = 'base_preview_btn';
                    break;
                case 'model_dropdown_refiner':
                    buttonId = 'refiner_preview_btn';
                    break;
                default:
                    if (modelDropdown.id.startsWith('lora_dropdown')) {
                        const indexMatch = modelDropdown.id.match(/lora_dropdown_(\d+)$/);
                        if (indexMatch) {
                            buttonId = `lora_preview_btn_${indexMatch[1]}`;
                        } else {
                            console.warn('LORA ID格式异常:', modelDropdown.id);
                        }
                    }
                }

            const btn = gradioApp().querySelector(`#${buttonId}`);
            if (btn) {
                const event = new MouseEvent('click', { bubbles: true });
                btn.dispatchEvent(event);
            }
            return;
        }
        const styleItem = e.target.closest('.style_item');
        const styleLabel = e.target.closest('.style_selections label');

        if (styleItem) {
            const container = e.target.closest('.style_item');
            if (!container) return;

            e.preventDefault();

            if (e.button !== 2) return;

            const dataInput = container.querySelector('.style_data_input textarea');
            if (!dataInput) return;

            const styleButton = container.querySelector('.style-button');
            if (styleButton) {
                styleButton.classList.add('disable-hover');
                setTimeout(() => {
                    styleButton.classList.remove('disable-hover');
                }, 3000);
            }

            handleStyleData(dataInput, e);

        } else if (styleLabel) {
            e.preventDefault();
            const styleName = styleLabel.querySelector('span')?.getAttribute('data-original-text')?.trim()
            || styleLabel.querySelector('span')?.textContent?.trim();
            if (!styleName) {
                console.error('无法获取样式名称:', styleLabel);
                return;
            }

            const escapedName = styleName.replace(/ /g, '\\ ');
            const dataInput = gradioApp().querySelector(`#style_data_${escapedName} textarea`);

            if (!dataInput?.value) {
                console.error('数据输入框未找到:', {
                    styleName,
                    selector: `#style_data_${escapedName} textarea`,
                    element: dataInput
                });
                return;
            }
            handleStyleData(dataInput, e);
        }
    });
}
const style = document.createElement('style');
style.textContent = `
@keyframes flash {
    0% { box-shadow: inset 0 0 0 3px rgba(0, 150, 255, 0.5); }
    50% { box-shadow: inset 0 0 0 4px rgba(0, 150, 255, 0.75); }
    100% { box-shadow: inset 0 0 0 0px rgba(0, 150, 255, 0); }
}
.flash-border {
    animation: flash 3s ease-in-out;
    position: relative;
    z-index: 3;
    pointer-events: none;
    overflow: visible !important;
}.style-button.disable-hover {
    transform: scale(1) !important;
    transition: none !important;
}`;
document.head.appendChild(style);
function handleStyleData(dataInput, e) {
    const targetElement = e.target.closest('.style_item, .style_selections label');
    if (targetElement) {
        targetElement.classList.add('flash-border');
        setTimeout(() => {
            targetElement.classList.remove('flash-border');
        }, 3000);
    }
    try {
        const styleData = JSON.parse(dataInput.value || '{}');
        const positivePrompt = gradioApp().querySelector('#positive_prompt textarea');
        const negativePrompt = gradioApp().querySelector('#negative_prompt textarea');

        if (styleData.prompt && positivePrompt) {
            const currentPrompt = (positivePrompt.value || '').trim();
            positivePrompt.value = styleData.prompt.replace('{prompt}', currentPrompt);
            positivePrompt.dispatchEvent(new Event('input', { bubbles: true }));
        }

        if (styleData.negative_prompt && negativePrompt) {
            negativePrompt.value += styleData.negative_prompt;
            negativePrompt.dispatchEvent(new Event('input', { bubbles: true }));
        }

        e.stopPropagation();
    } catch (error) {
        console.error('Error handling style click:', error);
    }
}

/**
 * checks that a UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
    if (el === document) {
        return true;
    }

    const computedStyle = getComputedStyle(el);
    const isVisible = computedStyle.display !== 'none';

    if (!isVisible) return false;
    return uiElementIsVisible(el.parentNode);
}

function uiElementInSight(el) {
    const clRect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;

    return isOnScreen;
}

function playNotification() {
    gradioApp().querySelector('#audio_notification audio')?.play();
}

function set_theme(theme) {
    var gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function htmlDecode(input) {
  var doc = new DOMParser().parseFromString(input, "text/html");
  return doc.documentElement.textContent;
}

(function() {
    let previewOverlay = null;
    const modelPreviewCache = {};
    function createPreviewOverlay() {
        if (!previewOverlay) {
            previewOverlay = document.createElement('div');
            Object.assign(previewOverlay.style, {
                position: 'fixed',
                pointerEvents: 'none',
                zIndex: 9999999999,
                maxWidth: '320px',
                borderRadius: '8px',
                overflow: 'hidden',
                boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                transition: 'opacity 0.2s',
                opacity: 0
            });
            gradioApp().appendChild(previewOverlay);
        }
    }

    function initModelPreviews() {
        createPreviewOverlay();

        const baseDropdown = gradioApp().getElementById('model_dropdown_base');
        if (baseDropdown) initDropdownPreview(baseDropdown, 'checkpoints');

        const refinerDropdown = gradioApp().getElementById('model_dropdown_refiner');
        if (refinerDropdown) initDropdownPreview(refinerDropdown, 'checkpoints');

        const loraDropdowns = gradioApp().querySelectorAll('[id^="lora_dropdown"], [id^="scene_lora_dropdown"]');
        loraDropdowns.forEach(dropdown => initDropdownPreview(dropdown, 'loras'));
    }
    autoHideTimer = 3000;
    function initDropdownPreview(dropdown, folder) {
        if (!modelPreviewCache[folder]) modelPreviewCache[folder] = {};
        const folderCache = modelPreviewCache[folder];
        const pathMeta = document.querySelector(`meta[name='${folder}-paths']`).getAttribute("content");
        const basePaths = pathMeta.split(',').map(p => p.split('?')[0].replace(/\\/g, '/'));
        const handleMouseOver = (e) => {
            clearTimeout(autoHideTimer);
            const option = e.target.closest('li.item[role="button"]');
            if (!option) return;

            const modelPath = option.dataset.value
            .replace(/\.[^/.]+$/, "")
            .replace(/\\/g, '/');

        if (modelPath.toLowerCase() === 'none') {
            previewOverlay.style.opacity = '0';
            return;
        }
        // 检查缓存
        const cached = folderCache[modelPath];
        if (cached) {
            const testImg = new Image();
            testImg.style.maxWidth = '150px';
            testImg.style.display = 'block';
            testImg.onload = () => {
                previewOverlay.innerHTML = '';
                previewOverlay.appendChild(testImg);
                previewOverlay.style.opacity = '1';
            };
            testImg.src = cached.url;
            return;
        }
        const img = new Image();
            img.style.maxWidth = '150px';
            img.style.display = 'block';
            const extensions = ['jpg', 'jpeg', 'png', 'webp'];
            let currentPath = 0;
            let currentExtension = 0;

            const tryNextPath = () => {
                if (currentPath >= basePaths.length) {
                    const fallbackImg = new Image();
                    fallbackImg.onload = () => {
                        previewOverlay.innerHTML = '';
                        previewOverlay.appendChild(fallbackImg);
                        previewOverlay.style.opacity = '1';
                    };
                    fallbackImg.src = '/file=presets/samples/noimage.jpg';
                    return;
                }

                const tryNextFormat = () => {
                    if (currentExtension >= extensions.length) {
                        currentPath++;
                        currentExtension = 0;
                        tryNextPath();
                        return;
                    }

                    const testImg = new Image();
                    testImg.onerror = () => {
                        currentExtension++;
                        tryNextFormat();
                    };
                    testImg.onload = () => {
                        folderCache[modelPath] = {
                            url: testImg.src,
                            timestamp: Date.now()
                        };
                        img.src = testImg.src;
                        previewOverlay.innerHTML = '';
                        previewOverlay.appendChild(img);
                        previewOverlay.style.opacity = '1';
                    };
                    testImg.src = `${basePaths[currentPath]}/${encodeURIComponent(modelPath)}.${extensions[currentExtension]}?${Date.now()}`;
                };

                tryNextFormat();
            };

            tryNextPath();

            img.onload = () => {
                previewOverlay.innerHTML = '';
                previewOverlay.appendChild(img);
                previewOverlay.style.opacity = '1';
            };
        };

        const handleMouseMove = (e) => {
            previewOverlay.style.left = `${e.clientX + 15}px`;
            previewOverlay.style.top = `${e.clientY + 15}px`;
        };

        const handleMouseOut = (e) => {
            if (!e.relatedTarget?.closest('li.item')) {
                previewOverlay.style.opacity = '0';
                clearTimeout(autoHideTimer);
            }
            autoHideTimer = setTimeout(() => {
                previewOverlay.style.opacity = '0';
            }, 3000);
        };

        dropdown.addEventListener('mouseover', handleMouseOver);
        dropdown.addEventListener('mousemove', handleMouseMove);
        dropdown.addEventListener('mouseout', handleMouseOut);
    }

    document.addEventListener('DOMContentLoaded', () => {
        const observer = new MutationObserver((mutations) => {
            if (document.getElementById('model_dropdown_base') || document.getElementById('lora_dropdown')) {
                initModelPreviews();
                observer.disconnect();
            }
        });
        observer.observe(gradioApp(), { childList: true, subtree: true });
    });
})();

window.highlightModelDropdown = function(activeTarget) {
    const baseDropdown = document.getElementById('model_dropdown_base');
    const refinerDropdown = document.getElementById('model_dropdown_refiner');

    const dropdowns = [baseDropdown, refinerDropdown].filter(Boolean);

    dropdowns.forEach(dropdown => {
        dropdown.style.removeProperty('--block-border-width');
        dropdown.style.removeProperty('--border-color-primary');
    });

    let activeDropdown;
    if (activeTarget === 'base' && baseDropdown) {
        activeDropdown = baseDropdown;
    } else if (activeTarget === 'refiner' && refinerDropdown) {
        activeDropdown = refinerDropdown;
    }

    if (activeDropdown) {
        activeDropdown.style.setProperty('--block-border-width', '2px');
        activeDropdown.style.setProperty('--border-color-primary', '#005CC8');
    }
}
function initTranslationPreview() {
    window.addEventListener('error', (e) => {
        console.error('全局JS错误:', e.message, e.filename, e.lineno);
    });

    let retryCount = 0;
    function tryInit() {
        const accordionElement = gradioApp().getElementById('translation_preview_accordion');
        if (!accordionElement) {
            retryCount++;
            if (retryCount < 10) {
                setTimeout(tryInit, 1000);
            } else {
                console.error('未找到翻译预览面板');
            }
            return;
        }

        const accordionHeader = accordionElement.querySelector('.label-wrap');

        if (!accordionHeader) {
            console.error('未找到翻译预览标题');
            return;
        }

        accordionHeader.removeEventListener('click', handleHeaderClick);
        accordionHeader.addEventListener('click', handleHeaderClick);
    }

    function handleHeaderClick() {
        const translationPreviewOpenContainer = gradioApp().getElementById('translation_preview_open');
        const translationPreviewOpenCheckbox = translationPreviewOpenContainer?.querySelector('input[type="checkbox"]');
        const promptContainer = gradioApp().getElementById('positive_prompt');
        const promptInput = promptContainer?.querySelector('textarea, input');

        if (!translationPreviewOpenCheckbox || !promptInput) {
            console.error('组件未找到！');
            return;
        }

        const isOpen = translationPreviewOpenCheckbox.checked;
        const promptValue = promptInput.value.trim();

        translationPreviewOpenCheckbox.checked = !isOpen;
        const changeEvent = new Event('change');
        translationPreviewOpenCheckbox.dispatchEvent(changeEvent);

        const triggerBtn = gradioApp().getElementById('trigger_translation_btn');
        if (triggerBtn) {
            triggerBtn.click();
        } else {
            console.error('未找到触发翻译的按钮');
        }
    }

    const observer = new MutationObserver(() => {
        if (gradioApp().getElementById('translation_preview_accordion')) {
            observer.disconnect();
            tryInit();
        }
    });
    observer.observe(gradioApp(), { childList: true, subtree: true });
}

onUiLoaded(() => {
    initTranslationPreview();
});

function setupAutoTranslate() {
    const promptContainer = gradioApp().getElementById('positive_prompt');
    const promptInput = promptContainer?.querySelector('textarea, input');
    const translateBtn = gradioApp().getElementById('trigger_translation_btn');

    if (promptInput && translateBtn) {
        let timer = null;
        let lastContent = '';

        const handleInput = () => {
            const currentContent = promptInput.value;
            if (currentContent === lastContent) return;

            if (timer) clearTimeout(timer);
            timer = setTimeout(() => {
                translateBtn.click();
                lastContent = currentContent;
            }, 1000);
        };

        let lastManualCheck = promptInput.value;
        setInterval(() => {
            if (promptInput.value !== lastManualCheck) {
                lastManualCheck = promptInput.value;
                handleInput();
            }
        }, 1500);

        promptInput.addEventListener('input', handleInput);
    }
}

onUiLoaded(setupAutoTranslate);

function setupSam3AutoTranslate() {
    let retryCount = 0;

    function tryBind() {
        const promptContainer = gradioApp().getElementById('sam3_prompt_text');
        const promptInput = promptContainer?.querySelector('textarea, input');
        const translateBtn = gradioApp().getElementById('sam3_trigger_translate_btn');

        if (!promptInput || !translateBtn) {
            return false;
        }

        let timer = null;
        let lastContent = '';

        const trigger = () => {
            const currentContent = promptInput.value;
            if (currentContent === lastContent) return;
            translateBtn.click();
            lastContent = currentContent;
        };

        const handleInput = () => {
            const currentContent = promptInput.value;
            if (currentContent === lastContent) return;
            if (timer) clearTimeout(timer);
            timer = setTimeout(() => {
                trigger();
            }, 500);
        };

        const handleBlur = () => {
            if (timer) {
                clearTimeout(timer);
                timer = null;
            }
            trigger();
        };

        let lastManualCheck = promptInput.value;
        setInterval(() => {
            if (promptInput.value !== lastManualCheck) {
                lastManualCheck = promptInput.value;
                handleInput();
            }
        }, 1500);

        promptInput.addEventListener('input', handleInput);
        promptInput.addEventListener('blur', handleBlur);
        return true;
    }

    const retryTimer = setInterval(() => {
        retryCount += 1;
        if (tryBind()) {
            clearInterval(retryTimer);
            return;
        }
        if (retryCount >= 20) {
            clearInterval(retryTimer);
        }
    }, 1500);
}

onUiLoaded(setupSam3AutoTranslate);

function _ro_getSliderValue(elemId) {
    const root = gradioApp().getElementById(elemId);
    if (!root) return null;
    const numberInput = root.querySelector('input[type="number"]');
    const rangeInput = root.querySelector('input[type="range"]');
    const raw = (numberInput?.value ?? rangeInput?.value);
    const v = parseInt(raw, 10);
    return Number.isFinite(v) ? v : null;
}

function _ro_setSliderValue(elemId, value) {
    const root = gradioApp().getElementById(elemId);
    if (!root) return false;
    const numberInput = root.querySelector('input[type="number"]');
    const rangeInput = root.querySelector('input[type="range"]');
    const v = String(value);
    if (rangeInput) {
        rangeInput.value = v;
        rangeInput.dispatchEvent(new Event('input', { bubbles: true }));
        rangeInput.dispatchEvent(new Event('change', { bubbles: true }));
    }
    if (numberInput) {
        numberInput.value = v;
        numberInput.dispatchEvent(new Event('input', { bubbles: true }));
        numberInput.dispatchEvent(new Event('change', { bubbles: true }));
    }
    return true;
}

function _ro_gcd(a, b) {
    a = Math.abs(a);
    b = Math.abs(b);
    while (b) {
        const t = b;
        b = a % b;
        a = t;
    }
    return a || 1;
}

function initResolutionOverrideWidget() {
    const widget = gradioApp().getElementById('resolution_override_widget');
    if (!widget) return;
    if (widget.dataset.initialized === '1') {
        if (typeof widget.__ro_sync === 'function') {
            widget.__ro_sync();
        }
        return;
    }

    const pad = widget.querySelector('[data-role="pad"]');
    const rect = widget.querySelector('[data-role="rect"]');
    const wInput = widget.querySelector('[data-role="winput"]');
    const hInput = widget.querySelector('[data-role="hinput"]');
    const btnUp = widget.querySelector('[data-role="scale_up"]');
    const btnDown = widget.querySelector('[data-role="scale_down"]');
    if (!pad || !rect || !wInput || !hInput) return;

    const syncFromSliders = () => {
        const w = _ro_getSliderValue('overwrite_width');
        const h = _ro_getSliderValue('overwrite_height');

        const enabled = (w != null && h != null && w > 0 && h > 0);
        if (!enabled) {
            wInput.value = '-1';
            hInput.value = '-1';
            rect.style.width = Math.round(pad.clientWidth * 0.5) + 'px';
            rect.style.height = Math.round(pad.clientHeight * 0.5) + 'px';
            rect.style.left = '0px';
            rect.style.top = '0px';
            return;
        }

        wInput.value = String(w);
        hInput.value = String(h);

        const pw = Math.max(1, pad.clientWidth);
        const ph = Math.max(1, pad.clientHeight);
        const maxSide = 2048;
        const dw = Math.max(6, Math.round((w / maxSide) * pw));
        const dh = Math.max(6, Math.round((h / maxSide) * ph));
        rect.style.width = dw + 'px';
        rect.style.height = dh + 'px';
        rect.style.left = '0px';
        rect.style.top = '0px';
    };

    const setFromPointer = (clientX, clientY) => {
        const r = pad.getBoundingClientRect();
        const x = Math.max(0, Math.min(r.width, clientX - r.left));
        const y = Math.max(0, Math.min(r.height, clientY - r.top));
        const maxSide = 2048;
        const minSide = 512;
        const step = 8;

        let w = Math.round((x / Math.max(1, r.width)) * maxSide);
        let h = Math.round((y / Math.max(1, r.height)) * maxSide);
        w = Math.max(minSide, Math.min(maxSide, w));
        h = Math.max(minSide, Math.min(maxSide, h));
        w = Math.round(w / step) * step;
        h = Math.round(h / step) * step;
        if (w < minSide) w = minSide;
        if (h < minSide) h = minSide;

        _ro_setSliderValue('overwrite_width', w);
        _ro_setSliderValue('overwrite_height', h);
        syncFromSliders();
    };

    const disable = () => {
        _ro_setSliderValue('overwrite_width', -1);
        _ro_setSliderValue('overwrite_height', -1);
        syncFromSliders();
    };

    const commitManualInput = () => {
        const w = parseInt(wInput.value, 10);
        const h = parseInt(hInput.value, 10);
        const clamp = (v) => {
            if (!Number.isFinite(v)) return -1;
            if (v <= 0) return -1;
            if (v < 512) return 512;
            if (v > 2048) return 2048;
            return Math.round(v / 8) * 8;
        };
        const wv = clamp(w);
        const hv = clamp(h);
        _ro_setSliderValue('overwrite_width', wv);
        _ro_setSliderValue('overwrite_height', hv);
        syncFromSliders();
    };

    wInput.addEventListener('change', commitManualInput);
    hInput.addEventListener('change', commitManualInput);

    const scaleFromCurrent = (factor) => {
        const step = 8;
        const minSide = 512;
        const maxSide = 2048;
        const clamp = (v) => {
            if (!Number.isFinite(v)) return minSide;
            v = Math.round(v / step) * step;
            if (v < minSide) v = minSide;
            if (v > maxSide) v = maxSide;
            return v;
        };

        let w = _ro_getSliderValue('overwrite_width');
        let h = _ro_getSliderValue('overwrite_height');
        if (!(w != null && h != null && w > 0 && h > 0)) {
            w = minSide;
            h = minSide;
        }
        const nw = clamp(Math.round(w * factor));
        const nh = clamp(Math.round(h * factor));
        _ro_setSliderValue('overwrite_width', nw);
        _ro_setSliderValue('overwrite_height', nh);
        syncFromSliders();
    };

    btnUp?.addEventListener('click', () => scaleFromCurrent(1.1));
    btnDown?.addEventListener('click', () => scaleFromCurrent(0.9));

    const widthRoot = gradioApp().getElementById('overwrite_width');
    const heightRoot = gradioApp().getElementById('overwrite_height');
    for (const root of [widthRoot, heightRoot]) {
        if (!root) continue;
        const rangeInput = root.querySelector('input[type="range"]');
        const numberInput = root.querySelector('input[type="number"]');
        rangeInput?.addEventListener('input', syncFromSliders);
        numberInput?.addEventListener('input', syncFromSliders);
        rangeInput?.addEventListener('change', syncFromSliders);
        numberInput?.addEventListener('change', syncFromSliders);
    }

    let dragging = false;
    const onMove = (e) => {
        if (!dragging) return;
        if (e.touches && e.touches.length) {
            setFromPointer(e.touches[0].clientX, e.touches[0].clientY);
        } else {
            setFromPointer(e.clientX, e.clientY);
        }
    };
    const onUp = () => {
        dragging = false;
        window.removeEventListener('pointermove', onMove, true);
        window.removeEventListener('pointerup', onUp, true);
        window.removeEventListener('touchmove', onMove, true);
        window.removeEventListener('touchend', onUp, true);
    };

    const onDown = (e) => {
        dragging = true;
        if (e.touches && e.touches.length) {
            setFromPointer(e.touches[0].clientX, e.touches[0].clientY);
        } else {
            setFromPointer(e.clientX, e.clientY);
        }
        window.addEventListener('pointermove', onMove, true);
        window.addEventListener('pointerup', onUp, true);
        window.addEventListener('touchmove', onMove, true);
        window.addEventListener('touchend', onUp, true);
    };

    pad.addEventListener('pointerdown', onDown, { passive: true });
    pad.addEventListener('touchstart', onDown, { passive: true });
    pad.addEventListener('dblclick', disable);

    const requestSync = () => syncFromSliders();
    if ('ResizeObserver' in window) {
        const ro = new ResizeObserver(requestSync);
        ro.observe(pad);
        widget.__ro_resize_observer = ro;
    }
    window.addEventListener('resize', requestSync, { passive: true });

    widget.dataset.initialized = '1';
    widget.__ro_sync = syncFromSliders;
    syncFromSliders();
}

onUiLoaded(initResolutionOverrideWidget);
onAfterUiUpdate(initResolutionOverrideWidget);

function initPersonalWildcardsPopup() {
    const app = (typeof gradioApp === 'function') ? gradioApp() : null;
    if (!app) return;

    const modal = app.getElementById('user_personal_wildcards_modal');
    const content = app.getElementById('user_personal_wildcards_modal_content');
    const handle = app.getElementById('user_personal_wildcards_modal_handle');
    if (!modal || !content || !handle) return;
    if (content.dataset.pw_inited === '1') return;
    content.dataset.pw_inited = '1';

    const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

    const placeDefault = () => {
        const w = content.getBoundingClientRect().width || 970;
        const h = content.getBoundingClientRect().height || 520;
        const margin = 10;
        let top = 220;
        let left = ((window.innerWidth - w) / 2) - 260;
        top = clamp(top, margin, window.innerHeight - margin - h);
        left = clamp(left, margin, window.innerWidth - margin - w);
        content.style.top = `${top}px`;
        content.style.left = `${left}px`;
        content.style.right = 'auto';
        content.style.bottom = 'auto';
    };

    const onShow = () => {
        if (modal.style.display === 'none') return;
        if (!content.style.left && !content.style.top) {
            placeDefault();
        }
    };

    const observer = new MutationObserver(onShow);
    observer.observe(modal, { attributes: true, attributeFilter: ['style'] });
    onShow();

    let dragging = false;
    let offsetX = 0;
    let offsetY = 0;

    const onMove = (e) => {
        if (!dragging) return;
        const x = e.clientX ?? 0;
        const y = e.clientY ?? 0;
        const rect = content.getBoundingClientRect();
        const margin = 10;
        const nextLeft = clamp(x - offsetX, margin, window.innerWidth - margin - rect.width);
        const nextTop = clamp(y - offsetY, margin, window.innerHeight - margin - rect.height);
        content.style.left = `${nextLeft}px`;
        content.style.top = `${nextTop}px`;
    };

    const onUp = () => {
        dragging = false;
        window.removeEventListener('pointermove', onMove, true);
        window.removeEventListener('pointerup', onUp, true);
    };

    const onDown = (e) => {
        if (e.button !== 0) return;
        if (modal.style.display === 'none') return;
        const rect = content.getBoundingClientRect();
        dragging = true;
        offsetX = e.clientX - rect.left;
        offsetY = e.clientY - rect.top;
        window.addEventListener('pointermove', onMove, true);
        window.addEventListener('pointerup', onUp, true);
        e.preventDefault();
    };

    handle.addEventListener('pointerdown', onDown, { passive: false });
}

onUiLoaded(initPersonalWildcardsPopup);
onAfterUiUpdate(initPersonalWildcardsPopup);
