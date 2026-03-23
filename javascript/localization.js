var re_num = /^[.\d]+$/;

window.globalAutoAddLoraTriggerWord = function(triggerWordElemId, modelElemId, directTriggerWord) {
    try {
        function getGradioRoot() {
            const elems = document.getElementsByTagName('gradio-app');
            const elem = elems.length == 0 ? document : elems[0];
            return elem.shadowRoot ? elem.shadowRoot : elem;
        }

        function addTriggerWordToPrompt(triggerWord) {
            const root = getGradioRoot();
            const positivePrompt = root.querySelector('#positive_prompt textarea');
            if (positivePrompt) {
                const currentText = positivePrompt.value.trim();
                const separator = currentText ? ', ' : '';
                positivePrompt.value = currentText + separator + triggerWord;
                positivePrompt.dispatchEvent(new Event('input', { bubbles: true }));
                console.log('Added trigger word to prompt:', triggerWord);
            } else {
                console.error('Positive prompt textarea not found');
            }
        }

        if (typeof directTriggerWord === 'string') {
            addTriggerWordToPrompt(directTriggerWord);
        } else {
            const root = getGradioRoot();
            const triggerWordElem = root.querySelector(`#${triggerWordElemId} textarea`);
            const modelElem = root.querySelector(`#${modelElemId}`);

            if (modelElem && modelElem.value !== 'None' && triggerWordElem && triggerWordElem.value) {
                addTriggerWordToPrompt(triggerWordElem.value);
            }
        }
    } catch (error) {
        console.error('Error in globalAutoAddLoraTriggerWord:', error);
    }
};

// Alias for backward compatibility if needed, but we will update webui.py
window.autoAddLoraTriggerWord = window.globalAutoAddLoraTriggerWord;

var original_lines = {};
var translated_lines = {};
var reverseLocalization = null;

function getReverseLocalization() {
    if (reverseLocalization === null && window.localization) {
        reverseLocalization = {};
        for (const [en, cn] of Object.entries(window.localization)) {
            reverseLocalization[cn] = en;
        }
    }
    return reverseLocalization;
}

const browser={
    device: function(){
           var u = navigator.userAgent;
           return {
                is_mobile: !!u.match(/AppleWebKit.*Mobile.*/),
                is_pc: (u.indexOf('Macintosh') > -1 || u.indexOf('Windows NT') > -1),
		is_wx_mini: (u.indexOf('miniProgram') > -1),
            };
         }(),
    language: (navigator.browserLanguage || navigator.language).toLowerCase()
}

function hasLocalization() {
    return window.localization && Object.keys(window.localization).length > 0;
}

function textNodesUnder(el) {
    var n, a = [], walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
    while ((n = walk.nextNode())) a.push(n);
    return a;
}

function canBeTranslated(node, text) {
    if (!text) return false;
    if (!node.parentElement) return false;
    var parentType = node.parentElement.nodeName;
    if (parentType == 'SCRIPT' || parentType == 'STYLE' || parentType == 'TEXTAREA') return false;
    if (re_num.test(text)) return false;
    return true;
}

function getTranslation(text) {
    if (!text) return undefined;

    if (translated_lines[text] === undefined) {
        original_lines[text] = 1;
    }

    var tl = localization[text];
    if (tl !== undefined) {
        translated_lines[tl] = 1;
    }

    return tl;
}

function processTextNode(node) {
    var text = node.textContent.trim();
    if (!canBeTranslated(node, text)) return;

    var tl = getTranslation(text);
    let originalText = text;

    if (tl === undefined) {
        const rev = getReverseLocalization();
        if (rev && rev[text]) {
            originalText = rev[text];
        } else {
            tl = text;
        }
    }

    if (tl !== undefined) {
        if (node.textContent.trim() !== tl) {
            node.textContent = tl;
        }
    }

    if (originalText && node.parentElement) {
        let p = node.parentElement;

        if ((p.nodeName === 'SPAN' || p.nodeName === 'LABEL') && p.getAttribute("data-original-text") !== originalText) {
             p.setAttribute("data-original-text", originalText);
        }

        let label = p.closest('label');
        if (label && label.getAttribute("data-original-text") !== originalText) {
             label.setAttribute("data-original-text", originalText);
        }

        let closestSpan = p.closest('span');
        if (closestSpan && closestSpan.getAttribute("data-original-text") !== originalText) {
            closestSpan.setAttribute("data-original-text", originalText);
        }

        let galleryDiv = p.closest('div.gallery');
        if (galleryDiv && galleryDiv.getAttribute("data-original-text") !== originalText) {
             galleryDiv.setAttribute("data-original-text", originalText);
        }
    }
}

function processNode(node) {
    if (node.nodeType == 3) {
        processTextNode(node);
        return;
    }

    if (node.title) {
        let tl = getTranslation(node.title);
        if (tl !== undefined) {
            node.title = tl;
        }
    }

    if (node.placeholder) {
        let tl = getTranslation(node.placeholder);
        if (tl !== undefined) {
            node.placeholder = tl;
        }
    }

    textNodesUnder(node).forEach(function(node) {
	processTextNode(node);
    });
}

function refresh_style_localization() {
    processNode(document.querySelector('.style_selections'));
}

let styleGridOriginalElements = [];
let styleSelectionsOriginalElements = [];
let styleGridHandlersAttached = false;
let isHandlingClick = false;
let lastKnownGoodStyles = new Set();
let lastAvailableStyleSet = new Set(); // 记录上一次看到的可用风格集合

function init_style_grid_handlers() {
    if (styleGridHandlersAttached) {
        const currentContainer = document.querySelector(".style_grid");
        if (currentContainer && currentContainer.getAttribute('data-handlers-attached') !== 'true') {
            styleGridHandlersAttached = false;
        } else {
            return;
        }
    }
    const container = document.querySelector(".style_grid");
    if (!container) return;

    container.setAttribute('data-handlers-attached', 'true');

    container.addEventListener('click', (e) => {
        const btn = e.target.closest('.style-button');
        if (!btn) return;

        const styleName = btn.getAttribute('data-style-name');
        if (!styleName) return;

        const selections = document.querySelector('.style_selections');
        if (!selections) return;

        const labels = selections.querySelectorAll('label');
        let found = false;
        const cleanStyle = styleName.toLowerCase().replace(/[- _]/g, '');

        const rev = getReverseLocalization();

        for (const label of labels) {
            const input = label.querySelector('input[type="checkbox"]');
            if (!input) continue;

            const labelText = label.textContent.trim();

            let identity = (input.value && input.value !== 'on') ? input.value : null;
            
            if (!identity) {
                identity = label.getAttribute('data-original-text') || 
                           (label.querySelector('span') ? label.querySelector('span').getAttribute('data-original-text') : null) ||
                           (rev ? rev[labelText] : null) ||
                           labelText;
            }

            if (!identity) continue;
            const cleanIdentity = identity.toLowerCase().replace(/[- _]/g, '');

            if (cleanIdentity === cleanStyle) {
                const willBeChecked = !input.checked;
                isHandlingClick = true;

                if (willBeChecked) {
                    lastKnownGoodStyles.add(cleanStyle);
                } else {
                    lastKnownGoodStyles.delete(cleanStyle);
                }

                input.click();

                window.styleExpectedState = {
                    name: cleanStyle,
                    state: willBeChecked,
                    timestamp: Date.now()
                };

                sync_style_grid_state();

                setTimeout(() => {
                    if (isHandlingClick) {
                        isHandlingClick = false;
                        sync_style_grid_state();
                    }
                }, 2000);

                found = true;
                break;
            }
        }
        if (!found) {
            console.warn(`[StyleClick] Could not find checkbox matching style: ${styleName}`);
        }
    });

    container.addEventListener('contextmenu', (e) => {
        const btn = e.target.closest('.style-button');
        if (!btn) return;
        e.preventDefault();
        const styleDataRaw = btn.parentElement.getAttribute('data-style-data');
        if (!styleDataRaw) return;
        try {
            const styleData = JSON.parse(styleDataRaw);
            const prompt = styleData.prompt || '';
            const negativePrompt = styleData.negative_prompt || '';
            const promptTextarea = document.querySelector('#positive_prompt textarea, #positive_prompt [data-testid="textbox"]');
            const negativePromptTextarea = document.querySelector('#negative_prompt textarea, #negative_prompt [data-testid="textbox"]');
            const applyTemplate = (template, userText) => {
                const t = (template || '').trim();
                const u = (userText || '').trim();
                if (!t) return u;
                if (t.includes('{prompt}')) {
                    const replaced = t.replace(/\{prompt\}/gi, u);
                    return replaced.replace(/\s+\.\s+/g, '. ').replace(/\s{2,}/g, ' ').trim();
                }
                return u ? `${u}, ${t}` : t;
            };
            if (promptTextarea && (prompt || promptTextarea.value)) {
                const current = promptTextarea.value;
                promptTextarea.value = applyTemplate(prompt, current);
                promptTextarea.dispatchEvent(new Event('input', { bubbles: true }));
            }
            if (negativePromptTextarea && (negativePrompt || negativePromptTextarea.value)) {
                const currentNeg = negativePromptTextarea.value;
                negativePromptTextarea.value = applyTemplate(negativePrompt, currentNeg);
                negativePromptTextarea.dispatchEvent(new Event('input', { bubbles: true }));
            }
        } catch (err) {}
    });

    styleGridHandlersAttached = true;

    ['mousedown', 'click'].forEach(eventType => {
        document.addEventListener(eventType, (e) => {
            const target = e.target;

            let targetBtn = null;
            if (target.classList.contains('bar_button')) {
                targetBtn = target;
            } else {
                targetBtn = target.closest('button');
                if (targetBtn && !targetBtn.closest('.preset_store') && !targetBtn.classList.contains('bar_button')) {
                    targetBtn = null;
                }
            }

            if (targetBtn) {
                let isAlreadyActive = false;
                if (targetBtn) {
                    const bg = targetBtn.style.background || '';
                    const color = targetBtn.style.color || '';

                    if (color === 'white' || bg.includes('secondary-200') || (targetBtn.closest('.preset_store') && targetBtn.classList.contains('primary'))) {
                         isAlreadyActive = true;
                    }
                }

                if (isAlreadyActive) {
                    e.stopImmediatePropagation();
                    e.stopPropagation();
                    e.preventDefault();
                    return;
                }

                if (eventType === 'mousedown') {
                    if (targetBtn.closest('.preset_store') || targetBtn.classList.contains('bar_button')) {
                         isHandlingClick = false;
                         lastKnownGoodStyles.clear();
                         window.styleExpectedState = null; // Clear any pending style enforcement

                         // Set a global lock to prevent style syncing from interfering with preset loading
                         window.presetLoadingLock = Date.now();

                         return;
                    }

                    isHandlingClick = false;
                    lastKnownGoodStyles.clear();
                    lastAvailableStyleSet.clear();

                    const selections = document.querySelector('.style_selections');
                    if (selections) {
                        const inputs = selections.querySelectorAll('input[type="checkbox"]');
                        inputs.forEach(input => {
                            if (input.checked) {
                                input.checked = false;
                            }
                        });
                    }

                    sync_style_grid_state();
                }
            }
        }, true);
    });

    const selections = document.querySelector('.style_selections');
    if (selections) {
        const observer = new MutationObserver(() => { sync_style_grid_state(); });
        observer.observe(selections, { childList: true, subtree: true, attributes: true });
        sync_style_grid_state();
    }
}

function sync_style_grid_state() {
    const selections = document.querySelector('.style_selections');
    if (!selections) return;
    const container = document.querySelector(".style_grid");
    if (!container) return;

    const labels = selections.querySelectorAll('label');
    const currentAvailableStyles = new Set();
    const rev = getReverseLocalization();

    labels.forEach(label => {
        const input = label.querySelector('input[type="checkbox"]');
        if (!input) return;
        let identity = (input.value && input.value !== 'on') ? input.value : null;
        if (!identity) {
            const labelText = label.textContent.trim();
            identity = label.getAttribute('data-original-text') ||
                       (label.querySelector('span') ? label.querySelector('span').getAttribute('data-original-text') : null) ||
                       (rev ? rev[labelText] : null) ||
                       labelText;
        }
        if (identity) {
            currentAvailableStyles.add(identity.toLowerCase().replace(/[- _]/g, ''));
        }
    });

    if (lastAvailableStyleSet.size > 0) {
        let changed = false;
        if (currentAvailableStyles.size !== lastAvailableStyleSet.size) {
            changed = true;
        } else {
            for (let s of currentAvailableStyles) {
                if (!lastAvailableStyleSet.has(s)) {
                    changed = true;
                    break;
                }
            }
        }
        if (changed && isHandlingClick) {
            isHandlingClick = false;
        }
    }
    lastAvailableStyleSet = currentAvailableStyles;

    const selectedStyles = new Set();
    const currentDOMSelected = new Set();

    labels.forEach(label => {
        const input = label.querySelector('input[type="checkbox"]');
        if (!input) return;

        const labelText = label.textContent.trim();
        let identity = (input.value && input.value !== 'on') ? input.value : null;

        if (!identity) {
            identity = label.getAttribute('data-original-text') ||
                       (label.querySelector('span') ? label.querySelector('span').getAttribute('data-original-text') : null) ||
                       (rev ? rev[labelText] : null) ||
                       labelText;
        }

        if (identity) {
            const cleanText = identity.toLowerCase().replace(/[- _]/g, '');

            // Check lock first
            if (window.presetLoadingLock && Date.now() - window.presetLoadingLock < 500) {
                isHandlingClick = false;
                lastKnownGoodStyles = new Set(currentDOMSelected);
                // We still let the function run to update UI classes
            }
            // Enforce expected state if mismatch detected within 1s (Only if not locked)
            else if (window.styleExpectedState &&
                window.styleExpectedState.name === cleanText &&
                Date.now() - window.styleExpectedState.timestamp < 1000) {

                if (input.checked !== window.styleExpectedState.state) {
                    // console.warn(`[StyleSync] State mismatch for ${identity}! Expected ${window.styleExpectedState.state}, got ${input.checked}. Enforcing and notifying.`);
                    input.checked = window.styleExpectedState.state;
                    // Dispatch event to ensure Gradio frontend framework is aware of the change
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }

            if (isHandlingClick && (!window.presetLoadingLock || Date.now() - window.presetLoadingLock >= 500)) {
                if (input.checked) {
                    if (!lastKnownGoodStyles.has(cleanText)) {
                            lastKnownGoodStyles.add(cleanText);
                    }
                    selectedStyles.add(cleanText);
                } else {

                    if (lastKnownGoodStyles.has(cleanText)) {
                            lastKnownGoodStyles.delete(cleanText);
                    }
                }
            } else {
                if (input.checked) {
                    selectedStyles.add(cleanText);
                    currentDOMSelected.add(cleanText);
                }
            }
        }
    });

    if (!isHandlingClick) {
        lastKnownGoodStyles = new Set(currentDOMSelected);
    }

    const buttons = container.querySelectorAll('.style-button');
    buttons.forEach(btn => {
        const styleName = btn.getAttribute('data-style-name');
        if (!styleName) return;
        const cleanName = styleName.toLowerCase().replace(/[- _]/g, '');
        if (selectedStyles.has(cleanName)) {
            btn.classList.add('primary');
            btn.classList.remove('secondary');
        } else {
            btn.classList.remove('primary');
            btn.classList.add('secondary');
        }
    });

    const styleItems = container.querySelectorAll('.style_item');
    styleItems.forEach(item => {
        const btn = item.querySelector('button');
        const rawName = btn?.getAttribute('data-style-name');
        if (!rawName) return;
        const cleanName = rawName.toLowerCase().replace(/[- _]/g, '');
        const isSelected = selectedStyles.has(cleanName);

        if (isSelected) {
            btn.classList.add('primary');
            btn.classList.remove('secondary');
            item.style.order = 0;
        } else {
            btn.classList.remove('primary');
            btn.classList.add('secondary');
            item.style.order = 1;
        }
    });
}

function refresh_style_layout() {
    const start = performance.now();
    const gridContainer = document.querySelector(".style_grid");
    const selectionsContainer = document.querySelector('.style_selections');
    
    if (!gridContainer && !selectionsContainer) return;

    if (gridContainer) {
        init_style_grid_handlers();
    }

    if (!selectionsContainer) return;

    const labels = selectionsContainer.querySelectorAll('label');
    let selectedStylesClean = new Set();
    const rev = getReverseLocalization();

    labels.forEach(label => {
        const cb = label.querySelector('input[type="checkbox"]');
        if (!cb) return;

        const labelText = label.textContent.trim();
        let identity = (cb.value && cb.value !== 'on') ? cb.value : null;

        if (!identity) {
            identity = label.getAttribute('data-original-text') ||
                       (label.querySelector('span') ? label.querySelector('span').getAttribute('data-original-text') : null) ||
                       (rev ? rev[labelText] : null) ||
                       labelText;
        }

        if (identity) {
            const cleanText = identity.toLowerCase().replace(/[- _]/g, '');
            if (cb.checked) {
                selectedStylesClean.add(cleanText);
            }
        }
    });

    if (isHandlingClick) {
        lastKnownGoodStyles.forEach(s => selectedStylesClean.add(s));
    }

    const searchBar = gradioApp().querySelector('textarea[data-testid="textbox"][placeholder*="搜索风格"], textarea[data-testid="textbox"][placeholder*="search styles"]');
    const searchText = (searchBar?.value?.trim() || '').toLowerCase();
    const cleanSearchText = searchText.replace(/[- _]/g, '');

    if (gridContainer) {
        if (styleGridOriginalElements.length === 0) {
            styleGridOriginalElements = [...gridContainer.querySelectorAll('.style_item')];
        }

        styleGridOriginalElements.forEach((item) => {
            const btn = item.querySelector('button');
            const btnText = btn?.textContent.trim();
            const rawName = btn?.getAttribute('data-style-name') || btnText;
            if (!rawName) return;

            const cleanName = rawName.toLowerCase().replace(/[- _]/g, '');
            const isSelected = selectedStylesClean.has(cleanName);

            const matchesSearch = cleanName.includes(cleanSearchText) || 
                                  btnText.toLowerCase().includes(searchText);
            const isVisible = isSelected || matchesSearch;

            if (isVisible) {
                item.style.setProperty('display', 'block', 'important');
                item.style.order = isSelected ? 0 : 1;

                if (!btn.style.backgroundImage || btn.style.backgroundImage === 'none') {
                    const styleName = rawName.toLowerCase().replace(/ /g, '_').replace(/[^a-z0-9_]/g, '');
                    const defaultUrl = 'file=sdxl_styles/samples/default_style.jpg';
                    const candidateUrls = [
                        `file=sdxl_styles/samples/${styleName}.jpg`
                    ];

                    btn.style.backgroundImage = `url("${defaultUrl}")`;

                    let candidateIndex = 0;
                    const probe = new Image();
                    probe.onload = () => {
                        const url = candidateUrls[candidateIndex];
                        if (url) {
                            btn.style.backgroundImage = `url("${url}")`;
                        }
                    };
                    probe.onerror = () => {
                        candidateIndex += 1;
                        if (candidateIndex < candidateUrls.length) {
                            probe.src = candidateUrls[candidateIndex];
                        }
                    };
                    probe.src = candidateUrls[candidateIndex];
                }

                if (isSelected) {
                    btn.classList.add('primary');
                    btn.classList.remove('secondary');
                } else {
                    btn.classList.add('secondary');
                    btn.classList.remove('primary');
                }
            } else {
                item.style.setProperty('display', 'none', 'important');
            }
        });
    }

    const checkboxGroup = selectionsContainer.querySelector('.wrap[data-testid="checkbox-group"]');
    if (checkboxGroup) {
        const labels = checkboxGroup.querySelectorAll('label');
        labels.forEach((label) => {
            const labelText = label.textContent.trim();
            const identity = label.getAttribute('data-original-text') ||
                             (label.querySelector('span') ? label.querySelector('span').getAttribute('data-original-text') : null) ||
                             (rev ? rev[labelText] : null) ||
                             labelText;

            if (!identity) return;

            const cleanText = identity.toLowerCase().replace(/[- _]/g, '');
            const isSelected = selectedStylesClean.has(cleanText);
            
            const translatedText = label.textContent.trim().toLowerCase();
            
            const matchesSearch = cleanText.includes(cleanSearchText) || 
                                  translatedText.includes(searchText);
            const isVisible = isSelected || matchesSearch;

            if (isVisible) {
                label.style.setProperty('display', 'flex', 'important');
                label.style.order = isSelected ? 0 : 1;
            } else {
                label.style.setProperty('display', 'none', 'important');
            }
        });
    }
}


function refresh_scene_localization() {
    processNode(document.querySelector('.scene_aspect_ratio_selections'));
}

function refresh_aspect_ratios_label(value) {
    var label = document.querySelector('#aspect_ratios_accordion div span');
    var translation = getTranslation("Aspect Ratios");
    if (typeof translation == "undefined") {
        translation = "Aspect Ratios";
    }
    value = (value || "").split(",")[0];

    var multiplier = 1.0;
    try {
        var mRoot = document.getElementById('resolution_multiplier');
        var mNumberInput = mRoot ? mRoot.querySelector('input[type="number"]') : null;
        var mRangeInput = mRoot ? mRoot.querySelector('input[type="range"]') : null;
        var mRaw = (mNumberInput && mNumberInput.value) ? mNumberInput.value : (mRangeInput ? mRangeInput.value : null);
        var m = parseFloat(mRaw);
        if (Number.isFinite(m) && m > 0) {
            multiplier = m;
        }
    } catch (e) {}

    var baseW = null;
    var baseH = null;
    try {
        var owRoot = document.getElementById('overwrite_width');
        var ohRoot = document.getElementById('overwrite_height');
        var owInput = owRoot ? owRoot.querySelector('input[type="number"]') : null;
        var ohInput = ohRoot ? ohRoot.querySelector('input[type="number"]') : null;
        var ow = owInput ? parseInt(owInput.value, 10) : NaN;
        var oh = ohInput ? parseInt(ohInput.value, 10) : NaN;
        if (Number.isFinite(ow) && Number.isFinite(oh) && ow > 0 && oh > 0) {
            baseW = ow;
            baseH = oh;
        }
    } catch (e) {}

    if (baseW == null || baseH == null) {
        try {
            var m2 = String(value).replace('×', 'x').match(/(\d+)\D+(\d+)/);
            if (m2) {
                baseW = parseInt(m2[1], 10);
                baseH = parseInt(m2[2], 10);
            }
        } catch (e) {}
    }

    var suffix = "";
    if (baseW != null && baseH != null && multiplier > 1.0) {
        var readQuantizeStep = function() {
            try {
                var sRoot = document.getElementById('resolution_quantize_step');
                var sInput = sRoot ? sRoot.querySelector('input[type="number"]') : null;
                var raw = sInput ? parseInt(sInput.value, 10) : NaN;
                var step = Number.isFinite(raw) ? raw : 8;
                return [8, 16, 32, 64].includes(step) ? step : 8;
            } catch (e) {
                return 8;
            }
        };
        var quantizeByStep = function(v) {
            var step = readQuantizeStep();
            var q = Math.round(v / step) * step;
            if (!(q > 0)) q = step;
            return q;
        };
        var effW = quantizeByStep(baseW * multiplier);
        var effH = quantizeByStep(baseH * multiplier);
        if (Number.isFinite(effW) && Number.isFinite(effH) && effW > 0 && effH > 0) {
            suffix = " \u2192 " + effW + "\u00d7" + effH;
        }
    }

    label.textContent = translation + " - " + htmlDecode(value) + suffix;
}

function init_aspect_ratios_label_multiplier_binding() {
    try {
        var mRoot = document.getElementById('resolution_multiplier');
        if (!mRoot) return;
        if (mRoot.dataset.aspectRatiosBound === '1') return;
        mRoot.dataset.aspectRatiosBound = '1';

        var mNumberInput = mRoot.querySelector('input[type="number"]');
        var mRangeInput = mRoot.querySelector('input[type="range"]');
        var sRoot = document.getElementById('resolution_quantize_step');
        var sNumberInput = sRoot ? sRoot.querySelector('input[type="number"]') : null;
        var readAspectValue = function() {
            var root = document.getElementById('aspect_ratios_selection');
            if (!root) return "";
            var input = root.querySelector('input, textarea');
            return input ? (input.value || "") : "";
        };
        var refresh = function() {
            refresh_aspect_ratios_label(readAspectValue());
        };

        if (mNumberInput) {
            mNumberInput.addEventListener('input', refresh, { passive: true });
            mNumberInput.addEventListener('change', refresh, { passive: true });
        }
        if (mRangeInput) {
            mRangeInput.addEventListener('input', refresh, { passive: true });
            mRangeInput.addEventListener('change', refresh, { passive: true });
        }
        if (sNumberInput) {
            sNumberInput.addEventListener('input', refresh, { passive: true });
            sNumberInput.addEventListener('change', refresh, { passive: true });
        }
        refresh();
    } catch (e) {}
}

if (typeof onUiLoaded === 'function') {
    onUiLoaded(init_aspect_ratios_label_multiplier_binding);
}
if (typeof onAfterUiUpdate === 'function') {
    onAfterUiUpdate(init_aspect_ratios_label_multiplier_binding);
}


function refresh_finished_images_catalog_label(value, type) {
    var label = document.querySelector('#finished_images_catalog div span');
    var translation = getTranslation("Finished Images Catalog");
    if (typeof translation == "undefined") {
        translation = "'s Finished Images Catalog";
    } else { translation = "的" + translation; }
    var translation_stat = getTranslation("total: xxx images and yyy pages");
    if (typeof translation_stat == "undefined") {
        translation_stat = "total: xxx images and yyy pages";
    }
    if (type == "video") {
        translation = getTranslation("Finished Videoes");
	translation_stat = getTranslation("total: xxx videoes");
	if (typeof translation == "undefined") {
            translation = "'s Finished Videoes";
    	} else { translation = "的" + translation; }
	if (typeof translation_stat == "undefined") {
	    translation_stat = "total: xxx videoes";
	}
    }
    var xxx = value.split(",")[0];
    var yyy = value.split(",")[1];
    var finished_label = nickname + translation + " - " + htmlDecode(translation_stat.replace(/xxx/g, xxx).replace(/yyy/g, yyy));
    const randomTip = getRandomTip();
    if (randomTip && !browser.device.is_mobile) {
	var space_num = 56 - randomTip.length;
	const spaces = space_num > 0 ? '&nbsp;'.repeat(space_num) : '';
        label.innerHTML = finished_label + spaces + randomTip; 
    } else { label.innerHTML = finished_label; }
}

function refresh_identity_center_label(role, upstream) {
    let label = document.getElementById("identity_center");
    var translation = getTranslation("IdentityCenter");
    if (typeof translation == "undefined") {
        translation = "IdentityCenter";
    }
    var display_name = nickname;
    if (role=="admin") {
	display_name = nickname + ", admin";
    }
    let display_upstream = upstream
        ? upstream.includes(":P2P")
            ? "On-P2P"
            : "On"
        : "Off";
    label.textContent = translation + "(" + display_name + ") - " + display_upstream;
}

function refresh_input_image_tab_label() {
    var items = ["Image Prompt", "Upscale or Variation", "Inpaint or Outpaint"]
    var imageInputTabs = document.getElementById('image_input_tabs');
    var tabNav = imageInputTabs.querySelector('.tab-nav');
    var buttons = tabNav.querySelectorAll('button');
    buttons.forEach(function(button) {
	let itemText = button.getAttribute('data-original-text');
	if (items.includes(itemText)) {
	    var translation = getTranslation(itemText);
	    if (typeof translation == "undefined") {
                translation = itemText;
            }
	    let class_name = task_class_name !== "Fooocus" ? "." + task_class_name : "";
	    const localizedText = translation + class_name;
	    button.textContent = localizedText;
	    button.dataset.localizedLabel = localizedText;
	    if (button.dataset.localizedLabelBound !== '1') {
	        button.dataset.localizedLabelBound = '1';
	        button.addEventListener('click', function() {
                    const nextText = button.dataset.localizedLabel;
                    if (nextText && button.textContent !== nextText) {
                        button.textContent = nextText;
                    }
                });
	    }
	}
    });
}

function localizeWholePage() {
    processNode(gradioApp());

    function elem(comp) {
        var elem_id = comp.props.elem_id ? comp.props.elem_id : "component-" + comp.id;
        return gradioApp().getElementById(elem_id);
    }

    for (var comp of window.gradio_config.components) {
        if (comp.props.webui_tooltip) {
            let e = elem(comp);

            let tl = e ? getTranslation(e.title) : undefined;
            if (tl !== undefined) {
                e.title = tl;
            }
        }
        if (comp.props.placeholder) {
            let e = elem(comp);
            let textbox = e ? e.querySelector('[placeholder]') : null;

            let tl = textbox ? getTranslation(textbox.placeholder) : undefined;
            if (tl !== undefined) {
                textbox.placeholder = tl;
            }
        }
    }
}

document.addEventListener("DOMContentLoaded", function() {
    if (!hasLocalization()) {
        return;
    }

    onUiUpdate(function(m) {
        m.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                processNode(node);
            });
            if (mutation.target) {
                processNode(mutation.target);
            }
        });
    });

    localizeWholePage();

    function bind_dynamic_localization_observers() {
        const btn = gradioApp().getElementById('super_prompter_button');
        if (btn && btn.dataset.localizationObserverBound !== '1') {
            btn.dataset.localizationObserverBound = '1';
            (new MutationObserver(() => {
                processNode(btn);
            })).observe(btn, { childList: true, subtree: true, characterData: true });
            processNode(btn);
        }
    }

    bind_dynamic_localization_observers();
    if (typeof onAfterUiUpdate === 'function') {
        onAfterUiUpdate(bind_dynamic_localization_observers);
    }

    if (localization.rtl) { // if the language is from right to left,
        (new MutationObserver((mutations, observer) => { // wait for the style to load
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.tagName === 'STYLE') {
                        observer.disconnect();

                        for (const x of node.sheet.rules) { // find all rtl media rules
                            if (Array.from(x.media || []).includes('rtl')) {
                                x.media.appendMedium('all'); // enable them
                            }
                        }
                    }
                });
            });
        })).observe(gradioApp(), {childList: true});
    }
});
