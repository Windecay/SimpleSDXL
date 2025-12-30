var re_num = /^[.\d]+$/;

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
           // console.log(navigator);
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

    // 新增反向查找逻辑以修复自定义风格悬浮图错位的问题
    let originalText = text;
    if (tl === undefined) {
        const rev = getReverseLocalization();
        if (rev && rev[text]) {
            originalText = rev[text];
        }
    }

    if (tl !== undefined) {
        node.textContent = tl;
    }

    if (originalText && node.parentElement) {
        node.parentElement.setAttribute("data-original-text", originalText);
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
    const start = performance.now();
    processNode(document.querySelector('.style_selections'));
    console.log(`[Timing] refresh_style_localization took ${performance.now() - start}ms`);
}

let styleGridOriginalElements = [];
let styleGridHandlersAttached = false;
let isHandlingClick = false;
let lastKnownGoodStyles = new Set();

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

        for (const label of labels) {
            const originalText = label.getAttribute('data-original-text');
            const innerSpan = label.querySelector('span');
            const spanOriginalText = innerSpan ? innerSpan.getAttribute('data-original-text') : null;
            const labelText = label.textContent.trim();

            const checkTexts = [originalText, spanOriginalText, labelText];
            for (let text of checkTexts) {
                if (!text) continue;
                const cleanText = text.toLowerCase().replace(/[- _]/g, '');
                if (cleanText === cleanStyle) {
                    const input = label.querySelector('input[type="checkbox"]');
                    if (input) {
                        isHandlingClick = true;

                        const isCurrentlyChecked = input.checked;
                        const willBeChecked = !isCurrentlyChecked;

                        if (willBeChecked) {
                            lastKnownGoodStyles.add(cleanStyle);
                        } else {
                            lastKnownGoodStyles.delete(cleanStyle);
                        }

                        input.checked = willBeChecked;
                        input.dispatchEvent(new Event('change', { bubbles: true }));

                        const isSpecialStyle = cleanStyle.includes('fooocusv2') || cleanStyle.includes('fooocuspony') || cleanStyle.includes('sd15');
                        if (isSpecialStyle) {
                            label.click();
                        }

                        sync_style_grid_state();

                        setTimeout(() => {
                            isHandlingClick = false;
                            sync_style_grid_state();
                        }, 800);
                        found = true;
                        break;
                    }
                }
            }
            if (found) break;
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
            if (promptTextarea && prompt) {
                const current = promptTextarea.value.trim();
                promptTextarea.value = current ? current + ", " + prompt : prompt;
                promptTextarea.dispatchEvent(new Event('input', { bubbles: true }));
            }
            if (negativePromptTextarea && negativePrompt) {
                const current = negativePromptTextarea.value.trim();
                negativePromptTextarea.value = current ? current + ", " + negativePrompt : negativePrompt;
                negativePromptTextarea.dispatchEvent(new Event('input', { bubbles: true }));
            }
        } catch (err) {}
    });

    styleGridHandlersAttached = true;
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
    const selectedStyles = new Set();
    const rev = getReverseLocalization();

    labels.forEach(label => {
        const input = label.querySelector('input[type="checkbox"]');
        if (input && input.checked) {
            const innerSpan = label.querySelector('span');
            let text = label.getAttribute('data-original-text') || 
                       (innerSpan ? innerSpan.getAttribute('data-original-text') : null) ||
                       label.textContent.trim();

            const cleanText = text.toLowerCase().replace(/[- _]/g, '');
            const revText = (rev && rev[text]) ? rev[text] : null;
            const cleanRevText = revText ? revText.toLowerCase().replace(/[- _]/g, '') : null;

            if (cleanText) selectedStyles.add(cleanText);
            if (cleanRevText) selectedStyles.add(cleanRevText);
        }
    });

    if (isHandlingClick) {
        selectedStyles.clear();
        lastKnownGoodStyles.forEach(s => selectedStyles.add(s));
    } else {
        if (selectedStyles.size > 0) {
            selectedStyles.forEach(s => lastKnownGoodStyles.add(s));
        } else {
        }
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
    const container = document.querySelector(".style_grid");
    if (!container) return;

    init_style_grid_handlers();

    const selections = document.querySelector('.style_selections');
    if (!selections) return;

    const checkedInputs = selections.querySelectorAll('input:checked');
    let selectedStylesClean = new Set();
    const rev = getReverseLocalization();

    checkedInputs.forEach(cb => {
        const label = cb.parentElement;
        const innerSpan = label.querySelector('span');
        let text = label.getAttribute('data-original-text') ||
                   (innerSpan ? innerSpan.getAttribute('data-original-text') : null) ||
                   label.textContent.trim();

        const revText = (rev && rev[text]) ? rev[text] : null;

        if (text) {
            selectedStylesClean.add(text.toLowerCase().replace(/[- _]/g, ''));
        }
        if (revText) {
            selectedStylesClean.add(revText.toLowerCase().replace(/[- _]/g, ''));
        }
    });

    if (isHandlingClick) {
        selectedStylesClean.clear();
        lastKnownGoodStyles.forEach(s => selectedStylesClean.add(s));
    } else {
        if (selectedStylesClean.size > 0) {
            selectedStylesClean.forEach(s => lastKnownGoodStyles.add(s));
        }
    }

    if (styleGridOriginalElements.length === 0) {
        styleGridOriginalElements = [...container.querySelectorAll('.style_item')];
        console.log("[Style] Cached original elements:", styleGridOriginalElements.length);
    }

    const searchBar = gradioApp().querySelector('textarea[data-testid="textbox"][placeholder*="搜索风格"], textarea[data-testid="textbox"][placeholder*="search styles"]');
    const searchText = (searchBar?.value?.trim() || '').toLowerCase();

    styleGridOriginalElements.forEach((item) => {
        const btn = item.querySelector('button');
        const btnText = btn?.textContent.trim();
        const rawName = btn?.getAttribute('data-style-name') || btnText;
        if (!rawName) return;

        const cleanName = rawName.toLowerCase().replace(/[- _]/g, '');
        const isSelected = selectedStylesClean.has(cleanName);

        const matchesSearch = cleanName.includes(searchText.replace(/[- _]/g, '')) || 
                              btnText.toLowerCase().includes(searchText);
        const isVisible = isSelected || matchesSearch;

        if (isVisible) {
            item.style.setProperty('display', 'block', 'important');
            item.style.order = isSelected ? 0 : 1;

            if (!btn.style.backgroundImage || btn.style.backgroundImage === 'none') {
                const styleName = rawName.toLowerCase().replace(/ /g, '_').replace(/[^a-z0-9_]/g, '');
                btn.style.backgroundImage = `url("file=sdxl_styles/samples/${styleName}.jpg")`;
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


function refresh_scene_localization() {
    processNode(document.querySelector('.scene_aspect_ratio_selections'));
}

function refresh_aspect_ratios_label(value) {
    var label = document.querySelector('#aspect_ratios_accordion div span');
    var translation = getTranslation("Aspect Ratios");
    if (typeof translation == "undefined") {
        translation = "Aspect Ratios";
    }
    value = value.split(",")[0]
    label.textContent = translation + " - " + htmlDecode(value);
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
	    button.textContent = translation + class_name;
	    button.addEventListener('click', function() {
                button.textContent = translation + class_name;
            });
	}
    });
}

function localizeWholePage() {
    console.log("in localize")
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
        });
    });

    localizeWholePage();

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
