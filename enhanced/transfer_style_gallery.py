import json
import os
import html as html_lib
from typing import Optional

from modules.ui_gradio_extensions import webpath


def _load_style_items(styles_json_path: str, images_dir: str):
    items = []
    if not os.path.isfile(styles_json_path):
        return items

    try:
        with open(styles_json_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    except Exception:
        return items

    if not isinstance(data_list, list):
        return items

    for data in data_list:
        if not isinstance(data, dict):
            continue

        title = str(data.get("name") or "")
        if not title:
            continue

        desc = str(data.get("description") or "")
        prompt = str(data.get("prompt") or "")
        negative = str(data.get("negative") or "")

        preview = data.get("preview")
        if not preview:
            continue

        img_path = os.path.join(images_dir, str(preview))
        if not os.path.isfile(img_path):
            continue

        payload = {"name": title, "description": desc, "prompt": prompt, "negative": negative}
        items.append((img_path, payload))

    return items


def get_viewer_html(
    positive_prompt_elem_id: str = "positive_prompt",
    negative_prompt_elem_id: str = "negative_prompt",
    styles_dir: Optional[str] = None,
):
    if styles_dir is None:
        styles_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "style_transfer_assets"))

    styles_json_path = os.path.join(styles_dir, "styles.json")
    images_dir = os.path.join(styles_dir, "images")
    items = _load_style_items(styles_json_path, images_dir)
    if not items:
        escaped_dir = html_lib.escape(styles_json_path)
        return f"""
        <div id="transfer_style_gallery_root" style="padding: 10px; opacity: 0.85;">
            <div style="font-size: 13px; color: #bbb;">未找到转绘风格资源：{escaped_dir}</div>
        </div>
        """

    cards_html = []
    for img_path, payload in items:
        img_url = webpath(os.path.abspath(img_path))
        payload_json = json.dumps(payload, ensure_ascii=False)
        payload_attr = html_lib.escape(payload_json, quote=True)
        title = html_lib.escape(payload.get("name", ""), quote=True)
        desc = html_lib.escape(payload.get("description", ""), quote=True)
        cards_html.append(
            f"""
            <div class="tsg-card" data-style="{payload_attr}" title="{title}">
                <div class="tsg-thumb">
                    <img src="{img_url}" loading="lazy" />
                </div>
            </div>
            """
        )

    cards_html_str = "\n".join(cards_html)

    glue_js = f"""
    <img src="x" style="display:none" onerror='(function(){{
        if (window.transferStyleGalleryInitialized) return;
        window.transferStyleGalleryInitialized = true;

        function updateGradioInput(elemId, value) {{
            const container = document.getElementById(elemId);
            if (!container) return false;
            const input = container.querySelector("input, textarea");
            if (!input) return false;
            const next = (value ?? "").toString();
            if (input.value === next) return true;
            input.value = next;
            input.dispatchEvent(new Event("input", {{ bubbles: true }}));
            input.dispatchEvent(new Event("change", {{ bubbles: true }}));
            return true;
        }}

        function bind() {{
            const root = document.getElementById("transfer_style_gallery_root");
            if (!root) return false;
            if (root.dataset.bound === "1") return true;
            root.dataset.bound = "1";

            const grid = root.querySelector(".tsg-grid");
            if (!grid) return false;

            grid.addEventListener("click", (e) => {{
                const card = e.target.closest(".tsg-card");
                if (!card) return;
                e.preventDefault();
                e.stopPropagation();

                let data;
                try {{
                    data = JSON.parse(card.dataset.style || "{{}}");
                }} catch (err) {{
                    data = {{}};
                }}

                const ok1 = updateGradioInput("{positive_prompt_elem_id}", data.prompt || "");
                const ok2 = (data.negative && String(data.negative).trim().length > 0)
                    ? updateGradioInput("{negative_prompt_elem_id}", data.negative)
                    : true;

                if (ok1 && ok2) {{
                    const selected = grid.querySelector(".tsg-card.is-selected");
                    if (selected) selected.classList.remove("is-selected");
                    card.classList.add("is-selected");
                }}
            }});

            const search = root.querySelector(".tsg-search");
            if (search) {{
                search.addEventListener("input", () => {{
                    const q = (search.value || "").trim().toLowerCase();
                    const cards = Array.from(grid.querySelectorAll(".tsg-card"));
                    for (const c of cards) {{
                        const raw = c.dataset.style || "";
                        const t = (c.querySelector(".tsg-title")?.textContent || "").toLowerCase();
                        const d = (c.querySelector(".tsg-desc")?.textContent || "").toLowerCase();
                        const hay = (t + " " + d + " " + raw.toLowerCase());
                        c.style.display = (!q || hay.includes(q)) ? "" : "none";
                    }}
                }});
            }}

            return true;
        }}

        (function retry(i){{
            if (bind()) return;
            if (i >= 60) return;
            requestAnimationFrame(() => retry(i + 1));
        }})(0);
    }})()'>
    """

    return f"""
    <style>
        #transfer_style_gallery_root {{
            width: 100%;
            padding: 10px 6px;
        }}
        #transfer_style_gallery_root .tsg-toolbar {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 10px;
        }}
        #transfer_style_gallery_root .tsg-titlebar {{
            font-size: 12px;
            color: rgba(255,255,255,0.72);
            white-space: nowrap;
        }}
        #transfer_style_gallery_root .tsg-search {{
            flex: 1;
            min-width: 120px;
            padding: 6px 10px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(0,0,0,0.25);
            color: rgba(255,255,255,0.9);
            outline: none;
            font-size: 12px;
        }}
        #transfer_style_gallery_root .tsg-search:focus {{
            border-color: rgba(233, 61, 130, 0.55);
            box-shadow: 0 0 0 2px rgba(233, 61, 130, 0.12);
        }}
        #transfer_style_gallery_root .tsg-grid {{
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 6px;
            max-height: 520px;
            overflow-y: auto;
            padding-right: 4px;
        }}
        @media (max-width: 1100px) {{
            #transfer_style_gallery_root .tsg-grid {{
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }}
        }}
        @media (max-width: 900px) {{
            #transfer_style_gallery_root .tsg-grid {{
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }}
        }}
        @media (max-width: 650px) {{
            #transfer_style_gallery_root .tsg-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
        #transfer_style_gallery_root .tsg-card {{
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(0,0,0,0.25);
            transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
            cursor: pointer;
        }}
        #transfer_style_gallery_root .tsg-card:hover {{
            transform: translateY(-1px);
            border-color: rgba(233, 61, 130, 0.45);
            box-shadow: 0 6px 14px rgba(0,0,0,0.35);
        }}
        #transfer_style_gallery_root .tsg-card.is-selected {{
            border-color: rgba(233, 61, 130, 0.9);
            box-shadow: 0 0 0 2px rgba(233, 61, 130, 0.18), 0 6px 14px rgba(0,0,0,0.35);
        }}
        #transfer_style_gallery_root .tsg-thumb {{
            width: 100%;
            aspect-ratio: 1 / 1;
            min-height: 64px;
            background: rgba(255,255,255,0.04);
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        #transfer_style_gallery_root .tsg-thumb img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}
    </style>

    <div id="transfer_style_gallery_root">
        <div class="tsg-toolbar">
            <div class="tsg-titlebar">转绘风格（点击图片覆盖 Prompt）</div>
            <input class="tsg-search" type="text" placeholder="搜索风格..." />
        </div>
        <div class="tsg-grid">
            {cards_html_str}
        </div>
    </div>
    {glue_js}
    """
