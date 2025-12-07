# src/nodes.py
import os
import copy
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Optional
try:
    from server import PromptServer
except Exception:
    PromptServer = None

# --- Helpers ---
def prettify_xml(elem: ET.Element) -> str:
    """Return a pretty-printed XML string for the Element.

    Falls back to a compact string if minidom fails (e.g. invalid tag names).
    """
    try:
        rough = ET.tostring(elem, encoding="utf-8")
        parsed = minidom.parseString(rough)
        return parsed.toprettyxml(indent="  ")
    except Exception:
        # fallback: attempt simple pretty by inserting newlines between tags
        try:
            compact = ET.tostring(elem, encoding="unicode")
            # simple newline between close/open tags
            pretty = compact.replace('><', '>' + os.linesep + '<')
            return pretty
        except Exception:
            return ""


def safe_parse_fragment(fragment: str) -> Optional[ET.Element]:
    fragment = (fragment or "").strip()
    if not fragment:
        return None
    try:
        return ET.fromstring(fragment)
    except Exception:
        return None


def sanitize_tag(text: Optional[str]) -> str:
    """Produce a safe XML tag name derived from user text.
    - replace spaces with underscores
    - strip invalid characters
    - ensure it starts with a letter or underscore, otherwise prefix with 'char_'
    """
    if not text:
        return "character"
    s = text.strip()
    s = s.replace(" ", "_")
    # allow letters, digits, underscore, hyphen, dot
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    if not re.match(r"^[A-Za-z_]", s):
        s = "char_" + s
    return s or "character"


def extract_elements_from_fragment(fragment: str) -> Optional[List[ET.Element]]:
    """
    Try to parse fragment and return a list of Element objects to be appended.

    Accepts:
    - A single-root XML string (e.g. "<custom_section>...children...</custom_section>")
    - A single-root element of another name (returns that element as single item)
    - A raw children fragment like "<left>...</left><right>...</right>" (attempts to wrap
      it with <custom_section> and parse)
    Returns None if parsing fails entirely.
    """
    fragment = (fragment or "").strip()
    if not fragment:
        return None

    parsed = safe_parse_fragment(fragment)
    if parsed is not None:
        if parsed.tag == "custom_section":
            return [copy.deepcopy(child) for child in parsed]
        else:
            return [copy.deepcopy(parsed)]

    # If direct parse failed, try wrapping with a <custom_section> to treat it as children
    wrapped = f"<custom_section>{fragment}</custom_section>"
    parsed_wrapped = safe_parse_fragment(wrapped)
    if parsed_wrapped is not None and parsed_wrapped.tag == "custom_section":
        return [copy.deepcopy(child) for child in parsed_wrapped]

    return None


def append_elements_to_parent(parent: ET.Element, elements: List[ET.Element]):
    """Append elements to parent, flattening any <custom_section> wrappers recursively."""
    for el in elements:
        if el.tag == "custom_section":
            # recursively append its children instead of the wrapper
            append_elements_to_parent(parent, list(el))
        else:
            parent.append(copy.deepcopy(el))


def find_name_text_in_elements(elements: List[ET.Element]) -> Optional[str]:
    """Search recursively through elements and their descendants for a <name> tag's text."""
    for el in elements:
        # check element itself
        if el.tag == "name" and (el.text or "").strip():
            return (el.text or "").strip()
        # search descendants using iter
        for desc in el.iter():
            if desc.tag == "name" and (desc.text or "").strip():
                return (desc.text or "").strip()
    return None


def add_child_if_text(parent: ET.Element, tag: str, text: str):
    """Add child element only if text contains non-whitespace characters."""
    if text is None:
        return None
    t = (text or "").strip()
    if t == "":
        return None
    child = ET.SubElement(parent, tag)
    child.text = t
    return child


# -------------------------
# Character Node (multiline descriptive fields + optional custom_section)
# -------------------------
class CharacterNode:
    CATEGORY = "Prompt/Builder"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING",),     # single-line name for the element tag
                "gender": ("STRING",),
            },
            "optional": {
                "appearance": ("STRING", {"multiline": True}),
                "clothing": ("STRING", {"multiline": True}),
                "body_type": ("STRING", {"multiline": True}),
                "expression": ("STRING", {"multiline": True}),
                "action": ("STRING", {"multiline": True}),
                "interaction": ("STRING", {"multiline": True}),
                "position": ("STRING", {"multiline": True}),
                # custom_section: receive output of CharacterCustomBuilder (XML fragment string)
                "custom_section": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_character_xml"

    def build_character_xml(self, name, gender,
                            appearance="", clothing="", body_type="",
                            expression="", action="", interaction="", position="",
                            custom_section=""):
        elem_name = (name or "character").strip() or "character"
        root = ET.Element(elem_name)

        # Required/simple children (only added when non-empty)
        add_child_if_text(root, "name", name)
        add_child_if_text(root, "gender", gender)

        # Optional descriptive children - only create if non-empty
        add_child_if_text(root, "appearance", appearance)
        add_child_if_text(root, "clothing", clothing)
        add_child_if_text(root, "body_type", body_type)
        add_child_if_text(root, "expression", expression)
        add_child_if_text(root, "action", action)
        add_child_if_text(root, "interaction", interaction)
        add_child_if_text(root, "position", position)

        # If a custom_section fragment was provided:
        # - parse it and append its child elements (flatten any nested custom_section wrappers)
        cs = (custom_section or "").strip()
        if cs:
            elems = extract_elements_from_fragment(cs)
            if elems is not None:
                append_elements_to_parent(root, elems)
            else:
                # fallback: treat as text inside <custom_section>
                add_child_if_text(root, "custom_section", cs)

        # Return compact XML fragment (no XML declaration) so downstream nodes can parse it
        return (ET.tostring(root, encoding="unicode"),)


# -------------------------
# Character Custom Builder (user-specified tag names + multiline contents)
#   - No custom_count param: all 10 pairs are exposed; only non-empty pairs are used.
#   - Produces: <custom_section>...children...</custom_section>
# -------------------------
class CharacterCustomBuilder:
    CATEGORY = "Prompt/Builder"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        # Provide up to 10 pairs: tag_name_i (single-line) + tag_content_i (multiline)
        for i in range(1, 11):
            optional[f"tag_name_{i}"] = ("STRING",)          # single-line tag name
            optional[f"tag_content_{i}"] = ("STRING", {"multiline": True})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_character_custom"

    def build_character_custom(self, **kwargs):
        root = ET.Element("custom_section")
        for i in range(1, 11):
            name_key = f"tag_name_{i}"
            content_key = f"tag_content_{i}"
            tag_name = (kwargs.get(name_key, "") or "").strip()
            tag_content = (kwargs.get(content_key, "") or "").strip()
            if tag_name and tag_content:
                safe_tag = sanitize_tag(tag_name)
                child = ET.SubElement(root, safe_tag)
                child.text = tag_content
        return (ET.tostring(root, encoding="unicode"),)


# -------------------------
# General Tag Builder (standard: only create tags that have content)
#   - Added custom_section input to accept output from GeneralTagCustomBuilder
# -------------------------
class GeneralTagBuilder:
    CATEGORY = "Prompt/Builder"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "count": ("STRING", {"multiline": True}),
                "artists": ("STRING", {"multiline": True}),
                "style": ("STRING", {"multiline": True}),
                "background": ("STRING", {"multiline": True}),
                "environment": ("STRING", {"multiline": True}),
                "perspective": ("STRING", {"multiline": True}),
                "atmosphere": ("STRING", {"multiline": True}),
                "lighting": ("STRING", {"multiline": True}),
                "quality": ("STRING", {"multiline": True}),
                "objects": ("STRING", {"multiline": True}),
                "other": ("STRING", {"multiline": True}),
                # custom_section: receive output of GeneralTagCustomBuilder (XML fragment string)
                "custom_section": ("STRING",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_general_tags"

    def build_general_tags(self, count=None, artists=None, style=None, background=None,
                           environment=None, perspective=None, atmosphere=None,
                           lighting=None, quality=None, objects=None, other=None,
                           custom_section=""):
        root = ET.Element("general_tags")
        # Add only non-empty children
        add_child_if_text(root, "count", count)
        add_child_if_text(root, "artists", artists)
        add_child_if_text(root, "style", style)
        add_child_if_text(root, "background", background)
        add_child_if_text(root, "environment", environment)
        add_child_if_text(root, "perspective", perspective)
        add_child_if_text(root, "atmosphere", atmosphere)
        add_child_if_text(root, "lighting", lighting)
        add_child_if_text(root, "quality", quality)
        add_child_if_text(root, "objects", objects)
        add_child_if_text(root, "other", other)

        # If a custom_section XML fragment was provided, parse and append its children
        cs = (custom_section or "").strip()
        if cs:
            elems = extract_elements_from_fragment(cs)
            if elems is not None:
                # If the provided fragment is a single <general_tags> element (i.e. it came
                # from GeneralTagCustomBuilder connected into this node's custom_section),
                # strip that inner <general_tags> wrapper and append *its children* into
                # the outer <general_tags> (avoid nesting).
                # If it's anything else (multiple elements or a different single element),
                # append normally.
                if len(elems) == 1 and elems[0].tag == "general_tags":
                    append_elements_to_parent(root, list(elems[0]))
                else:
                    append_elements_to_parent(root, elems)
            else:
                add_child_if_text(root, "custom_section", cs)

        return (ET.tostring(root, encoding="unicode"),)


# -------------------------
# General Tag Custom Builder (customizable tag names/content, 10 pairs)
# -------------------------
class GeneralTagCustomBuilder:
    CATEGORY = "Prompt/Builder"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        # reduce custom tag pairs to 10 (tag_name_i single-line + tag_content_i multiline)
        for i in range(1, 11):
            optional[f"tag_name_{i}"] = ("STRING",)
            optional[f"tag_content_{i}"] = ("STRING", {"multiline": True})
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build_general_tag_custom"

    def build_general_tag_custom(self, **kwargs):
        root = ET.Element("general_tags")
        for i in range(1, 11):
            name_key = f"tag_name_{i}"
            content_key = f"tag_content_{i}"
            tag_name = (kwargs.get(name_key, "") or "").strip()
            tag_content = (kwargs.get(content_key, "") or "").strip()
            if tag_name and tag_content:
                safe_tag = sanitize_tag(tag_name)
                child = ET.SubElement(root, safe_tag)
                child.text = tag_content
        return (ET.tostring(root, encoding="unicode"),)


# -------------------------
# XML Assembler with SLOTS (slots can contain either character fragment, general_tags fragment, custom fragments, or raw text)
# - No slot_count param: all slot_1..slot_11 inputs are present (single-line).
# - When a parsed fragment is <custom_section>, wrap its children with a tag named after the <name> child if present,
#   otherwise wrap with <character>.
# - Non-XML text will be wrapped in <character>.
# - Final output is pretty-printed XML.
# -------------------------
class XMLAssemblerSlots:
    CATEGORY = "Prompt/Builder"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        # provide up to 11 slot inputs (all always present) - single-line
        for i in range(1, 12):
            optional[f"slot_{i}"] = ("STRING",)
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "assemble_xml"

    def assemble_xml(self, **kwargs):
        root = ET.Element("document")

        # process each slot in order 1..11 (all inputs exist; use those with non-empty text)
        for i in range(1, 12):
            key = f"slot_{i}"
            slot_text = (kwargs.get(key, "") or "").strip()
            if not slot_text:
                continue

            parsed = safe_parse_fragment(slot_text)
            if parsed is not None:
                # If the parsed fragment is a <custom_section>, wrap its children with a
                # <character>-like element (use <name> content as tag if available)
                if parsed.tag == "custom_section":
                    # look for a <name> child anywhere inside parsed
                    name_child = parsed.find('.//name')
                    wrapper_name = "character"
                    if name_child is not None and (name_child.text or "").strip():
                        wrapper_name = sanitize_tag(name_child.text)
                    wrapper = ET.Element(wrapper_name)
                    append_elements_to_parent(wrapper, list(parsed))
                    root.append(wrapper)
                else:
                    root.append(copy.deepcopy(parsed))
            else:
                # Maybe it's an unrooted fragment (e.g. "<left>...</left><right>...</right>")
                elems = extract_elements_from_fragment(slot_text)
                if elems is not None:
                    # If multiple top-level child elements, wrap them in a character-like tag.
                    if len(elems) > 1:
                        # find possible name child among them (search recursively)
                        name_text = find_name_text_in_elements(elems)
                        wrapper_name = sanitize_tag(name_text) if name_text else "character"
                        wrapper = ET.Element(wrapper_name)
                        append_elements_to_parent(wrapper, elems)
                        root.append(wrapper)
                    else:
                        # single element: append as-is (but flatten if it's a custom_section)
                        single = elems[0]
                        if single.tag == "custom_section":
                            # treat like parsed == custom_section case
                            name_text = find_name_text_in_elements(list(single))
                            wrapper_name = sanitize_tag(name_text) if name_text else "character"
                            wrapper = ET.Element(wrapper_name)
                            append_elements_to_parent(wrapper, list(single))
                            root.append(wrapper)
                        else:
                            root.append(single)
                else:
                    # not XML: wrap as <character>
                    wrapper = ET.Element("character")
                    wrapper.text = slot_text
                    root.append(wrapper)

        pretty = prettify_xml(root)

        # save fallback preview files
        try:
            out_dir = os.path.join("outputs", "xml_prompts")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "latest_preview.xml"), "w", encoding="utf-8") as f:
                f.write(pretty)
            with open(os.path.join(out_dir, "latest_preview.html"), "w", encoding="utf-8") as f:
                f.write("<pre>" + pretty.replace('<','&lt;').replace('>','&gt;') + "</pre>")
        except Exception:
            pass

        # broadcast best-effort
        if PromptServer:
            try:
                PromptServer.instance.send_sync("xml_prompt_builder.preview", {"xml": pretty})
            except Exception:
                pass

        return (pretty,)


# Node registration map
NODE_CLASS_MAPPINGS = {
    "Character Node": CharacterNode,
    "Character Custom Builder": CharacterCustomBuilder,
    "General Tag Builder": GeneralTagBuilder,
    "General Tag Custom Builder": GeneralTagCustomBuilder,
    "XML Assembler (slots)": XMLAssemblerSlots,
}
