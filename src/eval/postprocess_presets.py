"""Named SVG postprocess methods (aligned with notebook 11 ablations)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Callable

from src.inference.postprocess import extract_svg_fragment, sanitize_svg_prediction
from src.svg.cleaning import MAX_SVG_CHARS, clean_svg, validate_svg_constraints
from src.training.lora.eval import svg_metrics


def strip_markdown_fences(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"```(?:svg|xml)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def repair_svg_wrapper(text: str) -> str:
    text = strip_markdown_fences(text)
    start = re.search(r"<svg\b", text, flags=re.IGNORECASE)
    if start:
        text = text[start.start() :]
    close_match = re.search(r"</svg>", text, flags=re.IGNORECASE)
    if close_match:
        text = text[: close_match.end()]
    elif re.search(r"<svg\b", text, flags=re.IGNORECASE):
        text = text + "</svg>"
    return text.strip()


def conservative_clean(text: str) -> str:
    text = repair_svg_wrapper(text)
    fragment = extract_svg_fragment(text)
    fragment = fragment.strip()
    if not fragment:
        return ""
    return fragment[:MAX_SVG_CHARS]


def aggressive_clean(text: str) -> str:
    text = repair_svg_wrapper(text)
    fragment = extract_svg_fragment(text)
    return clean_svg(fragment)


def repair_then_clean(text: str) -> str:
    return clean_svg(repair_svg_wrapper(text))


def hybrid_extract_if_valid_else_repair(text: str) -> str:
    extracted = extract_svg_fragment(repair_svg_wrapper(text)).strip()
    if extracted:
        extracted_constraints = validate_svg_constraints(extracted)
        extracted_metrics = svg_metrics(extracted)
        if extracted_constraints["is_valid_submission_svg"] and extracted_metrics["render_ok"]:
            return extracted
    return repair_then_clean(text)


def truncate_last_nodes_then_clean(text: str) -> str:
    text = repair_svg_wrapper(text)
    fragment = extract_svg_fragment(text)
    if not fragment:
        return clean_svg(fragment)
    try:
        root = ET.fromstring(fragment)
    except Exception:
        return clean_svg(fragment)
    while len(ET.tostring(root, encoding="unicode")) > MAX_SVG_CHARS:
        removed = False
        for parent in list(root.iter()):
            children = list(parent)
            if children:
                parent.remove(children[-1])
                removed = True
                break
        if not removed:
            break
    return clean_svg(ET.tostring(root, encoding="unicode"))


def truncate_then_clean_default(text: str) -> str:
    """Repair wrapper, extract fragment, then ``clean_svg`` (ablation baseline)."""
    return clean_svg(extract_svg_fragment(repair_svg_wrapper(text)))


def truncate_non_path_first_then_clean(text: str) -> str:
    text = repair_svg_wrapper(text)
    fragment = extract_svg_fragment(text)
    if not fragment:
        return clean_svg(fragment)
    try:
        root = ET.fromstring(fragment)
    except Exception:
        return clean_svg(fragment)
    while len(ET.tostring(root, encoding="unicode")) > MAX_SVG_CHARS:
        removed = False
        for parent in list(root.iter()):
            children = list(parent)
            for child in reversed(children):
                local_name = child.tag.split("}", 1)[-1] if "}" in child.tag else child.tag
                if local_name != "path":
                    parent.remove(child)
                    removed = True
                    break
            if removed:
                break
        if removed:
            continue
        for parent in list(root.iter()):
            children = list(parent)
            if children:
                parent.remove(children[-1])
                removed = True
                break
        if not removed:
            break
    return clean_svg(ET.tostring(root, encoding="unicode"))


def postprocess_raw_output(text: str) -> str:
    return "" if text is None else str(text)


def postprocess_extract_only(text: str) -> str:
    return extract_svg_fragment(text)


POSTPROCESS_METHODS: dict[str, Callable[[str], str]] = {
    "raw_output": postprocess_raw_output,
    "extract_only": postprocess_extract_only,
    "current_default_sanitizer": sanitize_svg_prediction,
    "conservative_cleaner": conservative_clean,
    "aggressive_cleaner": aggressive_clean,
    "repair_then_clean": repair_then_clean,
    "hybrid_extract_if_valid_else_repair": hybrid_extract_if_valid_else_repair,
    "truncate_then_clean_default": truncate_then_clean_default,
    "truncate_last_nodes_then_clean": truncate_last_nodes_then_clean,
    "truncate_non_path_first_then_clean": truncate_non_path_first_then_clean,
}

POSTPROCESS_DESCRIPTIONS: dict[str, str] = {
    "raw_output": "Decoded model text as-is (no postprocessing).",
    "extract_only": "First SVG fragment only; no clean_svg.",
    "current_default_sanitizer": "extract_svg_fragment + clean_svg (project default).",
    "conservative_cleaner": "Light repair + extract; cap length without full clean_svg tree rebuild.",
    "aggressive_cleaner": "repair wrapper + extract + clean_svg.",
    "repair_then_clean": "repair_svg_wrapper then clean_svg.",
    "hybrid_extract_if_valid_else_repair": "Use extract if valid+renderable; else repair_then_clean.",
    "truncate_then_clean_default": "Repair + extract + clean_svg (default-style truncation baseline).",
    "truncate_last_nodes_then_clean": "Truncate XML nodes from end until under MAX_SVG_CHARS, then clean_svg.",
    "truncate_non_path_first_then_clean": "Prefer removing non-path nodes for length, then clean_svg.",
}


def get_postprocess_fn(name: str) -> Callable[[str], str]:
    if name not in POSTPROCESS_METHODS:
        raise KeyError(f"Unknown postprocess {name!r}. Choose from {sorted(POSTPROCESS_METHODS)}")
    return POSTPROCESS_METHODS[name]
