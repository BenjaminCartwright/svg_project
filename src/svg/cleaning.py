import re
from collections import OrderedDict
from copy import deepcopy
from xml.etree import ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
MAX_SVG_CHARS = 16000
MAX_PATH_ELEMENTS = 256
ALLOWED_TAGS = {
    "svg",
    "g",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
    "defs",
    "use",
    "symbol",
    "clipPath",
    "mask",
    "linearGradient",
    "radialGradient",
    "stop",
    "text",
    "tspan",
    "title",
    "desc",
    "style",
    "pattern",
    "marker",
    "filter",
}
DRAWABLE_TAGS = {
    "path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "g"
}
DISALLOWED_REF_PREFIXES = ("http://", "https://", "//", "javascript:", "data:")
DEFAULT_EMPTY_SVG = (
    f'<svg xmlns="{SVG_NS}" width="256" height="256" viewBox="0 0 256 256"></svg>'
)


def _to_str(x):
    """Convert ``None`` to an empty string and coerce other values to ``str``.

    Args:
        x (Any): Value to normalize.

    Returns:
        str: ``""`` when ``x`` is ``None``, otherwise ``str(x)``.
    """
    if x is None:
        return ""
    return str(x)


def is_valid_svg(svg_text: str) -> bool:
    """Check whether an SVG string parses as XML.

    Args:
        svg_text (str): Candidate SVG or XML text.

    Returns:
        bool: ``True`` if the text is non-empty and ``xml.etree.ElementTree`` can parse it,
            otherwise ``False``.
    """
    svg_text = _to_str(svg_text).strip()
    if not svg_text:
        return False
    try:
        ET.fromstring(svg_text)
        return True
    except Exception:
        return False


def has_svg_wrapper(svg_text: str) -> bool:
    """Detect whether text contains both opening and closing ``svg`` tags.

    Args:
        svg_text (str): Raw SVG-like text.

    Returns:
        bool: ``True`` when both ``<svg ...>`` and ``</svg>`` are present.
    """
    svg_text = _to_str(svg_text)
    has_open = re.search(r"<svg\b", svg_text, flags=re.IGNORECASE) is not None
    has_close = re.search(r"</svg>", svg_text, flags=re.IGNORECASE) is not None
    return has_open and has_close


def has_namespace(svg_text: str) -> bool:
    """Check for the standard SVG XML namespace declaration.

    Args:
        svg_text (str): Raw SVG text.

    Returns:
        bool: ``True`` if the string contains ``xmlns="http://www.w3.org/2000/svg"``.
    """
    svg_text = _to_str(svg_text)
    return re.search(r'xmlns\s*=\s*["\']http://www\.w3\.org/2000/svg["\']', svg_text) is not None


def extract_svg_tags(svg_text: str):
    """Extract XML tag names from SVG text.

    Args:
        svg_text (str): SVG or XML string to scan.

    Returns:
        list[str]: Lower-cased tag names in encounter order, excluding the XML declaration.
    """
    svg_text = _to_str(svg_text)
    tags = re.findall(r"<\s*/?\s*([a-zA-Z][\w:-]*)", svg_text)
    cleaned = []
    for tag in tags:
        tag = tag.lower()
        if tag.startswith("?xml"):
            continue
        cleaned.append(tag)
    return cleaned


def extract_opening_svg_tag(svg_text: str):
    """Return the opening ``<svg ...>`` tag from a string.

    Args:
        svg_text (str): SVG or SVG-like text.

    Returns:
        str: The first opening ``svg`` tag if present, otherwise ``""``.
    """
    svg_text = _to_str(svg_text)
    match = re.search(r"<svg\b[^>]*>", svg_text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(0) if match else ""


def extract_svg_attributes(opening_svg_tag: str):
    """Parse attributes from an opening ``svg`` tag.

    Args:
        opening_svg_tag (str): Opening ``<svg ...>`` tag text.

    Returns:
        OrderedDict[str, str]: Attribute names mapped to string values in source order.
    """
    opening_svg_tag = _to_str(opening_svg_tag)
    if not opening_svg_tag:
        return OrderedDict()

    attr_matches = re.findall(r'([a-zA-Z_:][\w:.-]*)\s*=\s*["\'](.*?)["\']', opening_svg_tag, flags=re.DOTALL)
    attrs = OrderedDict()
    for key, value in attr_matches:
        if key.lower() == "svg":
            continue
        attrs[key] = value
    return attrs


def parse_viewbox(viewbox_value: str):
    """Parse a ``viewBox`` attribute into four numeric components.

    Args:
        viewbox_value (str): Raw ``viewBox`` string, typically ``"min_x min_y width height"``.

    Returns:
        tuple[float, float, float, float] | None: Parsed numeric values, or ``None`` if the
            string is missing or malformed.
    """
    viewbox_value = _to_str(viewbox_value).strip()
    if not viewbox_value:
        return None
    try:
        parts = re.split(r"[\s,]+", viewbox_value)
        parts = [float(x) for x in parts if x != ""]
        if len(parts) != 4:
            return None
        return tuple(parts)
    except Exception:
        return None


def _local_name(tag: str) -> str:
    """Strip an XML namespace prefix from a tag name.

    Args:
        tag (str): Raw ElementTree tag such as ``"{namespace}svg"``.

    Returns:
        str: Local tag name without the namespace prefix.
    """
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _normalized_name(tag: str) -> str:
    """Normalize namespace-qualified or lowercase tag names to allowed SVG casing.

    Args:
        tag (str): Raw tag name from parsed XML.

    Returns:
        str: Normalized local tag name, preserving canonical SVG casing for tags such as
            ``clipPath`` and gradients.
    """
    local = _local_name(tag)
    if local == "clippath":
        return "clipPath"
    if local == "lineargradient":
        return "linearGradient"
    if local == "radialgradient":
        return "radialGradient"
    return local


def _is_disallowed_attr(name: str, value: str) -> bool:
    """Check whether an SVG attribute is unsafe for submission use.

    Args:
        name (str): Attribute name such as ``href`` or ``onclick``.
        value (str): Attribute value to inspect.

    Returns:
        bool: ``True`` when the attribute looks like an event handler, external reference, or
            non-local ``url(...)`` dependency.
    """
    name_lower = str(name).strip().lower()
    value_lower = _to_str(value).strip().lower()
    if name_lower.startswith("on"):
        return True
    if name_lower in {"href", "xlink:href"}:
        if value_lower.startswith("#"):
            return False
        return value_lower.startswith(DISALLOWED_REF_PREFIXES) or value_lower != ""
    if "url(" in value_lower:
        match = re.search(r"url\((.*?)\)", value_lower)
        if match:
            ref = match.group(1).strip().strip("\"'")
            if ref and not ref.startswith("#"):
                return True
    return False


def _strip_unsafe_attributes(elem: ET.Element) -> None:
    """Remove disallowed attributes from an XML element in place.

    Args:
        elem (xml.etree.ElementTree.Element): Parsed XML element whose attributes may be
            mutated.

    Returns:
        None: ``elem.attrib`` is edited directly.
    """
    for attr_name in list(elem.attrib.keys()):
        if _is_disallowed_attr(attr_name, elem.attrib[attr_name]):
            del elem.attrib[attr_name]


def _clone_allowed_children(src_elem: ET.Element, dst_elem: ET.Element) -> int:
    """Recursively copy allowed descendant nodes into a sanitized SVG tree.

    Args:
        src_elem (xml.etree.ElementTree.Element): Source element from the original SVG tree.
        dst_elem (xml.etree.ElementTree.Element): Destination element that receives sanitized
            child copies.

    Returns:
        int: Number of ``path`` elements copied beneath ``dst_elem`` including nested children.
    """
    path_count = 1 if _normalized_name(dst_elem.tag) == "path" else 0
    for child in list(src_elem):
        child_name = _normalized_name(child.tag)
        if child_name not in ALLOWED_TAGS:
            continue
        child_copy = ET.Element(child_name)
        child_copy.attrib.update(deepcopy(child.attrib))
        _strip_unsafe_attributes(child_copy)
        child_copy.text = child.text
        child_copy.tail = child.tail
        child_paths = _clone_allowed_children(child, child_copy)
        if path_count + child_paths > MAX_PATH_ELEMENTS:
            remaining = MAX_PATH_ELEMENTS - path_count
            if remaining <= 0:
                break
            if child_name == "path":
                break
        dst_elem.append(child_copy)
        path_count += child_paths
        if path_count >= MAX_PATH_ELEMENTS:
            break
    return path_count


def _rebuild_allowed_svg_tree(root: ET.Element) -> ET.Element:
    """Build a sanitized SVG root with fixed submission-safe attributes.

    Args:
        root (xml.etree.ElementTree.Element): Parsed root element from the original SVG.

    Returns:
        xml.etree.ElementTree.Element: New ``svg`` element containing only allowed tags,
            safe attributes, and the standard 256x256 viewport metadata.
    """
    clean_root = ET.Element("svg")
    clean_root.attrib.update(deepcopy(root.attrib))
    _strip_unsafe_attributes(clean_root)
    clean_root.attrib["xmlns"] = SVG_NS
    clean_root.attrib["width"] = "256"
    clean_root.attrib["height"] = "256"
    clean_root.attrib["viewBox"] = "0 0 256 256"
    _clone_allowed_children(root, clean_root)
    return clean_root


def _truncate_paths(root: ET.Element) -> None:
    """Remove extra ``path`` nodes beyond the configured submission limit.

    Args:
        root (xml.etree.ElementTree.Element): Parsed SVG tree to edit in place.

    Returns:
        None: Child nodes may be removed from ``root`` and its descendants.
    """
    path_seen = 0
    for parent in list(root.iter()):
        for child in list(parent):
            if _normalized_name(child.tag) == "path":
                path_seen += 1
                if path_seen > MAX_PATH_ELEMENTS:
                    parent.remove(child)


def _trim_to_char_limit(svg_text: str) -> str:
    """Shrink an SVG string until it fits the character budget.

    Args:
        svg_text (str): Serialized SVG string, typically already sanitized and XML-valid.

    Returns:
        str: The original SVG if it already fits, a simplified version with trailing children
            removed, or ``DEFAULT_EMPTY_SVG`` as a final fallback.
    """
    if len(svg_text) <= MAX_SVG_CHARS:
        return svg_text
    empty_with_same_header = DEFAULT_EMPTY_SVG
    root = ET.fromstring(svg_text)
    for parent in list(root.iter()):
        for child in reversed(list(parent)):
            parent.remove(child)
            candidate = ET.tostring(root, encoding="unicode", method="xml")
            if len(candidate) <= MAX_SVG_CHARS:
                return candidate
    return empty_with_same_header


def validate_svg_constraints(svg_text: str) -> dict:
    """Compute submission-style validity checks for an SVG string.

    Args:
        svg_text (str): Serialized SVG candidate to inspect.

    Returns:
        dict[str, bool | int]: Validation metrics including XML parse status, path count,
            viewport checks, safety checks, and the aggregate ``is_valid_submission_svg`` flag.
    """
    svg_text = _to_str(svg_text).strip()
    result = {
        "starts_with_svg": svg_text.startswith("<svg"),
        "xml_valid": False,
        "within_char_limit": len(svg_text) <= MAX_SVG_CHARS,
        "path_count": 0,
        "within_path_limit": False,
        "viewbox_ok": False,
        "width_ok": False,
        "height_ok": False,
        "allowed_tags_only": False,
        "has_disallowed_refs": False,
        "has_event_handlers": False,
        "is_valid_submission_svg": False,
    }
    if not svg_text:
        return result
    try:
        root = ET.fromstring(svg_text)
    except Exception:
        return result
    result["xml_valid"] = True
    tags = [_normalized_name(elem.tag) for elem in root.iter()]
    result["allowed_tags_only"] = all(tag in ALLOWED_TAGS for tag in tags)
    result["path_count"] = sum(tag == "path" for tag in tags)
    result["within_path_limit"] = result["path_count"] <= MAX_PATH_ELEMENTS
    result["viewbox_ok"] = _to_str(root.attrib.get("viewBox")).strip() == "0 0 256 256"
    result["width_ok"] = _to_str(root.attrib.get("width")).strip() == "256"
    result["height_ok"] = _to_str(root.attrib.get("height")).strip() == "256"
    for elem in root.iter():
        for attr_name, attr_value in elem.attrib.items():
            attr_name_lower = str(attr_name).strip().lower()
            if attr_name_lower.startswith("on"):
                result["has_event_handlers"] = True
            if _is_disallowed_attr(attr_name, attr_value):
                result["has_disallowed_refs"] = True
    result["is_valid_submission_svg"] = all(
        [
            result["starts_with_svg"],
            result["xml_valid"],
            result["within_char_limit"],
            result["within_path_limit"],
            result["viewbox_ok"],
            result["width_ok"],
            result["height_ok"],
            result["allowed_tags_only"],
            not result["has_disallowed_refs"],
            not result["has_event_handlers"],
        ]
    )
    return result


def is_valid_viewbox(viewbox_value: str) -> bool:
    """Check whether a ``viewBox`` string has positive width and height.

    Args:
        viewbox_value (str): Raw ``viewBox`` attribute value.

    Returns:
        bool: ``True`` when the string parses into four numbers and the width/height components
            are both positive.
    """
    parsed = parse_viewbox(viewbox_value)
    if parsed is None:
        return False
    _, _, width, height = parsed
    return width > 0 and height > 0


def detect_drawable_tags(svg_text: str):
    """List drawable SVG tags found in a string.

    Args:
        svg_text (str): Raw SVG text.

    Returns:
        list[str]: Tag names from ``DRAWABLE_TAGS`` that appear in the SVG source.
    """
    tags = extract_svg_tags(svg_text)
    return [tag for tag in tags if tag in DRAWABLE_TAGS]


def clean_svg(svg_text: str) -> str:
    """Sanitize model output into a submission-safe SVG string.

    Args:
        svg_text (str): Raw model output or SVG-like text. The value may be malformed, partial,
            or missing the ``svg`` wrapper.

    Returns:
        str: Cleaned SVG string that satisfies the repository's submission constraints, or
            ``DEFAULT_EMPTY_SVG`` if sanitization fails or the final result remains invalid.
    """
    svg_text = _to_str(svg_text).strip()

    if not svg_text:
        return DEFAULT_EMPTY_SVG

    svg_text = re.sub(r"<\?xml.*?\?>", "", svg_text, flags=re.IGNORECASE | re.DOTALL).strip()
    svg_text = re.sub(r"<!DOCTYPE.*?>", "", svg_text, flags=re.IGNORECASE | re.DOTALL).strip()

    if re.search(r"<svg\b", svg_text, flags=re.IGNORECASE) is None:
        svg_text = f'<svg xmlns="{SVG_NS}" width="256" height="256" viewBox="0 0 256 256">{svg_text}</svg>'

    if not has_namespace(svg_text):
        svg_text = re.sub(
            r"<svg\b",
            f'<svg xmlns="{SVG_NS}"',
            svg_text,
            count=1,
            flags=re.IGNORECASE,
        )

    if re.search(r"<svg\b", svg_text, flags=re.IGNORECASE) and re.search(r"</svg>", svg_text, flags=re.IGNORECASE) is None:
        svg_text = svg_text + "</svg>"

    svg_text = re.sub(r">\s+<", "><", svg_text)
    svg_text = re.sub(r"\s+", " ", svg_text).strip()
    try:
        root = ET.fromstring(svg_text)
        clean_root = _rebuild_allowed_svg_tree(root)
        _truncate_paths(clean_root)
        svg_text = ET.tostring(clean_root, encoding="unicode", method="xml")
    except Exception:
        svg_text = DEFAULT_EMPTY_SVG
    svg_text = _trim_to_char_limit(svg_text)
    if not validate_svg_constraints(svg_text)["is_valid_submission_svg"]:
        return DEFAULT_EMPTY_SVG
    return svg_text
