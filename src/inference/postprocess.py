import re

from src.svg.cleaning import clean_svg


def extract_svg_fragment(text: str) -> str:
    """Extract the most SVG-like fragment from model output text.

    Args:
        text (str): Raw model output, which may contain explanatory text before or after the SVG.

    Returns:
        str: The first complete ``<svg>...</svg>`` block if present, otherwise the text starting
            at the first opening ``svg`` tag, or the raw text when no ``svg`` tag is found.
    """
    if text is None:
        return ""
    text = str(text)
    full_match = re.search(r"<svg.*?</svg>", text, flags=re.IGNORECASE | re.DOTALL)
    if full_match:
        return full_match.group(0)
    open_match = re.search(r"<svg.*", text, flags=re.IGNORECASE | re.DOTALL)
    if open_match:
        return open_match.group(0)
    return text


def sanitize_svg_prediction(text: str) -> str:
    """Extract and sanitize an SVG prediction.

    Args:
        text (str): Raw model output text, typically containing an SVG fragment.

    Returns:
        str: Submission-safe SVG string produced by fragment extraction followed by
            ``src.svg.cleaning.clean_svg``.
    """
    return clean_svg(extract_svg_fragment(text))
