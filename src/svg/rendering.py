import io

import cairosvg
import numpy as np
from PIL import Image, ImageOps


def render_svg_to_pil(svg_text: str, output_width: int = 256, output_height: int = 256):
    """Render SVG text to a PIL image with a white fallback canvas.

    Args:
        svg_text (str): Serialized SVG string to render.
        output_width (int, optional): Target image width in pixels. Defaults to ``256``.
        output_height (int, optional): Target image height in pixels. Defaults to ``256``.

    Returns:
        PIL.Image.Image: RGBA image containing the rendered SVG, or a blank white image of the
            requested size if rendering fails.
    """
    try:
        png_bytes = cairosvg.svg2png(
            bytestring=str(svg_text).encode("utf-8"),
            output_width=output_width,
            output_height=output_height,
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        img = ImageOps.contain(img, (output_width, output_height))
        return img
    except Exception:
        return Image.new("RGBA", (output_width, output_height), (255, 255, 255, 255))


def render_svg_or_none(svg_text):
    """Render SVG text to a NumPy image array when possible.

    Args:
        svg_text (str): Serialized SVG string to render.

    Returns:
        np.ndarray | None: Image array produced by CairoSVG and PIL, or ``None`` if rendering
            fails.
    """
    try:
        png_bytes = cairosvg.svg2png(bytestring=str(svg_text).encode("utf-8"))
        img = Image.open(io.BytesIO(png_bytes))
        img.load()
        return np.array(img)
    except Exception:
        return None
