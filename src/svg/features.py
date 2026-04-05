import re
from collections import Counter

import numpy as np
import pandas as pd


def count_prompt_sentences_like_splits(text: str) -> int:
    """Estimate the number of sentence-like prompt segments.

    Args:
        text (str): Prompt text or any value coercible to a string.

    Returns:
        int: Heuristic count based on punctuation and newline separators. Returns ``0`` for blank
            text and at least ``1`` for non-empty text.
    """
    text = "" if text is None else str(text)
    if not text.strip():
        return 0
    separators = [".", ";", ":", "!", "?", "\n"]
    count = 1
    for sep in separators:
        count += text.count(sep)
    return count


def count_tag_frequencies(tag_series):
    """Aggregate SVG tag counts across an iterable of tag lists.

    Args:
        tag_series (Iterable[Iterable[str] | None]): Sequence whose items are tag-name iterables
            such as lists returned by ``extract_svg_tags``.

    Returns:
        collections.Counter: Counter mapping each tag name to the total number of appearances.
    """
    counter = Counter()
    for tags in tag_series:
        if tags is None:
            continue
        for tag in tags:
            counter[tag] += 1
    return counter


def extract_path_strings(svg_text: str):
    """Extract ``d`` attribute strings from ``path`` elements.

    Args:
        svg_text (str): Serialized SVG string.

    Returns:
        list[str]: Path command strings in encounter order. Returns an empty list for non-string
            inputs.
    """
    if not isinstance(svg_text, str):
        return []
    return re.findall(r'<path\b[^>]*\bd="([^"]*)"', svg_text, flags=re.IGNORECASE)


def count_tag_occurrences(svg_text: str, tag: str) -> int:
    """Count how many times an SVG tag appears in raw source text.

    Args:
        svg_text (str): Serialized SVG text.
        tag (str): Tag name to count, such as ``"circle"`` or ``"path"``.

    Returns:
        int: Number of matching opening tags, or ``0`` for non-string inputs.
    """
    if not isinstance(svg_text, str):
        return 0
    return len(re.findall(rf"<{tag}\b", svg_text, flags=re.IGNORECASE))


def count_numeric_tokens(text: str) -> int:
    """Count numeric literals in a string.

    Args:
        text (str): Input text that may contain integers, decimals, or scientific notation.

    Returns:
        int: Number of numeric tokens detected, or ``0`` for non-string inputs.
    """
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text))


def count_path_command_types(path_d: str):
    """Count SVG path command categories within one path string.

    Args:
        path_d (str): Value of a ``path`` element's ``d`` attribute.

    Returns:
        dict[str, int]: Counts for line, curve, arc, move, close, and total SVG path commands.
    """
    if not isinstance(path_d, str):
        return {
            "line_cmds": 0,
            "curve_cmds": 0,
            "arc_cmds": 0,
            "move_cmds": 0,
            "close_cmds": 0,
            "total_cmds": 0,
        }
    cmds = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]", path_d)
    return {
        "line_cmds": sum(c in "LlHhVv" for c in cmds),
        "curve_cmds": sum(c in "CcSsQqTt" for c in cmds),
        "arc_cmds": sum(c in "Aa" for c in cmds),
        "move_cmds": sum(c in "Mm" for c in cmds),
        "close_cmds": sum(c in "Zz" for c in cmds),
        "total_cmds": len(cmds),
    }


def count_poly_points(svg_text: str, tag: str) -> int:
    """Count coordinate pairs in polygon-like ``points`` attributes.

    Args:
        svg_text (str): Serialized SVG text.
        tag (str): Polygon-like tag name, typically ``"polygon"`` or ``"polyline"``.

    Returns:
        int: Total number of ``x,y`` coordinate pairs found across matching elements.
    """
    if not isinstance(svg_text, str):
        return 0
    point_attrs = re.findall(
        rf"<{tag}\b[^>]*\bpoints=\"([^\"]*)\"",
        svg_text,
        flags=re.IGNORECASE,
    )
    total = 0
    for pts in point_attrs:
        pairs = re.findall(
            r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
            pts,
        )
        total += len(pairs)
    return total


def max_group_depth(svg_text: str) -> int:
    """Estimate the maximum nesting depth of ``g`` groups.

    Args:
        svg_text (str): Serialized SVG text.

    Returns:
        int: Maximum nested ``g`` depth inferred from opening and closing group tags.
    """
    if not isinstance(svg_text, str):
        return 0
    tokens = re.findall(r"</?g\b[^>]*>", svg_text, flags=re.IGNORECASE)
    depth = 0
    max_depth = 0
    for tok in tokens:
        if tok.startswith("</"):
            depth = max(depth - 1, 0)
        else:
            depth += 1
            max_depth = max(max_depth, depth)
    return max_depth


def svg_complexity_features(svg_text: str):
    """Compute structural complexity features for an SVG string.

    Args:
        svg_text (str): Serialized SVG text. Non-string values are treated as empty SVG text.

    Returns:
        dict[str, int | float]: Feature dictionary covering path statistics, shape counts,
            grouping depth, advanced SVG feature flags, and numeric-token counts.
    """
    if not isinstance(svg_text, str):
        svg_text = ""
    path_strings = extract_path_strings(svg_text)
    path_stats = [count_path_command_types(p) for p in path_strings]
    num_paths = len(path_strings)
    path_lens = [len(p) for p in path_strings]
    max_path_len = max(path_lens) if path_lens else 0
    sum_path_len = sum(path_lens)
    total_path_cmds = sum(d["total_cmds"] for d in path_stats)
    max_path_cmds = max([d["total_cmds"] for d in path_stats], default=0)
    line_cmds = sum(d["line_cmds"] for d in path_stats)
    curve_cmds = sum(d["curve_cmds"] for d in path_stats)
    arc_cmds = sum(d["arc_cmds"] for d in path_stats)
    move_cmds = sum(d["move_cmds"] for d in path_stats)
    close_cmds = sum(d["close_cmds"] for d in path_stats)
    num_circles = count_tag_occurrences(svg_text, "circle")
    num_rects = count_tag_occurrences(svg_text, "rect")
    num_ellipses = count_tag_occurrences(svg_text, "ellipse")
    num_polygons = count_tag_occurrences(svg_text, "polygon")
    num_polylines = count_tag_occurrences(svg_text, "polyline")
    num_lines = count_tag_occurrences(svg_text, "line")
    num_text = count_tag_occurrences(svg_text, "text")
    num_groups = count_tag_occurrences(svg_text, "g")
    num_uses = count_tag_occurrences(svg_text, "use")
    polygon_points = count_poly_points(svg_text, "polygon")
    polyline_points = count_poly_points(svg_text, "polyline")
    has_transform = int(bool(re.search(r"\btransform\s*=", svg_text, flags=re.IGNORECASE)))
    has_defs = int(bool(re.search(r"<defs\b", svg_text, flags=re.IGNORECASE)))
    has_linear_gradient = int(bool(re.search(r"<linearGradient\b", svg_text, flags=re.IGNORECASE)))
    has_radial_gradient = int(bool(re.search(r"<radialGradient\b", svg_text, flags=re.IGNORECASE)))
    has_clip_path = int(bool(re.search(r"<clipPath\b", svg_text, flags=re.IGNORECASE)))
    has_mask = int(bool(re.search(r"<mask\b", svg_text, flags=re.IGNORECASE)))
    has_filter = int(bool(re.search(r"<filter\b", svg_text, flags=re.IGNORECASE)))
    has_style = int(bool(re.search(r"<style\b", svg_text, flags=re.IGNORECASE)))
    has_opacity = int(bool(re.search(r"\bopacity\s*=", svg_text, flags=re.IGNORECASE)))

    num_drawable_tags = (
        num_paths + num_circles + num_rects + num_ellipses + num_polygons + num_polylines + num_lines + num_text
    )
    advanced_feature_count = (
        has_transform
        + has_defs
        + has_linear_gradient
        + has_radial_gradient
        + has_clip_path
        + has_mask
        + has_filter
        + has_style
        + has_opacity
        + int(num_uses > 0)
    )
    weighted_path_difficulty = 1.0 * line_cmds + 2.0 * curve_cmds + 2.5 * arc_cmds + 0.5 * move_cmds
    return {
        "num_paths": num_paths,
        "max_path_len": max_path_len,
        "sum_path_len": sum_path_len,
        "sum_path_cmds": total_path_cmds,
        "max_path_cmds": max_path_cmds,
        "line_cmds": line_cmds,
        "curve_cmds": curve_cmds,
        "arc_cmds": arc_cmds,
        "move_cmds": move_cmds,
        "close_cmds": close_cmds,
        "weighted_path_difficulty": weighted_path_difficulty,
        "num_circles": num_circles,
        "num_rects": num_rects,
        "num_ellipses": num_ellipses,
        "num_polygons": num_polygons,
        "num_polylines": num_polylines,
        "num_lines": num_lines,
        "num_text": num_text,
        "num_groups": num_groups,
        "group_depth": max_group_depth(svg_text),
        "num_uses": num_uses,
        "polygon_points": polygon_points,
        "polyline_points": polyline_points,
        "num_drawable_tags": num_drawable_tags,
        "num_numeric_tokens": count_numeric_tokens(svg_text),
        "has_transform": has_transform,
        "has_defs": has_defs,
        "has_linear_gradient": has_linear_gradient,
        "has_radial_gradient": has_radial_gradient,
        "has_clip_path": has_clip_path,
        "has_mask": has_mask,
        "has_filter": has_filter,
        "has_style": has_style,
        "has_opacity": has_opacity,
        "advanced_feature_count": advanced_feature_count,
    }


def prompt_complexity_features(prompt_text: str):
    """Compute simple lexical features for a natural-language prompt.

    Args:
        prompt_text (str): Prompt text describing the desired SVG image.

    Returns:
        dict[str, int]: Feature dictionary covering prompt length, token counts, and heuristic
            counts for colors, relations, quantity words, style words, commas, and conjunctions.
    """
    if not isinstance(prompt_text, str):
        prompt_text = ""
    text = prompt_text.strip().lower()
    tokens = re.findall(r"\b[\w'-]+\b", text)
    color_words = {
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "gray", "grey",
        "brown", "gold", "silver", "cyan", "magenta", "teal", "navy", "beige",
    }
    relation_words = {
        "above", "below", "under", "over", "behind", "in", "inside", "outside", "next", "between",
        "around", "through", "across", "left", "right", "top", "bottom", "center", "middle", "near",
    }
    count_words = {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "single", "double", "pair", "multiple", "many", "several",
    }
    style_words = {
        "minimalist", "cartoon", "outline", "filled", "flat", "3d", "geometric", "abstract",
        "icon", "logo", "sketch", "vintage", "cute", "simple", "detailed", "symmetrical",
    }
    return {
        "prompt_num_chars": len(text),
        "prompt_num_tokens": len(tokens),
        "prompt_num_color_words": sum(t in color_words for t in tokens),
        "prompt_num_relation_words": sum(t in relation_words for t in tokens),
        "prompt_num_count_words": sum(t in count_words for t in tokens),
        "prompt_num_style_words": sum(t in style_words for t in tokens),
        "prompt_comma_count": text.count(","),
        "prompt_and_count": len(re.findall(r"\band\b", text)),
        "prompt_with_count": len(re.findall(r"\bwith\b", text)),
    }


def rank01(series):
    """Convert numeric values to percentile-like ranks in ``[0, 1]``.

    Args:
        series (pd.Series | Sequence): Numeric data to rank. Non-numeric values are coerced to
            ``NaN`` and then filled with ``0``.

    Returns:
        pd.Series: Ranked values using average-percentile ranking. Constant inputs yield zeros.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if s.nunique() <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return s.rank(pct=True, method="average")


def weighted_rank_score(df, feature_weights):
    """Combine multiple feature columns into a weighted rank score.

    Args:
        df (pd.DataFrame): DataFrame containing one or more feature columns.
        feature_weights (dict[str, float]): Mapping from column name to weight. Missing columns
            are ignored.

    Returns:
        pd.Series: Weighted average of percentile-ranked feature columns. Returns all zeros if no
            requested feature column exists.
    """
    parts = []
    total_weight = 0.0
    for col, weight in feature_weights.items():
        if col in df.columns:
            parts.append(weight * rank01(df[col]))
            total_weight += weight
    if total_weight == 0:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return sum(parts) / total_weight
