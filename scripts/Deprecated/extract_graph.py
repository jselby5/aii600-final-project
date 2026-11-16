from pathlib import Path
from dataclasses import dataclass
import argparse
import csv
import math
import os
import re
import shutil
import sys

import cv2
import numpy as np
import pytesseract
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.signal import find_peaks


DEFAULT_IMAGE = Path("data/images/graph_223.png")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_DIAGNOSTIC_DIR = Path("data/diagnostics")

MINOR_DIVISIONS = 10
MASK_WIDTH = 3
MIN_CROSSINGS = 4
MIN_AXIS_LABELS = 3
OCR_MIN_CONFIDENCE = 35
ANCHOR_MIN_CONFIDENCE = 50
AXIS_SEARCH_PADDING = 70
BOTTOM_LABEL_BAND_FRACTION = 0.12
COLUMN_PADDING = 8
ROW_HALF_HEIGHT_FACTOR = 1.05
TITLE_PADDING = 3
LABEL_STEP_TOLERANCE = 0.05

REFERENCE_GRID_WARN_ASPECT_ERROR = 0.05
REFERENCE_GRID_MAX_ASPECT_ERROR = 0.10
REFERENCE_GRID_MAX_MEDIAN_LABEL_ERROR = 0.25

DEFAULT_GRID_SEARCH = 12.0
GRID_MAX_SPACING_FRACTION = 0.45
GRID_NODE_INTERVALS = 4.0
GRID_WINDOW_INTERVALS = 6.0
GRID_MAX_LOCAL_SLOPE = 0.12
GRID_SLOPE_SAMPLES = 9
GRID_MIN_SUPPORT = 0.25
GRID_MIN_CONTRAST = 0.08
GRID_MIN_PEAK_MARGIN = 0.04
GRID_ALONG_SMOOTHNESS = 1.0
GRID_ACROSS_SMOOTHNESS = 0.75
GRID_REFERENCE_WEIGHT = 0.015
GRID_CG_ITERATIONS = 250

SPLINE_DEGREE = 3
SPLINE_BASIS_COUNTS = (8, 12, 16)
SPLINE_SMOOTHNESS = 0.002
SPLINE_ROBUST_SCALE = 2.0
SPLINE_MIN_SEPARATION = 0.10
SPLINE_ASSIGNMENT_ITERATIONS = 7
SPLINE_OPTIMIZER_ITERATIONS = 220
SPLINE_MAX_SAMPLE_COLUMNS = 450
DEFAULT_SHAPE_WEIGHT = 0.1

ASSIGN_MIN_GATE = 4.0
ASSIGN_MAX_GATE = 18.0
ASSIGN_MISSING_COST = 0.8
OBSERVED_TOLERANCE = 3.0
MIN_CURVE_SUPPORT_COLUMNS = 8
MIN_HIGH_SUPPORT_COLUMNS = 4
MIN_DISTINCT_STROKE_SEPARATION = 2.0

DEFAULT_INK_WEIGHT = 0.35
INK_DISTANCE_TOLERANCE = 3.0
INK_MASKED_WEIGHT = 0.25
SUPPORT_WINDOW_MAJOR = 0.5
SUPPORT_MAX_GAP_MAJOR = 1.0
SUPPORT_MIN_FRACTION = 0.55
SUPPORT_MIN_EVALUABLE_FRACTION = 0.25

JOIN_X = 1.0
JOIN_PIXEL_TOLERANCE = 3.0
DEFAULT_JOIN_MISMATCH = 0.05
LOW_JOIN_PRIOR_WEIGHT = 0.25
HIGH_JOIN_IMAGE_WEIGHT = 1.0
JOIN_STRIP_MAJOR_FRACTION = 0.25
JOIN_STRIP_MIN_WIDTH = 12
JOIN_STRIP_MAX_WIDTH = 50
JOIN_STRIP_PASSES = 3
JOIN_STRIP_MIN_POINTS = 4

# Diagnostic only. None of these parameters enter the baseline fit.
INIT_SEARCH_FRACTION = 0.40
INIT_CENTERS = 14
INIT_WIDTH_FACTORS = (0.22, 0.35, 0.50)
INIT_MIN_WIDTH = 20
INIT_MAX_WIDTH = 70
INIT_PEAK_HEIGHT = 0.10
INIT_PEAK_PROMINENCE = 0.045
INIT_PEAK_DISTANCE = 4
INIT_MIN_POINTS = 5
INIT_MAX_MEDIAN_RESIDUAL = 1.75
INIT_MATCH_MEDIAN = 3.0
INIT_MATCH_MAX = 6.0

PLAIN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"


@dataclass(frozen=True)
class Word:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int

    @property
    def cx(self):
        return self.left + self.width / 2

    @property
    def cy(self):
        return self.top + self.height / 2

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height


@dataclass(frozen=True)
class ReferenceGrid:
    intervals_x: int
    intervals_y: float
    major_x: float
    major_y: float
    pooled_major: float
    median_label_error_x: float
    median_label_error_y: float
    aspect_error: float

    @property
    def minor_x(self):
        return self.major_x / MINOR_DIVISIONS

    @property
    def minor_y(self):
        return self.major_y / MINOR_DIVISIONS

    @property
    def minor_intervals_x(self):
        return self.intervals_x * MINOR_DIVISIONS

    @property
    def minor_intervals_y(self):
        return int(round(self.intervals_y * MINOR_DIVISIONS))


@dataclass
class GridFit:
    mask: np.ndarray
    horizontal: list
    vertical: list
    measurements: list
    accepted: int
    attempted: int
    reference: ReferenceGrid


@dataclass(frozen=True)
class JoinAnchor:
    tag: str
    value: float
    sigma: float
    low_supported: bool


@dataclass
class JoinGuidance:
    pixel_x: float
    u: float
    initial_pixels: np.ndarray
    initial_slopes: np.ndarray
    prior_indices: np.ndarray
    prior_targets: np.ndarray
    prior_sigmas: np.ndarray
    local_indices: np.ndarray
    local_targets: np.ndarray
    local_sigmas: np.ndarray
    local_xs: np.ndarray
    local_observations: np.ndarray


@dataclass
class InkEvidence:
    distance: np.ndarray
    grid_mask: np.ndarray
    bounds: tuple


@dataclass
class ImageSupport:
    dense_xs: np.ndarray
    near_ink: np.ndarray
    obscured: np.ndarray
    inside: np.ndarray
    right_edges: np.ndarray
    sampled_mask: np.ndarray
    window_states: np.ndarray
    retained_fraction: np.ndarray
    valid: np.ndarray


@dataclass
class CurveModel:
    basis: object
    coefficients: np.ndarray
    tags: list
    paths: list
    seed_x: int
    seed_ys: np.ndarray
    basis_count: int
    validation_cost: float
    sample_xs: np.ndarray
    assigned: np.ndarray
    support: object
    guidance: object


@dataclass
class PageResult:
    image: Path
    segment: str
    bounds: tuple
    x_fit: np.ndarray
    y_fit: np.ndarray
    model: CurveModel


@dataclass
class InitWindow:
    center: float
    xs: np.ndarray
    slope: float
    peak_count: int
    coefficients: object
    observations: object
    residuals: object
    score: float
    reason: str
    matches: int = 0


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def normalize(text):
    return re.sub(
        r"\s+", "",
        text.strip().replace("−", "-").replace("–", "-").replace(",", "."),
    )


def number(text):
    text = normalize(text)
    if not re.fullmatch(PLAIN, text):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def clip(box, shape):
    x0, y0, x1, y1 = box
    return (
        max(0, min(shape[1], math.floor(x0))),
        max(0, min(shape[0], math.floor(y0))),
        max(0, min(shape[1], math.ceil(x1))),
        max(0, min(shape[0], math.ceil(y1))),
    )


def crop(image, box):
    x0, y0, x1, y1 = clip(box, image.shape)
    return image[y0:y1, x0:x1]


def intersection(a, b):
    return (
        max(0, min(a[2], b[2]) - max(a[0], b[0]))
        * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    )


def box_of(word):
    return word.left, word.top, word.right, word.bottom


def binary_image(gray):
    block = max(15, min(81, (min(gray.shape) // 30) | 1))
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, 12,
    ) > 0


def threshold_ink(gray):
    return cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]


def draw_frame(image, bounds):
    left, top, right, bottom = bounds
    cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 0), 2)


def tint(image, mask, color):
    result = image.copy()
    result[mask] = (
        0.35 * result[mask].astype(float)
        + 0.65 * np.asarray(color, float)
    ).astype(np.uint8)
    return result


def detect_lines(ink, horizontal, audit=None):
    height, width = ink.shape
    length = max(25, (width if horizontal else height) // 30)
    opened = cv2.morphologyEx(
        ink.astype(np.uint8) * 255, cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (length, 1) if horizontal else (1, length)
        ),
    )
    count, _, stats, centers = cv2.connectedComponentsWithStats(opened)
    thickness = max(8, int(0.006 * (height if horizontal else width)))
    lines, raw = [], []

    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        along, across = (w, h) if horizontal else (h, w)
        too_short = along < length
        too_thick = (
            across > thickness or area / max(1, along) > thickness
        )
        accepted = not (too_short or too_thick)

        if audit is not None:
            raw.append({
                "box": (x, y, x + w - 1, y + h - 1),
                "accepted": accepted,
                "reason": ("L" if too_short else "") + ("T" if too_thick else ""),
            })

        if accepted:
            if horizontal:
                lines.append((float(centers[index][1]), x, x + w - 1))
            else:
                lines.append((float(centers[index][0]), y, y + h - 1))

    groups = []
    for line in sorted(lines):
        if (
            groups
            and abs(line[0] - np.median([v[0] for v in groups[-1]])) <= 4
        ):
            groups[-1].append(line)
        else:
            groups.append([line])

    grouped = [
        (
            float(np.median([v[0] for v in group])),
            float(min(v[1] for v in group)),
            float(max(v[2] for v in group)),
        )
        for group in groups
    ]

    if audit is not None:
        key = "horizontal" if horizontal else "vertical"
        audit[f"{key}_raw"] = raw
        audit[f"{key}_accepted"] = grouped.copy()
        audit[f"{key}_length_limit"] = length
        audit[f"{key}_thickness_limit"] = thickness

    return grouped


def intersections(horizontal, vertical):
    result = np.zeros((len(horizontal), len(vertical)), bool)
    for i, (y, x0, x1) in enumerate(horizontal):
        for j, (x, y0, y1) in enumerate(vertical):
            result[i, j] = (
                x0 - 6 <= x <= x1 + 6 and y0 - 6 <= y <= y1 + 6
            )
    return result


def components(matrix):
    seen_h, seen_v, result = set(), set(), []
    for initial in range(matrix.shape[0]):
        if initial in seen_h:
            continue
        pending = [(True, initial)]
        hs, vs = set(), set()

        while pending:
            horizontal, index = pending.pop()
            if horizontal:
                if index in seen_h:
                    continue
                seen_h.add(index)
                hs.add(index)
                pending.extend(
                    (False, int(j))
                    for j in np.flatnonzero(matrix[index])
                    if int(j) not in seen_v
                )
            else:
                if index in seen_v:
                    continue
                seen_v.add(index)
                vs.add(index)
                pending.extend(
                    (True, int(i))
                    for i in np.flatnonzero(matrix[:, index])
                    if int(i) not in seen_h
                )

        if hs and vs:
            result.append((sorted(hs), sorted(vs)))

    return result


def outer_endpoint(values, lower, tolerance):
    groups = []
    for value in sorted(values):
        if groups and value - np.mean(groups[-1]) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])

    minimum = max(2, math.ceil(0.08 * len(values)))
    supported = [g for g in groups if len(g) >= minimum]
    if supported:
        return float(np.median(supported[0] if lower else supported[-1]))
    return float(np.percentile(values, 10 if lower else 90))


def detect_frame(ink, audit=None):
    if audit is not None:
        audit.clear()
        for family in ("horizontal", "vertical"):
            for stage in ("raw", "accepted", "pruned", "selected"):
                audit[f"{family}_{stage}"] = []
        audit["bounds"] = None
        audit["component_count"] = 0

    horizontal = detect_lines(ink, True, audit)
    vertical = detect_lines(ink, False, audit)
    if not horizontal or not vertical:
        return None

    matrix = intersections(horizontal, vertical)
    keep_h = np.ones(len(horizontal), bool)
    keep_v = np.ones(len(vertical), bool)

    for _ in range(10):
        new_h = keep_h & (matrix[:, keep_v].sum(axis=1) >= MIN_CROSSINGS)
        new_v = keep_v & (matrix[keep_h, :].sum(axis=0) >= MIN_CROSSINGS)
        if np.array_equal(new_h, keep_h) and np.array_equal(new_v, keep_v):
            break
        keep_h, keep_v = new_h, new_v

    horizontal = [v for v, keep in zip(horizontal, keep_h) if keep]
    vertical = [v for v, keep in zip(vertical, keep_v) if keep]

    if audit is not None:
        audit["horizontal_pruned"] = horizontal.copy()
        audit["vertical_pruned"] = vertical.copy()

    if not horizontal or not vertical:
        return None

    matrix = intersections(horizontal, vertical)
    choices = []
    for hi, vi in components(matrix):
        if min(len(hi), len(vi)) < MIN_CROSSINGS:
            continue
        crossings = int(matrix[np.ix_(hi, vi)].sum())
        choices.append((
            crossings * crossings / (len(hi) * len(vi)), hi, vi
        ))

    if audit is not None:
        audit["component_count"] = len(choices)
    if not choices:
        return None

    _, hi, vi = max(choices, key=lambda value: value[0])
    horizontal = [horizontal[i] for i in hi]
    vertical = [vertical[i] for i in vi]

    if audit is not None:
        audit["horizontal_selected"] = horizontal.copy()
        audit["vertical_selected"] = vertical.copy()

    height, width = ink.shape
    bounds = (
        max(0, round(outer_endpoint(
            [v[1] for v in horizontal], True, max(6, 0.004 * width)
        ))),
        max(0, round(outer_endpoint(
            [v[1] for v in vertical], True, max(6, 0.004 * height)
        ))),
        min(width - 1, round(outer_endpoint(
            [v[2] for v in horizontal], False, max(6, 0.004 * width)
        ))),
        min(height - 1, round(outer_endpoint(
            [v[2] for v in vertical], False, max(6, 0.004 * height)
        ))),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        return None
    if audit is not None:
        audit["bounds"] = bounds
    return bounds


def frame_detection_preview(gray, audit):
    source = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    height, width = gray.shape
    scale = min(1.0, 1100.0 / max(height, width))
    pw, ph = max(1, round(width * scale)), max(1, round(height * scale))
    header = 76

    def point(x, y):
        return round(x * scale), round(y * scale) + header

    def panel(title, subtitle):
        canvas = np.full((ph + header, pw, 3), 255, np.uint8)
        canvas[header:] = cv2.resize(
            source, (pw, ph), interpolation=cv2.INTER_AREA
        )
        cv2.putText(
            canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (0, 0, 0), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, subtitle, (10, 49), cv2.FONT_HERSHEY_SIMPLEX,
            0.39, (40, 40, 40), 1, cv2.LINE_AA,
        )
        return canvas

    def draw_segments(image, stage):
        for y, x0, x1 in audit.get(f"horizontal_{stage}", []):
            cv2.line(image, point(x0, y), point(x1, y), (255, 80, 0), 2)
        for x, y0, y1 in audit.get(f"vertical_{stage}", []):
            cv2.line(image, point(x, y0), point(x, y1), (0, 140, 255), 2)

    first = panel(
        "1. Components after directional opening",
        "Green=accepted; magenta=rejected; T=thickness; L=length.",
    )
    raw = sorted(
        audit.get("horizontal_raw", []) + audit.get("vertical_raw", []),
        key=lambda item: not item["accepted"],
    )
    for item in raw:
        x0, y0, x1, y1 = item["box"]
        color = (0, 170, 0) if item["accepted"] else (180, 0, 180)
        cv2.rectangle(first, point(x0, y0), point(x1, y1), color, 1)
        if not item["accepted"]:
            px, py = point(x0, y0)
            cv2.putText(
                first, item["reason"],
                (px + 2, max(header + 12, py - 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    second = panel(
        "2. Accepted and grouped segments",
        f"H={len(audit.get('horizontal_accepted', []))}; "
        f"V={len(audit.get('vertical_accepted', []))}.",
    )
    draw_segments(second, "accepted")

    third = panel(
        "3. After intersection pruning",
        f"H={len(audit.get('horizontal_pruned', []))}; "
        f"V={len(audit.get('vertical_pruned', []))}.",
    )
    draw_segments(third, "pruned")

    bounds = audit.get("bounds")
    fourth = panel(
        "4. Selected component and inferred frame",
        f"bounds={bounds}; components={audit.get('component_count', 0)}.",
    )
    draw_segments(fourth, "selected")
    if bounds is not None:
        left, top, right, bottom = bounds
        cv2.rectangle(
            fourth, point(left, top), point(right, bottom),
            (255, 255, 0), 3,
        )

    return np.vstack((
        np.hstack((first, second)), np.hstack((third, fourth))
    ))


def find_tesseract():
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        pytesseract.pytesseract.tesseract_cmd,
    ]
    if os.name == "nt":
        for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            root = os.environ.get(variable)
            if root:
                candidates.extend([
                    Path(root) / "Tesseract-OCR" / "tesseract.exe",
                    Path(root) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
                ])

    for candidate in candidates:
        if not candidate:
            continue
        executable = shutil.which(str(candidate))
        if executable is None and Path(candidate).is_file():
            executable = str(Path(candidate).resolve())
        if executable:
            pytesseract.pytesseract.tesseract_cmd = executable
            return True
    return False


def read_words(gray, modes=(11, 6), minimum=OCR_MIN_CONFIDENCE):
    result = []
    for psm in modes:
        data = pytesseract.image_to_data(
            gray, config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
        )
        for i, text in enumerate(data["text"]):
            text = text.strip()
            confidence = float(data["conf"][i])
            if text and confidence >= minimum:
                result.append(Word(
                    text, confidence,
                    int(data["left"][i]), int(data["top"][i]),
                    int(data["width"][i]), int(data["height"][i]),
                ))
    return result


def unique_words(words):
    result = []
    for word in sorted(words, key=lambda w: -w.confidence):
        if any(
            normalize(word.text) == normalize(old.text)
            and intersection(box_of(word), box_of(old))
            / max(1, min(
                word.width * word.height, old.width * old.height
            )) >= 0.6
            for old in result
        ):
            continue
        result.append(word)
    return result


def ocr_region_words(gray, box):
    x0, y0, x1, y1 = clip(box, gray.shape)
    image = gray[y0:y1, x0:x1]
    if image.size == 0:
        return []

    scale, border = 3, 20
    prepared = cv2.copyMakeBorder(
        cv2.resize(
            image, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        ),
        border, border, border, border,
        cv2.BORDER_CONSTANT, value=255,
    )
    result = []
    for word in read_words(prepared, minimum=0):
        left = max(x0, math.floor(x0 + (word.left - border) / scale))
        top = max(y0, math.floor(y0 + (word.top - border) / scale))
        right = min(x1, math.ceil(x0 + (word.right - border) / scale))
        bottom = min(y1, math.ceil(y0 + (word.bottom - border) / scale))
        if right > left and bottom > top:
            result.append(Word(
                word.text, word.confidence,
                left, top, right - left, bottom - top,
            ))
    return result


def numeric_readings(gray, box):
    image = crop(gray, box)
    if image.size == 0:
        return []

    enlarged = cv2.resize(
        image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC
    )
    binary = 255 - threshold_ink(enlarged)
    result = []
    for variant in (enlarged, binary):
        prepared = cv2.copyMakeBorder(
            variant, 24, 24, 24, 24,
            cv2.BORDER_CONSTANT, value=255,
        )
        for psm in (7, 13):
            text = normalize(pytesseract.image_to_string(
                prepared,
                config=(
                    f"--psm {psm} "
                    "-c tessedit_char_whitelist=0123456789.+-"
                ),
            ))
            if number(text) is not None:
                result.append(text)
    return result


def consistent_sequence(words, horizontal):
    words = [w for w in words if number(w.text) is not None]
    if len(words) < MIN_AXIS_LABELS:
        return []

    positions = np.array([w.cx if horizontal else w.cy for w in words])
    values = np.array([number(w.text) for w in words], float)
    tolerance = max(3.0, 0.30 * np.median([w.height for w in words]))
    best = None

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if abs(positions[j] - positions[i]) < 10:
                continue
            slope = (values[j] - values[i]) / (positions[j] - positions[i])
            if slope == 0 or (slope > 0) != horizontal:
                continue

            intercept = values[i] - slope * positions[i]
            residual = np.abs((values - intercept) / slope - positions)
            chosen = []
            for k in sorted(
                np.flatnonzero(residual <= tolerance),
                key=lambda k: (residual[k], -words[k].confidence),
            ):
                if not any(
                    abs(positions[k] - positions[old]) < 10 for old in chosen
                ):
                    chosen.append(int(k))

            if (
                len(chosen) < MIN_AXIS_LABELS
                or len(np.unique(values[chosen])) < MIN_AXIS_LABELS
            ):
                continue

            score = (
                len(chosen), float(np.ptp(positions[chosen])),
                -float(residual[chosen].mean()),
                float(np.mean([words[k].confidence for k in chosen])),
            )
            if best is None or score > best[0]:
                best = (score, chosen)

    if best is None:
        return []
    return sorted(
        [words[k] for k in best[1]],
        key=lambda w: w.cx if horizontal else w.cy,
    )


def select_bottom(words, bounds):
    left, top, right, bottom = bounds
    margin = max(45, int(BOTTOM_LABEL_BAND_FRACTION * (bottom - top)))
    candidates = [
        w for w in words
        if number(w.text) is not None
        and left - 15 <= w.cx <= right + 15
        and bottom <= w.cy <= bottom + margin
    ]
    if not candidates:
        return []

    tolerance = max(8, 0.7 * np.median([w.height for w in candidates]))
    choices = []
    for anchor in candidates:
        selected = consistent_sequence(
            [
                w for w in candidates
                if abs(w.cy - anchor.cy) <= tolerance
            ], True,
        )
        if selected:
            choices.append((
                float(np.median([w.cy - bottom for w in selected])),
                -len(selected), selected,
            ))
    return (
        min(choices, key=lambda value: (value[0], value[1]))[2]
        if choices else []
    )


def pooled_major_spacing(bottom, right):
    xp = np.array(sorted(w.cx for w in bottom), float)
    yp = np.array(sorted(w.cy for w in right), float)
    gaps = np.concatenate((np.diff(xp), np.diff(yp)))
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if not len(gaps):
        raise ValueError("Insufficient label-centre gaps.")

    spacing = float(np.median(gaps))
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Invalid pooled grid spacing.")
    return spacing


def reference_axis_candidates(
    positions, start, end, pooled, allow_half=False, anchor_at_end=False
):
    span = float(end - start)
    if allow_half:
        central = max(1, int(round(2.0 * span / pooled)))
        counts = [
            value / 2.0
            for value in sorted({
                max(1, central + offset) for offset in (-2, -1, 0, 1, 2)
            })
        ]
    else:
        central = max(1, int(round(span / pooled)))
        counts = sorted({max(1, central - 1), central, central + 1})

    result = []
    for count in counts:
        spacing = span / count
        largest = int(math.floor(count + 1e-9))
        if anchor_at_end:
            indices = np.clip(
                np.rint((end - positions) / spacing), 0, largest
            )
            predicted = end - indices * spacing
        else:
            indices = np.clip(
                np.rint((positions - start) / spacing), 0, largest
            )
            predicted = start + indices * spacing

        residual = positions - predicted
        normalized = residual / max(3.0, 0.12 * spacing)
        alignment = float(np.mean(
            2.0 * (np.sqrt(1.0 + normalized ** 2) - 1.0)
        ))
        prior = 0.25 * ((spacing - pooled) / (0.10 * pooled)) ** 2
        result.append((
            count, spacing, alignment + prior,
            float(np.median(np.abs(residual))),
        ))
    return result


def fit_reference_grid(bottom, right, bounds):
    left, top, end, bottom_y = bounds
    pooled = pooled_major_spacing(bottom, right)
    xp = np.array(sorted(w.cx for w in bottom), float)
    yp = np.array(sorted(w.cy for w in right), float)
    if not len(xp) or not len(yp):
        raise ValueError("Need bottom and right labels.")

    horizontal = reference_axis_candidates(xp, left, end, pooled)
    vertical = reference_axis_candidates(
        yp, top, bottom_y, pooled,
        allow_half=True, anchor_at_end=True,
    )
    choices = []
    for nx, gx, sx, ex in horizontal:
        for ny, gy, sy, ey in vertical:
            square = 0.25 * ((gx - gy) / (0.05 * pooled)) ** 2
            choices.append((sx + sy + square, nx, ny, gx, gy, ex, ey))

    _, nx, ny, gx, gy, ex, ey = min(
        choices, key=lambda choice: choice[0]
    )
    return ReferenceGrid(
        int(nx), float(ny), gx, gy, pooled, ex, ey,
        abs(gx - gy) / ((gx + gy) / 2.0),
    )


def validate_reference_grid(reference):
    if reference.aspect_error > REFERENCE_GRID_MAX_ASPECT_ERROR:
        raise ValueError(
            "Horizontal/vertical major spacings disagree by "
            f"{reference.aspect_error:.1%}."
        )
    if (
        reference.median_label_error_x
        > REFERENCE_GRID_MAX_MEDIAN_LABEL_ERROR * reference.major_x
        or reference.median_label_error_y
        > REFERENCE_GRID_MAX_MEDIAN_LABEL_ERROR * reference.major_y
    ):
        raise ValueError(
            "Boundary-fitted major positions disagree with labels."
        )
    if reference.aspect_error > REFERENCE_GRID_WARN_ASPECT_ERROR:
        print(
            "Warning: horizontal/vertical major spacings differ by "
            f"{reference.aspect_error:.1%}.", file=sys.stderr,
        )


def recover_bottom_labels(gray, labels, right_labels, bounds):
    if len(labels) < MIN_AXIS_LABELS or len(right_labels) < MIN_AXIS_LABELS:
        return labels

    labels = sorted(labels, key=lambda w: w.cx)
    reference = fit_reference_grid(labels, right_labels, bounds)
    spacing = reference.major_x
    slope, intercept = np.polyfit(
        np.array([w.cx for w in labels]),
        np.array([number(w.text) for w in labels], float), 1,
    )
    if not math.isfinite(slope) or slope <= 0:
        return labels

    tolerance = max(1e-9, abs(slope * spacing) * LABEL_STEP_TOLERANCE)
    left, _, right, bottom = bounds
    y0 = max(bottom + 1, min(w.top for w in labels) - 4)
    y1 = min(gray.shape[0], max(w.bottom for w in labels) + 4)
    if y1 <= y0:
        return labels

    half_width = max(
        8.0, min(0.42 * spacing, max(w.width for w in labels) * 0.8 + 6)
    )
    additions, unread = [], 0

    for center in np.linspace(left, right, reference.intervals_x + 1):
        if any(abs(center - w.cx) < 0.35 * spacing for w in labels):
            continue

        box = clip(
            (center - half_width, y0, center + half_width, y1), gray.shape
        )
        readings = numeric_readings(gray, box)
        matches = [
            text for text in readings
            if math.isclose(
                number(text), slope * center + intercept,
                rel_tol=0, abs_tol=tolerance,
            )
        ]
        if not matches:
            unread += 1
            continue

        text = max(sorted(set(matches)), key=matches.count)
        x0, by0, x1, by1 = box
        image = gray[by0:by1, x0:x1]
        if image.size == 0:
            continue

        count, labels_image, stats, _ = cv2.connectedComponentsWithStats(
            threshold_ink(image)
        )
        mask = np.zeros(image.shape, bool)
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] >= 3:
                mask |= labels_image == i

        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue

        lx, ty = x0 + int(xs.min()), by0 + int(ys.min())
        rx, by = x0 + int(xs.max()) + 1, by0 + int(ys.max()) + 1
        additions.append(Word(
            text, 50.0 + 5.0 * matches.count(text),
            lx, ty, rx - lx, by - ty,
        ))

    if unread:
        print(
            f"Warning: {unread} candidate bottom-label position(s) "
            "were unread; no labels were synthesized.",
            file=sys.stderr,
        )
    if not additions:
        return labels

    selected = consistent_sequence(labels + additions, True)
    return selected if len(selected) >= len(labels) else labels


def in_right_region(word, bounds):
    return (
        word.cx >= bounds[2]
        and word.bottom >= bounds[1] - AXIS_SEARCH_PADDING
        and word.top <= bounds[3] + AXIS_SEARCH_PADDING
    )


def find_numeric_column(words, bounds, width):
    anchors = [
        w for w in words
        if in_right_region(w, bounds)
        and w.confidence >= ANCHOR_MIN_CONFIDENCE
        and re.fullmatch(PLAIN, normalize(w.text))
        and "." in normalize(w.text)
        and w.height <= 2.5 * max(1, w.width)
    ]
    choices = []
    for anchor in anchors:
        group = [
            w for w in anchors
            if abs(w.left - anchor.left) <= max(12, anchor.height)
        ]
        rows = []
        for word in sorted(group, key=lambda w: w.cy):
            if not rows or abs(word.cy - rows[-1]) > 10:
                rows.append(word.cy)
        if len(rows) >= 2:
            choices.append((
                float(np.median([w.left for w in group])),
                -len(rows), group,
            ))

    if not choices:
        raise ValueError(
            "Need two confident aligned right-axis decimal labels."
        )

    anchors = unique_words(
        min(choices, key=lambda value: (value[0], value[1]))[2]
    )
    distinct = []
    for word in sorted(anchors, key=lambda w: -w.confidence):
        if not any(abs(word.cy - old.cy) < 10 for old in distinct):
            distinct.append(word)

    anchors = sorted(distinct, key=lambda w: w.cy)
    height = max(10.0, float(np.median([w.height for w in anchors])))
    return (
        max(bounds[2] + 1, min(w.left for w in anchors) - COLUMN_PADDING),
        min(width, max(w.right for w in anchors) + COLUMN_PADDING),
        height, anchors,
    )


def propose_rows(words, bounds, column, anchors):
    x0, x1, height = column
    rows = []

    def add(y):
        if (
            bounds[1] - AXIS_SEARCH_PADDING
            <= y <= bounds[3] + AXIS_SEARCH_PADDING
            and not any(abs(y - old) < 0.7 * height for old in rows)
        ):
            rows.append(float(y))

    for word in anchors:
        add(word.cy)

    spacing = float(np.median(np.diff([w.cy for w in anchors])))
    if spacing > 2 * height:
        reference = anchors[0].cy
        first = math.ceil(
            (bounds[1] - AXIS_SEARCH_PADDING - reference) / spacing
        )
        last = math.floor(
            (bounds[3] + AXIS_SEARCH_PADDING - reference) / spacing
        )
        for index in range(first, last + 1):
            add(reference + index * spacing)

    for word in sorted(words, key=lambda w: -w.confidence):
        if (
            in_right_region(word, bounds)
            and x0 <= word.cx <= x1
            and word.height <= 2.5 * height
        ):
            add(word.cy)

    add(bounds[3])
    return sorted(rows)


def nearest_ink_box(gray, box, center, height):
    x0, y0, x1, y1 = clip(box, gray.shape)
    image = gray[y0:y1, x0:x1]
    if image.size == 0:
        return None

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        threshold_ink(image)
    )
    mask = np.zeros(image.shape, bool)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 3:
            mask |= labels == i

    occupied = np.flatnonzero(mask.any(axis=1))
    if not len(occupied):
        return None

    groups = np.split(
        occupied,
        np.flatnonzero(
            np.diff(occupied) > max(2, round(0.18 * height))
        ) + 1,
    )
    candidates = []
    for group in groups:
        top, bottom = int(group[0]), int(group[-1]) + 1
        _, xs = np.nonzero(mask[top:bottom])
        if not len(xs):
            continue

        result = (
            x0 + int(xs.min()), y0 + top,
            x0 + int(xs.max()) + 1, y0 + bottom,
        )
        offset = abs((result[1] + result[3]) / 2 - center)
        if bottom - top <= 1.8 * height and offset <= 0.90 * height:
            candidates.append((offset, -(bottom - top), result))

    return (
        min(candidates, key=lambda value: (value[0], value[1]))[2]
        if candidates else None
    )


def read_label_rows(gray, rows, column, anchors):
    x0, x1, height = column
    candidates = list(anchors)
    for center in rows:
        broad = clip((
            x0, center - ROW_HALF_HEIGHT_FACTOR * height,
            x1, center + ROW_HALF_HEIGHT_FACTOR * height,
        ), gray.shape)
        found = numeric_readings(gray, broad)
        box = nearest_ink_box(gray, broad, center, height)
        if box is None:
            box = broad
        else:
            left, top, right, bottom = box
            found += numeric_readings(
                gray, (left - 3, top - 3, right + 3, bottom + 3)
            )

        left, top, right, bottom = box
        for text in set(found):
            candidates.append(Word(
                text, 40 + 5 * found.count(text),
                left, top, max(1, right - left), max(1, bottom - top),
            ))
    return unique_words(candidates)


def detect_title_box(gray, bounds, top_band):
    image = gray.copy()
    x0, y0, x1, y1 = top_band
    image[y0:y1, x0:x1] = 255

    x0, y0, x1, y1 = clip((
        bounds[2] + 1, bounds[1] - AXIS_SEARCH_PADDING,
        gray.shape[1], bounds[3] + AXIS_SEARCH_PADDING,
    ), gray.shape)
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        return None

    rotated = cv2.rotate(region, cv2.ROTATE_90_CLOCKWISE)
    boxes = []
    for word in read_words(rotated):
        if not re.search(r"[A-Za-z]{3,}", word.text):
            continue
        box = (
            x0 + word.top,
            y0 + region.shape[0] - word.right,
            x0 + word.bottom,
            y0 + region.shape[0] - word.left,
        )
        if not intersection(box, top_band):
            boxes.append(box)

    if not boxes:
        return None
    return clip((
        min(b[0] for b in boxes) - TITLE_PADDING,
        min(b[1] for b in boxes) - TITLE_PADDING,
        max(b[2] for b in boxes) + TITLE_PADDING,
        max(b[3] for b in boxes) + TITLE_PADDING,
    ), gray.shape)


def grid_nodes(length, spacing):
    step = max(8.0, GRID_NODE_INTERVALS * spacing)
    count = max(3, int(math.ceil((length - 1) / step)) + 1)
    return np.linspace(0, length - 1, count)


def measure_grid_offsets(
    stripe, references, nodes,
    along_spacing, across_spacing, max_search, prior=None,
):
    across_length, along_length = stripe.shape
    limit = min(max_search, GRID_MAX_SPACING_FRACTION * across_spacing)
    values = np.zeros((len(references), len(nodes)), float)
    weights = np.zeros_like(values)
    if limit < 1 or len(references) == 0:
        return values, weights, limit

    if prior is None:
        prior = np.zeros_like(values)
        radius = limit
    else:
        radius = min(limit, max(2.0, 0.20 * across_spacing))

    half_window = max(
        6, int(round(0.5 * GRID_WINDOW_INTERVALS * along_spacing))
    )
    offsets = np.arange(
        -math.floor(radius), math.floor(radius) + 1, dtype=float
    )
    slopes = np.linspace(
        -GRID_MAX_LOCAL_SLOPE, GRID_MAX_LOCAL_SLOPE, GRID_SLOPE_SAMPLES
    )

    for i, reference in enumerate(references):
        if reference < limit + 2 or reference > across_length - limit - 3:
            continue

        for j, node in enumerate(nodes):
            a = max(2, int(round(node)) - half_window)
            b = min(along_length - 2, int(round(node)) + half_window + 1)
            if b - a < max(8, half_window):
                continue

            columns = np.arange(a, b, 2, dtype=int)
            along = columns.astype(float)
            relative = along - node
            base = np.interp(along, nodes, prior[i])
            center_prior = float(prior[i, j])
            scores = np.full(len(offsets), -np.inf)

            for k, offset in enumerate(offsets):
                if abs(center_prior + offset) > limit:
                    continue
                displacement = (
                    base[None, :] + offset
                    + slopes[:, None] * relative[None, :]
                )
                rows = reference + displacement
                valid = (
                    (np.abs(displacement) <= limit)
                    & (rows >= 1) & (rows < across_length - 2)
                )
                counts = valid.sum(axis=1)
                usable = counts >= 0.85 * len(columns)
                if not np.any(usable):
                    continue

                safe = np.clip(rows, 0, across_length - 2)
                lower = np.floor(safe).astype(np.intp)
                fraction = safe - lower
                samples = (
                    (1.0 - fraction) * stripe[lower, columns]
                    + fraction * stripe[lower + 1, columns]
                )
                support = np.sum(
                    np.where(valid, samples, 0.0), axis=1
                ) / np.maximum(counts, 1)
                support[~usable] = -np.inf
                scores[k] = float(support.max())

            finite = np.isfinite(scores)
            if np.count_nonzero(finite) < 3:
                continue

            peak = float(scores[finite].max())
            baseline = float(np.median(scores[finite]))
            contrast = peak - baseline
            if peak < GRID_MIN_SUPPORT or contrast < GRID_MIN_CONTRAST:
                continue

            best = int(np.argmax(scores))
            if (
                best == 0 or best == len(scores) - 1
                or not finite[best - 1] or not finite[best + 1]
            ):
                continue

            cutoff = baseline + 0.65 * contrast
            lo = hi = best
            while lo > 0 and scores[lo - 1] >= cutoff:
                lo -= 1
            while hi + 1 < len(scores) and scores[hi + 1] >= cutoff:
                hi += 1

            if (
                lo == 0 or hi == len(scores) - 1
                or not finite[lo - 1] or not finite[hi + 1]
            ):
                continue

            outside = finite.copy()
            outside[max(0, lo - 1):min(len(scores), hi + 2)] = False
            competing = (
                float(scores[outside].max()) if outside.any() else baseline
            )
            margin = peak - competing
            if margin < GRID_MIN_PEAK_MARGIN:
                continue

            selected = np.arange(lo, hi + 1)
            mass = np.maximum(scores[selected] - baseline, 1e-9)
            correction = float(np.average(offsets[selected], weights=mass))
            values[i, j] = center_prior + correction
            weights[i, j] = min(
                1.0, max(0.05, peak * min(1.0, margin / 0.15))
            )

    return values, weights, limit


def solve_displacement(values, weights, initial=None):
    if values.size == 0 or not np.any(weights):
        return np.zeros_like(values)

    def apply(field):
        result = (weights + GRID_REFERENCE_WEIGHT) * field
        along = field[:, 1:] - field[:, :-1]
        result[:, :-1] -= GRID_ALONG_SMOOTHNESS * along
        result[:, 1:] += GRID_ALONG_SMOOTHNESS * along
        across = field[1:, :] - field[:-1, :]
        result[:-1, :] -= GRID_ACROSS_SMOOTHNESS * across
        result[1:, :] += GRID_ACROSS_SMOOTHNESS * across
        return result

    diagonal = (weights + GRID_REFERENCE_WEIGHT).copy()
    diagonal[:, :-1] += GRID_ALONG_SMOOTHNESS
    diagonal[:, 1:] += GRID_ALONG_SMOOTHNESS
    diagonal[:-1, :] += GRID_ACROSS_SMOOTHNESS
    diagonal[1:, :] += GRID_ACROSS_SMOOTHNESS

    rhs = weights * values
    field = np.zeros_like(values) if initial is None else initial.copy()
    residual = rhs - apply(field)
    z = residual / diagonal
    direction = z.copy()
    rz = float(np.sum(residual * z))
    tolerance = 1e-7 * max(1.0, float(np.linalg.norm(rhs)))

    for _ in range(GRID_CG_ITERATIONS):
        if float(np.linalg.norm(residual)) <= tolerance:
            break
        product = apply(direction)
        denominator = float(np.sum(direction * product))
        if denominator <= 0 or rz <= 0:
            break
        alpha = rz / denominator
        field += alpha * direction
        residual -= alpha * product
        z = residual / diagonal
        new_rz = float(np.sum(residual * z))
        if new_rz <= 0:
            break
        direction = z + (new_rz / rz) * direction
        rz = new_rz

    return field


def robust_grid_fit(values, weights, limit, initial=None):
    fitted = solve_displacement(values, weights, initial)
    adjusted = weights.copy()
    if np.any(weights):
        residual = np.abs(values - fitted)
        scale = max(1.0, 0.15 * limit)
        large = residual > scale
        adjusted[large] *= (scale / residual[large]) ** 2
        fitted = solve_displacement(values, adjusted, initial=fitted)
    return np.clip(fitted, -limit, limit), adjusted


def fit_grid_family(
    ink, references, along_spacing, across_spacing, max_search
):
    nodes = grid_nodes(ink.shape[1], along_spacing)
    stripe = cv2.blur(
        ink.astype(np.float32), (1, 3), borderType=cv2.BORDER_REPLICATE
    )
    v1, w1, limit = measure_grid_offsets(
        stripe, references, nodes,
        along_spacing, across_spacing, max_search,
    )
    f1, w1 = robust_grid_fit(v1, w1, limit)
    v2, w2, _ = measure_grid_offsets(
        stripe, references, nodes,
        along_spacing, across_spacing, max_search, prior=f1,
    )

    retained = w1.copy()
    retained[w2 > 0] *= 0.35
    weights = retained + w2
    values = np.divide(
        retained * v1 + w2 * v2,
        weights, out=np.zeros_like(v1), where=weights > 0,
    )
    fitted, weights = robust_grid_fit(values, weights, limit, initial=f1)

    along = np.arange(ink.shape[1], dtype=float)
    paths, measurements = [], []
    for i, reference in enumerate(references):
        displacement = np.interp(along, nodes, fitted[i])
        if i == 0 or i == len(references) - 1:
            displacement = np.zeros_like(displacement)
        paths.append(np.column_stack((along, reference + displacement)))
        for j in np.flatnonzero(weights[i] >= 0.05):
            measurements.append((
                float(nodes[j]), float(reference + values[i, j])
            ))

    return (
        paths, measurements,
        int(np.count_nonzero(weights >= 0.05)), int(weights.size),
    )


def generate_grid(ink, bounds, bottom, right, max_search, straight=False):
    reference = fit_reference_grid(bottom, right, bounds)
    validate_reference_grid(reference)

    left, top, end, bottom_y = bounds
    width, height = end - left + 1, bottom_y - top + 1
    xs = np.linspace(
        0.0, float(end - left), reference.minor_intervals_x + 1
    )
    ys = np.linspace(
        0.0, float(bottom_y - top), reference.minor_intervals_y + 1
    )
    local_ink = ink[top:bottom_y + 1, left:end + 1]
    measurements = []
    accepted = attempted = 0

    if straight:
        horizontal = [
            np.column_stack((
                np.arange(width, dtype=float), np.full(width, y)
            ))
            for y in ys
        ]
        vertical = [
            np.column_stack((
                np.full(height, x), np.arange(height, dtype=float)
            ))
            for x in xs
        ]
    else:
        hp, hm, ha, ht = fit_grid_family(
            local_ink, ys, reference.minor_x, reference.minor_y, max_search
        )
        vp, vm, va, vt = fit_grid_family(
            local_ink.T, xs, reference.minor_y, reference.minor_x, max_search
        )
        horizontal = hp
        vertical = [path[:, ::-1].copy() for path in vp]
        measurements.extend((x + left, y + top) for x, y in hm)
        measurements.extend((x + left, y + top) for y, x in vm)
        accepted, attempted = ha + va, ht + vt

    origin = np.array([left, top])
    horizontal = [path + origin for path in horizontal]
    vertical = [path + origin for path in vertical]
    local_mask = np.zeros((height, width), np.uint8)

    for path in horizontal + vertical:
        cv2.polylines(
            local_mask, [np.rint(path - origin).astype(np.int32)],
            False, 255, MASK_WIDTH, lineType=cv2.LINE_8,
        )
    cv2.rectangle(
        local_mask, (0, 0), (width - 1, height - 1), 255, MASK_WIDTH
    )
    mask = np.zeros(ink.shape, bool)
    mask[top:bottom_y + 1, left:end + 1] = local_mask > 0

    grid = GridFit(
        mask, horizontal, vertical, measurements,
        accepted, attempted, reference,
    )
    references = [
        np.array([[left, top + y], [end, top + y]], float) for y in ys
    ] + [
        np.array([[left + x, top], [left + x, bottom_y]], float) for x in xs
    ]
    return grid, references


def grid_fit_preview(base, bounds, grid, references):
    image = base.copy()
    for path in references:
        cv2.polylines(
            image, [np.rint(path).astype(np.int32)], False, (255, 80, 0), 1
        )
    for path in grid.horizontal + grid.vertical:
        cv2.polylines(
            image, [np.rint(path).astype(np.int32)], False, (0, 180, 0), 1
        )
    for x, y in grid.measurements:
        cv2.circle(image, (round(x), round(y)), 2, (0, 140, 255), -1)
    draw_frame(image, bounds)
    return image


def fit_axis(labels, horizontal, multiplier=1):
    positions = np.array([w.cx if horizontal else w.cy for w in labels])
    values = np.array(
        [number(w.text) for w in labels], float
    ) * multiplier
    return np.polyfit(positions, values, 1)


def run_centers(column):
    rows = np.flatnonzero(column)
    if not len(rows):
        return np.array([], float)
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    return np.array([
        group.mean() for group in groups if len(group) <= 12
    ])


def column_candidates(clean, bounds, x):
    _, top, _, bottom = bounds
    return run_centers(clean[top + 2:bottom - 1, x]) + top + 2


def tags_for_count(count):
    if count == 7:
        return [str(value) for value in range(7, 0, -1)]
    return [f"{6.5 - 0.5 * i:g}" for i in range(12)]


def choose_seeds(clean, bounds, requested_count=None, high_speed=False):
    left, top, right, bottom = bounds
    margin = max(6, (bottom - top) // 100)
    allowed = (requested_count,) if requested_count is not None else (7, 12)
    choices = []

    for x in np.linspace(left + margin, right - margin, 180).astype(int):
        centers = (
            run_centers(clean[top + margin:bottom - margin + 1, x])
            + top + margin
        )
        separated = []
        for y in centers:
            if not separated or y - separated[-1] >= 5:
                separated.append(float(y))
        if len(separated) in allowed:
            choices.append((int(x), np.array(separated)))

    if not choices:
        return None

    count = max(
        allowed, key=lambda n: sum(len(ys) == n for _, ys in choices)
    )
    choices = [value for value in choices if len(value[1]) == count]

    if high_speed:
        earliest = min(value[0] for value in choices)
        band = max(5, int(0.03 * (right - left)))
        choices = [
            value for value in choices if value[0] <= earliest + band
        ]

    return max(
        choices,
        key=lambda value: (
            float(np.min(np.diff(value[1])))
            + 5 * min(value[0] - left, right - value[0])
            / max(1, right - left)
        ),
    )


def assignment_gates(predicted):
    if len(predicted) == 1:
        return np.array([ASSIGN_MAX_GATE], float)
    distances = np.maximum(np.diff(predicted), 1.0)
    local = np.full(len(predicted), float(np.median(distances)))
    local[:-1] = np.minimum(local[:-1], distances)
    local[1:] = np.minimum(local[1:], distances)
    return np.clip(0.45 * local, ASSIGN_MIN_GATE, ASSIGN_MAX_GATE)


def ordered_assignment(predicted, candidates, gates, lower=None, upper=None):
    predicted = np.asarray(predicted, float)
    candidates = np.asarray(candidates, float)
    count, total = len(predicted), len(candidates)
    assigned = np.full(count, np.nan)
    if count == 0:
        return assigned, 0.0
    if total == 0:
        return assigned, count * ASSIGN_MISSING_COST

    if lower is None:
        lower = np.full(count, -np.inf)
    if upper is None:
        upper = np.full(count, np.inf)

    cost = np.full((count + 1, total + 1), np.inf)
    action = np.zeros((count + 1, total + 1), np.uint8)
    cost[0, :] = 0.0

    for i in range(1, count + 1):
        cost[i, 0] = i * ASSIGN_MISSING_COST
        action[i, 0] = 2

    for i in range(1, count + 1):
        for j in range(1, total + 1):
            best, operation = cost[i, j - 1], 1
            missing = cost[i - 1, j] + ASSIGN_MISSING_COST
            if missing < best:
                best, operation = missing, 2

            candidate = candidates[j - 1]
            distance = abs(predicted[i - 1] - candidate)
            if (
                distance <= gates[i - 1]
                and lower[i - 1] <= candidate <= upper[i - 1]
            ):
                matched = (
                    cost[i - 1, j - 1] + (distance / gates[i - 1]) ** 2
                )
                if matched < best:
                    best, operation = matched, 3

            cost[i, j] = best
            action[i, j] = operation

    i, j = count, total
    while i > 0:
        operation = action[i, j]
        if operation == 3:
            assigned[i - 1] = candidates[j - 1]
            i -= 1
            j -= 1
        elif operation == 2:
            i -= 1
        elif j > 0:
            j -= 1
        else:
            break

    return assigned, float(cost[count, total])


def visible_assignment(
    predicted, candidates, bounds,
    tolerance=None, allowed=None, reject_ambiguous=False,
):
    _, top, _, bottom = bounds
    visible = (
        np.isfinite(predicted)
        & (predicted >= top + 2) & (predicted <= bottom - 2)
    )
    if allowed is not None:
        visible &= allowed

    active = np.flatnonzero(visible)
    result = np.full(len(predicted), np.nan)

    if reject_ambiguous and len(active) > 1:
        close = (
            np.diff(predicted[active]) < MIN_DISTINCT_STROKE_SEPARATION
        )
        ambiguous = np.zeros(len(active), bool)
        ambiguous[:-1] |= close
        ambiguous[1:] |= close
        active = active[~ambiguous]

    if not len(active):
        return result, 0.0, 0

    gates = (
        assignment_gates(predicted[active])
        if tolerance is None else np.full(len(active), tolerance)
    )
    assigned, cost = ordered_assignment(
        predicted[active], candidates, gates
    )
    result[active] = assigned
    return result, cost, len(active)


def ordered_initial_positions(values, weights):
    index = np.arange(len(values), dtype=float)
    shifted = np.asarray(values, float) - SPLINE_MIN_SEPARATION * index
    blocks = []

    for i, (value, weight) in enumerate(zip(shifted, weights)):
        blocks.append([i, i + 1, float(weight), float(value)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            b, a = blocks.pop(), blocks.pop()
            weight_sum = a[2] + b[2]
            mean = (a[2] * a[3] + b[2] * b[3]) / weight_sum
            blocks.append([a[0], b[1], weight_sum, mean])

    result = np.empty(len(values), float)
    for start, end, _, mean in blocks:
        result[start:end] = mean
    return result + SPLINE_MIN_SEPARATION * index


def robust_local_line(xs, ys, join_x):
    valid = np.isfinite(ys)
    x = np.asarray(xs[valid], float) - join_x
    y = np.asarray(ys[valid], float)
    if len(x) < JOIN_STRIP_MIN_POINTS or np.ptp(x) < 4:
        return None

    design = np.column_stack((np.ones(len(x)), x))
    weights = np.ones(len(x))
    coefficients = np.zeros(2)

    for _ in range(3):
        coefficients = np.linalg.lstsq(
            design * np.sqrt(weights)[:, None],
            y * np.sqrt(weights), rcond=None,
        )[0]
        residual = y - design @ coefficients
        scale = max(
            1.0,
            1.4826 * float(np.median(
                np.abs(residual - np.median(residual))
            )),
        )
        weights = 1.0 / np.sqrt(
            1.0 + (residual / (2.0 * scale)) ** 2
        )

    residual = y - design @ coefficients
    if float(np.median(np.abs(residual))) > 3.0:
        return None

    center = float(np.mean(x))
    spread = float(np.sum((x - center) ** 2))
    leverage = 1.0 / len(x) + center * center / max(spread, 1.0)
    sigma = max(1.5, scale * math.sqrt(leverage))
    return float(coefficients[0]), float(coefficients[1]), sigma


def recover_high_join_strip(
    clean, bounds, join_pixel, initial_pixels,
    major_spacing, print_sigma_pixels,
):
    left, _, right, _ = bounds
    width = int(np.clip(
        JOIN_STRIP_MAJOR_FRACTION * major_spacing,
        JOIN_STRIP_MIN_WIDTH, JOIN_STRIP_MAX_WIDTH,
    ))
    first = max(left + 2, int(math.ceil(join_pixel)))
    last = min(right - 2, first + width)
    xs = np.arange(first, last + 1, 2, dtype=int)

    if len(xs) < JOIN_STRIP_MIN_POINTS:
        return (
            xs, np.full((len(initial_pixels), len(xs)), np.nan),
            np.full(len(initial_pixels), np.nan),
            np.zeros(len(initial_pixels)),
            np.full(len(initial_pixels), np.nan),
        )

    candidates = [
        column_candidates(clean, bounds, int(x)) for x in xs
    ]
    intercepts = initial_pixels.copy()
    slopes = np.zeros(len(intercepts))
    observations = np.full((len(intercepts), len(xs)), np.nan)
    estimated = np.full(len(intercepts), np.nan)
    sigmas = np.full(len(intercepts), np.nan)

    separation = float(np.median(
        np.maximum(np.diff(initial_pixels), 1.0)
    ))
    search_radius = min(
        max(8.0, 2.0 * print_sigma_pixels + JOIN_PIXEL_TOLERANCE),
        max(8.0, 0.8 * separation),
    )

    for _ in range(JOIN_STRIP_PASSES):
        observations[:] = np.nan
        for j, x in enumerate(xs):
            prediction = intercepts + slopes * (x - join_pixel)
            prediction = ordered_initial_positions(
                prediction, np.ones(len(prediction))
            )
            observations[:, j], _, _ = visible_assignment(
                prediction, candidates[j], bounds, tolerance=search_radius
            )

        proposed = initial_pixels.copy()
        proposed_slopes = np.zeros(len(intercepts))
        proposed_sigmas = np.full(len(intercepts), np.nan)
        estimated[:] = np.nan
        weights = np.ones(len(intercepts))

        for k in range(len(intercepts)):
            fit = robust_local_line(xs, observations[k], join_pixel)
            if fit is None:
                continue
            value, slope, sigma = fit
            estimated[k] = value
            proposed[k] = value
            proposed_slopes[k] = slope
            proposed_sigmas[k] = sigma
            weights[k] = 4.0

        intercepts = ordered_initial_positions(proposed, weights)
        slopes, sigmas = proposed_slopes, proposed_sigmas

    return xs, observations, estimated, slopes, sigmas


def pixel_for_numerical_x(bounds, x_fit, numerical_x):
    left, _, right, _ = bounds
    if x_fit[0] <= 0:
        raise ValueError("Invalid horizontal calibration.")

    pixel = float((numerical_x - x_fit[1]) / x_fit[0])
    tolerance = max(3.0, 0.01 * (right - left))
    if pixel < left - tolerance or pixel > right + tolerance:
        raise ValueError(
            f"Speed-length ratio {numerical_x:g} is outside "
            "the calibrated page domain."
        )
    return pixel


def build_join_guidance(
    anchors, tags, bounds, x_fit, y_fit,
    clean, major_spacing, print_sigma,
):
    if not anchors:
        return None

    left, top, right, bottom = bounds
    height = float(bottom - top)
    if y_fit[0] >= 0:
        raise ValueError("Invalid right-axis calibration direction.")

    by_tag = {anchor.tag: anchor for anchor in anchors}
    for value in range(1, 8):
        if str(value) not in by_tag:
            raise ValueError(
                "Paired initialization requires low-speed tags 1–7."
            )

    pixel_x = pixel_for_numerical_x(bounds, x_fit, JOIN_X)
    initial_pixels = []
    prior_indices, prior_targets, prior_sigmas = [], [], []

    def pixel_y(cr):
        return (cr - y_fit[1]) / y_fit[0]

    for index, tag in enumerate(tags):
        value = float(tag)
        if value.is_integer():
            anchor = by_tag[str(int(value))]
            target_pixel = pixel_y(anchor.value)
            sigma_numeric = math.sqrt(
                anchor.sigma ** 2
                + (JOIN_PIXEL_TOLERANCE * abs(y_fit[0])) ** 2
                + print_sigma ** 2
            )
            prior_indices.append(index)
            prior_targets.append((target_pixel - top) / height)
            prior_sigmas.append(max(
                sigma_numeric / abs(y_fit[0]) / height, 1e-9
            ))
            initial_pixels.append(target_pixel)
        else:
            a = pixel_y(by_tag[str(int(math.floor(value)))].value)
            b = pixel_y(by_tag[str(int(math.ceil(value)))].value)
            initial_pixels.append(0.5 * (a + b))

    initial_pixels = ordered_initial_positions(
        np.asarray(initial_pixels, float), np.ones(len(tags))
    )
    local_xs, local_observations, estimates, slopes, sigmas = (
        recover_high_join_strip(
            clean, bounds, pixel_x, initial_pixels,
            major_spacing, print_sigma / abs(y_fit[0]),
        )
    )
    local_indices = np.flatnonzero(np.isfinite(estimates))
    local_targets = (estimates[local_indices] - top) / height
    local_sigmas = sigmas[local_indices] / height

    proposed = initial_pixels.copy()
    proposed[local_indices] = estimates[local_indices]
    weights = np.ones(len(tags))
    weights[local_indices] = 4.0
    initial_pixels = ordered_initial_positions(proposed, weights)

    return JoinGuidance(
        pixel_x, (pixel_x - left) / (right - left),
        initial_pixels, slopes,
        np.asarray(prior_indices, int),
        np.asarray(prior_targets, float),
        np.asarray(prior_sigmas, float),
        local_indices,
        np.asarray(local_targets, float),
        np.maximum(np.asarray(local_sigmas, float), 1e-9),
        local_xs, local_observations,
    )


def initial_assignments(
    xs, candidates, seed_x, seed_ys,
    bounds, high_speed, guidance=None,
):
    count = len(seed_ys)
    observations = np.full((count, len(xs)), np.nan)
    seed_index = int(np.argmin(np.abs(xs - seed_x)))
    _, top, _, bottom = bounds

    if guidance is None:
        observations[:, seed_index] = seed_ys
        start_positions = seed_ys.astype(float).copy()
        starting_slopes = np.zeros(count)
    else:
        assigned, _, _ = visible_assignment(
            seed_ys, candidates[seed_index], bounds
        )
        observations[:, seed_index] = assigned
        start_positions = seed_ys.astype(float).copy()
        valid = np.isfinite(assigned)
        start_positions[valid] = assigned[valid]
        starting_slopes = guidance.initial_slopes.copy()

    for direction in (-1, 1):
        positions = start_positions.copy()
        velocity = starting_slopes.copy()
        last_x = np.full(count, float(seed_x))
        last_y = start_positions.copy()
        previous_x = seed_x
        indices = (
            range(seed_index - 1, -1, -1)
            if direction < 0 else range(seed_index + 1, len(xs))
        )

        for j in indices:
            x = int(xs[j])
            prediction = positions + velocity * (x - previous_x)
            if not high_speed:
                prediction = np.clip(prediction, top + 2, bottom - 2)
            prediction = np.maximum.accumulate(prediction)
            assigned, _, _ = visible_assignment(
                prediction, candidates[j], bounds
            )
            valid = np.isfinite(assigned)
            observations[valid, j] = assigned[valid]

            for k in np.flatnonzero(valid):
                distance = x - last_x[k]
                if distance != 0:
                    slope = (assigned[k] - last_y[k]) / distance
                    velocity[k] = (
                        0.75 * velocity[k]
                        + 0.25 * np.clip(slope, -3.0, 3.0)
                    )
                last_x[k], last_y[k] = x, assigned[k]

            positions = prediction
            positions[valid] = assigned[valid]
            previous_x = x

    if guidance is not None:
        for local_j, x in enumerate(guidance.local_xs):
            matches = np.flatnonzero(xs == x)
            if not len(matches):
                continue
            j = int(matches[0])
            values = guidance.local_observations[:, local_j]
            valid = np.isfinite(values)
            observations[valid, j] = values[valid]

    return observations


def build_ink_evidence(clean, grid_mask, bounds):
    left, top, right, bottom = bounds
    local = clean[top:bottom + 1, left:right + 1]
    if not np.any(local):
        raise ValueError("No surviving ink inside the plotting frame.")

    background = np.where(local, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(
        background, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    ).astype(float)
    return InkEvidence(
        distance, grid_mask[top:bottom + 1, left:right + 1].copy(), bounds
    )


def sample_ink(evidence, xs, predicted):
    left, top, right, bottom = evidence.bounds
    local_x = np.clip(
        np.rint(np.asarray(xs) - left).astype(int),
        0, evidence.distance.shape[1] - 1,
    )
    local_y = predicted - top
    finite = np.isfinite(local_y)
    inside = (
        finite & (predicted >= top + 1) & (predicted <= bottom - 1)
    )
    safe_y = np.clip(
        np.where(finite, local_y, 0.0),
        0, evidence.distance.shape[0] - 1.000001,
    )
    lower = np.floor(safe_y).astype(np.intp)
    fraction = safe_y - lower
    a = evidence.distance[lower, local_x]
    b = evidence.distance[lower + 1, local_x]
    distances = (1.0 - fraction) * a + fraction * b
    derivative = b - a
    nearest_row = np.clip(
        np.rint(safe_y).astype(np.intp),
        0, evidence.grid_mask.shape[0] - 1,
    )
    masked = evidence.grid_mask[nearest_row, local_x]
    derivative[~inside] = 0.0
    return distances, derivative, inside, masked


def assess_image_support(
    bounds, basis, coefficients, sample_xs, evidence, major_spacing
):
    left, top, right, bottom = bounds
    dense_xs = np.arange(left, right + 1, dtype=int)
    predicted = evaluate_family(bounds, basis, coefficients, dense_xs)
    distances, _, inside, masked = sample_ink(
        evidence, dense_xs, predicted
    )
    near = inside & (distances <= INK_DISTANCE_TOLERANCE)
    obscured = inside & masked & ~near
    evaluable = inside & (~masked | near)

    window_width = max(12.0, SUPPORT_WINDOW_MAJOR * major_spacing)
    window_ids = np.floor(
        (dense_xs - left) / window_width
    ).astype(int)
    windows = int(window_ids.max()) + 1
    states = np.full((len(coefficients), windows), -1, np.int8)
    maximum_gap = max(1, int(math.ceil(
        SUPPORT_MAX_GAP_MAJOR * major_spacing / window_width
    )))
    right_edges = np.full(len(coefficients), float(left))
    fractions = np.zeros(len(coefficients), float)
    valid = np.zeros(len(coefficients), bool)

    for k in range(len(coefficients)):
        for window in range(windows):
            columns = window_ids == window
            n_inside = int(np.count_nonzero(inside[k] & columns))
            if n_inside == 0:
                states[k, window] = -2
                continue

            n_evaluable = int(np.count_nonzero(evaluable[k] & columns))
            required = max(4, int(math.ceil(
                SUPPORT_MIN_EVALUABLE_FRACTION * n_inside
            )))
            if n_evaluable < required:
                continue

            n_supported = int(np.count_nonzero(
                near[k] & evaluable[k] & columns
            ))
            fraction = n_supported / n_evaluable
            states[k, window] = (
                1 if fraction >= SUPPORT_MIN_FRACTION else 0
            )

        reliable = np.flatnonzero(states[k] == 1)
        if not len(reliable):
            continue

        last_reliable = int(reliable[0])
        for window in reliable[1:]:
            if int(window) - last_reliable - 1 > maximum_gap:
                break
            last_reliable = int(window)

        retained_columns = window_ids <= last_reliable
        supported_points = np.flatnonzero(near[k] & retained_columns)
        if len(supported_points) < MIN_HIGH_SUPPORT_COLUMNS:
            continue

        right_edges[k] = float(dense_xs[supported_points[-1]])
        domain = dense_xs <= right_edges[k]
        known = evaluable[k] & domain
        denominator = int(np.count_nonzero(known))
        fractions[k] = (
            np.count_nonzero(near[k] & known) / denominator
            if denominator else 0.0
        )
        valid[k] = (
            denominator >= MIN_HIGH_SUPPORT_COLUMNS
            and fractions[k] >= SUPPORT_MIN_FRACTION
        )

    sampled_mask = sample_xs[None, :] <= right_edges[:, None]
    return ImageSupport(
        dense_xs, near, obscured, inside, right_edges,
        sampled_mask, states, fractions, valid,
    )


def spline_basis(basis_count):
    interior_count = basis_count - SPLINE_DEGREE - 1
    interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
    knots = np.concatenate((
        np.zeros(SPLINE_DEGREE + 1), interior,
        np.ones(SPLINE_DEGREE + 1),
    ))
    return BSpline(
        knots, np.eye(basis_count), SPLINE_DEGREE, extrapolate=False
    )


def coefficient_constraints(count, basis_count, height, basis, guidance):
    matrix = np.zeros((
        (count - 1) * basis_count, count * basis_count
    ))
    for k in range(count - 1):
        for j in range(basis_count):
            row = k * basis_count + j
            matrix[row, k * basis_count + j] = -1.0
            matrix[row, (k + 1) * basis_count + j] = 1.0

    constraints = [
        LinearConstraint(matrix, SPLINE_MIN_SEPARATION / height, np.inf)
    ]
    if guidance is not None:
        join_design = np.asarray(
            basis(guidance.u, extrapolate=True), float
        )
        join_matrix = np.zeros((count - 1, count * basis_count))
        for k in range(count - 1):
            join_matrix[
                k, k * basis_count:(k + 1) * basis_count
            ] = -join_design
            join_matrix[
                k, (k + 1) * basis_count:(k + 2) * basis_count
            ] = join_design
        constraints.append(LinearConstraint(
            join_matrix, SPLINE_MIN_SEPARATION / height, np.inf
        ))

    return constraints


def robust_penalty(residual):
    root = np.sqrt(1.0 + residual * residual)
    return 2.0 * (root - 1.0), 2.0 * residual / root


def optimize_coefficients(
    coefficients, basis, xs, observations, included,
    seed_x, seed_ys, bounds, shape_weight,
    high_speed, guidance, evidence, ink_weight,
    support=None, snapshot=None,
):
    left, top, right, bottom = bounds
    height = float(bottom - top)
    count, basis_count = coefficients.shape

    design = np.asarray(basis((xs - left) / (right - left)), float)
    seed_design = np.asarray(
        basis((seed_x - left) / (right - left)), float
    )
    regularization_x = np.linspace(0.0, 1.0, 100)
    slope_design = np.asarray(
        basis.derivative(1)(regularization_x), float
    )
    curvature_design = np.asarray(
        basis.derivative(2)(regularization_x), float
    )

    valid = np.isfinite(observations) & included[None, :]
    if support is not None:
        valid &= support.sampled_mask
    curves, columns = np.nonzero(valid)
    targets = (observations[curves, columns] - top) / height
    if len(targets) < count * 3:
        raise ValueError("Too few assigned image observations.")

    rows = design[columns]
    scale = height / SPLINE_ROBUST_SCALE
    seed_targets = (seed_ys - top) / height

    if high_speed:
        counts = np.bincount(curves, minlength=count)
        if np.any(counts == 0):
            raise ValueError("A high-speed curve has no observations.")
        data_weights = 1.0 / (count * counts[curves])
    else:
        data_weights = np.full(len(targets), 1.0 / len(targets))

    constraints = coefficient_constraints(
        count, basis_count, height, basis, guidance
    )
    join_design = (
        np.asarray(basis(guidance.u, extrapolate=True), float)
        if guidance is not None else None
    )

    ink_weights = None
    if high_speed and ink_weight > 0:
        starting_prediction = top + height * (coefficients @ design.T)
        _, _, _, starting_mask = sample_ink(
            evidence, xs, starting_prediction
        )
        ink_weights = np.where(starting_mask, INK_MASKED_WEIGHT, 1.0)
        ink_weights *= included[None, :]
        if support is not None:
            ink_weights *= support.sampled_mask
        normalizers = np.maximum(
            np.sum(ink_weights, axis=1, keepdims=True), 1.0
        )
        ink_weights = ink_weights / normalizers / count

    def objective(flat):
        c = flat.reshape(count, basis_count)
        prediction = np.sum(c[curves] * rows, axis=1)
        residual = scale * (prediction - targets)
        loss, derivative = robust_penalty(residual)
        value = float(np.sum(data_weights * loss))
        gradient = np.zeros_like(c)
        factors = data_weights * derivative * scale
        np.add.at(gradient, curves, factors[:, None] * rows)

        bending = c @ curvature_design.T
        value += SPLINE_SMOOTHNESS * float(np.mean(bending ** 2))
        gradient += (
            2.0 * SPLINE_SMOOTHNESS / bending.size
            * (bending @ curvature_design)
        )

        if shape_weight > 0:
            differences = np.diff(c, axis=0) @ slope_design.T
            value += shape_weight * float(np.mean(differences ** 2))
            changes = (
                2.0 * shape_weight / differences.size
                * (differences @ slope_design)
            )
            gradient[:-1] -= changes
            gradient[1:] += changes

        if guidance is None:
            seed_residual = scale * (c @ seed_design - seed_targets)
            seed_weight = 0.05
            value += seed_weight * float(np.mean(seed_residual ** 2))
            gradient += (
                2.0 * seed_weight / count
                * seed_residual[:, None] * scale * seed_design[None, :]
            )

        if guidance is not None and join_design is not None:
            indices = guidance.prior_indices
            residual_prior = (
                c[indices] @ join_design - guidance.prior_targets
            ) / guidance.prior_sigmas
            prior_loss, prior_derivative = robust_penalty(residual_prior)
            value += LOW_JOIN_PRIOR_WEIGHT * float(np.mean(prior_loss))
            prior_factors = (
                LOW_JOIN_PRIOR_WEIGHT / len(indices)
                * prior_derivative / guidance.prior_sigmas
            )
            np.add.at(
                gradient, indices,
                prior_factors[:, None] * join_design[None, :],
            )

            local_indices = guidance.local_indices
            if len(local_indices):
                residual_local = (
                    c[local_indices] @ join_design - guidance.local_targets
                ) / guidance.local_sigmas
                local_loss, local_derivative = robust_penalty(residual_local)
                value += HIGH_JOIN_IMAGE_WEIGHT * float(np.mean(local_loss))
                local_factors = (
                    HIGH_JOIN_IMAGE_WEIGHT / len(local_indices)
                    * local_derivative / guidance.local_sigmas
                )
                np.add.at(
                    gradient, local_indices,
                    local_factors[:, None] * join_design[None, :],
                )

        if ink_weights is not None:
            all_prediction = top + height * (c @ design.T)
            distance, ink_derivative, inside, _ = sample_ink(
                evidence, xs, all_prediction
            )
            residual_ink = distance / INK_DISTANCE_TOLERANCE
            ink_loss, ink_loss_derivative = robust_penalty(residual_ink)
            weights = ink_weights * inside
            value += ink_weight * float(np.sum(weights * ink_loss))
            ordinate_gradient = (
                ink_weight * weights * ink_loss_derivative
                * ink_derivative / INK_DISTANCE_TOLERANCE * height
            )
            gradient += ordinate_gradient @ design

        return value, gradient.ravel()

    lower = -np.inf if high_speed else 2.0 / height
    upper = np.inf if high_speed else 1.0 - 2.0 / height
    result = minimize(
        objective, coefficients.ravel(), method="SLSQP", jac=True,
        bounds=Bounds(
            np.full(count * basis_count, lower),
            np.full(count * basis_count, upper),
        ),
        constraints=constraints,
        options={
            "maxiter": SPLINE_OPTIMIZER_ITERATIONS,
            "ftol": 1e-7, "disp": False,
        },
    )
    fitted = result.x.reshape(count, basis_count)

    if snapshot is not None and np.all(np.isfinite(fitted)):
        snapshot(
            fitted,
            "Optimizer candidate - not yet validated"
            if result.success
            else f"Rejected optimizer candidate: {result.message}",
        )

    if not result.success:
        raise ValueError(f"Spline optimization failed: {result.message}")
    if not np.all(np.isfinite(fitted)):
        raise ValueError("Nonfinite spline coefficients.")
    if np.any(
        np.diff(fitted, axis=0)
        < SPLINE_MIN_SEPARATION / height - 1e-7
    ):
        raise ValueError("Spline coefficient ordering failed.")
    return fitted


def evaluate_family(bounds, basis, coefficients, xs, extrapolate=False):
    left, top, right, bottom = bounds
    design = np.asarray(
        basis(
            (np.asarray(xs) - left) / (right - left),
            extrapolate=extrapolate,
        ), float,
    )
    return top + (bottom - top) * (coefficients @ design.T)


def assign_to_models(
    predicted, candidates, bounds,
    included=None, support=None, high_speed=False,
):
    assigned = np.full_like(predicted, np.nan)
    total_cost, evaluated = 0.0, 0
    for j in range(predicted.shape[1]):
        if included is not None and not included[j]:
            continue
        allowed = (
            support.sampled_mask[:, j] & support.valid
            if support is not None else None
        )
        assigned[:, j], cost, active = visible_assignment(
            predicted[:, j], candidates[j], bounds,
            allowed=allowed, reject_ambiguous=high_speed,
        )
        total_cost += cost
        evaluated += active
    return assigned, total_cost / max(1, evaluated)


def check_support(observations, included, high_speed, support, tags):
    minimum = (
        MIN_HIGH_SUPPORT_COLUMNS
        if high_speed else MIN_CURVE_SUPPORT_COLUMNS
    )
    for k, tag in enumerate(tags):
        if support is not None and not support.valid[k]:
            raise ValueError(
                f"Hull {tag}: insufficient direct cleaned-ink overlap."
            )
        valid = np.isfinite(observations[k]) & included
        if support is not None:
            valid &= support.sampled_mask[k]
        if np.count_nonzero(valid) < minimum:
            raise ValueError(
                f"Hull {tag}: too few distinct assigned observations."
            )


def fit_one_complexity(
    basis_count, xs, candidates, initial,
    seed_x, seed_ys, bounds, included,
    shape_weight, high_speed, guidance,
    evidence, major_spacing, ink_weight, tags, diagnostic_callback=None,
):
    left, top, right, bottom = bounds
    height = float(bottom - top)
    basis = spline_basis(basis_count)
    design = np.asarray(basis((xs - left) / (right - left)), float)
    coefficients = np.repeat(
        ((seed_ys - top) / height)[:, None], basis_count, axis=1
    )
    observations = initial.copy()
    observations[:, ~included] = np.nan
    support = None

    for iteration in range(SPLINE_ASSIGNMENT_ITERATIONS):
        previous = coefficients.copy()

        def snapshot(candidate, status):
            if diagnostic_callback is not None:
                candidate_support = (
                    assess_image_support(
                        bounds, basis, candidate, xs, evidence, major_spacing
                    ) if high_speed else None
                )
                diagnostic_callback(
                    xs, observations, basis, candidate,
                    tags, candidate_support,
                    f"{basis_count} coefficients; {status}",
                )

        coefficients = optimize_coefficients(
            coefficients, basis, xs, observations, included,
            seed_x, seed_ys, bounds, shape_weight,
            high_speed, guidance, evidence, ink_weight, support, snapshot,
        )
        support = (
            assess_image_support(
                bounds, basis, coefficients, xs, evidence, major_spacing
            ) if high_speed else None
        )
        predicted = top + height * (coefficients @ design.T)
        observations, _ = assign_to_models(
            predicted, candidates, bounds, included, support, high_speed
        )

        if diagnostic_callback is not None:
            diagnostic_callback(
                xs, observations, basis, coefficients, tags, support,
                "Candidate assessed against cleaned ink",
            )

        check_support(observations, included, high_speed, support, tags)
        movement = height * float(np.max(np.abs(
            (coefficients - previous) @ design.T
        )))
        if iteration >= 1 and movement < 0.15:
            break

    return basis, coefficients, support


def validate_family(bounds, basis, coefficients, guidance=None):
    left, top, right, bottom = bounds
    height = bottom - top
    predicted = evaluate_family(
        bounds, basis, coefficients,
        np.arange(left, right + 1, dtype=int),
    )
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Nonfinite fitted coordinates.")
    if np.any(
        np.diff(predicted, axis=0) < SPLINE_MIN_SEPARATION - 1e-3
    ):
        raise ValueError("Evaluated fitted curves cross.")

    if guidance is not None:
        join_design = np.asarray(
            basis(guidance.u, extrapolate=True), float
        )
        join_values = coefficients @ join_design
        if np.any(
            np.diff(join_values)
            < SPLINE_MIN_SEPARATION / height - 1e-6
        ):
            raise ValueError("Curve ordering failed at x=1.")


def contiguous_groups(indices):
    if not len(indices):
        return []
    return np.split(
        indices, np.flatnonzero(np.diff(indices) > 1) + 1
    )


def export_fitted_paths(
    clean, bounds, basis, coefficients, high_speed, support
):
    left, top, right, bottom = bounds
    xs = np.arange(left, right + 1, dtype=int)
    predicted = evaluate_family(bounds, basis, coefficients, xs)
    assignments = np.full(predicted.shape, np.nan)

    for j, x in enumerate(xs):
        if x < left + 2 or x > right - 2:
            continue
        allowed = (
            (x <= support.right_edges) & support.valid
            if support is not None else None
        )
        assignments[:, j], _, _ = visible_assignment(
            predicted[:, j], column_candidates(clean, bounds, int(x)),
            bounds, tolerance=OBSERVED_TOLERANCE,
            allowed=allowed, reject_ambiguous=high_speed,
        )

    minimum = (
        MIN_HIGH_SUPPORT_COLUMNS
        if high_speed else MIN_CURVE_SUPPORT_COLUMNS
    )
    paths = []
    for k in range(len(coefficients)):
        assigned_support = np.isfinite(assignments[k])
        if np.count_nonzero(assigned_support) < minimum:
            raise ValueError(
                f"Hull index {k + 1}: insufficient final assigned support."
            )

        visible = (
            (predicted[k] >= top) & (predicted[k] <= bottom)
        )
        if support is not None:
            visible &= xs <= support.right_edges[k]

        path = []
        for group in contiguous_groups(np.flatnonzero(visible)):
            supported = group[assigned_support[group]]
            if len(supported) < minimum:
                continue
            first_support, last_support = int(supported[0]), int(supported[-1])
            first_export = 0 if int(group[0]) == 0 else first_support

            for j in range(first_export, last_support + 1):
                observed = bool(assigned_support[j])
                kind = (
                    "image_supported" if observed
                    else "left_projected" if j < first_support
                    else "interpolated"
                )
                path.append((
                    float(xs[j]), float(predicted[k, j]), observed, kind
                ))

        if not path:
            raise ValueError(
                f"Hull index {k + 1}: no supported visible component."
            )
        paths.append(path)
    return paths


def fit_curve_family(
    clean, grid_mask, bounds, requested_count, shape_weight,
    high_speed, x_fit, y_fit, major_spacing, ink_weight,
    print_sigma, anchors=None, diagnostic_callback=None,
):
    left, top, right, bottom = bounds
    evidence = build_ink_evidence(clean, grid_mask, bounds)

    if high_speed and anchors:
        tags = tags_for_count(12)
        guidance = build_join_guidance(
            anchors, tags, bounds, x_fit, y_fit,
            clean, major_spacing, print_sigma,
        )
        if guidance is None:
            raise ValueError("Missing paired join guidance.")
        seed_x = int(np.clip(
            round(guidance.pixel_x), left + 2, right - 2
        ))
        seed_ys = ordered_initial_positions(
            guidance.initial_pixels
            + guidance.initial_slopes * (seed_x - guidance.pixel_x),
            np.ones(len(tags)),
        )
    else:
        guidance = None
        seeds = choose_seeds(
            clean, bounds, requested_count, high_speed
        )
        if seeds is None:
            raise ValueError("No complete seed column. Inspect clean.")
        seed_x, seed_ys = seeds
        tags = tags_for_count(len(seed_ys))

    step = max(1, int(math.ceil(
        (right - left + 1) / SPLINE_MAX_SAMPLE_COLUMNS
    )))
    parts = [
        np.arange(left + 2, right - 1, step, dtype=int),
        np.array([seed_x], dtype=int),
    ]
    if guidance is not None:
        parts.append(guidance.local_xs)

    xs = np.unique(np.concatenate(parts))
    candidates = [
        column_candidates(clean, bounds, int(x)) for x in xs
    ]
    initial = initial_assignments(
        xs, candidates, seed_x, seed_ys, bounds, high_speed, guidance
    )

    if diagnostic_callback is not None:
        initial_basis = spline_basis(SPLINE_BASIS_COUNTS[0])
        initial_coefficients = np.repeat(
            ((seed_ys - top) / (bottom - top))[:, None],
            SPLINE_BASIS_COUNTS[0], axis=1,
        )
        initial_support = (
            assess_image_support(
                bounds, initial_basis, initial_coefficients,
                xs, evidence, major_spacing,
            ) if high_speed else None
        )
        diagnostic_callback(
            xs, initial, initial_basis, initial_coefficients,
            tags, initial_support,
            "INITIALIZATION ONLY - no optimization performed",
        )

    validation = (np.arange(len(xs)) // 8) % 5 == 2
    validation[xs == seed_x] = False
    if guidance is not None:
        validation[np.isin(xs, guidance.local_xs)] = False

    training = ~validation
    choices, last_error = [], ""

    for basis_count in SPLINE_BASIS_COUNTS:
        try:
            basis, coefficients, support = fit_one_complexity(
                basis_count, xs, candidates, initial,
                seed_x, seed_ys, bounds, training,
                shape_weight, high_speed, guidance,
                evidence, major_spacing, ink_weight, tags, diagnostic_callback,
            )
            predicted = evaluate_family(bounds, basis, coefficients, xs)
            assigned, score = assign_to_models(
                predicted, candidates, bounds,
                validation, support, high_speed,
            )
            possible = (
                len(tags) * max(1, int(np.count_nonzero(validation)))
                if support is None
                else max(1, int(np.count_nonzero(
                    support.sampled_mask & validation[None, :]
                )))
            )
            observed_fraction = (
                np.count_nonzero(np.isfinite(assigned)) / possible
            )
            score += 0.25 * (1.0 - observed_fraction)
            if support is not None:
                score += float(np.mean(1.0 - support.retained_fraction))

            choices.append((
                score, basis_count, basis, coefficients, support
            ))
        except ValueError as error:
            last_error = str(error)

    if not choices:
        raise ValueError("No supported spline fit. " + last_error)

    best_score = min(choice[0] for choice in choices)
    eligible = [
        choice for choice in choices
        if choice[0] <= best_score + max(0.005, 0.02 * best_score)
    ]
    score, basis_count, basis, coefficients, support = min(
        eligible, key=lambda choice: choice[1]
    )

    all_columns = np.ones(len(xs), bool)
    design = np.asarray(basis((xs - left) / (right - left)), float)

    for iteration in range(SPLINE_ASSIGNMENT_ITERATIONS):
        predicted = top + (bottom - top) * (coefficients @ design.T)
        observations, _ = assign_to_models(
            predicted, candidates, bounds, None, support, high_speed
        )
        if diagnostic_callback is not None:
            diagnostic_callback(
                xs, observations, basis, coefficients, tags, support,
                "Final-stage candidate - not yet validated",
            )

        check_support(
            observations, all_columns, high_speed, support, tags
        )
        previous = coefficients.copy()

        def snapshot(candidate, status):
            if diagnostic_callback is not None:
                candidate_support = (
                    assess_image_support(
                        bounds, basis, candidate, xs, evidence, major_spacing
                    ) if high_speed else None
                )
                diagnostic_callback(
                    xs, observations, basis, candidate,
                    tags, candidate_support, status,
                )

        coefficients = optimize_coefficients(
            coefficients, basis, xs, observations, all_columns,
            seed_x, seed_ys, bounds, shape_weight,
            high_speed, guidance, evidence, ink_weight, support, snapshot,
        )
        support = (
            assess_image_support(
                bounds, basis, coefficients, xs, evidence, major_spacing
            ) if high_speed else None
        )

        movement = (bottom - top) * float(np.max(np.abs(
            (coefficients - previous) @ design.T
        )))
        if iteration >= 1 and movement < 0.15:
            break

    predicted = evaluate_family(bounds, basis, coefficients, xs)
    assigned, _ = assign_to_models(
        predicted, candidates, bounds, None, support, high_speed
    )
    if diagnostic_callback is not None:
        diagnostic_callback(
            xs, assigned, basis, coefficients, tags, support,
            "Final candidate - awaiting export validation",
        )

    validate_family(bounds, basis, coefficients, guidance)
    check_support(assigned, all_columns, high_speed, support, tags)
    paths = export_fitted_paths(
        clean, bounds, basis, coefficients, high_speed, support
    )

    return CurveModel(
        basis, coefficients, tags, paths,
        seed_x, seed_ys, basis_count, score,
        xs, assigned, support, guidance,
    )


def label_preview(base, bounds, bottom, right, title, top_band):
    image = base.copy()
    draw_frame(image, bounds)
    for box, color in (
        (title, (180, 0, 180)), (top_band, (0, 180, 0))
    ):
        if box:
            x0, y0, x1, y1 = box
            cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), color, 2)

    for labels, color in (
        (bottom, (255, 120, 0)), (right, (0, 160, 255))
    ):
        for word in labels:
            cv2.rectangle(
                image, (word.left, word.top),
                (word.right, word.bottom), color, 2,
            )
            cv2.circle(
                image, (round(word.cx), round(word.cy)), 3, color, -1
            )
    return image


def curve_colors(tags):
    tags = list(tags)
    if not tags:
        return []
    hsv = np.zeros((len(tags), 1, 3), np.uint8)
    for index, tag in enumerate(tags):
        value = float(tag)
        if not math.isfinite(value) or not 1.0 <= value <= 7.0:
            raise ValueError(f"Invalid curve tag for display palette: {tag!r}")
        hsv[index, 0, 0] = round((value - 1.0) * 180.0 / 7.0)
        hsv[index, 0, 1] = 220
        hsv[index, 0, 2] = 210
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return [
        tuple(int(value) for value in color)
        for color in bgr[:, 0]
    ]


def diagnostic_banner(image, status):
    header = 62
    result = np.full(
        (image.shape[0] + header, image.shape[1], 3), 255, np.uint8
    )
    result[header:] = image
    max_characters = max(30, image.shape[1] // 8)
    lines, current = [], ""

    for word in status.split():
        if current and len(current) + len(word) + 1 > max_characters:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)

    for index, line in enumerate(lines[:3]):
        cv2.putText(
            result, line, (10, 18 + 18 * index),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 0, 160), 1, cv2.LINE_AA,
        )
    return result


def draw_fitted_path(image, path, color):
    for previous, current in zip(path, path[1:]):
        x0, y0, observed0, _ = previous
        x1, y1, observed1, _ = current
        if x1 - x0 > 1.5:
            continue
        supported = observed0 and observed1
        cv2.line(
            image, (round(x0), round(y0)), (round(x1), round(y1)),
            color if supported else (130, 130, 130),
            2 if supported else 1,
        )
    for x, y, observed, _ in path:
        if not observed and round(x) % 8 == 0:
            cv2.circle(
                image, (round(x), round(y)), 1, (0, 140, 255), -1
            )


def candidate_preview(
    base, bounds, xs, assigned,
    basis, coefficients, tags, support, mode,
):
    image = base.copy()
    left, top, right, bottom = bounds
    colors = curve_colors(tags)
    dense_x = np.arange(left, right + 1, dtype=int)
    predicted = evaluate_family(bounds, basis, coefficients, dense_x)

    for k, color in enumerate(colors):
        visible = (
            np.isfinite(predicted[k])
            & (predicted[k] >= top) & (predicted[k] <= bottom)
        )
        for group in contiguous_groups(np.flatnonzero(visible)):
            if len(group) > 1:
                points = np.rint(np.column_stack((
                    dense_x[group], predicted[k, group]
                ))).astype(np.int32)
                cv2.polylines(
                    image, [points], False, (170, 170, 170), 1
                )

        retained = visible.copy()
        if support is not None:
            retained &= (
                dense_x <= support.right_edges[k]
            ) & support.valid[k]

        if mode != "support":
            for group in contiguous_groups(np.flatnonzero(retained)):
                if len(group) > 1:
                    points = np.rint(np.column_stack((
                        dense_x[group], predicted[k, group]
                    ))).astype(np.int32)
                    cv2.polylines(
                        image, [points], False, color,
                        1 if mode == "assignments" else 2,
                    )

        if mode == "support" and support is not None:
            for j in np.flatnonzero(visible)[::4]:
                if support.near_ink[k, j]:
                    point_color = (0, 180, 0)
                elif support.obscured[k, j]:
                    point_color = (0, 150, 255)
                else:
                    point_color = (0, 0, 230)
                cv2.circle(
                    image,
                    (int(dense_x[j]), round(float(predicted[k, j]))),
                    2, point_color, -1,
                )

        left_y = float(predicted[k, 0])
        if math.isfinite(left_y) and top <= left_y <= bottom:
            cv2.circle(image, (left, round(left_y)), 4, color, -1)
            cv2.putText(
                image, tags[k], (left + 7, round(left_y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

        if support is not None and support.valid[k]:
            edge = float(support.right_edges[k])
            y = float(evaluate_family(
                bounds, basis, coefficients, np.array([edge])
            )[k, 0])
            if math.isfinite(y) and top <= y <= bottom:
                cv2.line(
                    image, (round(edge), round(y) - 6),
                    (round(edge), round(y) + 6), color, 2,
                )

    if mode == "assignments":
        for k, color in enumerate(colors):
            for j in np.flatnonzero(np.isfinite(assigned[k])):
                cv2.circle(
                    image, (int(xs[j]), round(float(assigned[k, j]))),
                    3, color, 1,
                )

    draw_frame(image, bounds)
    return image


# Diagnostic-only interior initialization. It never supplies baseline seeds.

def init_mask(grid_mask, xs, ys):
    xs, ys = np.broadcast_arrays(np.asarray(xs), np.asarray(ys))
    valid = (
        np.isfinite(ys)
        & (xs >= 0) & (xs < grid_mask.shape[1])
        & (ys >= 0) & (ys < grid_mask.shape[0])
    )
    xx = np.clip(
        np.rint(xs).astype(int), 0, grid_mask.shape[1] - 1
    )
    yy = np.clip(
        np.rint(np.where(np.isfinite(ys), ys, 0)).astype(int),
        0, grid_mask.shape[0] - 1,
    )
    masked = np.zeros(xs.shape, bool)
    for offset in (-1, 0, 1):
        rr = np.clip(yy + offset, 0, grid_mask.shape[0] - 1)
        masked |= grid_mask[rr, xx]
    return masked | ~valid


def init_slopes(clean, bounds, xs):
    _, top, _, bottom = bounds
    local = clean[top + 2:bottom - 1, xs[0]:xs[-1] + 1]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        local.astype(np.uint8)
    )
    slopes = []
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if area < 6 or w < 4 or h > 8 * w:
            continue
        yy, xx = np.nonzero(
            labels[y:y + h, x:x + w] == index
        )
        if len(xx) < 6:
            continue
        xx = xx.astype(float)
        yy = yy.astype(float)
        xx -= xx.mean()
        denominator = float(np.sum(xx * xx))
        if denominator <= 0:
            continue
        slope = float(np.sum(xx * (yy - yy.mean())) / denominator)
        if abs(slope) <= 4:
            slopes.append(slope)

    if not slopes:
        return []
    values = np.quantile(slopes, (0.2, 0.5, 0.8))
    result = []
    for value in values:
        if not any(abs(value - old) < 0.12 for old in result):
            result.append(float(value))
    return result


def init_profile(clean, grid_mask, bounds, xs, center, slope):
    _, top, _, bottom = bounds
    rows = np.arange(top + 3, bottom - 2, dtype=float)
    yy = rows[:, None] + slope * (xs[None, :] - center)
    xx = np.broadcast_to(xs[None, :], yy.shape)
    inside = (yy >= top + 2) & (yy <= bottom - 2)
    masked = init_mask(grid_mask, xx, yy)
    evaluable = inside & ~masked

    rr = np.clip(
        np.rint(yy).astype(int), 0, clean.shape[0] - 1
    )
    hits = np.zeros(yy.shape, bool)
    for offset in (-1, 0, 1):
        sample_rows = np.clip(rr + offset, 0, clean.shape[0] - 1)
        hits |= clean[sample_rows, xx]

    denominator = evaluable.sum(axis=1)
    profile = np.divide(
        (hits & evaluable).sum(axis=1),
        denominator,
        out=np.zeros(len(rows), float),
        where=denominator >= max(6, int(0.20 * len(xs))),
    )
    profile = gaussian_filter1d(profile, 0.8)
    peaks, properties = find_peaks(
        profile,
        height=INIT_PEAK_HEIGHT,
        prominence=INIT_PEAK_PROMINENCE,
        distance=INIT_PEAK_DISTANCE,
    )
    return rows[peaks], properties["prominences"], profile


def init_column_candidates(clean, grid_mask, bounds, x):
    _, top, _, bottom = bounds
    occupied = np.flatnonzero(clean[top + 2:bottom - 1, x]) + top + 2
    if not len(occupied):
        return np.array([], float)

    runs = np.split(
        occupied, np.flatnonzero(np.diff(occupied) > 1) + 1
    )
    runs = [run for run in runs if len(run) <= 12]
    merged = []

    for run in runs:
        if merged:
            previous = merged[-1]
            gap = int(run[0] - previous[-1] - 1)
            if 0 < gap <= MASK_WIDTH + 4:
                between = grid_mask[previous[-1] + 1:run[0], x]
                if len(between) and np.mean(between) >= 0.80:
                    merged[-1] = np.concatenate((previous, run))
                    continue
        merged.append(run)

    return np.array([
        0.5 * (run[0] + run[-1]) for run in merged
    ], float)


def init_design(xs, center):
    dx = np.asarray(xs, float) - center
    return np.column_stack((np.ones(len(dx)), dx, dx * dx))


def init_local_fit(xs, ys, center):
    valid = np.isfinite(ys)
    if np.count_nonzero(valid) < INIT_MIN_POINTS:
        return None
    if np.ptp(xs[valid]) < max(6, 0.25 * np.ptp(xs)):
        return None

    design = init_design(xs[valid], center)
    targets = ys[valid]
    weights = np.ones(len(targets))

    for _ in range(4):
        coefficients = np.linalg.lstsq(
            design * np.sqrt(weights)[:, None],
            targets * np.sqrt(weights), rcond=None,
        )[0]
        residual = targets - design @ coefficients
        weights = 1.0 / np.sqrt(1.0 + (residual / 1.5) ** 2)

    median = float(np.median(np.abs(residual)))
    if median > INIT_MAX_MEDIAN_RESIDUAL:
        return None
    return coefficients, median


def init_test_window(clean, grid_mask, bounds, xs, center, slope, peaks, prominence):
    result = InitWindow(
        center, xs, slope, len(peaks),
        None, None, None, 0.0, f"ridge_count_{len(peaks)}",
    )
    if len(peaks) != 12:
        return result

    design = init_design(xs, center)
    coefficients = np.column_stack((
        peaks, np.full(12, slope), np.zeros(12)
    ))
    candidates = [
        init_column_candidates(clean, grid_mask, bounds, int(x))
        for x in xs
    ]
    observations = np.full((12, len(xs)), np.nan)
    residuals = np.full(12, np.nan)

    for _ in range(3):
        predicted = coefficients @ design.T
        if np.any(np.diff(predicted, axis=0) <= 2.0):
            result.reason = "local_tracks_not_distinct"
            return result

        observations[:] = np.nan
        for j, x in enumerate(xs):
            prediction = predicted[:, j]
            separation = np.diff(prediction)
            local_gap = np.full(12, float(np.median(separation)))
            local_gap[:-1] = np.minimum(local_gap[:-1], separation)
            local_gap[1:] = np.minimum(local_gap[1:], separation)
            gates = np.clip(0.35 * local_gap, 2.0, 4.0)

            assigned, _ = ordered_assignment(
                prediction, candidates[j], gates
            )
            masked = init_mask(
                grid_mask, np.full(12, x), prediction
            )
            assigned[masked] = np.nan
            observations[:, j] = assigned

        proposed = []
        for k in range(12):
            fit = init_local_fit(xs, observations[k], center)
            if fit is None:
                result.reason = f"insufficient_local_track_{tags_for_count(12)[k]}"
                return result
            proposed.append(fit[0])
            residuals[k] = fit[1]
        coefficients = np.asarray(proposed)

    predicted = coefficients @ design.T
    if np.any(np.diff(predicted, axis=0) <= 2.0):
        result.reason = "refined_tracks_not_distinct"
        return result

    result.coefficients = coefficients
    result.observations = observations.copy()
    result.residuals = residuals.copy()
    result.score = (
        float(np.median(prominence))
        + 0.10 * float(np.mean(np.isfinite(observations)))
        - 0.05 * float(np.mean(residuals))
    )
    result.reason = "twelve_local_tracks"
    return result


def init_evaluate(window, xs):
    return window.coefficients @ init_design(xs, window.center).T


def init_compare_windows(windows):
    valid = [window for window in windows if window.coefficients is not None]
    for i, a in enumerate(valid):
        for b in valid[i + 1:]:
            minimum_shift = max(6.0, 0.20 * min(len(a.xs), len(b.xs)))
            if abs(a.center - b.center) < minimum_shift:
                continue

            first = max(a.xs[0], b.xs[0])
            last = min(a.xs[-1], b.xs[-1])
            if last - first < 8:
                continue

            xs = np.linspace(first, last, 7)
            distance = np.abs(
                init_evaluate(a, xs) - init_evaluate(b, xs)
            )
            if (
                float(np.median(distance)) <= INIT_MATCH_MEDIAN
                and float(np.max(distance)) <= INIT_MATCH_MAX
            ):
                a.matches += 1
                b.matches += 1

    if not valid:
        return None
    return max(
        valid, key=lambda window: (window.matches, window.score)
    )


def init_write_windows(path, windows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "window", "center_x", "first_x", "last_x", "slope",
            "ridge_count", "status", "matching_windows", "score",
        ))
        for index, window in enumerate(windows, 1):
            writer.writerow((
                index, window.center, window.xs[0], window.xs[-1],
                window.slope, window.peak_count, window.reason,
                window.matches, f"{window.score:.6g}",
            ))


def init_write_result(path, window, bounds, model, x_fit, y_fit):
    fields = (
        "proposed_tag", "status", "reference_pixel_x",
        "proposed_pixel_y", "proposed_x", "proposed_y",
        "baseline_pixel_y", "proposed_minus_baseline_px",
        "local_slope", "observed_columns", "masked_columns",
        "median_residual_px", "matching_windows",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        if window is None:
            return

        baseline = evaluate_family(
            bounds, model.basis, model.coefficients,
            np.array([window.center]),
        )[:, 0]
        proposed = window.coefficients[:, 0]
        predicted = init_evaluate(window, window.xs)
        masks = init_mask(
            grid_mask=init_write_result.grid_mask,
            xs=window.xs[None, :],
            ys=predicted,
        )

        for k, tag in enumerate(tags_for_count(12)):
            baseline_y = (
                baseline[model.tags.index(tag)] if tag in model.tags else np.nan
            )
            writer.writerow((
                tag,
                "cross_window_checked" if window.matches else "provisional",
                f"{window.center:.3f}", f"{proposed[k]:.6f}",
                f"{x_fit[0] * window.center + x_fit[1]:.10g}",
                f"{y_fit[0] * proposed[k] + y_fit[1]:.10g}",
                f"{baseline_y:.6f}",
                f"{proposed[k] - baseline_y:.6f}",
                f"{window.coefficients[k, 1]:.6f}",
                int(np.count_nonzero(np.isfinite(window.observations[k]))),
                int(np.count_nonzero(masks[k])),
                f"{window.residuals[k]:.6f}", window.matches,
            ))


def init_preview(base, bounds, window, status):
    image = base.copy()
    draw_frame(image, bounds)

    if window is None:
        return diagnostic_banner(image, status)

    _, top, _, bottom = bounds
    cv2.line(
        image, (int(window.xs[0]), top),
        (int(window.xs[0]), bottom), (0, 140, 255), 1,
    )
    cv2.line(
        image, (int(window.xs[-1]), top),
        (int(window.xs[-1]), bottom), (0, 140, 255), 1,
    )

    tags = tags_for_count(12)
    colors = curve_colors(tags)
    predicted = init_evaluate(window, window.xs)

    for k, (tag, color) in enumerate(zip(tags, colors)):
        points = np.rint(np.column_stack((
            window.xs, predicted[k]
        ))).astype(np.int32)
        cv2.polylines(image, [points], False, color, 2)

        valid = np.flatnonzero(np.isfinite(window.observations[k]))
        for j in valid[::2]:
            cv2.circle(
                image,
                (int(window.xs[j]), round(window.observations[k, j])),
                2, color, -1,
            )

        y = float(window.coefficients[k, 0])
        cv2.circle(
            image, (round(window.center), round(y)), 5, color, 2
        )
        cv2.putText(
            image, tag,
            (int(window.xs[-1]) + 8, round(y) - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA,
        )

    return diagnostic_banner(
        image, status + " | Short colored tracks and proposed tags only; "
        "baseline fits and exports are unchanged",
    )


def diagnose_initialization(
    clean, grid_mask, bounds, major_spacing,
    base, image_path, diagnostic_dir, model, x_fit, y_fit,
):
    left, _, right, _ = bounds
    widths = sorted(set(
        int(np.clip(
            factor * major_spacing, INIT_MIN_WIDTH, INIT_MAX_WIDTH
        ))
        for factor in INIT_WIDTH_FACTORS
    ))
    first_center = left + max(widths) / 2 + 3
    last_center = min(
        right - max(widths) / 2 - 3,
        left + INIT_SEARCH_FRACTION * (right - left),
    )
    windows = []

    if last_center > first_center:
        centers = np.linspace(first_center, last_center, INIT_CENTERS)
        for center in centers:
            for width in widths:
                start = max(left + 2, int(round(center - width / 2)))
                stop = min(right - 2, start + width - 1)
                xs = np.arange(start, stop + 1, dtype=int)
                reference = float(np.mean(xs))

                for slope in init_slopes(clean, bounds, xs):
                    peaks, prominence, _ = init_profile(
                        clean, grid_mask, bounds, xs, reference, slope
                    )
                    windows.append(init_test_window(
                        clean, grid_mask, bounds, xs, reference,
                        slope, peaks, prominence,
                    ))

    chosen = init_compare_windows(windows)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    init_write_windows(
        diagnostic_dir / f"{stem}_init_windows.csv", windows
    )

    # Diagnostic data only; not passed back to the fitter.
    init_write_result.grid_mask = grid_mask
    init_write_result(
        diagnostic_dir / f"{stem}_initialization.csv",
        chosen, bounds, model, x_fit, y_fit,
    )

    valid_count = sum(
        window.coefficients is not None for window in windows
    )
    print(
        f"{image_path.name}: diagnostic initialization: "
        f"{len(windows)} strip tests; "
        f"{valid_count} twelve-track candidates."
    )

    if chosen is None:
        status = "No twelve-track strip established; baseline unchanged"
        counts = {}
        for window in windows:
            counts[window.reason] = counts.get(window.reason, 0) + 1
        for reason, count in sorted(
            counts.items(), key=lambda item: -item[1]
        )[:8]:
            print(f"  {count:4d} {reason}")
    else:
        status = (
            "Twelve proposed tracks: cross-window checked"
            if chosen.matches
            else "Twelve proposed tracks: PROVISIONAL, no overlapping-strip check"
        )
        print(
            f"  Selected strip x={chosen.xs[0]}..{chosen.xs[-1]}, "
            f"reference x={chosen.center:.1f}; "
            f"matching strips={chosen.matches}."
        )
        baseline = evaluate_family(
            bounds, model.basis, model.coefficients,
            np.array([chosen.center]),
        )[:, 0]
        print("  tag  proposed_y_px  baseline_y_px  difference_px  observations")
        for k, tag in enumerate(tags_for_count(12)):
            baseline_y = (
                baseline[model.tags.index(tag)] if tag in model.tags else np.nan
            )
            proposed_y = chosen.coefficients[k, 0]
            print(
                f"  {tag:>3s}  {proposed_y:13.3f}  "
                f"{baseline_y:13.3f}  "
                f"{proposed_y - baseline_y:13.3f}  "
                f"{np.count_nonzero(np.isfinite(chosen.observations[k])):12d}"
            )

    save(
        diagnostic_dir / f"{stem}_initialization.png",
        init_preview(base, bounds, chosen, status),
    )


def process_page(
    image_path, segment, args,
    requested_count, shape_weight, anchors=None,
):
    def diagnostic(name):
        return args.diagnostic_dir / f"{image_path.stem}_{name}.png"

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read {image_path}")

    ink = binary_image(gray)
    audit = {} if args.diagnostics else None
    bounds = detect_frame(ink, audit)
    if audit is not None:
        save(
            diagnostic("frame_detection"),
            frame_detection_preview(gray, audit),
        )
    if bounds is None:
        raise ValueError(f"{image_path.name}: no plotting frame.")

    original = read_words(gray)
    right_region = clip((
        bounds[2] + 1, bounds[1] - AXIS_SEARCH_PADDING,
        gray.shape[1], bounds[3] + AXIS_SEARCH_PADDING,
    ), gray.shape)
    raw = unique_words(
        [word for word in original if in_right_region(word, bounds)]
        + ocr_region_words(gray, right_region)
    )
    x0, x1, height, right_anchors = find_numeric_column(
        raw, bounds, gray.shape[1]
    )
    column = (x0, x1, height)
    top_anchor = right_anchors[0]
    top_band = clip((
        bounds[2] + 1, top_anchor.top - height,
        gray.shape[1], top_anchor.bottom + height,
    ), gray.shape)
    title = detect_title_box(gray, bounds, top_band)
    axis_gray = gray.copy()

    if title:
        tx0, ty0, tx1, ty1 = title
        axis_gray[ty0:ty1, tx0:tx1] = 255
        bx0, by0, bx1, by1 = top_band
        axis_gray[by0:by1, bx0:bx1] = gray[by0:by1, bx0:bx1]
        if max(w.right for w in right_anchors) <= title[0] < x1:
            column = (x0, title[0], height)

    rows = propose_rows(raw, bounds, column, right_anchors)
    right = consistent_sequence(
        read_label_rows(axis_gray, rows, column, right_anchors), False
    )
    bottom = select_bottom(original, bounds)
    bottom = recover_bottom_labels(gray, bottom, right, bounds)
    base = (
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if args.diagnostics else None
    )

    if base is not None:
        save(
            diagnostic("labels"),
            label_preview(base, bounds, bottom, right, title, top_band),
        )

    if len(bottom) < MIN_AXIS_LABELS or len(right) < MIN_AXIS_LABELS:
        raise ValueError(
            f"{image_path.name}: insufficient labels "
            f"({len(bottom)} bottom, {len(right)} right)."
        )

    x_fit = fit_axis(bottom, True)
    multiplier = 10.0 ** args.right_exponent
    y_fit = fit_axis(right, False, multiplier)
    grid, references = generate_grid(
        ink, bounds, bottom, right,
        args.grid_search, args.straight_grid,
    )

    left, top, end, bottom_y = bounds
    inside = np.zeros(ink.shape, bool)
    inside[top:bottom_y + 1, left:end + 1] = True
    clean = ink & ~grid.mask & inside

    if base is not None:
        save(
            diagnostic("grid_fit"),
            grid_fit_preview(base, bounds, grid, references),
        )
        grid_overlay = tint(base, grid.mask, (0, 180, 0))
        draw_frame(grid_overlay, bounds)
        save(diagnostic("grid_overlay"), grid_overlay)
        save(diagnostic("clean"), np.where(clean, 0, 255).astype(np.uint8))

        if segment == "high" and anchors:
            join_pixel = pixel_for_numerical_x(bounds, x_fit, JOIN_X)
            print(
                f"{image_path.name}: calibrated x=1 is "
                f"{join_pixel - left:.2f} px from the left frame."
            )

    def save_candidate(
        xs, assigned, basis, coefficients, tags, support, status
    ):
        if base is None:
            return
        for name in ("fits", "assignments", "support"):
            image = candidate_preview(
                base, bounds, xs, assigned,
                basis, coefficients, tags, support, name,
            )
            detail = status
            if name == "support":
                detail += (
                    " | Green=near cleaned ink; red=unsupported; "
                    "orange=grid-obscured; gray=full candidate"
                )
            elif name == "assignments":
                detail += " | Circles=assigned observations"
            save(diagnostic(name), diagnostic_banner(image, detail))

    model = fit_curve_family(
        clean, grid.mask, bounds,
        requested_count, shape_weight,
        segment == "high", x_fit, y_fit,
        grid.reference.major_x, args.ink_weight,
        args.join_mismatch * multiplier, anchors,
        save_candidate if base is not None else None,
    )

    if base is not None:
        overlay = base.copy()
        colors = curve_colors(model.tags)
        left_values = evaluate_family(
            bounds, model.basis, model.coefficients,
            np.array([float(left)]),
        )[:, 0]

        for index, (tag, path) in enumerate(zip(model.tags, model.paths)):
            color = colors[index]
            draw_fitted_path(overlay, path, color)

            if segment == "high" and anchors:
                marker_x, marker_y = left, float(left_values[index])
            else:
                marker_x = model.seed_x
                marker_y = float(model.seed_ys[index])

            if top <= marker_y <= bottom_y:
                cv2.circle(
                    overlay, (marker_x, round(marker_y)), 4, color, -1
                )
                cv2.putText(
                    overlay, tag,
                    (marker_x + 7, round(marker_y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA,
                )

        draw_frame(overlay, bounds)
        save(
            diagnostic("fits"),
            diagnostic_banner(
                overlay,
                "ACCEPTED FIT | Gray=unsupported estimates; "
                "right tails are not projected",
            ),
        )
        reference = grid.reference
        print(
            f"{image_path.name}: major spans "
            f"{reference.intervals_x}/{reference.intervals_y:g}; "
            f"{model.basis_count} coefficients/curve; "
            f"shape weight {shape_weight:g}."
        )
        if model.guidance is not None:
            print(
                f"{image_path.name}: local join estimates found for "
                f"{len(model.guidance.local_indices)}/{len(model.tags)} curves."
            )

        if segment == "high":
            try:
                diagnose_initialization(
                    clean, grid.mask, bounds, grid.reference.major_x,
                    base, image_path, args.diagnostic_dir,
                    model, x_fit, y_fit,
                )
            except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
                print(
                    f"Diagnostic initialization failed: {error}. "
                    "Baseline fit is retained.", file=sys.stderr,
                )

    return PageResult(
        image_path, segment, bounds, x_fit, y_fit, model
    )


def model_at_join(page, tag):
    pixel_x = pixel_for_numerical_x(page.bounds, page.x_fit, JOIN_X)
    index = page.model.tags.index(tag)
    pixel_y = float(evaluate_family(
        page.bounds, page.model.basis, page.model.coefficients,
        np.array([pixel_x]), extrapolate=True,
    )[index, 0])
    numerical_y = float(page.y_fit[0] * pixel_y + page.y_fit[1])
    observed_points = [
        point for point in page.model.paths[index] if point[2]
    ]
    distance = (
        min(abs(point[0] - pixel_x) for point in observed_points)
        if observed_points else float("inf")
    )
    supported = distance <= JOIN_PIXEL_TOLERANCE
    return numerical_y, pixel_x, pixel_y, supported, distance


def make_join_anchors(low_page):
    anchors = []
    for value in range(1, 8):
        tag = str(value)
        y, _, _, supported, distance = model_at_join(low_page, tag)
        inflation = min(5.0, 1.0 + distance / 20.0)
        sigma = (
            JOIN_PIXEL_TOLERANCE * abs(low_page.y_fit[0]) * inflation
        )
        anchors.append(JoinAnchor(
            tag, y, max(sigma, 1e-15), supported
        ))
    return anchors


def collect_rows(pages, paired):
    rows = []
    for page in pages:
        for tag, path in zip(page.model.tags, page.model.paths):
            for px, py, observed, kind in path:
                x = float(page.x_fit[0] * px + page.x_fit[1])
                y = float(page.y_fit[0] * py + page.y_fit[1])

                if paired:
                    if page.segment == "low" and x > JOIN_X:
                        continue
                    if page.segment == "high" and x < JOIN_X:
                        continue
                    if abs(x - JOIN_X) < 1e-10:
                        continue

                rows.append((
                    tag, x, y, px, py, int(observed),
                    page.image.name, page.segment, kind,
                ))

    if paired:
        for page in pages:
            for tag, path in zip(page.model.tags, page.model.paths):
                if page.segment == "high":
                    if not path or path[0][0] > page.bounds[0] + 0.5:
                        continue

                y, px, py, supported, _ = model_at_join(page, tag)
                if not page.bounds[1] <= py <= page.bounds[3]:
                    continue
                rows.append((
                    tag, JOIN_X, y, px, py, int(supported),
                    page.image.name, page.segment,
                    "image_supported" if supported else "join_projected",
                ))

    rows.sort(key=lambda row: (-float(row[0]), row[1], row[7]))
    return rows


def write_points(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow((
                "curve", "x", "y", "pixel_x", "pixel_y",
                "observed", "source_image", "segment", "estimate_kind",
            ))
            for tag, x, y, px, py, observed, source, segment, kind in rows:
                writer.writerow((
                    tag, f"{x:.10g}", f"{y:.10g}",
                    f"{px:.3f}", f"{py:.3f}",
                    observed, source, segment, kind,
                ))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_join_audit(path, low, high, anchors):
    temporary = path.with_name(path.name + ".tmp")
    by_tag = {anchor.tag: anchor for anchor in anchors}
    guidance = high.model.guidance
    local_values = {}

    if guidance is not None:
        _, top, _, bottom = high.bounds
        height = bottom - top
        for position, index in enumerate(guidance.local_indices):
            pixel_y = top + height * guidance.local_targets[position]
            local_values[high.model.tags[index]] = float(
                high.y_fit[0] * pixel_y + high.y_fit[1]
            )

    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow((
                "curve", "x",
                "low_y", "high_fitted_y", "high_minus_low",
                "high_local_join_y", "low_anchor_sigma",
                "low_image_supported", "high_image_supported",
                "low_source", "high_source",
            ))

            for tag in high.model.tags:
                high_y, _, _, supported, _ = model_at_join(high, tag)
                value = float(tag)
                local = local_values.get(tag)

                if value.is_integer():
                    anchor = by_tag[str(int(value))]
                    low_y = f"{anchor.value:.10g}"
                    difference = f"{high_y - anchor.value:.10g}"
                    sigma = f"{anchor.sigma:.10g}"
                    low_supported = int(anchor.low_supported)
                else:
                    low_y = difference = sigma = low_supported = ""

                writer.writerow((
                    tag, f"{JOIN_X:g}",
                    low_y, f"{high_y:.10g}", difference,
                    "" if local is None else f"{local:.10g}",
                    sigma, low_supported, int(supported),
                    low.image.name, high.image.name,
                ))

        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "images", nargs="*", type=Path,
        help="One image, or low-speed followed by high-speed image.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--right-exponent", type=int, required=True)
    parser.add_argument(
        "--curve-count", type=int, choices=(7, 12),
        help="Single-image mode only.",
    )
    parser.add_argument(
        "--high-speed", action="store_true",
        help="Treat one input as a high-speed segment.",
    )
    parser.add_argument(
        "--shape-weight", type=float, default=DEFAULT_SHAPE_WEIGHT,
        help="Low-speed/single ordinary image slope coupling.",
    )
    parser.add_argument(
        "--high-shape-weight", type=float, default=0.0,
        help="High-speed slope coupling; default disabled.",
    )
    parser.add_argument(
        "--ink-weight", type=float, default=DEFAULT_INK_WEIGHT,
        help="High-speed direct cleaned-ink distance penalty.",
    )
    parser.add_argument(
        "--join-mismatch", type=float, default=DEFAULT_JOIN_MISMATCH,
        help=(
            "Printed-data mismatch allowance in displayed right-axis "
            "units, before applying --right-exponent."
        ),
    )
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR
    )
    parser.add_argument(
        "--grid-search", type=float, default=DEFAULT_GRID_SEARCH
    )
    parser.add_argument("--straight-grid", action="store_true")
    args = parser.parse_args()

    images = args.images or [DEFAULT_IMAGE]
    if len(images) not in (1, 2):
        raise ValueError("Provide one image or exactly two paired images.")

    paired = len(images) == 2
    if paired and (args.high_speed or args.curve_count is not None):
        raise ValueError(
            "Paired mode uses low=7 and high=12 curves. "
            "Do not supply --high-speed or --curve-count."
        )
    if len({image.resolve() for image in images}) != len(images):
        raise ValueError("Input images must be distinct.")
    if not -100 <= args.right_exponent <= 100:
        raise ValueError("Right exponent must be between -100 and 100.")

    for name, value in (
        ("shape-weight", args.shape_weight),
        ("high-shape-weight", args.high_shape_weight),
        ("ink-weight", args.ink_weight),
        ("join-mismatch", args.join_mismatch),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and nonnegative.")

    if not math.isfinite(args.grid_search) or args.grid_search <= 0:
        raise ValueError("--grid-search must be positive and finite.")

    default_stem = "_".join(image.stem for image in images)
    output = args.output or DEFAULT_OUTPUT_DIR / f"{default_stem}_points.csv"
    join_output = output.with_name(output.stem + "_join.csv")

    generated = [
        output, output.with_name(output.name + ".tmp")
    ]
    if paired:
        generated += [
            join_output, join_output.with_name(join_output.name + ".tmp")
        ]

    if args.diagnostics:
        for image in images:
            generated += [
                args.diagnostic_dir / f"{image.stem}_{name}.png"
                for name in (
                    "frame_detection", "labels", "grid_fit",
                    "grid_overlay", "clean", "fits",
                    "assignments", "support",
                )
            ]
        high_images = (
            [images[1]] if paired
            else [images[0]] if args.high_speed else []
        )
        for image in high_images:
            generated += [
                args.diagnostic_dir / f"{image.stem}_initialization.png",
                args.diagnostic_dir / f"{image.stem}_initialization.csv",
                args.diagnostic_dir / f"{image.stem}_init_windows.csv",
            ]

    resolved = [path.resolve() for path in generated]
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "Generated paths collide; use distinct image stems "
            "or another output/diagnostic directory."
        )
    if any(image.resolve() in resolved for image in images):
        raise ValueError("An output path would overwrite an input.")

    for image in images:
        if not image.is_file():
            raise ValueError(f"Input does not exist: {image}")

    output.parent.mkdir(parents=True, exist_ok=True)
    for path in generated:
        path.unlink(missing_ok=True)

    if not find_tesseract():
        raise ValueError("Tesseract not found.")

    if paired:
        low = process_page(
            images[0], "low", args,
            requested_count=7, shape_weight=args.shape_weight,
        )
        anchors = make_join_anchors(low)
        high = process_page(
            images[1], "high", args,
            requested_count=12, shape_weight=args.high_shape_weight,
            anchors=anchors,
        )
        pages = [low, high]
    else:
        high_speed = args.high_speed
        requested_count = args.curve_count
        if high_speed and requested_count is None:
            requested_count = 12
        page = process_page(
            images[0], "high" if high_speed else "single", args,
            requested_count=requested_count,
            shape_weight=(
                args.high_shape_weight if high_speed else args.shape_weight
            ),
        )
        pages, anchors = [page], []

    rows = collect_rows(pages, paired)
    if not rows:
        raise ValueError("No fitted points available for export.")

    if paired:
        write_join_audit(join_output, pages[0], pages[1], anchors)

    write_points(output, rows)
    print(f"Wrote {output} ({len(rows)} fitted points).")
    if paired:
        print(f"Wrote {join_output}.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        OSError, ValueError, cv2.error, pytesseract.TesseractError
    ) as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        sys.exit(1)