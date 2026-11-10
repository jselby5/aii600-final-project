"""
Automatic, experimental digitizer for a stitched scanned graph.

Run:
    python scripts/extract_graph.py data/stitched_graph_223_224.png \
        --output data/points.csv \
        --overlay data/diagnostics.png

Install Python dependencies:
    python -m pip install numpy scipy opencv-python pytesseract

Tesseract itself must also be installed.

Grid model:
* Rotation is estimated from vertical evidence in the right half.
* Spacing is measured from repeated adjacent vertical lines in the right half.
* The same spacing is used horizontally and vertically.
* Horizontal evidence determines only the horizontal grid offset.
* The generated grid cannot bend along individual curves.
* Supported line extents can differ between plot sections.

Outputs:
    diagnostics_expected_grid.png
        Full generated template, including outside the plot.
    diagnostics_expected_overlay.png
        Full generated template in green over the image.
    diagnostics_grid.png
        Final exclusion mask.
    diagnostics.png
        Exclusion mask in red, plot bounds, and curve traces.
    diagnostics.json
        Grid measurements, OCR, calibration, warnings, and errors.
    points.csv
        Written only if all detected sections pass implemented checks.

Assumptions:
* Approximately upright image.
* One global square-grid spacing and rotation fit both sections.
* Linear axes.
* Curves are functions of x and do not cross.
* Tags descend from top to bottom.
* Seven curves correspond to 7, 6, ..., 1.
* Twelve curves correspond to 6.5, 6, ..., 1.
* If two sections are detected, the left has seven curves and the right twelve.

Limitations:
* No local geometric dewarping.
* Detached scientific-notation axis multipliers are not interpreted.
* Missing grid lines can still cause spacing ambiguity.
* OCR, bounds detection, and tracing remain experimental.
* A curve overlapping an actual grid line cannot be separated by geometry alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class Panel:
    left: int
    top: int
    right: int
    bottom: int


@dataclass
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


@dataclass
class Track:
    horizontal: bool
    points: np.ndarray
    support: float


# ----------------------------------------------------------------------
# Image preparation
# ----------------------------------------------------------------------

def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def binarize(gray):
    block = max(15, min(81, (min(gray.shape) // 30) | 1))
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        12,
    )
    return binary > 0


def true_runs(values):
    """Return [start, end) intervals of consecutive True values."""
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(
        np.flatnonzero(changes == 1),
        np.flatnonzero(changes == -1),
    ))


def fill_short_gaps(values, maximum):
    result = values.copy()
    for start, end in true_runs(~values):
        if (
            start > 0
            and end < len(values)
            and end - start <= maximum
        ):
            result[start:end] = True
    return result


# ----------------------------------------------------------------------
# Generated-grid fitting
# ----------------------------------------------------------------------

def rotation_matrix(shape, angle):
    height, width = shape
    return cv2.getRotationMatrix2D(
        ((width - 1) / 2, (height - 1) / 2),
        float(angle),
        1.0,
    )


def rotate_foreground(ink, angle):
    height, width = ink.shape
    matrix = rotation_matrix(ink.shape, angle)
    rotated = cv2.warpAffine(
        ink.astype(np.float32),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated, matrix


def directional_support(evidence):
    height, width = evidence.shape

    vertical = cv2.boxFilter(
        evidence,
        -1,
        (1, max(15, min(101, height // 12))),
        normalize=True,
        borderType=cv2.BORDER_CONSTANT,
    )
    horizontal = cv2.boxFilter(
        evidence,
        -1,
        (max(15, min(101, width // 12)), 1),
        normalize=True,
        borderType=cv2.BORDER_CONSTANT,
    )
    return vertical, horizontal


def prepare_profile(profile):
    profile = gaussian_filter1d(
        profile.astype(float), sigma=0.7
    )
    background = gaussian_filter1d(
        profile, sigma=max(8, len(profile) / 45)
    )
    result = np.maximum(profile - background, 0)
    maximum = float(result.max())

    if maximum <= 1e-12:
        raise ValueError("Insufficient directional grid evidence.")

    return result / maximum


def directional_profiles(evidence):
    vertical, horizontal = directional_support(evidence)
    x_profile = prepare_profile(np.mean(vertical ** 2, axis=0))
    y_profile = prepare_profile(np.mean(horizontal ** 2, axis=1))
    return x_profile, y_profile


def estimate_angle_from_right(ink):
    """Estimate a small global rotation using right-side vertical evidence."""
    reference = ink[:, ink.shape[1] // 2:]
    height, width = reference.shape
    scale = min(1.0, 1200 / max(height, width))

    reference = cv2.resize(
        reference.astype(np.float32),
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )

    def score(angle):
        aligned, _ = rotate_foreground(reference, angle)
        vertical = cv2.boxFilter(
            aligned,
            -1,
            (1, max(15, aligned.shape[0] // 10)),
            normalize=True,
            borderType=cv2.BORDER_CONSTANT,
        )
        profile = np.mean(vertical ** 2, axis=0)
        count = min(len(profile), max(3, len(profile) // 50))
        strongest = np.partition(
            profile, len(profile) - count
        )[-count:]
        return float(np.mean(strongest))

    coarse = np.linspace(-2.0, 2.0, 17)
    best = float(max(coarse, key=score))
    fine = np.linspace(best - 0.25, best + 0.25, 21)
    return float(max(fine, key=score))


def best_template_phase(profile, spacing):
    """Fit the phase of a generated regular comb at fixed spacing."""
    length = len(profile)
    phase_count = max(24, int(math.ceil(2 * spacing)))
    phases = np.linspace(0, spacing, phase_count, endpoint=False)

    line_count = int(math.ceil(length / spacing)) + 1
    locations = (
        phases[:, None]
        + spacing * np.arange(line_count)[None, :]
    )
    valid = locations <= length - 1

    values = np.interp(
        locations.ravel(),
        np.arange(length),
        profile,
        left=0,
        right=0,
    ).reshape(locations.shape)

    counts = valid.sum(axis=1)
    scores = np.full(phase_count, -np.inf)
    usable = counts >= 3

    if not usable.any():
        return None

    scores[usable] = (
        (values * valid).sum(axis=1)[usable] / counts[usable]
    )
    index = int(np.argmax(scores))
    return float(scores[index]), float(phases[index])


def fit_grid_from_right(aligned_ink):
    """
    Fit spacing from right-side vertical evidence.

    Tests subdivisions of detected gaps to avoid locking onto major
    grid lines while missing the finer grid.

    A candidate must:
    1. Explain observed vertical-line peaks.
    2. Avoid predicting many unsupported intermediate lines.

    Horizontal evidence determines only the horizontal-row phase.
    """
    height, width = aligned_ink.shape
    right_start = width // 2
    reference = aligned_ink[:, right_start:]

    x_profile, y_profile = directional_profiles(reference)

    # Retain faint lines as well as prominent major grid lines.
    peaks, properties = find_peaks(
        x_profile,
        distance=3,
        prominence=0.015,
        height=0.02,
    )

    if len(peaks) < 5:
        raise ValueError(
            "Too few vertical-line peaks in the right-hand region."
        )

    positions = peaks.astype(float)

    # Give faint lines meaningful influence without weighting every
    # tiny peak exactly like a strong grid line.
    weights = np.sqrt(properties["prominences"])
    weights = np.maximum(weights, 1e-6)

    gaps = np.diff(positions)
    minimum_spacing = 6.0
    maximum_spacing = min(reference.shape) / 3

    proposals = set()

    # Include several neighboring-peak separations and their subdivisions.
    # A separation can span multiple cells if faint peaks were missed.
    for offset in (1, 2, 3):
        if len(positions) <= offset:
            continue

        separations = positions[offset:] - positions[:-offset]

        for separation in separations:
            for divisor in range(1, 7):
                pitch = float(separation) / divisor

                if minimum_spacing <= pitch <= maximum_spacing:
                    proposals.add(round(pitch, 2))

    if not proposals:
        raise ValueError("No plausible grid-spacing proposals.")

    proposals = sorted(proposals)

    # Keep the search bounded for noisy pages.
    if len(proposals) > 600:
        indices = np.linspace(
            0, len(proposals) - 1, 600
        ).astype(int)
        proposals = [proposals[index] for index in indices]

    first_peak = float(positions[0])
    last_peak = float(positions[-1])

    def evaluate(spacing):
        # Candidate phases come from actual peak positions.
        phases = np.unique(
            np.round(np.mod(positions, spacing), 2)
        )

        if len(phases) > 100:
            indices = np.linspace(
                0, len(phases) - 1, 100
            ).astype(int)
            phases = phases[indices]

        tolerance = min(2.0, 0.10 * spacing)
        best_phase = None

        for phase in phases:
            grid_indices = np.rint(
                (positions - phase) / spacing
            ).astype(int)

            residual = np.abs(
                positions - (phase + grid_indices * spacing)
            )
            matched = residual <= tolerance

            if matched.sum() < 4:
                continue

            unique_indices = np.unique(grid_indices[matched])
            if len(unique_indices) < 4:
                continue

            # How much observed line evidence does this grid explain?
            coverage = float(
                weights[matched].sum() / weights.sum()
            )

            # Evaluate generated lines across the observed peak span.
            lower_index = int(math.ceil(
                (first_peak - phase) / spacing
            ))
            upper_index = int(math.floor(
                (last_peak - phase) / spacing
            ))

            generated = (
                phase
                + np.arange(lower_index, upper_index + 1) * spacing
            )

            if len(generated) < 4:
                continue

            nearest = np.min(
                np.abs(generated[:, None] - positions[None, :]),
                axis=1,
            )

            # Unsupported intermediate lines penalize an overly fine grid.
            occupancy = float(np.mean(nearest <= tolerance))

            # A coarse grid is penalized for unexplained observed lines.
            # A fine grid is penalized for unsupported generated lines.
            score = (
                2 * coverage * occupancy
                / max(coverage + occupancy, 1e-12)
            )

            error = float(np.average(
                residual[matched],
                weights=weights[matched],
            ))

            candidate = {
                "score": score,
                "error": error,
                "spacing": float(spacing),
                "phase": float(phase),
                "coverage": coverage,
                "occupancy": occupancy,
            }

            if best_phase is None or (
                candidate["score"], -candidate["error"]
            ) > (
                best_phase["score"], -best_phase["error"]
            ):
                best_phase = candidate

        return best_phase

    best = None

    for spacing in proposals:
        candidate = evaluate(spacing)

        if candidate is not None:
            if best is None or (
                candidate["score"], -candidate["error"]
            ) > (
                best["score"], -best["error"]
            ):
                best = candidate

    if best is None:
        raise ValueError(
            "No consistent grid spacing found in right-side evidence."
        )

    # Refine without allowing a jump to another spacing multiple.
    spacing = best["spacing"]
    phase = best["phase"]
    initial_spacing = spacing

    for _ in range(5):
        indices = np.rint(
            (positions - phase) / spacing
        ).astype(int)

        residual = positions - (phase + indices * spacing)
        tolerance = min(2.0, 0.10 * spacing)
        matched = np.abs(residual) <= tolerance

        if len(np.unique(indices[matched])) < 4:
            raise ValueError(
                "Too few consistent grid lines after refinement."
            )

        new_spacing, new_phase = np.polyfit(
            indices[matched],
            positions[matched],
            1,
            w=np.sqrt(weights[matched]),
        )

        if not (
            0.95 * initial_spacing
            <= new_spacing
            <= 1.05 * initial_spacing
        ):
            break

        spacing = float(new_spacing)
        phase = float(new_phase)

    indices = np.rint(
        (positions - phase) / spacing
    ).astype(int)
    residual = positions - (phase + indices * spacing)
    matched = np.abs(residual) <= min(2.0, 0.10 * spacing)

    # Merge any detections assigned to the same grid index.
    grouped = {}
    for index, position in zip(indices[matched], positions[matched]):
        grouped.setdefault(int(index), []).append(float(position))

    ordered = sorted(
        (index, float(np.median(values)))
        for index, values in grouped.items()
    )

    if len(ordered) < 4:
        raise ValueError("Fewer than four distinct fitted grid lines.")

    measured_spacings = [
        (p1 - p0) / (i1 - i0)
        for (i0, p0), (i1, p1) in zip(ordered, ordered[1:])
        if i1 > i0
    ]

    fit_positions = np.array([p for _, p in ordered])
    fit_indices = np.array([i for i, _ in ordered])
    fit_residual = fit_positions - (phase + fit_indices * spacing)

    y_fit = best_template_phase(y_profile, spacing)
    if y_fit is None:
        raise ValueError(
            "Could not place horizontal rows at the fitted spacing."
        )

    return {
        "spacing": float(spacing),
        "x_phase": float((phase + right_start) % spacing),
        "y_phase": float(y_fit[1]),
        "spacing_source": "right-side vertical lattice with subdivision tests",
        "reference_bounds_aligned": [
            right_start, 0, width - 1, height - 1
        ],
        "detected_vertical_peaks": int(len(peaks)),
        "trusted_vertical_lines": int(len(ordered)),
        "measured_interval_count": int(len(measured_spacings)),
        "mean_adjacent_spacing": float(np.mean(measured_spacings)),
        "adjacent_spacing_std": float(
            np.std(measured_spacings, ddof=1)
        ),
        "vertical_fit_residual_std": float(
            np.std(fit_residual, ddof=1)
        ),
        "reference_vertical_positions_aligned": (
            fit_positions + right_start
        ).tolist(),
        "horizontal_phase_score": float(y_fit[0]),
        "vertical_peak_coverage": float(best["coverage"]),
        "generated_line_occupancy": float(best["occupancy"]),
    }


def grid_positions(length, phase, spacing):
    return np.arange(phase, length - 1, spacing)


def supported_runs(support, position, horizontal, spacing):
    """
    Determine visible extents along a generated line.

    Evidence can activate a line but cannot move it or add another line.
    """
    coordinate = int(round(position))

    if horizontal:
        if not 0 <= coordinate < support.shape[0]:
            return []
        lo = max(0, coordinate - 1)
        hi = min(support.shape[0], coordinate + 2)
        evidence = np.max(support[lo:hi, :], axis=0)
    else:
        if not 0 <= coordinate < support.shape[1]:
            return []
        lo = max(0, coordinate - 1)
        hi = min(support.shape[1], coordinate + 2)
        evidence = np.max(support[:, lo:hi], axis=1)

    present = fill_short_gaps(
        evidence >= 0.60,
        maximum=max(3, int(0.15 * spacing)),
    )
    minimum_length = max(30, int(2 * spacing))

    return [
        (int(start), int(end))
        for start, end in true_runs(present)
        if end - start >= minimum_length
    ]


def render_expected_grid(shape, model):
    height, width = shape
    aligned = np.zeros(shape, dtype=np.uint8)

    for x in grid_positions(width, model["x_phase"], model["spacing"]):
        cv2.line(
            aligned,
            (int(round(x)), 0),
            (int(round(x)), height - 1),
            255,
            model["line_width"],
        )

    for y in grid_positions(height, model["y_phase"], model["spacing"]):
        cv2.line(
            aligned,
            (0, int(round(y))),
            (width - 1, int(round(y))),
            255,
            model["line_width"],
        )

    return cv2.warpAffine(
        aligned,
        np.array(model["inverse_matrix"], dtype=float),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def make_grid_mask(ink):
    angle = estimate_angle_from_right(ink)
    aligned, matrix = rotate_foreground(ink, angle)

    # Fit parameters using the right half, then apply them everywhere.
    model = fit_grid_from_right(aligned)
    model["alignment_angle_degrees"] = angle
    model["line_width"] = 5

    inverse = cv2.invertAffineTransform(matrix)
    model["inverse_matrix"] = inverse.tolist()

    vertical_support, horizontal_support = directional_support(aligned)
    height, width = ink.shape
    aligned_mask = np.zeros(ink.shape, dtype=np.uint8)
    tracks = []

    def add_track(horizontal, position, start, end):
        along = np.arange(start, end, dtype=float)
        across = np.full(end - start, position, dtype=float)

        points = (
            np.column_stack((along, across))
            if horizontal else np.column_stack((across, along))
        )

        cv2.polylines(
            aligned_mask,
            [np.rint(points).astype(np.int32)],
            False,
            255,
            thickness=model["line_width"],
        )

        original = cv2.transform(
            points.reshape(-1, 1, 2), inverse
        ).reshape(-1, 2)

        tracks.append(Track(horizontal, original, 1.0))

    for x in grid_positions(width, model["x_phase"], model["spacing"]):
        for start, end in supported_runs(
            vertical_support, x, False, model["spacing"]
        ):
            add_track(False, x, start, end)

    for y in grid_positions(height, model["y_phase"], model["spacing"]):
        for start, end in supported_runs(
            horizontal_support, y, True, model["spacing"]
        ):
            add_track(True, y, start, end)

    mask = cv2.warpAffine(
        aligned_mask,
        inverse,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    model["horizontal_segments"] = sum(t.horizontal for t in tracks)
    model["vertical_segments"] = sum(not t.horizontal for t in tracks)
    return mask, tracks, model


# ----------------------------------------------------------------------
# Automatic plot-section detection
# ----------------------------------------------------------------------

def detect_panels(mask, tracks):
    height, width = mask.shape
    joined = cv2.dilate(
        mask.astype(np.uint8), np.ones((5, 5), np.uint8)
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    panels = []

    for i in range(1, count):
        x, y, cw, ch, area = map(int, stats[i])
        if (
            cw >= 0.14 * width
            and ch >= 0.15 * height
            and area >= 0.001 * height * width
        ):
            ys, xs = np.nonzero(mask[y:y + ch, x:x + cw])
            if len(xs):
                panels.append(Panel(
                    x + int(xs.min()),
                    y + int(ys.min()),
                    x + int(xs.max()),
                    y + int(ys.max()),
                ))

    panels.sort(key=lambda p: p.left)
    if not panels:
        raise ValueError("No plausible plot region found in the grid mask.")
    if len(panels) != 1:
        return panels

    panel = panels[0]
    vertical = []

    for track in tracks:
        if track.horizontal:
            continue
        top = float(np.min(track.points[:, 1]))
        bottom = float(np.max(track.points[:, 1]))
        if bottom - top >= 0.25 * (panel.bottom - panel.top):
            vertical.append((
                float(np.median(track.points[:, 0])),
                top,
                bottom,
            ))

    groups = []
    for record in sorted(vertical):
        if (
            groups
            and abs(record[0] - np.mean(
                [item[0] for item in groups[-1]]
            )) < 5
        ):
            groups[-1].append(record)
        else:
            groups.append([record])

    vertical = [
        (
            float(np.mean([r[0] for r in group])),
            float(min(r[1] for r in group)),
            float(max(r[2] for r in group)),
        )
        for group in groups
    ]

    best = None
    for split in range(3, len(vertical) - 2):
        before = np.array(vertical[:split])
        after = np.array(vertical[split:])
        extent_a = np.median(before[:, 1:], axis=0)
        extent_b = np.median(after[:, 1:], axis=0)

        difference = float(np.max(np.abs(extent_a - extent_b)))
        scatter = float(
            np.median(np.abs(before[:, 1:] - extent_a))
            + np.median(np.abs(after[:, 1:] - extent_b))
        )
        score = difference - 3 * scatter
        threshold = max(15, 0.07 * (panel.bottom - panel.top))

        if difference >= threshold and score >= threshold / 2:
            if best is None or score > best[0]:
                best = (score, split, extent_a, extent_b)

    if best is None:
        return panels

    _, split, extent_a, extent_b = best
    seam = int(round(
        (vertical[split - 1][0] + vertical[split][0]) / 2
    ))

    return [
        Panel(
            panel.left,
            max(0, int(round(extent_a[0]))),
            seam,
            min(height - 1, int(round(extent_a[1]))),
        ),
        Panel(
            seam,
            max(0, int(round(extent_b[0]))),
            panel.right,
            min(height - 1, int(round(extent_b[1]))),
        ),
    ]


# ----------------------------------------------------------------------
# OCR and axis calibration
# ----------------------------------------------------------------------

def configure_tesseract():
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
    ]

    if os.name == "nt":
        for variable in (
            "ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"
        ):
            root = os.environ.get(variable)
            if root:
                candidates.extend([
                    Path(root) / "Tesseract-OCR" / "tesseract.exe",
                    Path(root) / "Programs" / "Tesseract-OCR"
                    / "tesseract.exe",
                ])

    candidates.append(pytesseract.pytesseract.tesseract_cmd)

    for candidate in candidates:
        if not candidate:
            continue
        executable = shutil.which(str(candidate))
        if executable is None and Path(candidate).is_file():
            executable = str(Path(candidate).resolve())

        if executable:
            pytesseract.pytesseract.tesseract_cmd = executable
            return {
                "executable": executable,
                "version": str(pytesseract.get_tesseract_version()),
            }

    raise RuntimeError(
        "Could not locate Tesseract. Install the engine or set "
        "TESSERACT_CMD to the full path of tesseract.exe."
    )


def read_words(gray):
    words = []
    for psm in (11, 6):
        data = pytesseract.image_to_data(
            gray,
            config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
        )

        for i, text in enumerate(data["text"]):
            text = text.strip()
            confidence = float(data["conf"][i])
            if not text or confidence < 15:
                continue

            word = Word(
                text, confidence,
                int(data["left"][i]), int(data["top"][i]),
                int(data["width"][i]), int(data["height"][i]),
            )

            if not any(
                word.text == old.text
                and abs(word.cx - old.cx) < 5
                and abs(word.cy - old.cy) < 5
                for old in words
            ):
                words.append(word)
    return words


def parse_number(text):
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace(",", ".")
        .strip()
    )
    if not re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
        text,
    ):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def fit_axis(samples, increasing):
    if len(samples) < 3:
        raise ValueError("Fewer than three readable axis-label candidates.")

    pixels = np.array([s[0] for s in samples], float)
    values = np.array([s[1] for s in samples], float)
    best = None

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if abs(pixels[j] - pixels[i]) < 15:
                continue

            a = (values[j] - values[i]) / (pixels[j] - pixels[i])
            if a == 0 or (a > 0) != increasing:
                continue

            b = values[i] - a * pixels[i]
            residual = np.abs((values - b) / a - pixels)
            selected = residual < 9

            if selected.sum() < 3:
                continue
            if (
                len(np.unique(np.round(pixels[selected] / 5))) < 3
                or len(np.unique(values[selected])) < 3
            ):
                continue

            score = (
                int(selected.sum()),
                float(np.ptp(pixels[selected])),
                -float(residual[selected].mean()),
            )
            if best is None or score > best[0]:
                best = (score, selected)

    if best is None:
        raise ValueError("No consistent linear axis calibration.")

    selected = best[1]
    a, b = np.polyfit(pixels[selected], values[selected], 1)
    return {
        "a": float(a),
        "b": float(b),
        "labels": [
            [float(p), float(v)]
            for p, v in zip(pixels[selected], values[selected])
        ],
    }


def calibrate(panel, words):
    width = panel.right - panel.left
    height = panel.bottom - panel.top
    x_samples, y_samples = [], []

    for word in words:
        value = parse_number(word.text)
        if value is None:
            continue

        if (
            panel.left - 12 <= word.cx <= panel.right + 12
            and panel.bottom - 5 <= word.top
            <= panel.bottom + max(45, 0.13 * height)
        ):
            x_samples.append((word.cx, value))

        if (
            panel.top - 12 <= word.cy <= panel.bottom + 12
            and panel.left - max(80, 0.16 * width)
            <= word.left + word.width <= panel.left + 8
        ):
            y_samples.append((word.cy, value))

    errors, fits = [], {}
    for name, samples, increasing in (
        ("x", x_samples, True),
        ("y", y_samples, False),
    ):
        try:
            fits[name] = fit_axis(samples, increasing)
        except ValueError as error:
            errors.append(f"{name}-axis: {error}")

    if errors:
        raise ValueError("; ".join(errors))
    return fits["x"], fits["y"]


# ----------------------------------------------------------------------
# Curve discovery and tracing
# ----------------------------------------------------------------------

def tracing_ink(ink, mask, words):
    result = ink & ~mask
    for word in words:
        x0 = max(0, word.left - 1)
        y0 = max(0, word.top - 1)
        x1 = min(ink.shape[1], word.left + word.width + 1)
        y1 = min(ink.shape[0], word.top + word.height + 1)
        result[y0:y1, x0:x1] = False
    return result


def run_centers(column):
    rows = np.flatnonzero(column)
    if not len(rows):
        return np.array([], dtype=float)
    groups = np.split(
        rows, np.flatnonzero(np.diff(rows) > 1) + 1
    )
    return np.array(
        [group.mean() for group in groups if len(group) <= 12],
        dtype=float,
    )


def choose_seeds(clean, panel, expected=None):
    margin = max(6, (panel.bottom - panel.top) // 100)
    left, right = panel.left + margin, panel.right - margin
    top, bottom = panel.top + margin, panel.bottom - margin

    if right <= left or bottom <= top:
        raise ValueError("Detected section is too small.")

    choices = []
    for x in np.linspace(left, right, 120).astype(int):
        centers = run_centers(clean[top:bottom + 1, x]) + top
        separated = []
        for y in centers:
            if not separated or y - separated[-1] >= 5:
                separated.append(float(y))
        if 2 <= len(separated) <= 20:
            choices.append((int(x), np.array(separated)))

    if not choices:
        raise ValueError("No plausible curve-seed columns.")

    if expected is None:
        votes = {
            count: sum(len(ys) == count for _, ys in choices)
            for count in (7, 12)
        }
        expected = max(votes, key=votes.get)
        if votes[expected] < 3:
            raise ValueError(
                "Could not identify a seven- or twelve-curve section."
            )

    candidates = [
        item for item in choices if len(item[1]) == expected
    ]
    if len(candidates) < 3:
        raise ValueError(
            f"Too few columns contain {expected} curve candidates."
        )

    def score(item):
        x, ys = item
        separation = float(np.min(np.diff(ys)))
        centrality = min(x - left, right - x) / max(1, right - left)
        return separation + 5 * centrality

    return max(candidates, key=score)


def trace_curve(clean, panel, seed_x, seed_y):
    max_gap = max(
        15, min(80, int(0.06 * (panel.right - panel.left)))
    )
    search = max(
        5, min(16, (panel.bottom - panel.top) // 65)
    )

    def follow(direction):
        observations = [(seed_x, float(seed_y))]
        path = [(seed_x, float(seed_y), True)]
        gap = 0
        stop = panel.left - 1 if direction < 0 else panel.right + 1

        for x in range(seed_x + direction, stop, direction):
            last_x, last_y = observations[-1]
            recent = observations[-10:]
            slope = 0.0

            if len(recent) >= 3:
                xs = np.array([p[0] for p in recent], float)
                ys = np.array([p[1] for p in recent], float)
                slope = float(
                    np.polyfit(xs - xs.mean(), ys, 1)[0]
                )

            slope = float(np.clip(slope, -3, 3))
            predicted = last_y + slope * (x - last_x)
            lo = max(panel.top + 2, int(predicted) - search)
            hi = min(panel.bottom - 1, int(predicted) + search + 1)

            candidates = (
                run_centers(clean[lo:hi, x]) + lo
                if hi > lo else np.array([])
            )

            if not len(candidates):
                gap += 1
                if gap > max_gap:
                    break
                continue

            y = float(
                candidates[np.argmin(abs(candidates - predicted))]
            )

            if gap:
                for missing_x in range(last_x + direction, x, direction):
                    fraction = (missing_x - last_x) / (x - last_x)
                    path.append((
                        missing_x,
                        last_y + fraction * (y - last_y),
                        False,
                    ))

            path.append((x, y, True))
            observations.append((x, y))
            observations = observations[-10:]
            gap = 0

        return path

    return list(reversed(follow(-1))) + follow(1)[1:]


def validate_order(traces):
    for upper, lower in zip(traces, traces[1:]):
        upper_map = {x: y for x, y, _ in upper[1]}
        lower_map = {x: y for x, y, _ in lower[1]}
        common = sorted(set(upper_map) & set(lower_map))

        if not common:
            raise ValueError("Adjacent curves have no shared x interval.")

        bad = sum(
            upper_map[x] >= lower_map[x] - 1 for x in common
        )
        if bad > 0.03 * len(common):
            raise ValueError(
                "Traces cross or merge; curve identities are unreliable."
            )


# ----------------------------------------------------------------------
# Diagnostics and main program
# ----------------------------------------------------------------------

def tint_mask(image, mask, color, opacity=0.65):
    result = image.copy()
    if np.any(mask):
        result[mask] = (
            (1 - opacity) * result[mask].astype(float)
            + opacity * np.array(color, dtype=float)
        ).astype(np.uint8)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("points.csv"))
    parser.add_argument(
        "--overlay", type=Path, default=Path("diagnostics.png")
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.overlay.parent.mkdir(parents=True, exist_ok=True)

    directory, stem = args.overlay.parent, args.overlay.stem
    report_path = args.overlay.with_suffix(".json")
    grid_path = directory / f"{stem}_grid.png"
    expected_path = directory / f"{stem}_expected_grid.png"
    expected_overlay_path = directory / f"{stem}_expected_overlay.png"

    report = {
        "image": str(args.image),
        "warnings": [
            "Experimental: inspect diagnostics before trusting CSV values.",
            "Spacing and rotation are estimated from the right half.",
            "One global square grid is assumed.",
            "No local geometric dewarping.",
            "Detached scientific-notation multipliers are not parsed.",
            "Axis calibration uses original-image coordinates.",
        ],
        "panels": [],
        "errors": [],
    }

    overlay = None
    rows = []
    failed = False
    colors = [
        (255, 100, 0), (0, 150, 0), (180, 0, 180),
        (0, 130, 255), (180, 100, 40), (130, 130, 0),
    ]

    try:
        gray = read_image(args.image)
        ink = binarize(gray)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay = base.copy()

        print("Fitting grid using the right-hand section...")
        mask, tracks, model = make_grid_mask(ink)
        report["grid_model"] = model

        print(
            f"Measured adjacent spacing: "
            f"{model['mean_adjacent_spacing']:.3f} px "
            f"(SD {model['adjacent_spacing_std']:.3f} px)"
        )
        print(
            f"Fitted spacing: {model['spacing']:.3f} px; "
            f"rotation: {model['alignment_angle_degrees']:.3f} degrees"
        )

        expected = render_expected_grid(ink.shape, model)
        cv2.imwrite(str(expected_path), expected)
        cv2.imwrite(
            str(expected_overlay_path),
            tint_mask(base, expected > 0, (0, 180, 0)),
        )
        cv2.imwrite(str(grid_path), mask.astype(np.uint8) * 255)

        overlay = tint_mask(base, mask, (0, 0, 255), opacity=0.60)
        cv2.imwrite(str(args.overlay), overlay)

        if not tracks:
            raise ValueError(
                "Grid fitted, but no sufficiently long supported "
                "line sections were found."
            )

        panels = detect_panels(mask, tracks)
        for index, panel in enumerate(panels):
            report["panels"].append({
                "panel": index + 1,
                "bounds": [
                    panel.left, panel.top, panel.right, panel.bottom
                ],
                "warnings": [],
            })
            cv2.rectangle(
                overlay,
                (panel.left, panel.top),
                (panel.right, panel.bottom),
                (255, 180, 0),
                2,
            )
            cv2.putText(
                overlay,
                f"Panel {index + 1}",
                (panel.left + 8, panel.top + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 180, 0),
                2,
            )
        cv2.imwrite(str(args.overlay), overlay)

        if len(panels) != 2:
            report["warnings"].append(
                f"Detected {len(panels)} section(s), not two. "
                "Curve counts will be inferred where possible."
            )

        print("Reading labels with Tesseract...")
        report["tesseract"] = configure_tesseract()
        words = read_words(gray)
        report["ocr_words"] = [
            {
                "text": word.text,
                "confidence": word.confidence,
                "box": [
                    word.left, word.top, word.width, word.height
                ],
            }
            for word in words
        ]
        clean = tracing_ink(ink, mask, words)

        for index, panel in enumerate(panels):
            info = report["panels"][index]
            try:
                expected_count = (
                    (7 if index == 0 else 12)
                    if len(panels) == 2 else None
                )
                seed_x, seed_ys = choose_seeds(
                    clean, panel, expected_count
                )

                tags = (
                    [str(v) for v in range(7, 0, -1)]
                    if len(seed_ys) == 7
                    else [f"{6.5 - 0.5 * i:g}" for i in range(12)]
                )

                info["tags"] = tags
                info["seed_x"] = int(seed_x)
                info["seed_ys"] = [float(y) for y in seed_ys]
                traces = []

                for curve_index, (tag, seed_y) in enumerate(
                    zip(tags, seed_ys)
                ):
                    path = trace_curve(clean, panel, seed_x, seed_y)
                    traces.append((tag, path))
                    color = colors[curve_index % len(colors)]
                    points = np.rint(
                        [[x, y] for x, y, _ in path]
                    ).astype(np.int32)

                    if len(points) > 1:
                        cv2.polylines(
                            overlay, [points], False, color, 2
                        )
                    cv2.circle(
                        overlay,
                        (seed_x, int(round(seed_y))),
                        5,
                        color,
                        -1,
                    )
                    cv2.putText(
                        overlay,
                        tag,
                        (seed_x + 8, int(round(seed_y)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

                    coverage = len(path) / max(
                        1, panel.right - panel.left + 1
                    )
                    observed_fraction = sum(
                        observed for _, _, observed in path
                    ) / max(1, len(path))

                    info.setdefault("curves", []).append({
                        "tag": tag,
                        "coverage": float(coverage),
                        "observed_fraction": float(observed_fraction),
                    })
                    if coverage < 0.60:
                        info["warnings"].append(
                            f"Curve {tag}: only {coverage:.0%} coverage."
                        )

                validate_order(traces)
                x_fit, y_fit = calibrate(panel, words)
                info["x_calibration"] = x_fit
                info["y_calibration"] = y_fit

                for tag, path in traces:
                    for px, py, observed in path:
                        rows.append({
                            "panel": index + 1,
                            "curve": tag,
                            "x": x_fit["a"] * px + x_fit["b"],
                            "y": y_fit["a"] * py + y_fit["b"],
                            "pixel_x": px,
                            "pixel_y": py,
                            "observed": int(observed),
                        })

            except ValueError as error:
                failed = True
                info["error"] = str(error)

    except Exception as error:
        failed = True
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()

    finally:
        if overlay is not None:
            cv2.imwrite(str(args.overlay), overlay)
        report_path.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    print(f"Expected grid: {expected_path}")
    print(f"Expected-grid overlay: {expected_overlay_path}")
    print(f"Final mask: {grid_path}")
    print(f"Diagnostic overlay: {args.overlay}")
    print(f"Report: {report_path}")

    if failed or not rows:
        print(
            "No new CSV written. Inspect the report and diagnostics.",
            file=sys.stderr,
        )
        if args.output.exists():
            print(
                f"Existing CSV was not updated: {args.output}",
                file=sys.stderr,
            )
        return 1

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "panel", "curve", "x", "y",
                "pixel_x", "pixel_y", "observed",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "x": f"{row['x']:.10g}",
                "y": f"{row['y']:.10g}",
                "pixel_y": f"{row['pixel_y']:.3f}",
            })

    print(f"Wrote {len(rows)} points to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())