"""
Render, deskew, and crop graph pages from a PDF.

Install:
    python -m pip install numpy opencv-python pypdfium2 \
        rapidocr-onnxruntime pillow

Run:
    python scripts/pull_graph_images.py

Configure settings below. Page numbers are 1-based PDF page positions.

Each output contains the detected plot and nearby recognized labels.
"""

from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR


# ======================================================================
# SETTINGS
# ======================================================================

PDF_PATH = Path("data/DTIC_AD0224171.pdf")
OUTPUT_DIR = Path("data/images")

PAGES = [223, 224]
DPI = 240

# Counterclockwise, matching PIL's rotation convention.
ROTATION_DEGREES = 270

CONTENT_PADDING_MM = 0.3
CONTINUE_ON_ERROR = True
ALLOW_GRID_DESKEW_FALLBACK = True

SAVE_CROP_PREVIEW = False
PREVIEW_DIR = OUTPUT_DIR / "diagnostics"

TICK_OCR_MIN_CONFIDENCE = 0.50
PROTECTED_TEXT_MIN_CONFIDENCE = 0.85

MAX_INITIAL_SKEW = 0.06
MIN_GRID_LINES_PER_DIRECTION = 3
MIN_GRID_SPAN_FRACTION = 0.12

# ======================================================================


def save(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def rectangle_points(bounds):
    left, top, right, bottom = bounds
    return np.array([
        [left, top], [right, top],
        [right, bottom], [left, bottom],
    ], dtype=np.float64)


def transform_box(matrix, points, shape):
    """Return enclosing bounds with exclusive right/bottom coordinates."""
    transformed = cv2.transform(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)

    minimum = np.floor(transformed.min(axis=0)).astype(int)
    maximum = np.ceil(transformed.max(axis=0)).astype(int)

    return (
        max(0, min(shape[1], int(minimum[0]))),
        max(0, min(shape[0], int(minimum[1]))),
        max(0, min(shape[1], int(maximum[0]) + 1)),
        max(0, min(shape[0], int(maximum[1]) + 1)),
    )


def parse_number(text):
    text = (
        str(text).strip()
        .replace("−", "-")
        .replace("–", "-")
        .replace(",", ".")
    )
    try:
        value = float(text)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def render_page(document, page_number):
    page = document[page_number - 1]
    try:
        bitmap = page.render(scale=DPI / 72)
        try:
            image = bitmap.to_pil().convert("L").copy()
        finally:
            bitmap.close()
    finally:
        page.close()

    if ROTATION_DEGREES:
        image = image.rotate(ROTATION_DEGREES, expand=True)

    return np.array(image, dtype=np.uint8, copy=True)


def detect_graph(image):
    height, width = image.shape
    block = min(151, max(31, round(61 * DPI / 240))) | 1

    foreground = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, 12,
    )
    minimum_length = max(
        round(60 * DPI / 240),
        round(MIN_GRID_SPAN_FRACTION * min(height, width)),
    )

    detected = cv2.HoughLinesP(
        foreground,
        rho=1,
        theta=np.pi / 1800,
        threshold=max(30, minimum_length // 3),
        minLineLength=minimum_length,
        maxLineGap=max(6, round(12 * DPI / 240)),
    )

    if detected is None or np.asarray(detected).size == 0:
        raise ValueError("No graph/grid line segments detected.")

    detected = np.asarray(detected)
    if detected.size % 4:
        raise ValueError(
            f"Unexpected HoughLinesP output shape: {detected.shape}"
        )

    horizontal, vertical = [], []
    for coordinates in detected.reshape(-1, 4):
        x0, y0, x1, y1 = map(int, coordinates)
        dx, dy = x1 - x0, y1 - y0

        if dx and abs(dy / dx) <= MAX_INITIAL_SKEW:
            horizontal.append((x0, y0, x1, y1))
        elif dy and abs(dx / dy) <= MAX_INITIAL_SKEW:
            vertical.append((x0, y0, x1, y1))

    if min(len(horizontal), len(vertical)) < MIN_GRID_LINES_PER_DIRECTION:
        raise ValueError("Insufficient horizontal/vertical grid lines.")

    network = np.zeros(image.shape, np.uint8)
    for x0, y0, x1, y1 in horizontal + vertical:
        cv2.line(network, (x0, y0), (x1, y1), 255, 3)

    join_size = max(3, round(9 * DPI / 240)) | 1
    network = cv2.morphologyEx(
        network,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (join_size, join_size)
        ),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(network)

    def belongs(line, component):
        x0, y0, x1, y1 = line
        x = max(0, min(width - 1, round((x0 + x1) / 2)))
        y = max(0, min(height - 1, round((y0 + y1) / 2)))
        return labels[y, x] == component

    candidates = []
    for component in range(1, count):
        x, y, w, h = map(int, stats[component, :4])

        if (
            w < MIN_GRID_SPAN_FRACTION * width
            or h < MIN_GRID_SPAN_FRACTION * height
        ):
            continue

        hs = [line for line in horizontal if belongs(line, component)]
        vs = [line for line in vertical if belongs(line, component)]

        if min(len(hs), len(vs)) < MIN_GRID_LINES_PER_DIRECTION:
            continue

        candidates.append((
            len(hs) * len(vs),
            w * h,
            (x, y, x + w, y + h),
            hs,
        ))

    if not candidates:
        raise ValueError("No suitable connected graph/grid network found.")

    _, _, bounds, lines = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    slope = float(np.median([
        (y1 - y0) / (x1 - x0)
        for x0, y0, x1, y1 in lines
        if x1 != x0
    ]))
    return bounds, slope


def detect_tick_slope(detections, plot, grid_slope):
    left, top, right, bottom = plot
    reference_x = (left + right) / 2
    band = max(
        round(40 * DPI / 240),
        round(0.15 * (bottom - top)),
    )
    candidates = []

    for box, text, confidence in detections:
        if (
            float(confidence) < TICK_OCR_MIN_CONFIDENCE
            or parse_number(text) is None
        ):
            continue

        points = np.asarray(box, dtype=float)
        x, y = points.mean(axis=0)
        if not left - 15 <= x <= right + 15:
            continue

        corrected_y = y - grid_slope * (x - reference_x)
        if min(abs(corrected_y - top), abs(corrected_y - bottom)) > band:
            continue

        candidates.append((
            float(x),
            float(y),
            float(corrected_y),
            max(5.0, float(np.ptp(points[:, 1]))),
        ))

    choices = []
    for anchor in candidates:
        tolerance = max(8.0, 0.65 * anchor[3])
        group = sorted(
            item for item in candidates
            if abs(item[2] - anchor[2]) <= tolerance
        )

        distinct = []
        for item in group:
            if not distinct or abs(item[0] - distinct[-1][0]) > 5:
                distinct.append(item)

        if len(distinct) < 3:
            continue

        xs = np.array([item[0] for item in distinct])
        ys = np.array([item[1] for item in distinct])
        span = float(np.ptp(xs))
        if span < 0.35 * (right - left):
            continue

        x_reference = float(xs.mean())
        slope, intercept = np.polyfit(xs - x_reference, ys, 1)
        if abs(slope) > MAX_INITIAL_SKEW:
            continue

        residual = ys - (intercept + slope * (xs - x_reference))
        rmse = float(np.sqrt(np.mean(residual ** 2)))

        choices.append((
            len(distinct), span, -rmse,
            float(slope), x_reference, rmse,
        ))

    if not choices:
        if not ALLOW_GRID_DESKEW_FALLBACK:
            raise ValueError("No suitable numerical tick-label row found.")
        return grid_slope, reference_x, None, "grid lines"

    selected = max(choices, key=lambda item: item[:3])
    return selected[3], selected[4], selected[5], "tick labels"


def deskew_page(image, slope, reference_x):
    height, width = image.shape
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [-slope, 1.0, slope * reference_x],
    ], dtype=np.float64)

    corners = cv2.transform(
        rectangle_points(
            (0, 0, width - 1, height - 1)
        ).reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)

    minimum = np.floor(corners.min(axis=0))
    maximum = np.ceil(corners.max(axis=0))
    matrix[:, 2] -= minimum

    result = cv2.warpAffine(
        image,
        matrix,
        (
            int(maximum[0] - minimum[0] + 1),
            int(maximum[1] - minimum[1] + 1),
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return result, matrix


def content_bounds(detections, matrix, plot, shape):
    boxes = [plot]
    left, top, right, bottom = plot
    width, height = right - left, bottom - top

    nearby = (
        max(0, left - 0.08 * width),
        max(0, top - 0.18 * height),
        min(shape[1], right + 0.20 * width),
        min(shape[0], bottom + 0.18 * height),
    )
    ignored = {"|", "||", "|||", "I", "l", "!", "-", "_"}

    for points, text, confidence in detections:
        text = str(text).strip()
        if (
            not text
            or text in ignored
            or float(confidence) < PROTECTED_TEXT_MIN_CONFIDENCE
        ):
            continue

        box = transform_box(matrix, points, shape)
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        x, y = (x0 + x1) / 2, (y0 + y1) / 2

        if w <= 0 or h <= 0:
            continue
        if len(text) == 1 and h > 5 * w:
            continue
        if nearby[0] <= x <= nearby[2] and nearby[1] <= y <= nearby[3]:
            boxes.append(box)

    padding = max(0, round(CONTENT_PADDING_MM * DPI / 25.4))
    bounds = (
        max(0, min(box[0] for box in boxes) - padding),
        max(0, min(box[1] for box in boxes) - padding),
        min(shape[1], max(box[2] for box in boxes) + padding),
        min(shape[0], max(box[3] for box in boxes) + padding),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Content crop is empty.")
    return bounds


def save_crop_preview(image, bounds, path):
    preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    left, top, right, bottom = bounds

    outside = np.ones(image.shape, dtype=bool)
    outside[top:bottom, left:right] = False
    preview[outside] = (
        0.35 * preview[outside].astype(float)
        + 0.65 * np.array([160, 160, 255], dtype=float)
    ).astype(np.uint8)

    cv2.rectangle(
        preview, (left, top), (right - 1, bottom - 1),
        (0, 180, 0), 2,
    )
    save(path, preview)


def validate_settings():
    if not PDF_PATH.is_file():
        raise ValueError(f"PDF not found: {PDF_PATH}")
    if not 72 <= DPI <= 600:
        raise ValueError("DPI must be between 72 and 600.")
    if ROTATION_DEGREES not in (0, 90, 180, 270):
        raise ValueError("ROTATION_DEGREES must be 0, 90, 180, or 270.")
    if not PAGES or any(
        isinstance(page, bool)
        or not isinstance(page, int)
        or page < 1
        for page in PAGES
    ):
        raise ValueError("PAGES must contain positive integer page numbers.")
    if len(set(PAGES)) != len(PAGES):
        raise ValueError("PAGES contains duplicate page numbers.")
    if CONTENT_PADDING_MM < 0:
        raise ValueError("CONTENT_PADDING_MM must not be negative.")


def process_page(document, page_number, ocr):
    print(f"Rendering page {page_number}...")
    image = render_page(document, page_number)

    print("Locating graph...")
    source_plot, grid_slope = detect_graph(image)

    print("Reading text and locating a tick-label row...")
    detections, _ = ocr(image)
    detections = detections or []

    slope, reference_x, rmse, method = detect_tick_slope(
        detections, source_plot, grid_slope
    )
    deskewed, matrix = deskew_page(image, slope, reference_x)
    plot = transform_box(
        matrix, rectangle_points(source_plot), deskewed.shape
    )
    bounds = content_bounds(detections, matrix, plot, deskewed.shape)

    if SAVE_CROP_PREVIEW:
        save_crop_preview(
            deskewed,
            bounds,
            PREVIEW_DIR / f"graph_{page_number}_crop.png",
        )

    left, top, right, bottom = bounds
    output = deskewed[top:bottom, left:right].copy()
    output_path = OUTPUT_DIR / f"graph_{page_number}.png"
    save(output_path, output)

    print(
        f"Saved {output_path} "
        f"({output.shape[1]} x {output.shape[0]} pixels)"
    )
    message = f"  Deskew source: {method}; slope: {slope:.7f}"
    if rmse is not None:
        message += f"; tick-row RMSE: {rmse:.2f} px"
    print(message)


def main():
    validate_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(PDF_PATH))
    failures = []

    try:
        invalid = [page for page in PAGES if page > len(document)]
        if invalid:
            raise ValueError(
                f"PDF has {len(document)} pages; invalid pages: {invalid}"
            )

        ocr = RapidOCR()
        for page_number in PAGES:
            output_path = OUTPUT_DIR / f"graph_{page_number}.png"
            output_path.unlink(missing_ok=True)

            try:
                process_page(document, page_number, ocr)
            except (
                OSError, ValueError, cv2.error, pdfium.PdfiumError
            ) as error:
                failures.append(page_number)
                print(f"Page {page_number} failed: {error}")
                if not CONTINUE_ON_ERROR:
                    raise
    finally:
        document.close()

    if failures:
        print(f"Failed pages: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())