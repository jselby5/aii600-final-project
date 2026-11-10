"""Extract and independently deskew report pages 223 and 224.

Run from the repository root:

    python scripts/stitch_graph_pages.py data/DTIC_AD0224171.pdf \
        --output-dir data

Outputs:
    data/graph_223.png
    data/graph_224.png

Each page is:
1. Rendered at the requested DPI.
2. Rotated 270 degrees.
3. Deskewed using the slope of its x-axis tick-label positions.
4. Saved independently.

No cross-page resizing, axis-scale matching, stitching, or compositing.

The complete deskewed pages are saved by default, preserving axis labels.

Optional:
    --plot-only

This crops to the transformed plot bounds from the original script.
Labels outside those plot bounds will be excluded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR


@dataclass(frozen=True)
class PageSpec:
    page_number: int
    x_tick_count: int
    x_tick_band: tuple[int, int]
    x_candidate_bounds: tuple[int, int]
    plot_bounds: tuple[int, int, int, int]


# Coordinates refer to rotated page images rendered at 300 DPI.
PAGE_SPECS = (
    PageSpec(
        page_number=223,
        x_tick_count=15,
        x_tick_band=(500, 575),
        x_candidate_bounds=(400, 3220),
        plot_bounds=(420, 617, 3204, 2224),
    ),
    PageSpec(
        page_number=224,
        x_tick_count=15,
        x_tick_band=(580, 660),
        x_candidate_bounds=(180, 3000),
        plot_bounds=(198, 705, 2976, 2465),
    ),
)


def scaled_spec(spec: PageSpec, scale: float) -> PageSpec:
    return PageSpec(
        page_number=spec.page_number,
        x_tick_count=spec.x_tick_count,
        x_tick_band=tuple(
            round(value * scale) for value in spec.x_tick_band
        ),
        x_candidate_bounds=tuple(
            round(value * scale) for value in spec.x_candidate_bounds
        ),
        plot_bounds=tuple(
            round(value * scale) for value in spec.plot_bounds
        ),
    )


def render_page(
    document: pdfium.PdfDocument,
    page_number: int,
    dpi: int,
) -> np.ndarray:
    page = document[page_number - 1]

    try:
        bitmap = page.render(scale=dpi / 72)

        try:
            image = bitmap.to_pil().convert("L").copy()
        finally:
            bitmap.close()

    finally:
        page.close()

    image = image.rotate(270, expand=True)
    return np.array(image, dtype=np.uint8, copy=True)


def detect_tick_slope(
    detections: Sequence,
    spec: PageSpec,
) -> tuple[float, float, float]:
    """
    Recover the original x-axis tick-label slope.

    Returns:
        slope, reference_x, vertical_fit_rmse_pixels
    """
    band_top, band_bottom = spec.x_tick_band
    min_x, max_x = spec.x_candidate_bounds
    centers = []

    for box, _, _ in detections:
        center_x, center_y = np.asarray(
            box, dtype=float
        ).mean(axis=0)

        if (
            min_x <= center_x <= max_x
            and band_top <= center_y <= band_bottom
        ):
            centers.append((float(center_x), float(center_y)))

    centers.sort()

    if len(centers) != spec.x_tick_count:
        raise ValueError(
            f"Page {spec.page_number}: found {len(centers)} "
            f"x-axis tick boxes; expected {spec.x_tick_count}. "
            "Deskewing was not applied."
        )

    tick_x = np.array([point[0] for point in centers])
    tick_y = np.array([point[1] for point in centers])

    reference_x = float(tick_x.mean())

    # Centering x improves numerical conditioning.
    slope, reference_y = np.polyfit(
        tick_x - reference_x,
        tick_y,
        1,
    )

    predicted_y = reference_y + slope * (tick_x - reference_x)
    rmse = float(np.sqrt(np.mean((tick_y - predicted_y) ** 2)))

    return float(slope), reference_x, rmse


def deskew_page(
    image: np.ndarray,
    slope: float,
    reference_x: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the original script's deskew correction:

        x' = x
        y' = y - slope * (x - reference_x)

    Padding is added so transformed image edges are not clipped.

    Returns:
        deskewed image, source-to-output affine matrix
    """
    height, width = image.shape

    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [-slope, 1.0, slope * reference_x],
        ],
        dtype=np.float64,
    )

    corners = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float64,
    )

    transformed = cv2.transform(
        corners.reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)

    minimum = np.floor(transformed.min(axis=0))
    maximum = np.ceil(transformed.max(axis=0))

    matrix[:, 2] -= minimum

    output_width = int(maximum[0] - minimum[0] + 1)
    output_height = int(maximum[1] - minimum[1] + 1)

    deskewed = cv2.warpAffine(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    return deskewed, matrix


def crop_transformed_plot(
    image: np.ndarray,
    matrix: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop to the bounding rectangle of the transformed plot."""
    left, top, right, bottom = bounds

    corners = np.array(
        [
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ],
        dtype=np.float64,
    )

    transformed = cv2.transform(
        corners.reshape(-1, 1, 2),
        matrix,
    ).reshape(-1, 2)

    minimum = np.floor(transformed.min(axis=0)).astype(int)
    maximum = np.ceil(transformed.max(axis=0)).astype(int)

    x0 = max(0, int(minimum[0]))
    y0 = max(0, int(minimum[1]))
    x1 = min(image.shape[1], int(maximum[0]) + 1)
    y1 = min(image.shape[0], int(maximum[1]) + 1)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "Transformed plot bounds do not overlap the rendered image."
        )

    return image[y0:y1, x0:x1].copy()


def extract_pages(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 240,
    plot_only: bool = False,
) -> None:
    if not 72 <= dpi <= 600:
        raise ValueError("DPI must be between 72 and 600")

    output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        scaled_spec(spec, dpi / 300)
        for spec in PAGE_SPECS
    ]

    document = pdfium.PdfDocument(str(pdf_path))

    try:
        for spec in specs:
            if not 1 <= spec.page_number <= len(document):
                raise ValueError(
                    f"PDF has {len(document)} pages; "
                    f"requested page {spec.page_number}"
                )

        ocr = RapidOCR()

        for spec in specs:
            print(f"Rendering page {spec.page_number}...")
            image = render_page(document, spec.page_number, dpi)

            print("Detecting x-axis tick positions...")
            detections, _ = ocr(image)

            slope, reference_x, rmse = detect_tick_slope(
                detections or [],
                spec,
            )

            deskewed, matrix = deskew_page(
                image,
                slope,
                reference_x,
            )

            if plot_only:
                deskewed = crop_transformed_plot(
                    deskewed,
                    matrix,
                    spec.plot_bounds,
                )

            output_path = output_dir / f"graph_{spec.page_number}.png"

            if not cv2.imwrite(str(output_path), deskewed):
                raise OSError(f"Could not write {output_path}")

            print(
                f"Saved {output_path} "
                f"({deskewed.shape[1]} x {deskewed.shape[0]} pixels)"
            )
            print(
                f"  Tick slope: {slope:.7f}; "
                f"tick-height fit RMSE: {rmse:.2f} pixels"
            )

    finally:
        document.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "pdf",
        type=Path,
        help="report PDF containing pages 223 and 224",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="directory for separate PNG files (default: data)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="render resolution (default: 240 DPI)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="crop to transformed plot bounds, excluding outside labels",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        extract_pages(
            args.pdf,
            args.output_dir,
            dpi=args.dpi,
            plot_only=args.plot_only,
        )
    except (OSError, ValueError, pdfium.PdfiumError) as error:
        parser.exit(1, f"Extraction failed: {error}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())