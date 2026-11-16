from pathlib import Path
from dataclasses import dataclass
from bisect import bisect_right
import argparse
import csv
import json
import math
import sys

import cv2
import networkx as nx
import numpy as np
from scipy.linalg import cho_factor, cho_solve


@dataclass
class Fragment:
    index: int
    xs: np.ndarray
    ys: np.ndarray
    slope: object
    first_y: float
    last_y: float


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def palette(count):
    hsv = np.zeros((count, 1, 3), np.uint8)
    hsv[:, 0, 0] = np.linspace(0, 165, count, dtype=np.uint8)
    hsv[:, 0, 1] = 220
    hsv[:, 0, 2] = 210
    return [
        tuple(map(int, color))
        for color in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]
    ]


def banner(image, text):
    width = image.shape[1]
    result = np.full((image.shape[0] + 55, width, 3), 255, np.uint8)
    result[55:] = image
    cv2.putText(
        result, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (0, 0, 140), 1, cv2.LINE_AA,
    )
    cv2.putText(
        result, "Diagnostic geometry only; curve identities are anonymous.",
        (10, 43), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (70, 70, 70), 1, cv2.LINE_AA,
    )
    return result


def load_cleaned(path, requested_crop):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot read {path}")

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    full_ink = binary > 0

    if requested_crop is None:
        ys, xs = np.nonzero(full_ink)
        if not len(xs):
            raise ValueError("The image contains no detected ink.")
        padding = 5
        x0 = max(0, int(xs.min()) - padding)
        y0 = max(0, int(ys.min()) - padding)
        x1 = min(gray.shape[1], int(xs.max()) + padding + 1)
        y1 = min(gray.shape[0], int(ys.max()) + padding + 1)
    else:
        x0, y0, x1, y1 = requested_crop
        if not (
            0 <= x0 < x1 <= gray.shape[1]
            and 0 <= y0 < y1 <= gray.shape[0]
        ):
            raise ValueError("--crop must lie within the input image.")

    local_gray = gray[y0:y1, x0:x1].copy()
    ink = full_ink[y0:y1, x0:x1].copy()

    if min(ink.shape) < 10 or not np.any(ink):
        raise ValueError("The working crop is empty or too small.")

    return local_gray, ink, (x0, y0, x1, y1)


def estimate_stroke(ink):
    distance = cv2.distanceTransform(
        ink.astype(np.uint8), cv2.DIST_L2, 5
    )
    local_maximum = cv2.dilate(distance, np.ones((3, 3), np.uint8))
    radii = distance[
        (distance > 0)
        & (distance >= local_maximum - 1e-6)
    ]
    if not len(radii):
        return 1.0
    return max(1.0, 2.0 * float(np.median(radii)) - 1.0)


def extract_fragments(ink, stroke):
    """
    Connected ink is divided into short x-local pieces.

    These are geometric fragments only. No attempt is made to identify
    grid removal, original dashes, or solid strokes.
    """
    height, width = ink.shape
    piece_width = max(8, int(round(width / 55)))
    minimum_area = max(3, int(round(stroke)))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    fragments = []

    for component in range(1, count):
        left, top, w, h, area = map(int, stats[component])
        if area < minimum_area:
            continue

        yy, xx = np.nonzero(
            labels[top:top + h, left:left + w] == component
        )
        xx = xx + left
        yy = yy + top

        # Order once; splitting then operates on contiguous arrays.
        order = np.argsort(xx, kind="stable")
        xx, yy = xx[order], yy[order]
        bins = (xx - left) // piece_width
        boundaries = np.r_[
            0, np.flatnonzero(np.diff(bins)) + 1, len(xx)
        ]

        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            px, py = xx[begin:end], yy[begin:end]
            columns, starts = np.unique(px, return_index=True)
            if len(columns) < 2:
                continue

            stops = np.r_[starts[1:], len(px)]
            centers = np.array([
                np.median(py[a:b]) for a, b in zip(starts, stops)
            ], float)
            columns = columns.astype(float)

            slope = None
            first_y, last_y = float(centers[0]), float(centers[-1])
            if np.ptp(columns) >= max(3.0, 2.0 * stroke):
                dx = columns - columns.mean()
                denominator = float(dx @ dx)
                if denominator > 0:
                    slope = float(
                        dx @ (centers - centers.mean()) / denominator
                    )
                    first_y = float(
                        centers.mean() + slope * (columns[0] - columns.mean())
                    )
                    last_y = float(
                        centers.mean() + slope * (columns[-1] - columns.mean())
                    )

            fragments.append(Fragment(
                len(fragments), columns, centers,
                slope, first_y, last_y,
            ))

    return fragments


def transition_cost(a, b, stroke):
    gap = float(b.xs[0] - a.xs[-1])
    dy = float(b.first_y - a.last_y)
    if gap <= 0:
        return None

    # A broad local direction bound, applied identically to all paths.
    if abs(dy) > 6.0 * gap + 4.0 * stroke:
        return None

    scale = 3.0 * stroke + 0.20 * gap
    errors = []

    if a.slope is not None:
        errors.append(abs(dy - a.slope * gap))
    if b.slope is not None:
        errors.append(abs(dy - b.slope * gap))

    if errors:
        if max(errors) > 4.0 * scale:
            return None
        geometric = float(np.mean(errors)) / scale
    else:
        geometric = abs(dy) / max(scale, gap)

    turn = 0.0
    if a.slope is not None and b.slope is not None:
        turn = abs(math.atan(a.slope) - math.atan(b.slope))

    # All white gaps receive the same treatment.
    return geometric + 2.0 * turn + 0.04 * gap


def initial_paths(fragments, width, stroke, requested_count, args):
    """
    Minimum-cost flow selects up to the requested number of paths.

    Fragment capacity is one: initial paths cannot reuse a fragment.
    An empty bypass allows fewer paths instead of forcing clutter
    into the requested count.
    """
    source, sink = "source", "sink"
    graph = nx.DiGraph()
    graph.add_node(source, demand=-requested_count)
    graph.add_node(sink, demand=requested_count)
    graph.add_edge(
        source, sink, capacity=requested_count, weight=0
    )

    birth_cost = max(8.0, 8.0 * stroke)
    scale = 100

    for fragment in fragments:
        incoming = 2 * fragment.index
        outgoing = incoming + 1
        reward = float(len(fragment.xs))
        graph.add_edge(
            source, incoming, capacity=1,
            weight=int(round(scale * birth_cost)),
        )
        graph.add_edge(
            incoming, outgoing, capacity=1,
            weight=-int(round(scale * reward)),
        )
        graph.add_edge(outgoing, sink, capacity=1, weight=0)

    ordered = sorted(fragments, key=lambda fragment: fragment.xs[0])
    starts = [float(fragment.xs[0]) for fragment in ordered]
    max_gap = max(12.0 * stroke, args.max_gap_fraction * width)

    for a in fragments:
        first = bisect_right(starts, float(a.xs[-1]))
        last = bisect_right(starts, float(a.xs[-1] + max_gap))
        choices = []

        for b in ordered[first:last]:
            cost = transition_cost(a, b, stroke)
            if cost is not None:
                choices.append((cost, b.index))

        for cost, index in sorted(choices)[:args.max_links]:
            graph.add_edge(
                2 * a.index + 1, 2 * index,
                capacity=1, weight=int(round(scale * cost)),
            )

    print(
        f"Initialization graph: {len(fragments)} fragments, "
        f"{graph.number_of_edges()} edges.",
        flush=True,
    )
    _, flow = nx.network_simplex(graph)

    paths = []
    for node, amount in flow[source].items():
        if node == sink or amount <= 0:
            continue

        fragment_ids = []
        while True:
            index = node // 2
            fragment_ids.append(index)
            outgoing = 2 * index + 1
            following = [
                target for target, value in flow[outgoing].items()
                if value > 0
            ]
            if not following or following[0] == sink:
                break
            node = following[0]

        paths.append(fragment_ids)

    return paths


def path_points(path, fragments):
    xs = np.concatenate([fragments[index].xs for index in path])
    ys = np.concatenate([fragments[index].ys for index in path])
    order = np.argsort(xs)
    return xs[order], ys[order]


def order_paths(paths, fragments, stroke):
    """
    Infer vertical order using overlapping path extents.

    Ambiguous or inconsistent initial ordering is reported. No physical
    labels or equal-spacing assumptions are used.
    """
    points = [path_points(path, fragments) for path in paths]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(paths)))
    warnings = []

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            ax, ay = points[i]
            bx, by = points[j]
            first = max(ax[0], bx[0])
            last = min(ax[-1], bx[-1])
            if last - first < max(4.0, 3.0 * stroke):
                continue

            xs = np.linspace(first, last, 25)
            difference = (
                np.interp(xs, ax, ay) - np.interp(xs, bx, by)
            )
            median = float(np.median(difference))

            if (
                np.any(difference < -stroke)
                and np.any(difference > stroke)
            ):
                warnings.append(
                    f"Initial paths {i + 1}/{j + 1} change relative order."
                )

            if abs(median) > 0.5 * stroke:
                above, below = (i, j) if median < 0 else (j, i)
                graph.add_edge(above, below, strength=abs(median))

    # Do not hide inconsistent initial path ordering.
    while not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph)
        edge = min(
            cycle,
            key=lambda pair: graph.edges[pair]["strength"],
        )
        graph.remove_edge(*edge)
        warnings.append(
            "An inconsistent initial-order relation was removed."
        )

    order = list(nx.lexicographical_topological_sort(
        graph, key=lambda index: float(np.median(points[index][1]))
    ))
    return [paths[index] for index in order], warnings


def project_order(values):
    """
    Euclidean projection onto nondecreasing y across curves at each node.

    Equal values are allowed. There is no minimum separation.
    """
    result = np.empty_like(values)
    count, nodes = values.shape

    for column in range(nodes):
        means = []
        weights = []
        lengths = []

        for value in values[:, column]:
            means.append(float(value))
            weights.append(1.0)
            lengths.append(1)

            while len(means) >= 2 and means[-2] > means[-1]:
                weight = weights[-2] + weights[-1]
                mean = (
                    means[-2] * weights[-2]
                    + means[-1] * weights[-1]
                ) / weight
                length = lengths[-2] + lengths[-1]
                means[-2:] = [mean]
                weights[-2:] = [weight]
                lengths[-2:] = [length]

        position = 0
        for mean, length in zip(means, lengths):
            result[position:position + length, column] = mean
            position += length

    return result


def interpolation_rows(xs, nodes):
    index = np.searchsorted(nodes, xs, side="right") - 1
    index = np.clip(index, 0, len(nodes) - 2)
    fraction = (
        (xs - nodes[index]) / (nodes[index + 1] - nodes[index])
    )
    design = np.zeros((len(xs), len(nodes)), float)
    rows = np.arange(len(xs))
    design[rows, index] = 1.0 - fraction
    design[rows, index + 1] = fraction
    return design


def build_quadratics(observations, nodes, current, stroke, smooth):
    count, node_count = current.shape
    du = 1.0 / (node_count - 1)
    second = np.diff(np.eye(node_count), n=2, axis=0) / (du * du)
    regularizer = smooth * (second.T @ second) / (node_count - 2)

    matrices, vectors = [], []
    for k, (xs, ys) in enumerate(observations):
        matrix = regularizer.copy()
        vector = np.zeros(node_count)

        if len(xs):
            design = interpolation_rows(xs, nodes)
            residual = design @ current[k] - ys

            # Robust weighting uses only this curve's own residuals.
            weights = 1.0 / np.sqrt(
                1.0 + (residual / max(1.0, 2.0 * stroke)) ** 2
            )
            weights /= max(float(weights.sum()), 1e-12)
            matrix += design.T @ (weights[:, None] * design)
            vector += design.T @ (weights * ys)

        matrices.append(matrix)
        vectors.append(vector)

    return matrices, np.asarray(vectors)


def ordered_smooth_fit(observations, nodes, initial, stroke, args):
    """
    Minimize separate robust quadratic fitting/smoothing objectives,
    subject only to nodewise order.

    Linear interpolation preserves this order between adjacent nodes.
    """
    current = project_order(initial)
    diagnostics = []

    for robust_pass in range(2):
        matrices, vectors = build_quadratics(
            observations, nodes, current, stroke, args.smooth
        )
        count, node_count = current.shape
        rho = 1.0 / node_count
        z = current.copy()
        dual = np.zeros_like(z)
        identity = np.eye(node_count)
        factors = None
        primal_rms = change_rms = float("inf")
        converged = False

        for iteration in range(args.admm_iterations):
            if factors is None:
                factors = [
                    cho_factor(
                        matrix + (rho + 1e-10) * identity,
                        lower=True, check_finite=False,
                    )
                    for matrix in matrices
                ]

            x = np.asarray([
                cho_solve(
                    factors[k],
                    vectors[k] + rho * (z[k] - dual[k]),
                    check_finite=False,
                )
                for k in range(count)
            ])

            previous = z
            z = project_order(x + dual)
            dual += x - z

            primal_rms = float(np.sqrt(np.mean((x - z) ** 2)))
            change_rms = float(np.sqrt(np.mean((z - previous) ** 2)))

            if primal_rms < args.fit_tolerance and change_rms < args.fit_tolerance:
                converged = True
                break

            if (iteration + 1) % 25 == 0:
                primal = float(np.linalg.norm(x - z))
                dual_residual = rho * float(np.linalg.norm(z - previous))
                old_rho = rho

                if primal > 10.0 * max(dual_residual, 1e-12):
                    rho *= 2.0
                elif dual_residual > 10.0 * max(primal, 1e-12):
                    rho /= 2.0

                if rho != old_rho:
                    dual *= old_rho / rho
                    factors = None

        current = z
        diagnostics.append({
            "robust_pass": robust_pass + 1,
            "iterations": iteration + 1,
            "converged": converged,
            "primal_rms_px": primal_rms,
            "change_rms_px": change_rms,
        })

    return current, diagnostics


def column_observations(ink, sample_step):
    columns = []
    for x in range(0, ink.shape[1], sample_step):
        rows = np.flatnonzero(ink[:, x])
        if not len(rows):
            columns.append((x, np.empty(0)))
            continue

        groups = np.split(
            rows, np.flatnonzero(np.diff(rows) > 1) + 1
        )
        centers = np.array([group.mean() for group in groups], float)
        columns.append((x, centers))

    return columns


def ordered_assignment(predicted, candidates, gate):
    """
    At one column:
      * an observation can belong to at most one curve;
      * a curve can receive at most one observation;
      * missing observations and unused ink are allowed.
    """
    count, total = len(predicted), len(candidates)
    costs = np.full((count + 1, total + 1), np.inf)
    actions = np.zeros((count + 1, total + 1), np.uint8)
    costs[0, :] = 0.0

    for i in range(1, count + 1):
        costs[i, 0] = float(i)
        actions[i, 0] = 2

    for i in range(1, count + 1):
        for j in range(1, total + 1):
            best, operation = costs[i, j - 1], 1

            missing = costs[i - 1, j] + 1.0
            if missing < best:
                best, operation = missing, 2

            distance = abs(predicted[i - 1] - candidates[j - 1])
            if distance <= gate:
                matched = costs[i - 1, j - 1] + (distance / gate) ** 2
                if matched < best:
                    best, operation = matched, 3

            costs[i, j] = best
            actions[i, j] = operation

    owners = np.full(total, -1, int)
    i, j = count, total
    while i > 0:
        operation = actions[i, j]
        if operation == 3:
            owners[j - 1] = i - 1
            i -= 1
            j -= 1
        elif operation == 2:
            i -= 1
        elif j:
            j -= 1
        else:
            break

    return owners


def assign_columns(columns, nodes, curves, gate):
    observations = [[] for _ in curves]
    records = []

    for x, candidates in columns:
        predicted = np.array([
            np.interp(x, nodes, curve) for curve in curves
        ])
        owners = ordered_assignment(predicted, candidates, gate)

        for y, owner in zip(candidates, owners):
            records.append((float(x), float(y), int(owner)))
            if owner >= 0:
                observations[owner].append((float(x), float(y)))

    result = []
    for points in observations:
        if points:
            array = np.asarray(points)
            result.append((array[:, 0], array[:, 1]))
        else:
            result.append((np.empty(0), np.empty(0)))
    return result, records


def render_curves(gray, nodes, curves, observations, colors):
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    width, height = gray.shape[1], gray.shape[0]
    all_xs = np.arange(width)

    for k, (curve, color) in enumerate(zip(curves, colors)):
        ys = np.interp(all_xs, nodes, curve)
        ox, _ = observations[k]
        if not len(ox):
            continue

        first, last = int(np.ceil(ox.min())), int(np.floor(ox.max()))
        for x in range(first, last):
            y0, y1 = ys[x], ys[x + 1]
            if not (0 <= y0 < height and 0 <= y1 < height):
                continue

            index = np.searchsorted(ox, x)
            nearby = []
            if index < len(ox):
                nearby.append(abs(ox[index] - x))
            if index:
                nearby.append(abs(ox[index - 1] - x))
            distance = min(nearby) if nearby else float("inf")

            # Display missing-data spans in gray; no gap-type inference.
            line_color = color if distance <= 3 else (150, 150, 150)
            cv2.line(
                image, (x, round(y0)), (x + 1, round(y1)),
                line_color, 2 if line_color == color else 1,
            )

        label_x = int(np.clip(first, 0, width - 1))
        label_y = int(np.clip(round(ys[label_x]), 12, height - 3))
        cv2.putText(
            image, f"C{k + 1:02d}", (label_x + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    return image


def render_assignments(gray, records, colors):
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x, y, owner in records:
        color = colors[owner] if owner >= 0 else (0, 0, 230)
        cv2.circle(image, (round(x), round(y)), 2, color, -1)
    return image


def curve_report(curves, nodes, observations, stroke):
    reports = []
    for k, (xs, ys) in enumerate(observations):
        item = {
            "curve": f"C{k + 1:02d}",
            "assigned_columns": int(len(xs)),
            "status": "provisional" if len(xs) >= 4 else "insufficient_support",
        }
        if len(xs):
            residual = np.interp(xs, nodes, curves[k]) - ys
            item.update({
                "first_local_x": float(xs.min()),
                "last_local_x": float(xs.max()),
                "span_px": float(np.ptp(xs)),
                "median_abs_residual_px": float(np.median(np.abs(residual))),
                "largest_observation_gap_px": (
                    float(np.max(np.diff(xs))) if len(xs) > 1 else None
                ),
            })
        reports.append(item)

    close_pairs = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            ax, _ = observations[i]
            bx, _ = observations[j]
            if not len(ax) or not len(bx):
                continue

            first, last = max(ax.min(), bx.min()), min(ax.max(), bx.max())
            if last <= first:
                continue

            xs = np.linspace(first, last, 200)
            separation = np.abs(
                np.interp(xs, nodes, curves[i])
                - np.interp(xs, nodes, curves[j])
            )
            fraction = float(np.mean(separation <= 2.0 * stroke))
            if fraction >= 0.10:
                close_pairs.append({
                    "curves": [f"C{i + 1:02d}", f"C{j + 1:02d}"],
                    "fraction_within_two_stroke_widths": fraction,
                    "median_separation_px": float(np.median(separation)),
                    "note": "Diagnostic warning, not automatic rejection.",
                })

    return reports, close_pairs


def write_outputs(
    out_dir, gray, crop_box, nodes, curves,
    observations, records, colors, summary,
):
    offset_x, offset_y = crop_box[:2]
    save_image(
        out_dir / "fits.png",
        banner(
            render_curves(gray, nodes, curves, observations, colors),
            "PROVISIONAL FITS | Colored=near assigned columns; gray=gaps",
        ),
    )
    save_image(
        out_dir / "assignments.png",
        banner(
            render_assignments(gray, records, colors),
            "ASSIGNMENTS | Colored=assigned; red=unused observation",
        ),
    )

    with (out_dir / "observations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("pixel_x", "pixel_y", "curve", "status"))
        for x, y, owner in records:
            writer.writerow((
                f"{x + offset_x:.3f}", f"{y + offset_y:.3f}",
                "" if owner < 0 else f"C{owner + 1:02d}",
                "unused" if owner < 0 else "assigned",
            ))

    with (out_dir / "curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "curve", "pixel_x", "fitted_pixel_y",
            "assigned_pixel_y", "estimate_kind",
        ))

        for k, (xs, ys) in enumerate(observations):
            if len(xs) < 4:
                continue
            assigned = {int(x): float(y) for x, y in zip(xs, ys)}

            # No export beyond this curve's first/last assigned column.
            for x in range(int(np.ceil(xs.min())), int(np.floor(xs.max())) + 1):
                fitted_y = float(np.interp(x, nodes, curves[k]))
                observed_y = assigned.get(x)
                writer.writerow((
                    f"C{k + 1:02d}", x + offset_x,
                    f"{fitted_y + offset_y:.6f}",
                    "" if observed_y is None else f"{observed_y + offset_y:.6f}",
                    "assigned_column" if observed_y is not None else "interpolated",
                ))

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone fitting from a cleaned image only. "
            "Independent smoothing objectives; noncrossing order constraints."
        )
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--curves", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/clean_fit_test"))
    parser.add_argument(
        "--crop", nargs=4, type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Optional input-image crop; upper bounds are exclusive.",
    )
    parser.add_argument(
        "--stroke-width", type=float,
        help="Optional stroke width in pixels; otherwise estimated.",
    )
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument(
        "--smooth", type=float, default=1e-5,
        help="Individual normalized-x curvature penalty.",
    )
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--sample-columns", type=int, default=600)
    parser.add_argument(
        "--gate-strokes", type=float, default=4.0,
        help="Final assignment radius, in estimated stroke widths.",
    )
    parser.add_argument(
        "--max-gap-fraction", type=float, default=0.12,
        help="Maximum initialization link gap as a fraction of crop width.",
    )
    parser.add_argument("--max-links", type=int, default=12)
    parser.add_argument("--admm-iterations", type=int, default=500)
    parser.add_argument("--fit-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    if args.curves < 1 or args.nodes < 4:
        raise ValueError("--curves must be positive; --nodes must be >= 4.")
    if min(
        args.iterations, args.sample_columns,
        args.max_links, args.admm_iterations,
    ) < 1:
        raise ValueError("Iteration, sampling and link counts must be positive.")
    for name in ("smooth", "gate_strokes", "max_gap_fraction", "fit_tolerance"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.stroke_width is not None:
        if not math.isfinite(args.stroke_width) or args.stroke_width <= 0:
            raise ValueError("--stroke-width must be positive.")

    output_names = (
        "working.png", "initialization.png", "fits.png",
        "assignments.png", "observations.csv", "curves.csv", "summary.json",
    )
    if any(
        (args.out_dir / name).resolve() == args.image.resolve()
        for name in output_names
    ):
        raise ValueError("An output would overwrite the input image.")

    gray, ink, crop_box = load_cleaned(args.image, args.crop)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Clear only this test's known outputs, never the input.
    for name in output_names:
        (args.out_dir / name).unlink(missing_ok=True)
    save_image(args.out_dir / "working.png", gray)

    height, width = ink.shape
    stroke = args.stroke_width or estimate_stroke(ink)
    print(
        f"Working image: {width} x {height}; crop={crop_box}; "
        f"stroke estimate={stroke:.2f} px.",
        flush=True,
    )

    fragments = extract_fragments(ink, stroke)
    if not fragments:
        raise ValueError("No usable ink fragments were found.")

    paths = initial_paths(
        fragments, width, stroke, args.curves, args
    )
    if not paths:
        failure = {
            "source": str(args.image),
            "crop": list(crop_box),
            "requested_curves": args.curves,
            "initialized_paths": 0,
            "fragments": len(fragments),
            "status": "initialization_failed",
            "note": "No data-supported path beat the empty-path alternative.",
        }
        with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, indent=2)
        raise ValueError(
            "Initialization selected no paths. working.png and summary.json "
            "were saved; no fitted curves were fabricated."
        )

    paths, warnings = order_paths(paths, fragments, stroke)
    observations = [path_points(path, fragments) for path in paths]
    nodes = np.linspace(0.0, float(width - 1), args.nodes)
    curves = np.asarray([
        np.interp(nodes, xs, ys) for xs, ys in observations
    ])
    curves = project_order(curves)
    colors = palette(len(paths))

    save_image(
        args.out_dir / "initialization.png",
        banner(
            render_curves(gray, nodes, curves, observations, colors),
            "INITIALIZATION ONLY | Fragment paths before smoothing",
        ),
    )

    print(
        f"Initialized {len(paths)}/{args.curves} paths. "
        "Missing paths are not filled with artificial curves.",
        flush=True,
    )

    sample_step = max(1, int(math.ceil(width / args.sample_columns)))
    columns = column_observations(ink, sample_step)
    history = []
    previous_signature = None

    # First fit uses the flow-selected fragments directly.
    curves, solver_log = ordered_smooth_fit(
        observations, nodes, curves, stroke, args
    )
    history.append({"stage": "initial_fit", "solver": solver_log})

    records = []
    for iteration in range(args.iterations):
        # Broad initial capture, then a fixed evidence-local gate.
        capture_factor = 1.5 if iteration < 2 else 1.0
        gate = max(2.0, args.gate_strokes * stroke * capture_factor)

        observations, records = assign_columns(
            columns, nodes, curves, gate
        )
        signature = tuple(record[2] for record in records)
        previous = curves.copy()

        curves, solver_log = ordered_smooth_fit(
            observations, nodes, curves, stroke, args
        )
        movement = float(np.max(np.abs(curves - previous)))
        counts = [len(xs) for xs, _ in observations]

        print(
            f"Iteration {iteration + 1:02d}: "
            f"maximum node movement={movement:.3f} px; "
            f"assigned columns={counts}",
            flush=True,
        )

        history.append({
            "stage": "assignment_fit",
            "iteration": iteration + 1,
            "gate_px": gate,
            "maximum_node_movement_px": movement,
            "assigned_columns": counts,
            "solver": solver_log,
        })

        # Save useful provisional outputs even before the test finishes.
        save_image(
            args.out_dir / "fits.png",
            banner(
                render_curves(gray, nodes, curves, observations, colors),
                f"PROVISIONAL FITS | Iteration {iteration + 1}",
            ),
        )

        if (
            iteration >= 2
            and signature == previous_signature
            and movement < 0.10
        ):
            break
        previous_signature = signature

    # Final observations correspond to final geometry and the final gate.
    observations, records = assign_columns(
        columns, nodes, curves,
        max(2.0, args.gate_strokes * stroke),
    )

    reports, close_pairs = curve_report(
        curves, nodes, observations, stroke
    )
    if len(curves) > 1:
        minimum_separation = float(np.min(np.diff(curves, axis=0)))
    else:
        minimum_separation = None

    supported_count = sum(
        report["assigned_columns"] >= 4 for report in reports
    )
    assigned_count = sum(owner >= 0 for _, _, owner in records)
    all_converged = all(
        item["converged"]
        for stage in history
        for item in stage["solver"]
    )

    summary = {
        "source": str(args.image),
        "input_evidence": "cleaned image only",
        "crop": list(crop_box),
        "coordinate_system": "original input-image pixels",
        "working_shape": [height, width],
        "estimated_stroke_width_px": float(stroke),
        "requested_curves": args.curves,
        "initialized_paths": len(paths),
        "paths_with_at_least_four_final_assignments": supported_count,
        "fragments": len(fragments),
        "sample_column_step_px": sample_step,
        "sampled_observations": len(records),
        "assigned_observations": assigned_count,
        "unused_observations": len(records) - assigned_count,
        "minimum_node_separation_px": minimum_separation,
        "noncrossing": minimum_separation is None or minimum_separation >= -1e-8,
        "all_optimizer_passes_converged": all_converged,
        "ordering_warnings": warnings,
        "curves": reports,
        "near_coincident_pairs": close_pairs,
        "history": history,
        "parameters": {
            "nodes": args.nodes,
            "smooth": args.smooth,
            "gate_strokes": args.gate_strokes,
            "max_gap_fraction": args.max_gap_fraction,
            "max_links": args.max_links,
        },
        "status": "provisional_test_output",
        "limitations": [
            "Initialization and observation correspondence are heuristic.",
            "Noncrossing does not establish correct curve identities.",
            "Equal or near-coincident curves are allowed and reported.",
            "All white gaps are treated alike.",
            "No curves are exported beyond their assigned x extent.",
            "No mask, original image, calibration, physical tags, or prior fits are used.",
        ],
    }

    write_outputs(
        args.out_dir, gray, crop_box, nodes, curves,
        observations, records, colors, summary,
    )

    print(f"\nWrote test outputs to {args.out_dir}")
    print(
        f"Paths: {len(paths)}/{args.curves}; "
        f"paths with >=4 assignments: {supported_count}; "
        f"near-coincident pairs: {len(close_pairs)}."
    )
    print(f"All optimizer passes converged: {all_converged}")

    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if len(paths) != args.curves:
        print(
            "Warning: initialization did not recover the requested count.",
            file=sys.stderr,
        )
    if not all_converged:
        print(
            "Warning: at least one optimizer pass reached its iteration limit. "
            "Geometry is ordered, but numerical convergence is not established.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, cv2.error, nx.NetworkXException) as error:
        print(f"Cleaned-image test failed: {error}", file=sys.stderr)
        sys.exit(1)