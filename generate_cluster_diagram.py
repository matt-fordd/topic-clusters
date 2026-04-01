#!/usr/bin/env python3
"""
Generate an interactive topic cluster scatter diagram from a Lumar-style CSV export.
Parses topic_similarities, positions URLs by semantic relevance, color-codes by primary topic.
"""

import csv
import json
import math
import os
import random
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "www-sourcewell-mn-gov_24-03-2026_All_Pages_basic_filtered.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "cluster_diagram.html")

# Distinct colors for dynamic topic assignment (cycles if more topics than entries)
TOPIC_COLOR_PALETTE = [
    "#3b82f6", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444", "#ec4899",
    "#84cc16", "#14b8a6", "#a855f7", "#f97316", "#6366f1", "#22c55e", "#eab308",
    "#d946ef", "#0ea5e9", "#64748b", "#f43f5e", "#2dd4bf", "#c084fc", "#fb923c",
    "#4ade80", "#38bdf8", "#f472b6", "#a3e635", "#fbbf24", "#94a3b8", "#7c3aed",
    "#059669", "#dc2626", "#2563eb",
]


def assign_topic_colors(topic_names):
    """Map each topic name to a distinct hex color."""
    return {name: TOPIC_COLOR_PALETTE[i % len(TOPIC_COLOR_PALETTE)] for i, name in enumerate(topic_names)}


def discover_topic_order(csv_path):
    """Scan CSV and return sorted list of unique topic_name values from topic_similarities."""
    names = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("topic_similarities", "").strip()
            if not ts:
                continue
            try:
                data = json.loads(ts)
                for item in data:
                    t = item.get("topic_name", "").strip()
                    if t:
                        names.add(t)
            except (json.JSONDecodeError, TypeError):
                pass
    return sorted(names)


def parse_csv(csv_path):
    """Parse CSV and return (urls, topic_colors). Each url has topic_vector aligned to topic order."""
    topic_order = discover_topic_order(csv_path)
    if not topic_order:
        return [], {}, []
    topic_idx = {t: i for i, t in enumerate(topic_order)}
    dim = len(topic_order)
    topic_colors = assign_topic_colors(topic_order)
    urls = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("topic_similarities", "").strip()
            url = row.get("url", "").strip()
            page_title = row.get("page_title", "").strip() or url
            if not ts or not url:
                continue
            try:
                data = json.loads(ts)
                if not data:
                    continue
                vec = [0.0] * dim
                best_topic, best_sim = None, 0.0
                for item in data:
                    topic = item.get("topic_name", "")
                    sim = item.get("cosine_similarity", 0)
                    if topic in topic_idx:
                        vec[topic_idx[topic]] = sim
                    if sim > best_sim:
                        best_sim = sim
                        best_topic = topic
                if best_topic:
                    urls.append({
                        "url": url,
                        "page_title": page_title,
                        "topic_vector": vec,
                        "primary_topic": best_topic,
                        "primary_sim": round(best_sim, 3),
                        "topic_index": topic_idx[best_topic],
                    })
            except (json.JSONDecodeError, KeyError):
                pass
    return urls, topic_colors, topic_order


def _embed_vectors(vectors, method, seed):
    """Return list of (x,y) from topic-dim vectors. method: 'tsne'|'umap'|'polar'."""
    n_topics = len(vectors[0]) if vectors else 0
    if method == "tsne":
        try:
            import numpy as np
            from sklearn.manifold import TSNE
            X = np.array(vectors, dtype=np.float64)
            n = len(vectors)
            perp = min(30, max(2, n - 1)) if n > 1 else 2
            embedded = TSNE(n_components=2, random_state=seed, perplexity=perp).fit_transform(X)
            return [(float(embedded[i][0]), float(embedded[i][1])) for i in range(len(vectors))]
        except ImportError:
            method = "polar"
    if method == "umap":
        try:
            import numpy as np
            import umap
            X = np.array(vectors, dtype=np.float64)
            embedded = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1).fit_transform(X)
            return [(float(embedded[i][0]), float(embedded[i][1])) for i in range(len(vectors))]
        except ImportError:
            return None
    # polar
    angles = [2 * math.pi * i / n_topics - math.pi / 2 for i in range(n_topics)]
    return [
        (sum(vec[i] * math.cos(angles[i]) for i in range(n_topics)),
         sum(vec[i] * math.sin(angles[i]) for i in range(n_topics)))
        for vec in vectors
    ]


def compute_scatter_positions(urls, method="tsne", width=800, height=500, margin_left=70, margin_right=20, margin_top=20, margin_bottom=55, jitter=5, seed=42):
    """Position URLs using specified method: tsne, umap, or polar."""
    random.seed(seed)
    vectors = [u["topic_vector"] for u in urls]

    raw_coords = _embed_vectors(vectors, method, seed)
    if raw_coords is None:
        return None

    points = []
    for i, u in enumerate(urls):
        points.append({**u, "raw_x": raw_coords[i][0], "raw_y": raw_coords[i][1]})

    # Scale to fit chart area with padding
    xs = [p["raw_x"] for p in points]
    ys = [p["raw_y"] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x or 1
    range_y = max_y - min_y or 1

    usable_w = width - margin_left - margin_right
    usable_h = height - margin_top - margin_bottom
    pad = 0.1  # Padding within chart so clusters don't touch edges
    scale_x = usable_w * (1 - 2 * pad) / range_x
    scale_y = usable_h * (1 - 2 * pad) / range_y
    scale = min(scale_x, scale_y) * 0.9
    cx = margin_left + usable_w / 2
    cy = margin_top + usable_h / 2

    result = []
    for p in points:
        nx = cx + (p["raw_x"] - (min_x + max_x) / 2) * scale
        ny = cy + (p["raw_y"] - (min_y + max_y) / 2) * scale
        # Add jitter to reveal density when points overlap
        nx += random.uniform(-jitter, jitter)
        ny += random.uniform(-jitter, jitter)
        result.append({**p, "x": nx, "y": ny})
    return result


def compute_cluster_outlines(urls_with_positions):
    """Compute ellipse bounds and label position for each topic cluster."""
    from collections import defaultdict
    by_topic = defaultdict(list)
    for p in urls_with_positions:
        by_topic[p["primary_topic"]].append((p["x"], p["y"]))

    outlines = []
    for topic, pts in by_topic.items():
        if len(pts) < 2:
            continue
        xs = [x for x, y in pts]
        ys = [y for x, y in pts]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        std_x = (sum((x - mean_x) ** 2 for x in xs) / len(xs)) ** 0.5
        std_y = (sum((y - mean_y) ** 2 for y in ys) / len(ys)) ** 0.5
        # Ellipse radius: 2 std + padding to encompass most points; cap for very spread clusters
        rx = min(max(std_x * 2.2 + 15, 25), 120)
        ry = min(max(std_y * 2.2 + 15, 25), 120)
        outlines.append({
            "topic": topic,
            "cx": mean_x,
            "cy": mean_y,
            "rx": rx,
            "ry": ry,
            "count": len(pts),
        })
    return outlines


def wrap_text(text, max_chars=28):
    """Wrap text at word boundaries, return list of lines."""
    words = text.split()
    lines = []
    current = []
    current_len = 0
    for w in words:
        need = len(w) + (1 if current else 0)
        if current_len + need <= max_chars:
            current.append(w)
            current_len += need
        else:
            if current:
                lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
    if current:
        lines.append(" ".join(current))
    return lines


def escape_html(s):
    """Escape HTML special characters for safe insertion."""
    if not s:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def build_chart_svg(urls_with_positions, topic_colors, topic_to_idx, width=700, height=500):
    """Build SVG string for one scatter chart (axes, cluster outlines, points)."""
    margin_left, margin_right = 70, 20
    margin_top, margin_bottom = 20, 55
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    # Axis lines and ticks
    x_axis_y = margin_top + chart_h
    y_axis_x = margin_left

    axis_style = 'stroke="#52525b" stroke-width="1"'
    tick_style = 'stroke="#52525b" stroke-width="1"'
    label_style = 'fill="#a1a1aa" font-size="11" font-family="system-ui,sans-serif"'

    axis_lines = []

    # X-axis line
    axis_lines.append(f'<line x1="{margin_left}" y1="{x_axis_y}" x2="{width - margin_right}" y2="{x_axis_y}" {axis_style}/>')
    # Y-axis line
    axis_lines.append(f'<line x1="{y_axis_x}" y1="{margin_top}" x2="{y_axis_x}" y2="{x_axis_y}" {axis_style}/>')

    # X-axis ticks (5 evenly spaced)
    for i, val in enumerate([-1, -0.5, 0, 0.5, 1]):
        x = margin_left + (i / 4) * chart_w
        axis_lines.append(f'<line x1="{x}" y1="{x_axis_y}" x2="{x}" y2="{x_axis_y + 5}" {tick_style}/>')
        axis_lines.append(f'<text x="{x}" y="{x_axis_y + 18}" text-anchor="middle" {label_style}>{val}</text>')

    # Y-axis ticks (5 evenly spaced, 1 at top)
    for i, val in enumerate([-1, -0.5, 0, 0.5, 1]):
        y = margin_top + (1 - i / 4) * chart_h  # -1 at bottom, 1 at top
        axis_lines.append(f'<line x1="{y_axis_x}" y1="{y}" x2="{y_axis_x - 5}" y2="{y}" {tick_style}/>')
        axis_lines.append(f'<text x="{y_axis_x - 8}" y="{y + 4}" text-anchor="end" {label_style}>{val}</text>')

    # Axis titles (polar projection: abstract dimensions; clusters = topic density)
    title_style = 'fill="#e4e4e7" font-size="12" font-weight="500" font-family="system-ui,sans-serif"'
    x_title_x = margin_left + chart_w / 2
    x_title_y = height - 10
    axis_lines.append(f'<text x="{x_title_x}" y="{x_title_y}" text-anchor="middle" {title_style}>Semantic dimension 1</text>')
    y_title_x = 22
    y_title_y = margin_top + chart_h / 2
    axis_lines.append(f'<text x="{y_title_x}" y="{y_title_y}" text-anchor="middle" transform="rotate(-90, {y_title_x}, {y_title_y})" {title_style}>Semantic dimension 2</text>')

    # Cluster outlines and labels (drawn before points so points appear on top)
    cluster_parts = []
    outlines = compute_cluster_outlines(urls_with_positions)
    for o in outlines:
        c = topic_colors.get(o["topic"], "#71717a")
        tidx = topic_to_idx.get(o["topic"], 0)
        cluster_parts.append(f'<g class="topic-cluster" data-topic-idx="{tidx}">')
        cluster_parts.append(
            f'<ellipse cx="{o["cx"]:.1f}" cy="{o["cy"]:.1f}" rx="{o["rx"]:.1f}" ry="{o["ry"]:.1f}" '
            f'fill="{c}" fill-opacity="0.08" stroke="{c}" stroke-width="2" stroke-opacity="0.6"/>'
        )
        # Label: topic name (wrapped) + count on last line, positioned above centroid
        lines = wrap_text(o["topic"], max_chars=28)
        count_suffix = f" ({o['count']})"
        if lines:
            lines[-1] += count_suffix
        else:
            lines = [str(o["count"])]
        tspan_parts = []
        for i, line in enumerate(lines):
            dy = "0" if i == 0 else "1.1em"
            tspan_parts.append(f'<tspan x="{o["cx"]:.1f}" dy="{dy}" text-anchor="middle">{escape_html(line)}</tspan>')
        cluster_parts.append(
            f'<text x="{o["cx"]:.1f}" y="{o["cy"] - o["ry"] - 5:.1f}" '
            f'fill="{c}" font-size="11" font-weight="600" font-family="system-ui,sans-serif">'
            + "".join(tspan_parts) + "</text>"
        )
        cluster_parts.append("</g>")

    # Generate scatter SVG - each point is a small circle
    static_svg_parts = []
    for i, p in enumerate(urls_with_positions):
        c = topic_colors.get(p["primary_topic"], "#71717a")
        tidx = p.get("topic_index", 0)
        static_svg_parts.append(
            f'<circle class="point" cx="{p["x"]:.1f}" cy="{p["y"]:.1f}" r="3" '
            f'fill="{c}" stroke="#27272a" stroke-width="1" data-index="{i}" data-topic-idx="{tidx}"/>'
        )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(axis_lines)
        + "".join(cluster_parts)
        + "".join(static_svg_parts)
        + "</svg>"
    )


def build_topic_counts_table(urls_for_data, topic_colors):
    """HTML table: each primary topic and number of URLs (sorted by count descending)."""
    counts = Counter(p["primary_topic"] for p in urls_for_data)
    rows = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    body_rows = []
    for topic, cnt in rows:
        c = topic_colors.get(topic, "#71717a")
        body_rows.append(
            "<tr>"
            f'<td><span class="topic-dot" style="background:{c}"></span>{escape_html(topic)}</td>'
            f'<td class="num">{cnt}</td>'
            "</tr>"
        )
    total = len(urls_for_data)
    return (
        '<section class="topic-table-section" aria-label="URLs per topic">'
        '<h2 class="topic-table-heading">URLs per primary topic</h2>'
        '<div class="topic-table-wrap">'
        '<table class="topic-table">'
        '<thead><tr><th scope="col">Topic</th><th scope="col" class="num">URLs</th></tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        f'<tfoot><tr><td>Total</td><td class="num">{total}</td></tr></tfoot>'
        "</table></div></section>"
    )


def build_topic_filters_html(topic_order, topic_colors):
    """Checkboxes to show/hide each topic in both charts."""
    rows = []
    for i, topic in enumerate(topic_order):
        c = topic_colors.get(topic, "#71717a")
        tid = f"topic-filter-{i}"
        rows.append(
            '<label class="topic-filter-item">'
            f'<input type="checkbox" class="topic-filter-cb" id="{tid}" data-topic-idx="{i}" checked />'
            f'<span class="topic-filter-dot" style="background:{c}"></span>'
            f'<span class="topic-filter-label">{escape_html(topic)}</span>'
            "</label>"
        )
    return (
        '<section class="topic-filters-section" aria-label="Topic visibility">'
        '<h2 class="topic-filters-heading">Show topics</h2>'
        '<p class="topic-filters-hint">Uncheck a topic to hide it in both charts. Hidden points are omitted from hover and clicks.</p>'
        '<div class="topic-filters-toolbar">'
        '<button type="button" class="topic-filters-btn" id="topic-show-all">Show all</button>'
        '<button type="button" class="topic-filters-btn" id="topic-hide-all">Hide all</button>'
        "</div>"
        f'<div class="topic-filters-list">{"".join(rows)}</div>'
        "</section>"
    )


def generate_html(urls_tsne, urls_umap, topic_colors, topic_order, method_tsne_label="t-SNE", page_title="Topic Cluster Scatter"):
    """Generate HTML with two side-by-side charts (t-SNE/Polar and UMAP)."""
    urls_for_data = urls_tsne if urls_tsne is not None else urls_umap
    if urls_for_data is None:
        raise ValueError("At least one chart must have data")

    topic_to_idx = {t: i for i, t in enumerate(topic_order)}

    points_data = [
        {
            "url": escape_html(p["url"]),
            "page_title": escape_html(p["page_title"]),
            "primary_topic": p["primary_topic"],
            "primary_sim": p["primary_sim"],
            "topic_index": p["topic_index"],
        }
        for p in urls_for_data
    ]
    points_json = json.dumps(points_data, indent=2)
    topic_order_json = json.dumps(topic_order)

    chart_w, chart_h = 700, 500
    svg_tsne = build_chart_svg(urls_tsne, topic_colors, topic_to_idx, chart_w, chart_h) if urls_tsne else ""
    svg_umap = build_chart_svg(urls_umap, topic_colors, topic_to_idx, chart_w, chart_h) if urls_umap else ""

    umap_placeholder = (
        '<div class="chart-placeholder">UMAP diagram requires umap-learn. Run: pip install umap-learn</div>'
        if urls_umap is None else ""
    )

    legend_parts = "".join(
        f'<span class="legend-item"><span class="legend-dot" style="background:{c}"></span>{escape_html(t)}</span>'
        for t, c in sorted(topic_colors.items(), key=lambda x: x[0])
    )

    topic_table_html = build_topic_counts_table(urls_for_data, topic_colors)
    topic_filters_html = build_topic_filters_html(topic_order, topic_colors)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape_html(page_title)}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f0f12; color: #e4e4e7; min-height: 100vh; padding: 2rem; }}
    h1 {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem; }}
    .subtitle {{ color: #71717a; font-size: 0.875rem; margin-bottom: 2rem; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem; font-size: 0.75rem; color: #a1a1aa; }}
    .legend-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.35rem; vertical-align: middle; }}
    .charts-container {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
    .chart-wrapper {{ min-width: 700px; flex: 1; }}
    .chart-title {{ font-size: 0.875rem; font-weight: 600; margin-bottom: 0.5rem; color: #a1a1aa; }}
    .chart-placeholder {{ min-width: 700px; height: 500px; display: flex; align-items: center; justify-content: center; background: #18181b; border-radius: 8px; color: #71717a; font-size: 0.875rem; padding: 2rem; text-align: center; }}
    .chart-wrapper svg {{ width: 100%; height: 500px; }}
    .point {{ cursor: pointer; transition: opacity 0.2s; }}
    .point:hover {{ opacity: 0.9; }}
    #tooltip {{ position: fixed; z-index: 200; background: #27272a; color: #e4e4e7; padding: 0.5rem 0.75rem; border-radius: 6px; font-size: 0.8rem; max-width: 320px; pointer-events: none; opacity: 0; transition: opacity 0.15s; border: 1px solid #3f3f46; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
    #tooltip.visible {{ opacity: 1; }}
    #tooltip .tt-title {{ font-weight: 500; margin-bottom: 0.25rem; word-break: break-word; }}
    #tooltip .tt-meta {{ color: #a1a1aa; font-size: 0.75rem; }}
    #url-panel {{ position: fixed; right: 0; top: 0; width: 380px; max-width: 100%; height: 100%; background: #18181b; border-left: 1px solid #27272a; overflow-y: auto; padding: 1.5rem; transform: translateX(100%); transition: transform 0.25s; z-index: 100; }}
    #url-panel.open {{ transform: translateX(0); }}
    #url-panel h2 {{ font-size: 1rem; margin-bottom: 1rem; color: #a1a1aa; }}
    #url-panel .close {{ position: absolute; top: 1rem; right: 1rem; background: none; border: none; color: #71717a; cursor: pointer; font-size: 1.25rem; }}
    #url-panel .close:hover {{ color: #fff; }}
    .url-item {{ padding: 0.5rem 0; border-bottom: 1px solid #27272a; font-size: 0.8rem; }}
    .url-item a {{ color: #60a5fa; text-decoration: none; word-break: break-all; }}
    .url-item a:hover {{ text-decoration: underline; }}
    .url-item .title {{ color: #e4e4e7; margin-top: 0.25rem; }}
    .url-item .sim {{ color: #71717a; font-size: 0.7rem; }}
    .overlay {{ position: fixed; inset: 0; background: rgba(0,0,0,0.5); opacity: 0; pointer-events: none; transition: opacity 0.25s; z-index: 99; }}
    .overlay.open {{ opacity: 1; pointer-events: auto; }}
    .topic-table-section {{ margin-top: 2.5rem; max-width: 960px; }}
    .topic-table-heading {{ font-size: 1rem; font-weight: 600; margin-bottom: 0.75rem; color: #a1a1aa; }}
    .topic-table-wrap {{ overflow-x: auto; border: 1px solid #27272a; border-radius: 8px; background: #18181b; }}
    .topic-table {{ width: 100%; border-collapse: collapse; font-size: 0.8125rem; }}
    .topic-table th, .topic-table td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #27272a; vertical-align: middle; }}
    .topic-table thead th {{ color: #a1a1aa; font-weight: 600; }}
    .topic-table tbody tr:last-child td {{ border-bottom: none; }}
    .topic-table td.num, .topic-table th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .topic-table tfoot td {{ border-top: 1px solid #3f3f46; background: #141416; font-weight: 500; color: #e4e4e7; }}
    .topic-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.5rem; vertical-align: middle; flex-shrink: 0; }}
    .topic-filters-section {{ margin: 1.5rem 0 1rem; max-width: 960px; }}
    .topic-filters-heading {{ font-size: 1rem; font-weight: 600; margin-bottom: 0.35rem; color: #a1a1aa; }}
    .topic-filters-hint {{ font-size: 0.8125rem; color: #71717a; margin-bottom: 0.75rem; line-height: 1.4; }}
    .topic-filters-toolbar {{ display: flex; gap: 0.5rem; margin-bottom: 0.75rem; }}
    .topic-filters-btn {{ background: #27272a; border: 1px solid #3f3f46; color: #e4e4e7; padding: 0.35rem 0.75rem; border-radius: 6px; font-size: 0.8125rem; cursor: pointer; }}
    .topic-filters-btn:hover {{ background: #3f3f46; }}
    .topic-filters-list {{ display: flex; flex-direction: column; gap: 0.35rem; max-height: 280px; overflow-y: auto; padding: 0.5rem; border: 1px solid #27272a; border-radius: 8px; background: #18181b; }}
    .topic-filter-item {{ display: flex; align-items: flex-start; gap: 0.5rem; font-size: 0.8125rem; color: #d4d4d8; cursor: pointer; line-height: 1.35; }}
    .topic-filter-item input {{ margin-top: 0.2rem; accent-color: #60a5fa; cursor: pointer; }}
    .topic-filter-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; margin-top: 0.35rem; }}
    .topic-filter-label {{ flex: 1; min-width: 0; }}
    .point.topic-hidden, .topic-cluster.topic-hidden {{ opacity: 0; pointer-events: none; }}
  </style>
</head>
<body>
  <h1>{escape_html(page_title)}</h1>
  <p class="subtitle">URLs clustered by topic similarity (Sourcewell). Compare t-SNE/Polar vs UMAP. Color = primary topic. Click a point for details.</p>
  <div class="legend">{legend_parts}</div>
  {topic_filters_html}
  <div class="charts-container">
    <div class="chart-wrapper">
      <div class="chart-title">{method_tsne_label}</div>
      <div id="chart-tsne">{svg_tsne}</div>
    </div>
    <div class="chart-wrapper">
      <div class="chart-title">UMAP</div>
      <div id="chart-umap">{svg_umap if urls_umap else umap_placeholder}</div>
    </div>
  </div>
  {topic_table_html}
  <div id="tooltip"></div>
  <div class="overlay" id="overlay"></div>
  <div id="url-panel">
    <button class="close" id="close-panel" aria-label="Close">&times;</button>
    <h2 id="panel-title">URLs</h2>
    <div id="url-list"></div>
  </div>

  <script>
    const pointsData = {points_json};
    const topicOrder = {topic_order_json};

    function initChart() {{
      const tooltip = document.getElementById("tooltip");
      const numTopics = topicOrder.length;
      const topicVisible = topicOrder.map(() => true);

      function applyTopicVisibility() {{
        document.querySelectorAll("#chart-tsne [data-topic-idx], #chart-umap [data-topic-idx]").forEach((el) => {{
          const idx = parseInt(el.getAttribute("data-topic-idx"), 10);
          if (Number.isNaN(idx) || idx < 0 || idx >= numTopics) return;
          el.classList.toggle("topic-hidden", !topicVisible[idx]);
        }});
      }}

      document.querySelectorAll(".topic-filter-cb").forEach((cb) => {{
        cb.addEventListener("change", () => {{
          const idx = parseInt(cb.getAttribute("data-topic-idx"), 10);
          if (!Number.isNaN(idx) && idx >= 0 && idx < numTopics) {{
            topicVisible[idx] = cb.checked;
            applyTopicVisibility();
          }}
        }});
      }});
      const showAll = document.getElementById("topic-show-all");
      const hideAll = document.getElementById("topic-hide-all");
      if (showAll) {{
        showAll.addEventListener("click", () => {{
          topicVisible.fill(true);
          document.querySelectorAll(".topic-filter-cb").forEach((cb) => {{ cb.checked = true; }});
          applyTopicVisibility();
        }});
      }}
      if (hideAll) {{
        hideAll.addEventListener("click", () => {{
          topicVisible.fill(false);
          document.querySelectorAll(".topic-filter-cb").forEach((cb) => {{ cb.checked = false; }});
          applyTopicVisibility();
        }});
      }}

      document.querySelectorAll("#chart-tsne .point, #chart-umap .point").forEach((el) => {{
        el.addEventListener("mouseenter", (e) => {{
          if (el.classList.contains("topic-hidden")) return;
          const i = parseInt(el.getAttribute("data-index"), 10);
          const p = pointsData[i];
          if (!p) return;
          const urlShort = p.url.replace(/^https?:\\/\\//, "").replace(/\\/$/, "").substring(0, 50);
          const title = p.page_title && p.page_title !== p.url ? p.page_title.substring(0, 60) : urlShort;
          tooltip.innerHTML = `<div class="tt-title">${{title}}${{(p.page_title || p.url).length > 60 ? "…" : ""}}</div><div class="tt-meta">${{p.primary_topic}} · similarity ${{p.primary_sim}}</div>`;
          tooltip.style.left = (e.clientX + 12) + "px";
          tooltip.style.top = (e.clientY + 12) + "px";
          tooltip.classList.add("visible");
        }});
        el.addEventListener("mousemove", (e) => {{
          if (tooltip.classList.contains("visible")) {{
            tooltip.style.left = (e.clientX + 12) + "px";
            tooltip.style.top = (e.clientY + 12) + "px";
          }}
        }});
        el.addEventListener("mouseleave", () => {{
          tooltip.classList.remove("visible");
        }});
        el.addEventListener("click", () => {{
          if (el.classList.contains("topic-hidden")) return;
          const i = parseInt(el.getAttribute("data-index"), 10);
          const p = pointsData[i];
          if (!p) return;
          document.getElementById("panel-title").textContent = p.primary_topic;
          const list = document.getElementById("url-list");
          list.innerHTML = `
            <div class="url-item">
              <a href="${{p.url}}" target="_blank" rel="noopener">${{p.url.replace(/^https?:\\/\\//, "")}}</a>
              ${{p.page_title ? `<div class="title">${{p.page_title.substring(0, 80)}}${{p.page_title.length > 80 ? "…" : ""}}</div>` : ""}}
              <div class="sim">${{p.primary_topic}} (similarity: ${{p.primary_sim}})</div>
            </div>
          `;
          document.getElementById("url-panel").classList.add("open");
          document.getElementById("overlay").classList.add("open");
        }});
      }});

      document.getElementById("close-panel").onclick = () => {{
        document.getElementById("url-panel").classList.remove("open");
        document.getElementById("overlay").classList.remove("open");
      }};
      document.getElementById("overlay").onclick = () => {{
        document.getElementById("url-panel").classList.remove("open");
        document.getElementById("overlay").classList.remove("open");
      }};
    }}

    if (document.readyState === "loading") {{
      document.addEventListener("DOMContentLoaded", initChart);
    }} else {{
      initChart();
    }}
  </script>
</body>
</html>
"""
    return html


def main():
    urls, topic_colors, topic_order = parse_csv(CSV_PATH)
    if not urls:
        print("No URLs with topic_similarities found; check CSV path and format.")
        return
    urls_tsne = compute_scatter_positions(urls, method="tsne", width=700, height=500)
    urls_umap = compute_scatter_positions(urls, method="umap", width=700, height=500)
    try:
        from sklearn.manifold import TSNE  # noqa: F401
        method_tsne_label = "t-SNE"
    except ImportError:
        method_tsne_label = "Polar"
    page_title = "Topic Cluster Scatter — Sourcewell"
    html = generate_html(urls_tsne, urls_umap, topic_colors, topic_order, method_tsne_label, page_title=page_title)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated {OUTPUT_PATH}")
    print(f"URLs: {len(urls_tsne or urls_umap)}")
    if urls_umap is None:
        print("UMAP skipped (install umap-learn for comparison)")


if __name__ == "__main__":
    main()
