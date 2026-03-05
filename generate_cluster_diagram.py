#!/usr/bin/env python3
"""
Generate an interactive topic cluster scatter diagram from the Lumar CSV export.
Parses topic_similarities, positions URLs by semantic relevance, color-codes by primary topic.
"""

import csv
import json
import math
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "full-lumar-export-with-clusters.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "cluster_diagram.html")

# Topic order for 5-dim vector (must match slot_no or topic names)
TOPIC_ORDER = [
    "Technical SEO at Scale",
    "GEO / AEO (Generative / Answer Engine Optimization)",
    "Website Accessibility at Scale",
    "Site Speed at Scale",
    "Content Optimization at Scale",
]
TOPIC_COLORS = {
    "Technical SEO at Scale": "#3b82f6",
    "GEO / AEO (Generative / Answer Engine Optimization)": "#8b5cf6",
    "Website Accessibility at Scale": "#06b6d4",
    "Site Speed at Scale": "#10b981",
    "Content Optimization at Scale": "#f59e0b",
}


def parse_csv():
    """Parse CSV and return list of {url, page_title, topic_vector, primary_topic, primary_sim}."""
    topic_idx = {t: i for i, t in enumerate(TOPIC_ORDER)}
    urls = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
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
                vec = [0.0] * 5
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
                    })
            except (json.JSONDecodeError, KeyError):
                pass
    return urls


def _embed_vectors(vectors, method, seed):
    """Return list of (x,y) from 5-dim vectors. method: 'tsne'|'umap'|'polar'."""
    if method == "tsne":
        try:
            import numpy as np
            from sklearn.manifold import TSNE
            X = np.array(vectors, dtype=np.float64)
            embedded = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(vectors) - 1)).fit_transform(X)
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
    n_topics = 5
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


def build_chart_svg(urls_with_positions, width=700, height=500):
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
        c = TOPIC_COLORS.get(o["topic"], "#71717a")
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

    # Generate scatter SVG - each point is a small circle
    static_svg_parts = []
    for i, p in enumerate(urls_with_positions):
        c = TOPIC_COLORS.get(p["primary_topic"], "#71717a")
        static_svg_parts.append(
            f'<circle class="point" cx="{p["x"]:.1f}" cy="{p["y"]:.1f}" r="3" '
            f'fill="{c}" stroke="#27272a" stroke-width="1" data-index="{i}"/>'
        )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        + "".join(axis_lines)
        + "".join(cluster_parts)
        + "".join(static_svg_parts)
        + "</svg>"
    )


def generate_html(urls_tsne, urls_umap, method_tsne_label="t-SNE"):
    """Generate HTML with two side-by-side charts (t-SNE/Polar and UMAP)."""
    urls_for_data = urls_tsne if urls_tsne is not None else urls_umap
    if urls_for_data is None:
        raise ValueError("At least one chart must have data")

    points_data = [
        {
            "url": escape_html(p["url"]),
            "page_title": escape_html(p["page_title"]),
            "primary_topic": p["primary_topic"],
            "primary_sim": p["primary_sim"],
        }
        for p in urls_for_data
    ]
    points_json = json.dumps(points_data, indent=2)

    chart_w, chart_h = 700, 500
    svg_tsne = build_chart_svg(urls_tsne, chart_w, chart_h) if urls_tsne else ""
    svg_umap = build_chart_svg(urls_umap, chart_w, chart_h) if urls_umap else ""

    umap_placeholder = (
        '<div class="chart-placeholder">UMAP diagram requires umap-learn. Run: pip install umap-learn</div>'
        if urls_umap is None else ""
    )

    legend_parts = "".join(
        f'<span class="legend-item"><span class="legend-dot" style="background:{c}"></span>{escape_html(t)}</span>'
        for t, c in TOPIC_COLORS.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Topic Cluster Scatter - Lumar</title>
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
  </style>
</head>
<body>
  <h1>Topic Cluster Scatter</h1>
  <p class="subtitle">URLs clustered by topic similarity. Compare t-SNE/Polar vs UMAP. Color = primary topic. Click a point for details.</p>
  <div class="legend">{legend_parts}</div>
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
  <div id="tooltip"></div>
  <div class="overlay" id="overlay"></div>
  <div id="url-panel">
    <button class="close" id="close-panel" aria-label="Close">&times;</button>
    <h2 id="panel-title">URLs</h2>
    <div id="url-list"></div>
  </div>

  <script>
    const pointsData = {points_json};

    function initChart() {{
      const tooltip = document.getElementById("tooltip");

      document.querySelectorAll("#chart-tsne .point, #chart-umap .point").forEach((el) => {{
        el.addEventListener("mouseenter", (e) => {{
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
    urls = parse_csv()
    urls_tsne = compute_scatter_positions(urls, method="tsne", width=700, height=500)
    urls_umap = compute_scatter_positions(urls, method="umap", width=700, height=500)
    try:
        from sklearn.manifold import TSNE
        method_tsne_label = "t-SNE"
    except ImportError:
        method_tsne_label = "Polar"
    html = generate_html(urls_tsne, urls_umap, method_tsne_label)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated {OUTPUT_PATH}")
    print(f"URLs: {len(urls_tsne or urls_umap)}")
    if urls_umap is None:
        print("UMAP skipped (install umap-learn for comparison)")


if __name__ == "__main__":
    main()
