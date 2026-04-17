#!/usr/bin/env python3
"""
compare_plots.py — Compare measurements across multiple phantomkit HTML output files.

Reads the embedded phantomkit-data JSON from each HTML, extracts per-vial
measurements (ADC, fitted T1, or fitted T2), and produces a new self-contained
interactive HTML showing all sessions on a single scatter plot.

Features
--------
* Each input file gets a unique colour (filled circles).
* Reference values shown as red open circles (loaded from template_data/ if available).
* Toggle buttons show/hide individual sessions.
* Checkboxes select which sessions contribute to a live group mean (filled squares).

Registered as ``phantomkit plot compare-plots`` via CLI auto-discovery.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click


# ---------------------------------------------------------------------------
# Colour palette (session colours — red #C62828 is reserved for reference)
# ---------------------------------------------------------------------------

_SESSION_PALETTE = [
    "#378ADD",  # blue
    "#1D9E75",  # teal-green
    "#D4537E",  # rose
    "#7F77DD",  # violet
    "#BA7517",  # amber
    "#639922",  # lime
    "#185FA5",  # navy
    "#D85A30",  # burnt orange
    "#993C1D",  # dark sienna
    "#888780",  # stone grey
]

_REFERENCE_COLOR = "#C62828"


# ---------------------------------------------------------------------------
# Embedded data extraction
# ---------------------------------------------------------------------------


def _load_embedded(html_path: str) -> dict:
    """Extract the phantomkit-data JSON block from a phantomkit HTML file."""
    text = Path(html_path).read_text(encoding="utf-8")
    m = re.search(
        r'<script\s+id="phantomkit-data"\s+type="application/json">\s*(\{.*?\})\s*</script>',
        text,
        re.DOTALL,
    )
    if not m:
        raise click.ClickException(
            f"No phantomkit-data block found in {html_path!r}. "
            "Is this a phantomkit HTML output file?"
        )
    return json.loads(m.group(1))


def _extract(data: dict) -> tuple[str, list[str], dict[str, float | None]]:
    """Return (metric_key, vials_in_order, {vial_upper: value_or_None}).

    metric_key is one of: "ADC", "FA", "Intensity", "T1", "T2".
    """
    dtype = data.get("type", "")

    if dtype == "vial_intensity":
        vials = data.get("vials", [])
        means = data.get("means", [])
        mode = data.get("contrast_mode", "generic")
        metric = {"adc": "ADC", "fa": "FA"}.get(mode, "Intensity")

        def _scalar(m):
            """Return a single float from a value that may be a list (multi-volume)."""
            if isinstance(m, list):
                valid = [x for x in m if x is not None]
                return sum(valid) / len(valid) if valid else None
            return float(m) if m is not None else None

        vals = {v.upper(): _scalar(m) for v, m in zip(vials, means)}
        return metric, vials, vals

    if dtype == "maps_ir":
        fit_results = data.get("fit_results", [])
        vials_order, vals = [], {}
        for r in fit_results:
            v = r.get("Vial", "")
            t1 = r.get("T1_ms")
            vals[v.upper()] = float(t1) if t1 is not None else None
            vials_order.append(v)
        return "T1", vials_order, vals

    if dtype == "maps_te":
        fit_results = data.get("fit_results", [])
        vials_order, vals = [], {}
        for r in fit_results:
            v = r.get("Vial", "")
            t2 = r.get("T2_ms")
            vals[v.upper()] = float(t2) if t2 is not None else None
            vials_order.append(v)
        return "T2", vials_order, vals

    raise click.ClickException(
        f"Unknown embedded data type {dtype!r}. "
        "Expected: vial_intensity, maps_ir, or maps_te."
    )


# ---------------------------------------------------------------------------
# Reference data loading
# ---------------------------------------------------------------------------


def _auto_template_dir() -> str | None:
    """Locate template_data/ from the installed phantomkit package."""
    import importlib.util

    spec = importlib.util.find_spec("phantomkit")
    if not spec:
        return None
    pkg_dir = Path(spec.origin).parent
    for candidate in [pkg_dir.parent / "template_data", pkg_dir / "template_data"]:
        if candidate.is_dir():
            return str(candidate)
    return None


def _load_reference(
    metric: str, template_dir: str, phantom: str
) -> tuple[list[str], dict[str, float]]:
    """Load reference values for *metric* from template_data.

    Returns (vials_ordered, {vial_upper: value}).
    ADC values are scaled to ×10⁻³ mm²/s to match the display units stored in
    the HTML.
    """
    ref_dir = Path(template_dir) / phantom

    if metric == "ADC":
        ref_file = ref_dir / "adc_reference.json"
        if not ref_file.exists():
            return [], {}
        with open(ref_file) as fh:
            raw = json.load(fh)
        vials = raw.get("vials", [])
        vals_raw = raw.get("adc_mm2_per_s", {})
        vals = {v.upper(): float(vals_raw[v]) * 1e3 for v in vials if v in vals_raw}
        return vials, vals

    if metric in ("T1", "T2"):
        ref_file = ref_dir / "t1t2_reference.json"
        if not ref_file.exists():
            return [], {}
        with open(ref_file) as fh:
            raw = json.load(fh)
        vials = raw.get("vials", [])
        key = "T1_ms" if metric == "T1" else "T2_ms"
        vals_raw = raw.get(key, {})
        vals = {v.upper(): float(vals_raw[v]) for v in vials if v in vals_raw}
        return vials, vals

    return [], {}


# ---------------------------------------------------------------------------
# X-axis vial ordering
# ---------------------------------------------------------------------------


def _vial_axis(sessions: list[dict], ref_vials: list[str]) -> list[str]:
    """Return an ordered vial list for the x-axis.

    Reference vials come first (preserving their prescribed order), then any
    measured vials that are absent from the reference.
    """
    if ref_vials:
        seen = {v.upper() for v in ref_vials}
        result = list(ref_vials)
    else:
        seen: set = set()
        result: list = []

    for sess in sessions:
        for v in sess["vials_order"]:
            if v.upper() not in seen:
                result.append(v)
                seen.add(v.upper())

    return result


# ---------------------------------------------------------------------------
# Y-axis labels
# ---------------------------------------------------------------------------

_Y_LABELS: dict[str, str] = {
    "ADC": "ADC \u00d710\u207b\u00b3 mm\u00b2/s",
    "FA": "Fractional Anisotropy",
    "T1": "T\u2081 (ms)",
    "T2": "T\u2082 (ms)",
    "Intensity": "Intensity",
}


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def _build_html(
    metric: str,
    vial_axis: list[str],
    sessions: list[dict],       # [{label, color, vials_order, vals}]
    ref_vals: dict[str, float], # {vial_upper: value}
) -> str:
    from phantomkit.plotting._html_common import html_head

    y_label = _Y_LABELS.get(metric, metric)
    title = f"Comparison: {metric} per vial"

    vial_upper_to_x = {v.upper(): i for i, v in enumerate(vial_axis)}

    # -- Per-session point data
    session_datasets = []
    for sess in sessions:
        pts = [
            {"x": vial_upper_to_x[v_up], "y": round(val, 4)}
            for v_up, val in sess["vals"].items()
            if val is not None and v_up in vial_upper_to_x
        ]
        pts.sort(key=lambda p: p["x"])
        session_datasets.append({
            "label": sess["label"],
            "color": sess["color"],
            "data": pts,
        })

    # -- Reference point data
    ref_pts = [
        {"x": vial_upper_to_x[v.upper()], "y": round(ref_vals[v.upper()], 4)}
        for v in vial_axis
        if ref_vals.get(v.upper()) is not None
    ]
    has_ref = bool(ref_pts)

    sessions_json   = json.dumps(session_datasets)
    ref_pts_json    = json.dumps(ref_pts)
    vial_labels_json = json.dumps(vial_axis)
    y_label_json    = json.dumps(y_label)
    ref_color_json  = json.dumps(_REFERENCE_COLOR)

    # -- Session control rows (toggle button + include-in-mean checkbox)
    session_controls_html = ""
    for i, sess in enumerate(sessions):
        c = sess["color"]
        lbl = sess["label"]
        session_controls_html += (
            f'\n    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{c};'
            f'display:inline-block;flex-shrink:0;"></span>'
            f'<button id="pk-sess-btn-{i}" data-visible="1" onclick="pkToggleSession({i},this)"'
            f' style="padding:4px 14px;border-radius:99px;border:1.5px solid var(--border);'
            f'background:var(--bg3);color:var(--text);font-size:12px;font-weight:500;'
            f'cursor:pointer;user-select:none;transition:opacity .15s;">{lbl}</button>'
            f'<label style="display:flex;align-items:center;gap:5px;font-size:12px;'
            f'color:var(--text2);cursor:pointer;">'
            f'<input type="checkbox" id="pk-incl-{i}" checked onchange="pkUpdateMean()">'
            f'include in mean</label>'
            f'</div>'
        )

    # -- Reference legend entry
    ref_legend_html = ""
    if has_ref:
        ref_legend_html = (
            f'\n    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:transparent;'
            f'border:2px solid {_REFERENCE_COLOR};display:inline-block;flex-shrink:0;"></span>'
            f'<span style="font-size:12px;color:var(--text2);">Reference</span></div>'
        )

    mean_legend_html = (
        '\n    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
        '<span id="pk-mean-swatch" style="width:12px;height:12px;'
        'display:inline-block;flex-shrink:0;"></span>'
        '<span style="font-size:12px;color:var(--text2);">Group mean (&#x25a0;)</span></div>'
    )

    head = html_head(title)

    return f"""{head}
<body>
<div class="page-wrap" style="max-width:960px;margin:0 auto;">
  <h1>{title}</h1>
  <p class="subtitle">{y_label}</p>

  <div class="chart-card" style="margin-bottom:20px;">
    <p class="chart-title">Sessions</p>
    <div id="pk-session-controls">
{session_controls_html}
{ref_legend_html}
{mean_legend_html}
    </div>
  </div>

  <div class="chart-card">
    <p class="chart-title">{metric} per vial</p>
    <div class="chart-wrap" style="height:440px;">
      <canvas id="compChart"></canvas>
    </div>
    <p style="font-size:11px;color:var(--text2);margin-top:8px;">
      Click session buttons to toggle visibility &middot;
      check/uncheck &ldquo;include in mean&rdquo; to update the group mean (&squ;)
    </p>
  </div>
</div>

<script>
const SESSIONS    = {sessions_json};
const REF_PTS     = {ref_pts_json};
const VIAL_LABELS = {vial_labels_json};
const REF_COLOR   = {ref_color_json};

const isDark   = window.matchMedia("(prefers-color-scheme: dark)").matches;
const gridCol  = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.07)";
const tickCol  = "#888780";
const MEAN_COLOR = isDark ? "#e0e0e0" : "#222222";

// Update mean-swatch colour to match
document.getElementById("pk-mean-swatch").style.background = MEAN_COLOR;

// ---------------------------------------------------------------------------
// Build Chart.js datasets
// ---------------------------------------------------------------------------
const datasets = [];

SESSIONS.forEach(function(sess) {{
  datasets.push({{
    label: sess.label,
    data: sess.data,
    backgroundColor: sess.color,
    borderColor: sess.color,
    pointBackgroundColor: sess.color,
    pointBorderColor: sess.color,
    pointRadius: 6,
    pointHoverRadius: 8,
    pointStyle: "circle",
    showLine: false,
    borderWidth: 0,
  }});
}});

// Reference — red open circles
datasets.push({{
  label: "Reference",
  data: REF_PTS,
  backgroundColor: "transparent",
  borderColor: "transparent",
  pointBackgroundColor: "transparent",
  pointBorderColor: REF_COLOR,
  pointBorderWidth: 2,
  pointRadius: 8,
  pointHoverRadius: 10,
  pointStyle: "circle",
  showLine: false,
  borderWidth: 0,
}});

// Group mean — filled squares (index = datasets.length before push)
const MEAN_IDX = datasets.length;
datasets.push({{
  label: "Group Mean",
  data: [],
  backgroundColor: MEAN_COLOR,
  borderColor: MEAN_COLOR,
  pointBackgroundColor: MEAN_COLOR,
  pointBorderColor: MEAN_COLOR,
  pointRadius: 8,
  pointHoverRadius: 10,
  pointStyle: "rect",
  showLine: false,
  borderWidth: 0,
}});

const chart = new Chart(
  document.getElementById("compChart").getContext("2d"),
  {{
    type: "scatter",
    data: {{ datasets: datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              const vialName = VIAL_LABELS[ctx.parsed.x] !== undefined
                ? VIAL_LABELS[ctx.parsed.x] : ctx.parsed.x;
              return ctx.dataset.label + " \u2014 " + vialName + ": " + ctx.parsed.y.toFixed(3);
            }}
          }}
        }},
        zoom: {{
          zoom: {{ wheel: {{ enabled: true }}, pinch: {{ enabled: true }}, mode: "xy" }},
          pan:  {{ enabled: true, mode: "xy" }},
        }}
      }},
      scales: {{
        x: {{
          type: "linear",
          title: {{ display: true, text: "Vial", color: tickCol, font: {{ size: 13 }} }},
          ticks: {{
            color: tickCol,
            font: {{ size: 12 }},
            stepSize: 1,
            callback: function(v) {{
              return VIAL_LABELS[v] !== undefined ? VIAL_LABELS[v] : v;
            }}
          }},
          grid: {{ color: gridCol }},
        }},
        y: {{
          title: {{ display: true, text: {y_label_json}, color: tickCol, font: {{ size: 13 }} }},
          ticks: {{ color: tickCol, font: {{ size: 12 }} }},
          grid: {{ color: gridCol }},
        }}
      }}
    }}
  }}
);

// Seed group mean from all sessions (all checked by default)
pkUpdateMean();

// ---------------------------------------------------------------------------
// Interactivity
// ---------------------------------------------------------------------------

function pkToggleSession(idx, btn) {{
  const ds = chart.data.datasets[idx];
  ds.hidden = !ds.hidden;
  btn.setAttribute("data-visible", ds.hidden ? "0" : "1");
  btn.style.opacity = ds.hidden ? "0.35" : "1.0";
  chart.update();
}}

function pkUpdateMean() {{
  const byX = {{}};
  SESSIONS.forEach(function(sess, i) {{
    const cb = document.getElementById("pk-incl-" + i);
    if (!cb || !cb.checked) return;
    sess.data.forEach(function(pt) {{
      if (byX[pt.x] === undefined) byX[pt.x] = [];
      byX[pt.x].push(pt.y);
    }});
  }});
  const meanPts = Object.keys(byX)
    .map(function(x) {{
      const ys = byX[x];
      return {{ x: Number(x), y: ys.reduce(function(a, b) {{ return a + b; }}, 0) / ys.length }};
    }})
    .sort(function(a, b) {{ return a.x - b.x; }});
  chart.data.datasets[MEAN_IDX].data = meanPts;
  chart.update();
}}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


@click.command("compare-plots")
@click.argument("html_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    "--output", "-o", required=True, type=click.Path(),
    help="Output HTML file path.",
)
@click.option(
    "--template-dir", default=None, metavar="DIR",
    help="Path to template_data/ directory (auto-detected if omitted).",
)
@click.option(
    "--phantom", default=None, metavar="NAME",
    help="Phantom name, e.g. SPIRIT (auto-detected from input files if omitted).",
)
@click.option(
    "--label", "labels", multiple=True, metavar="TEXT",
    help="Label for each input file (repeat once per file; defaults to filename stem).",
)
def main(
    html_files: tuple[str, ...],
    output: str,
    template_dir: str | None,
    phantom: str | None,
    labels: tuple[str, ...],
) -> None:
    """Compare measurements from multiple phantomkit HTML output files.

    Reads the embedded JSON from each HTML, overlays all sessions on one
    scatter plot, and provides a toggleable group-mean marker (filled square).

    \b
    Example:
        phantomkit plot compare-plots \\
            session01/metrics/plots/ADC.html \\
            session02/metrics/plots/ADC.html \\
            -o comparison_ADC.html
    """
    if labels and len(labels) != len(html_files):
        raise click.ClickException(
            f"--label provided {len(labels)} time(s) but "
            f"{len(html_files)} input file(s) given — counts must match."
        )

    # ---- Load and parse each HTML file
    sessions_raw: list[dict] = []
    metrics_seen: list[str] = []
    for path in html_files:
        data = _load_embedded(path)
        metric, vials_order, vals = _extract(data)
        metrics_seen.append(metric)
        sessions_raw.append({
            "path": path,
            "data": data,
            "metric": metric,
            "vials_order": vials_order,
            "vals": vals,
        })

    # ---- Validate that all inputs share the same metric
    unique_metrics = list(dict.fromkeys(metrics_seen))
    if len(unique_metrics) > 1:
        raise click.ClickException(
            f"Input files contain mixed metrics: {unique_metrics}. "
            "All files must be the same type (all ADC, all T1, or all T2)."
        )
    metric = unique_metrics[0]

    # ---- Determine session labels
    if labels:
        session_labels = list(labels)
    else:
        stems = [Path(p).stem for p in html_files]
        # If every stem is the same (e.g. all files named "ADC.html"),
        # walk up the path skipping standard phantomkit directory names
        # ('plots', 'metrics') to find the first meaningful ancestor.
        if len(set(stems)) == 1 and len(stems) > 1:
            _skip = {"plots", "metrics"}

            def _meaningful_ancestor(path: str) -> str:
                for part in reversed(Path(path).resolve().parts[:-1]):
                    if part.lower() not in _skip:
                        return part
                return Path(path).stem

            session_labels = [_meaningful_ancestor(p) for p in html_files]
        else:
            session_labels = stems

    # ---- Assign colours
    colors = [_SESSION_PALETTE[i % len(_SESSION_PALETTE)] for i in range(len(html_files))]

    sessions = [
        {
            "label": session_labels[i],
            "color": colors[i],
            "vials_order": sessions_raw[i]["vials_order"],
            "vals": sessions_raw[i]["vals"],
        }
        for i in range(len(html_files))
    ]

    # ---- Auto-detect phantom from embedded data (vial_intensity stores it)
    if phantom is None:
        for sr in sessions_raw:
            p = sr["data"].get("phantom")
            if p:
                phantom = p
                break

    # ---- Load reference values
    ref_vials: list[str] = []
    ref_vals: dict[str, float] = {}
    resolved_template_dir = template_dir or _auto_template_dir()
    if resolved_template_dir and phantom:
        ref_vials, ref_vals = _load_reference(metric, resolved_template_dir, phantom)
    elif not ref_vals:
        click.echo(
            "[WARN] Reference values not loaded — provide --template-dir and --phantom "
            "to overlay reference data.",
            err=True,
        )

    # ---- Build x-axis vial order
    vial_axis = _vial_axis(sessions, ref_vials)

    # ---- Generate and write HTML
    html = _build_html(metric, vial_axis, sessions, ref_vals)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    click.echo(f"[INFO] Comparison plot saved to: {out_path}")
