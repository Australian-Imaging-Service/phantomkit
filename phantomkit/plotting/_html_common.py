"""
Shared HTML generation utilities for phantomkit interactive plots.

Provides CDN constants, CSS, JavaScript plugins, and data embedding
utilities used by vial_intensity, maps_ir, and maps_te.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# CDN constants (pinned versions)
# ---------------------------------------------------------------------------

CHARTJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"
HAMMERJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"
ZOOM_CDN = (
    "https://cdnjs.cloudflare.com/ajax/libs/"
    "chartjs-plugin-zoom/1.2.1/chartjs-plugin-zoom.min.js"
)

_NIIVUE_CDN = (
    "https://cdn.jsdelivr.net/npm/@niivue/niivue@0.46.0/dist/niivue.umd.min.js"
)
_NIIVUE_BUNDLE = Path(__file__).parent / "niivue.umd.min.js"


def _niivue_script_tag() -> str:
    """Return a <script> tag that loads NiiVue — inlined bundle or CDN fallback."""
    if _NIIVUE_BUNDLE.exists():
        src = _NIIVUE_BUNDLE.read_text(encoding="utf-8")
        src = re.sub(r"(?i)</script>", r"<\/script>", src)
        return f"<script>/* niivue@0.46.0 inlined */\n{src}\n</script>"
    return f'<script src="{_NIIVUE_CDN}"></script>'


def nifti_to_base64(path: str) -> str:
    """Base64-encode a NIfTI file for embedding in an HTML page.

    If the file stores float64 voxels it is re-packed as float32 in-memory
    before encoding — halving the payload with no perceptible visual change.
    The result is always a valid ``.nii.gz`` byte stream.
    """
    import gzip
    import io
    import struct

    import numpy as np  # type: ignore[import]

    raw = Path(path).read_bytes()

    # Decompress if gzipped so we can inspect/rewrite the header
    try:
        data = gzip.decompress(raw)
    except OSError:
        data = raw  # already uncompressed .nii

    # NIfTI-1 datatype field is a little-endian int16 at byte offset 70
    (dtype_code,) = struct.unpack_from("<h", data, 70)

    if dtype_code == 64:  # float64 → repack as float32 (code 16)
        # Read vox_offset (float32 at offset 108) to find where voxel data starts
        (vox_offset,) = struct.unpack_from("<f", data, 108)
        vox_start = int(vox_offset) if vox_offset >= 352 else 352

        hdr = bytearray(data[:vox_start])
        # Patch datatype (offset 70) and bitpix (offset 72) for float32
        struct.pack_into("<h", hdr, 70, 16)  # datatype = float32
        struct.pack_into("<h", hdr, 72, 32)  # bitpix   = 32

        voxels = np.frombuffer(data[vox_start:], dtype="<f8").astype("<f4")
        repack = bytes(hdr) + voxels.tobytes()

        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
            gz.write(repack)
        raw = buf.getvalue()

    return base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Color palette (Chart.js datasets)
# ---------------------------------------------------------------------------

_PALETTE = [
    "#378ADD",
    "#D85A30",
    "#1D9E75",
    "#D4537E",
    "#7F77DD",
    "#BA7517",
    "#639922",
    "#888780",
    "#185FA5",
    "#993C1D",
]


def chart_color(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #ffffff; --bg2: #f4f4f2; --bg3: #e9e9e6;
    --text: #1a1a18; --text2: #5f5e5a; --border: rgba(0,0,0,0.12);
    --radius: 8px; --font: system-ui, -apple-system, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #1c1c1a; --bg2: #252523; --bg3: #2e2e2b;
      --text: #e8e8e4; --text2: #888780; --border: rgba(255,255,255,0.10);
    }
  }
  body { font-family: var(--font); background: var(--bg); color: var(--text); padding: 24px; }
  h1   { font-size: 18px; font-weight: 500; margin-bottom: 4px; }
  .subtitle { font-size: 13px; color: var(--text2); margin-bottom: 20px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; margin-bottom: 20px; }
  label { font-size: 12px; color: var(--text2); font-weight: 500; }
  .panel-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
  }
  @media (max-width: 900px) { .panel-row { grid-template-columns: 1fr; } }
  .chart-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }
  @media (max-width: 900px) { .chart-grid { grid-template-columns: 1fr; } }
  .chart-card {
    background: var(--bg2); border: 0.5px solid var(--border);
    border-radius: 12px; padding: 16px;
  }
  .chart-title { font-size: 13px; font-weight: 500; margin-bottom: 8px; }
  .chart-wrap  { position: relative; width: 100%; height: 280px; }
  /* Legend */
  .legend-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
  .legend-item {
    display: flex; align-items: center; gap: 6px; font-size: 12px;
    color: var(--text2); padding: 4px 10px; border-radius: var(--radius);
    background: var(--bg2); border: 0.5px solid var(--border);
  }
  .legend-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  /* Stats table */
  .stats-section { margin-top: 28px; }
  .stats-title  { font-size: 13px; font-weight: 500; color: var(--text2); margin-bottom: 10px; }
  .stats-table  { width: 100%; border-collapse: collapse; font-size: 12px; }
  .stats-table th {
    text-align: left; padding: 6px 10px; background: var(--bg3);
    border-bottom: 0.5px solid var(--border); color: var(--text2); font-weight: 500;
  }
  .stats-table td { padding: 5px 10px; border-bottom: 0.5px solid var(--border); }
  .stats-table tr:last-child td { border-bottom: none; }
"""

# ---------------------------------------------------------------------------
# Shared JavaScript — error bar plugin and base chart options
# ---------------------------------------------------------------------------

ERROR_BAR_PLUGIN_JS = """
const errorBarPlugin = {
  id: "errorBar",
  afterDatasetsDraw(chart) {
    const ctx = chart.ctx;
    chart.data.datasets.forEach((ds, di) => {
      if (!ds.errorBars) return;
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      ctx.save(); ctx.strokeStyle = ds.borderColor; ctx.lineWidth = 1.2;
      Object.entries(ds.errorBars).forEach(([i, eb]) => {
        const el = meta.data[+i]; if (!el) return;
        const x = el.x, cap = 4;
        const yMin = chart.scales.y.getPixelForValue(eb.yMin);
        const yMax = chart.scales.y.getPixelForValue(eb.yMax);
        ctx.beginPath();
        ctx.moveTo(x, yMin); ctx.lineTo(x, yMax);
        ctx.moveTo(x-cap, yMin); ctx.lineTo(x+cap, yMin);
        ctx.moveTo(x-cap, yMax); ctx.lineTo(x+cap, yMax);
        ctx.stroke();
      });
      ctx.restore();
    });
  }
};
"""


def base_opts_js(
    x_label: str = "x", y_label: str = "Intensity", enable_zoom: bool = True
) -> str:
    """Return a JS baseOpts() function for Chart.js.

    Parameters
    ----------
    enable_zoom : bool
        When True (default) zoom/pan via the chartjs-plugin-zoom is active.
        Pass False to disable zoom and pan entirely (e.g. for ADC, FA, vial
        intensity charts where scroll-to-zoom is undesirable).
    """
    if enable_zoom:
        zoom_plugin = """\
      zoom: {
        zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: "xy" },
        pan:  { enabled: true, mode: "xy" },
      }"""
    else:
        zoom_plugin = """\
      zoom: {
        zoom: { wheel: { enabled: false }, pinch: { enabled: false } },
        pan:  { enabled: false },
      }"""
    return f"""
function baseOpts(xLabel, yLabel) {{
  xLabel = xLabel || {json.dumps(x_label)};
  yLabel = yLabel || {json.dumps(y_label)};
  const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const grid = isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.07)";
  const tick = "#888780";
  return {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(2)}}` }} }},
{zoom_plugin}
    }},
    scales: {{
      x: {{ type: "linear",
           title: {{ display: true, text: xLabel, color: tick, font: {{ size: 13 }} }},
           ticks: {{ color: tick, font: {{ size: 13 }} }}, grid: {{ color: grid }} }},
      y: {{ title: {{ display: true, text: yLabel, color: tick, font: {{ size: 13 }} }},
           ticks: {{ color: tick, font: {{ size: 13 }} }}, grid: {{ color: grid }} }}
    }}
  }};
}}
"""


# ---------------------------------------------------------------------------
# Embedded data tag
# ---------------------------------------------------------------------------


def phantomkit_data_tag(data: dict) -> str:
    """Return a <script> tag embedding data as JSON for downstream parsing."""
    return (
        '<script id="phantomkit-data" type="application/json">\n'
        + json.dumps(data, indent=2)
        + "\n</script>"
    )


# ---------------------------------------------------------------------------
# NiiVue viewer panel (base64-embedded NIfTI — no server required)
# ---------------------------------------------------------------------------


def _niivue_viewer_panel(
    nifti_image: str,
    vial_niftis: dict[str, str],
    bg_cal_min: float | None = None,
    bg_cal_max: float | None = None,
) -> tuple[str, str]:
    """Build an embedded NiiVue viewer panel for a static HTML page.

    NIfTI files are base64-encoded and converted to blob URLs at runtime so
    the HTML is fully self-contained with no HTTP server required.

    Parameters
    ----------
    nifti_image:
        Path to the background NIfTI file.
    vial_niftis:
        ``{vial_name: path}`` mapping for ROI overlay NIfTI files.  Only
        files that exist on disk are included.
    bg_cal_min, bg_cal_max:
        Optional intensity window for the background volume (e.g. 0/0.004 for ADC).

    Returns
    -------
    html_panel : str
        The ``<div>`` element containing the canvas and vial toggle chips.
    js_block : str
        JavaScript that initialises NiiVue and defines ``pkToggleVial``.
    """
    bg_b64 = nifti_to_base64(nifti_image)

    vials_data = [
        {"name": name, "b64": nifti_to_base64(vpath)}
        for name, vpath in sorted(vial_niftis.items())
        if vpath and Path(vpath).exists()
    ]

    chips = "".join(
        f'<button id="pk-chip-{i}" data-active="1" onclick="pkToggleVial({i + 1},this)"'
        f' style="padding:4px 12px;border-radius:99px;border:1.5px solid var(--border);'
        f"background:var(--bg3);color:var(--text);font-size:11px;font-weight:500;"
        f'cursor:pointer;user-select:none;transition:opacity .15s;">'
        f'{v["name"]}</button>'
        for i, v in enumerate(vials_data)
    )
    chips_section = (
        (
            '<p style="font-size:11px;color:var(--text2);margin-top:10px;margin-bottom:6px;">'
            "Toggle vial ROIs</p>"
            f'<div style="display:flex;flex-wrap:wrap;gap:6px;">{chips}</div>'
        )
        if vials_data
        else ""
    )

    html_panel = f"""<div class="chart-card" style="margin-bottom:20px;">
  <p class="chart-title">MRI Viewer</p>
  <canvas id="nv-canvas" style="width:100%;height:400px;display:block;background:#000;border-radius:6px;cursor:crosshair;"></canvas>
  <p style="font-size:11px;color:var(--text2);margin-top:8px;">Scroll to change slice &middot; drag to adjust contrast</p>
  {chips_section}
</div>"""

    bg_b64_js = json.dumps(bg_b64)
    vials_js = json.dumps(vials_data)
    cal_min_js = "null" if bg_cal_min is None else str(float(bg_cal_min))
    cal_max_js = "null" if bg_cal_max is None else str(float(bg_cal_max))

    js_block = f"""
var _NV_BG = {bg_b64_js};
var _NV_VIALS = {vials_js};
var _NV_CAL_MIN = {cal_min_js};
var _NV_CAL_MAX = {cal_max_js};

(function() {{
  var canvas = document.getElementById("nv-canvas");
  if (!canvas) return;
  var ro = new ResizeObserver(function(entries) {{
    for (var i = 0; i < entries.length; i++) {{
      var r = entries[i].contentRect;
      var w = Math.round(r.width), h = Math.round(r.height);
      if (w > 0 && h > 0) {{ ro.disconnect(); _pkInitNv(canvas, w, h); break; }}
    }}
  }});
  ro.observe(canvas);
}})();

function _pkB64ToUrl(b64) {{
  var bin = atob(b64), arr = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return URL.createObjectURL(new Blob([arr], {{type:"application/octet-stream"}}));
}}

function _pkInitNv(canvas, w, h) {{
  canvas.width = w; canvas.height = h;
  // Prevent the browser from scrolling the page when the user scrolls over
  // the canvas — NiiVue's own wheelListener handles slice navigation.
  // dragMode defaults to 1 (contrast-on-drag); dragMode:2 (pan) would
  // hijack the wheel for zoom and break slice scrolling.
  canvas.addEventListener("wheel", function(e) {{ e.preventDefault(); }}, {{ passive: false }});
  window._pkNv = new niivue.Niivue({{
    isColorbar: false, crosshairWidth: 1, isResizeCanvas: false,
    isAntiAlias: false,
  }});
  window._pkNv.attachToCanvas(canvas);
  window._pkNv.opts.sliceType = 0;
  var vols = [{{url: _pkB64ToUrl(_NV_BG), name: "background.nii.gz", colormap: "gray"}}];
  for (var i = 0; i < _NV_VIALS.length; i++) {{
    vols.push({{
      url: _pkB64ToUrl(_NV_VIALS[i].b64),
      name: _NV_VIALS[i].name + ".nii.gz",
      colormap: "red",
      opacity: 0.5,
    }});
  }}
  window._pkNv.loadVolumes(vols).then(function() {{
    if (_NV_CAL_MIN !== null && _NV_CAL_MAX !== null) {{
      if (typeof window._pkNv.setCalMinMax === "function") {{
        window._pkNv.setCalMinMax(0, _NV_CAL_MIN, _NV_CAL_MAX);
      }} else if (window._pkNv.volumes && window._pkNv.volumes[0]) {{
        window._pkNv.volumes[0].cal_min = _NV_CAL_MIN;
        window._pkNv.volumes[0].cal_max = _NV_CAL_MAX;
        window._pkNv.updateGLVolume();
      }}
    }}
  }});
}}

function pkToggleVial(idx, btn) {{
  if (!window._pkNv) return;
  var wasActive = btn.dataset.active === "1";
  btn.dataset.active = wasActive ? "0" : "1";
  btn.style.opacity = wasActive ? "0.35" : "1.0";
  window._pkNv.setOpacity(idx, wasActive ? 0.0 : 0.5);
}}
"""
    return html_panel, js_block


# ---------------------------------------------------------------------------
# Relaxometry (IR / TE) 3×3 HTML builder
# ---------------------------------------------------------------------------


def build_relaxometry_html(
    *,
    title: str,
    subtitle: str,
    x_label: str,
    fit_label: str,
    fit_key: str,
    vial_groups: list,
    contrast_numbers,
    mean_matrix,
    std_matrix,
    vial_labels: list,
    subplot_fits: list,
    embedded_data: dict,
    nifti_image: str | None = None,
    vial_niftis: dict | None = None,
    ref_data: dict | None = None,
) -> str:
    """Build the interactive HTML for a relaxometry (IR or TE) plot.

    Parameters
    ----------
    title : str
        Page/plot title (e.g. "T1 Inversion Recovery Mapping").
    subtitle : str
        Short description line.
    x_label : str
        X-axis label (e.g. "Inversion Time (ms)").
    fit_label : str
        Legend label for fit curve (e.g. "T₁ fit").
    fit_key : str
        Key in fit_results for the relaxation time (e.g. "T1_ms").
    vial_groups : list of list of str
        Groups of vial names for the subplot grid.
    contrast_numbers : array-like
        Sorted numeric values (TI or TE) used as x-axis.
    mean_matrix : ndarray
        Shape (n_vials, n_contrasts) — mean intensities, rows=vials, cols=contrasts.
    std_matrix : ndarray
        Shape (n_vials, n_contrasts) — std deviations, same layout.
    vial_labels : list of str
        Ordered list of vial names matching row order in mean_matrix.
    subplot_fits : list of dict
        One entry per vial group, each with pre-computed fit data:
        {"vial": str, "color": str, "x_fit": list, "y_fit": list,
         "ci_upper": list or None, "ci_lower": list or None}
    embedded_data : dict
        Data to embed as machine-readable JSON in the HTML.
    """
    import numpy as np

    vial_to_idx = {label: i for i, label in enumerate(vial_labels)}
    # Convert to plain Python float so json.dumps handles them without error
    # (contrast_numbers may be a numpy array of int64 or float64)
    x_arr = [float(x) for x in contrast_numbers]

    # Build per-subplot Chart.js dataset lists
    subplot_datasets = []
    for g_idx, group in enumerate(vial_groups):
        group_datasets = []
        group_fits_this = [f for f in subplot_fits if f["vial"] in group]

        for j, vial in enumerate(group):
            if vial not in vial_to_idx:
                continue
            row = vial_to_idx[vial]
            color = chart_color(j if len(group) > 1 else 8)  # black-ish for single

            means = mean_matrix[row, :].tolist()
            stds = std_matrix[row, :].tolist()

            scatter_ds = {
                "label": f"Vial {vial}",
                "data": [{"x": x_arr[k], "y": means[k]} for k in range(len(x_arr))],
                "borderColor": color,
                "backgroundColor": color + "33",
                "pointBackgroundColor": color,
                "pointRadius": 5,
                "pointHoverRadius": 7,
                "borderWidth": 0,
                "showLine": False,
                "errorBars": {
                    str(k): {"yMin": means[k] - stds[k], "yMax": means[k] + stds[k]}
                    for k in range(len(x_arr))
                },
            }
            group_datasets.append(scatter_ds)

            # Fitted curve + CI band for this vial
            fit_entry = next((f for f in group_fits_this if f["vial"] == vial), None)
            if fit_entry and fit_entry.get("x_fit"):
                xf = fit_entry["x_fit"]
                yf = fit_entry["y_fit"]
                ci_lo = fit_entry.get("ci_lower")
                ci_hi = fit_entry.get("ci_upper")

                if ci_hi and ci_lo:
                    # Upper CI line — fills to the next dataset (lower CI)
                    group_datasets.append(
                        {
                            "label": "_ci_upper",
                            "data": [
                                {"x": xf[k], "y": ci_hi[k]} for k in range(len(xf))
                            ],
                            "borderColor": "transparent",
                            "backgroundColor": color + "22",
                            "pointRadius": 0,
                            "borderWidth": 0,
                            "showLine": True,
                            "tension": 0.3,
                            "fill": "+1",
                        }
                    )
                    group_datasets.append(
                        {
                            "label": "_ci_lower",
                            "data": [
                                {"x": xf[k], "y": ci_lo[k]} for k in range(len(xf))
                            ],
                            "borderColor": "transparent",
                            "backgroundColor": "transparent",
                            "pointRadius": 0,
                            "borderWidth": 0,
                            "showLine": True,
                            "tension": 0.3,
                            "fill": False,
                        }
                    )

                group_datasets.append(
                    {
                        "label": fit_label,
                        "data": [{"x": xf[k], "y": yf[k]} for k in range(len(xf))],
                        "borderColor": color + "bb",
                        "backgroundColor": "transparent",
                        "pointRadius": 0,
                        "borderWidth": 1.5,
                        "borderDash": [4, 3],
                        "showLine": True,
                        "tension": 0.3,
                        "fill": False,
                    }
                )

        subplot_datasets.append(group_datasets)

    datasets_json = json.dumps(subplot_datasets)
    x_label_json = json.dumps(x_label)
    data_tag = phantomkit_data_tag(embedded_data)
    opts_js = base_opts_js(x_label=x_label, y_label="Intensity", enable_zoom=True)

    # ------------------------------------------------------------------
    # Standardised axis bounds — shared across every subplot on the page
    # ------------------------------------------------------------------
    x_min_plot = 0.0
    x_max_plot = float(np.max(contrast_numbers)) * 1.05  # 5 % right padding

    y_lower = float((mean_matrix - std_matrix).min())
    y_upper = float((mean_matrix + std_matrix).max())
    for _fit in subplot_fits:
        for _key in ("y_fit", "ci_lower", "ci_upper"):
            if _fit.get(_key):
                y_lower = min(y_lower, min(_fit[_key]))
                y_upper = max(y_upper, max(_fit[_key]))
    _y_pad = (y_upper - y_lower) * 0.12
    y_min_raw = y_lower - _y_pad
    y_max_raw = y_upper + _y_pad

    # Round to a sensible grid unit based on the data range so axis tick
    # labels are clean integers (e.g. -200, 0, 1000, 2000 … not -187.3).
    _range = y_max_raw - y_min_raw
    _unit = 10 ** int(np.floor(np.log10(_range / 5))) if _range > 0 else 1
    y_min_plot = int(np.floor(y_min_raw / _unit)) * _unit
    y_max_plot = int(np.ceil(y_max_raw / _unit)) * _unit

    axis_consts_js = (
        f"const X_MIN = {x_min_plot:.1f};\n"
        f"const X_MAX = {x_max_plot:.1f};\n"
        f"const Y_MIN = {y_min_plot};\n"
        f"const Y_MAX = {y_max_plot};\n"
    )

    # ------------------------------------------------------------------
    # Fit-results table (T1 / T2 per vial)
    # ------------------------------------------------------------------
    fit_results = embedded_data.get("fit_results", [])
    fit_unit_label = "T\u2081 (ms)" if fit_key == "T1_ms" else "T\u2082 (ms)"
    if fit_results:
        fit_rows_html = ""
        for r in sorted(fit_results, key=lambda r: r.get(fit_key) or 0, reverse=True):
            vial = r.get("Vial", "")
            t_val = r.get(fit_key, "")
            s0_val = r.get("S0", "")
            r2_val = r.get("R2", "")
            t_str = f"{t_val:.1f}" if isinstance(t_val, (int, float)) else str(t_val)
            s0_str = (
                f"{s0_val:.1f}" if isinstance(s0_val, (int, float)) else str(s0_val)
            )
            r2_str = (
                f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else str(r2_val)
            )
            fit_rows_html += (
                f"    <tr>"
                f"<td>{vial}</td>"
                f"<td>{t_str}</td>"
                f"<td>{s0_str}</td>"
                f"<td>{r2_str}</td>"
                f"</tr>\n"
            )
        fits_table_html = f"""<div class="stats-section">
  <div class="stats-title">Relaxometry Fit Results</div>
  <table class="stats-table">
    <thead>
      <tr><th>Vial</th><th>{fit_unit_label}</th><th>S0</th><th>R\u00b2</th></tr>
    </thead>
    <tbody>
{fit_rows_html}    </tbody>
  </table>
</div>"""
    else:
        fits_table_html = ""

    # ------------------------------------------------------------------
    # Optional measured-vs-reference relaxometry chart
    # ------------------------------------------------------------------
    ref_chart_html = ""
    ref_chart_js = ""
    if ref_data is not None and fit_results:
        _ref_vials = ref_data.get("vials", [])
        _ref_vals = ref_data.get(fit_key, {})
        _ref_upper = {k.upper(): v for k, v in _ref_vals.items()}
        _meas_map = {
            r.get("Vial", "").upper(): r.get(fit_key)
            for r in fit_results
            if r.get(fit_key) is not None
        }

        _meas_pts, _ref_pts = [], []
        for _j, _v in enumerate(_ref_vials):
            _mu = _v.upper()
            if _meas_map.get(_mu) is not None:
                _meas_pts.append({"x": _j, "y": round(float(_meas_map[_mu]), 1)})
            if _ref_upper.get(_mu) is not None:
                _ref_pts.append({"x": _j, "y": float(_ref_upper[_mu])})

        _unit_label = "T\u2081 (ms)" if fit_key == "T1_ms" else "T\u2082 (ms)"
        _meas_color = "#378ADD"   # blue filled  — measured
        _ref_color = "#C62828"    # red open circles — reference
        _meas_ds = {
            "label": f"Measured {_unit_label}",
            "data": _meas_pts,
            "borderColor": _meas_color,
            "backgroundColor": _meas_color,
            "pointBackgroundColor": _meas_color,
            "pointRadius": 6,
            "pointHoverRadius": 8,
            "showLine": False,
            "borderWidth": 0,
        }
        _ref_ds = {
            "label": f"Reference {_unit_label}",
            "data": _ref_pts,
            "borderColor": "transparent",
            "backgroundColor": "transparent",
            "pointBackgroundColor": "transparent",
            "pointBorderColor": _ref_color,
            "pointBorderWidth": 2,
            "pointRadius": 8,
            "pointHoverRadius": 10,
            "showLine": False,
            "borderWidth": 0,
        }
        _vial_labels_json = json.dumps(_ref_vials)
        _meas_ds_json = json.dumps(_meas_ds)
        _ref_ds_json = json.dumps(_ref_ds)
        _unit_label_json = json.dumps(_unit_label)

        ref_chart_html = f"""<div class="chart-card" style="margin-bottom:20px;">
  <div class="chart-title">Measured vs Reference {_unit_label}</div>
  <div class="chart-wrap" style="height:260px"><canvas id="refChart"></canvas></div>
</div>"""

        ref_chart_js = f"""
const _REF_VIALS = {_vial_labels_json};
(function() {{
  const opts = baseOpts("Vial", {_unit_label_json});
  opts.scales.x.type = "linear";
  opts.scales.x.ticks.stepSize = 1;
  opts.scales.x.ticks.callback = v => _REF_VIALS[v] ?? v;
  opts.plugins.legend.display = true;
  opts.plugins.legend.labels = {{ color: "#888780", font: {{ size: 12 }} }};
  new Chart(
    document.getElementById("refChart").getContext("2d"),
    {{ type: "line", data: {{ datasets: [{_meas_ds_json}, {_ref_ds_json}] }}, options: opts }}
  );
}})();"""

    # ------------------------------------------------------------------
    # Optional NiiVue viewer panel
    # ------------------------------------------------------------------
    _has_viewer = bool(nifti_image and Path(nifti_image).exists())
    if _has_viewer:
        # Only show vials present in the reference JSON (if provided)
        _viewer_vials = dict(vial_niftis or {})
        if ref_data is not None:
            _ref_set = {v.upper() for v in ref_data.get("vials", [])}
            _viewer_vials = {k: v for k, v in _viewer_vials.items() if k.upper() in _ref_set}
        viewer_html, viewer_js = _niivue_viewer_panel(
            nifti_image,  # type: ignore[arg-type]
            _viewer_vials,
        )
    else:
        viewer_html = viewer_js = ""

    head = html_head(title, include_niivue=_has_viewer)

    # Build chart card divs — 2-column grid
    chart_cards_html = ""
    for g_idx, group in enumerate(vial_groups):
        group_title = " & ".join(f"Vial {v}" for v in group)
        chart_cards_html += f"""  <div class="chart-card">
    <div class="chart-title">{group_title}</div>
    <div class="chart-wrap"><canvas id="chart{g_idx}"></canvas></div>
  </div>
"""

    return f"""{head}
<body>
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>

{viewer_html}

<div class="chart-grid">
{chart_cards_html}
</div>

{ref_chart_html}

{fits_table_html}

{data_tag}

<script>
const SUBPLOTS = {datasets_json};
{axis_consts_js}
{ERROR_BAR_PLUGIN_JS}
{opts_js}

function makeChart(id, datasets) {{
  const opts = baseOpts({x_label_json}, "Intensity");
  opts.scales.x.min = X_MIN;
  opts.scales.x.max = X_MAX;
  opts.scales.y.min = Y_MIN;
  opts.scales.y.max = Y_MAX;
  const ctx = document.getElementById(id).getContext("2d");
  const c = new Chart(ctx, {{
    type: "line",
    data: {{ datasets }},
    options: opts,
    plugins: [errorBarPlugin],
  }});
  return c;
}}

SUBPLOTS.forEach((datasets, i) => makeChart("chart" + i, datasets));
{ref_chart_js}
{viewer_js}
</script>
</body>
</html>"""


def parse_phantomkit_html(html_path: str) -> dict:
    """Parse the embedded phantomkit-data JSON from an HTML output file.

    Works without BeautifulSoup by using a simple regex, making it a
    zero-dependency helper for downstream scripts.
    """
    text = Path(html_path).read_text(encoding="utf-8")
    m = re.search(
        r'<script\s+id="phantomkit-data"\s+type="application/json">\s*(\{.*?\})\s*</script>',
        text,
        re.DOTALL,
    )
    if not m:
        raise ValueError(
            f"No phantomkit-data JSON block found in {html_path!r}. "
            "Is this a phantomkit HTML output file?"
        )
    return json.loads(m.group(1))


# ---------------------------------------------------------------------------
# HTML head builder
# ---------------------------------------------------------------------------


def html_head(title: str, include_niivue: bool = False) -> str:
    """Return the ``<head>`` block with Chart.js CDN scripts and shared CSS.

    Parameters
    ----------
    include_niivue:
        When True, the NiiVue script is added to the head (inlined bundle or
        CDN fallback) for pages that embed an MRI viewer panel.
    """
    niivue_tag = _niivue_script_tag() if include_niivue else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="{CHARTJS_CDN}"></script>
<script src="{HAMMERJS_CDN}"></script>
<script src="{ZOOM_CDN}"></script>
{niivue_tag}
<style>{_CSS}</style>
</head>
"""
