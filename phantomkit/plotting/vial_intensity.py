#!/usr/bin/env python3
"""
plot_vial_intensity.py
Plot vial vs intensity (mean ± std) for 3D or 4D contrasts, with optional ROI overlay.

Plot mode is detected automatically from the csv_file filename:

  ADC mode  – filename contains 'ADC' (case-insensitive)
    · Y-axis: ADC ×10⁻³ mm²/s
    · Reference values + ±5%/±10% tolerance bands drawn (blue crosshairs / shading)
    · Measured values plotted as filled red circles
    · Requires --phantom and --template_dir so the reference JSON can be loaded

  FA mode   – filename contains standalone 'FA' (case-insensitive, whole-word)
    · Y-axis: Fractional Anisotropy, fixed range 0–1

  Generic   – all other filenames → standard intensity plot

Output formats
  html  (default) – interactive HTML with NiiVue MRI viewer and Chart.js plot
  png             – static matplotlib figure (original behaviour)
"""

import click
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; required when plotting runs
# in a background thread (e.g. ThreadPoolExecutor on macOS)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import sys
import csv
import os
import json
import re
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_separator(file_path):
    """Detects the most likely CSV separator by sniffing the first line."""
    try:
        with open(file_path, "r", newline="") as f:
            first_line = f.readline()
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(first_line)
            return dialect.delimiter
    except Exception as e:
        sys.exit(f"Error detecting separator: {e}")


def load_adc_reference(template_dir: str, phantom: str) -> dict:
    """
    Load ADC reference values from TemplateData/<phantom>/adc_reference.json.

    Returns a dict with keys:
        'vials'        : list of vial labels in order
        'adc_mm2_per_s': dict mapping vial label -> reference ADC (mm²/s)
    """
    ref_path = os.path.join(template_dir, phantom, "adc_reference.json")
    if not os.path.isfile(ref_path):
        sys.exit(
            f"Error: ADC reference file not found at '{ref_path}'.\n"
            f"Expected structure: <template_dir>/<phantom>/adc_reference.json"
        )
    with open(ref_path) as fh:
        data = json.load(fh)
    required_keys = {"vials", "adc_mm2_per_s"}
    if not required_keys.issubset(data.keys()):
        sys.exit(f"Error: adc_reference.json must contain keys: {required_keys}")
    return data


def overlay_adc_reference(ax: plt.Axes, ref_data: dict):
    """
    Draw per-vial reference ADC values as open circles onto *ax*.
    Values are converted to ×10⁻³ for display.

    Open circles (no fill) are drawn after the measured scatter so both
    are visible when they coincide.
    """
    vials = ref_data["vials"]
    ref_vals = np.array([ref_data["adc_mm2_per_s"][v] for v in vials])
    x = np.arange(len(vials))
    ref_display = ref_vals * 1e3  # convert to ×10⁻³ for display

    # Open circle — no fill, steelblue edge, 1.5× the measured markersize (7→10.5)
    ax.scatter(
        x,
        ref_display,
        marker="o",
        s=110,  # approx 10.5² ≈ 110
        facecolors="none",
        edgecolors="steelblue",
        linewidths=1.5,
        zorder=6,
    )


def build_adc_legend(has_measured: bool) -> list:
    """Return legend handles for the ADC overlay."""
    ref_handle = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markeredgecolor="steelblue",
        markeredgewidth=1.5,
        linestyle="None",
        markersize=10,
        label="Reference ADC",
    )
    handles = [ref_handle]
    if has_measured:
        meas_handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="crimson",
            linestyle="None",
            markersize=7,
            label="Mean (SD) ADC",
        )
        handles.append(meas_handle)
    return handles


# ---------------------------------------------------------------------------
# HTML generation helpers
# ---------------------------------------------------------------------------


def _build_vial_intensity_html(
    *,
    vials: list,
    display_values: np.ndarray,
    display_stds,
    n_vols: int,
    contrast_mode: str,
    ref_data: dict | None,
    plot_type: str,
    phantom: str | None,
    title: str,
    y_label: str,
    embedded_data: dict,
    nifti_image: str | None = None,
    vial_niftis: dict | None = None,
) -> str:
    """Build a self-contained HTML string for the vial intensity plot."""
    from phantomkit.plotting._html_common import (
        html_head,
        ERROR_BAR_PLUGIN_JS,
        base_opts_js,
        phantomkit_data_tag,
        chart_color,
        _niivue_viewer_panel,
    )
    from pathlib import Path as _Path

    _has_viewer = bool(nifti_image and _Path(nifti_image).exists())
    if _has_viewer:
        _adc_cal = contrast_mode == "adc"
        # For ADC mode, only show the vials defined in the reference (E–L);
        # the other vials are unrelated to ADC calibration.
        _viewer_vials = dict(vial_niftis or {})
        if _adc_cal and ref_data is not None:
            _adc_set = {v.upper() for v in ref_data.get("vials", [])}
            _viewer_vials = {k: v for k, v in _viewer_vials.items() if k.upper() in _adc_set}
        elif _adc_cal:
            _adc_set = {"E", "F", "G", "H", "I", "J", "K", "L"}
            _viewer_vials = {k: v for k, v in _viewer_vials.items() if k.upper() in _adc_set}
        viewer_html, viewer_js = _niivue_viewer_panel(
            nifti_image,  # type: ignore[arg-type]
            _viewer_vials,
            bg_cal_min=0.0 if _adc_cal else None,
            bg_cal_max=0.004 if _adc_cal else None,
        )
    else:
        viewer_html = viewer_js = ""

    # ---- Build Chart.js datasets -------------------------------------------
    datasets = []
    for vol_idx in range(n_vols):
        means = display_values[:, vol_idx].tolist()
        stds = (
            display_stds[:, vol_idx].tolist()
            if display_stds is not None
            else None
        )

        if contrast_mode == "adc":
            color = "#378ADD"  # blue filled circles for measured ADC
            vol_label = f"Vol {vol_idx}" if n_vols > 1 else "Mean (SD) ADC"
        else:
            color = chart_color(vol_idx)
            vol_label = f"Vol {vol_idx}"

        scatter_ds = {
            "label": vol_label,
            "data": [{"x": j, "y": means[j]} for j in range(len(vials))],
            "borderColor": color,
            "backgroundColor": color + "33",
            "pointBackgroundColor": color,
            "pointRadius": 5,
            "pointHoverRadius": 7,
            "borderWidth": 1 if plot_type == "scatter" else 2,
            "showLine": plot_type != "scatter",
            "tension": 0.3 if plot_type == "line" else 0,
        }
        if stds is not None:
            scatter_ds["errorBars"] = {
                str(j): {
                    "yMin": means[j] - stds[j],
                    "yMax": means[j] + stds[j],
                }
                for j in range(len(vials))
            }
        datasets.append(scatter_ds)

    # Reference ADC overlay dataset
    if ref_data is not None:
        ref_vials = ref_data["vials"]
        ref_vals_display = {
            v: ref_data["adc_mm2_per_s"][v] * 1e3 for v in ref_vials
        }
        # Case-insensitive lookup so CSV vial labels match the reference JSON
        ref_vals_upper = {k.upper(): val for k, val in ref_vals_display.items()}
        ref_pts = []
        for j, v in enumerate(vials):
            val = ref_vals_upper.get(v.upper())
            if val is not None:
                ref_pts.append({"x": j, "y": val})
        if ref_pts:
            datasets.append({
                "label": "Reference ADC",
                "data": ref_pts,
                "borderColor": "transparent",
                "backgroundColor": "transparent",
                "pointBackgroundColor": "transparent",
                "pointBorderColor": "#C62828",  # red open circles for reference
                "pointBorderWidth": 2,
                "pointRadius": 8,
                "pointStyle": "circle",
                "borderWidth": 0,
                "showLine": False,
            })

    datasets_json = json.dumps(datasets)
    vials_json = json.dumps(list(vials))
    title_json = json.dumps(title)
    y_label_json = json.dumps(y_label)

    data_tag = phantomkit_data_tag(embedded_data)
    opts_js = base_opts_js(x_label="Vial", y_label=y_label, enable_zoom=False)
    head = html_head(title, include_niivue=_has_viewer)

    # Always show one tick per vial position
    tick_step_js = "opts.scales.x.ticks.stepSize = 1;"

    # FA y-axis constraint JS
    fa_ylim_js = ""
    if contrast_mode == "fa":
        fa_ylim_js = "opts.scales.y.min = 0; opts.scales.y.max = 1;"

    # ADC mode: show legend to distinguish measured vs reference
    adc_legend_js = ""
    if contrast_mode == "adc":
        adc_legend_js = (
            "opts.plugins.legend.display = true;"
            " opts.plugins.legend.labels = { color: '#888780', font: { size: 12 } };"
        )

    return f"""{head}
<body>
<h1>{title}</h1>
<p class="subtitle">Interactive plot · scroll to zoom · drag to pan · double-click to reset view</p>

{viewer_html}

<div class="chart-card">
  <div class="chart-title">{title}</div>
  <div class="chart-wrap" style="height:340px"><canvas id="intensityChart"></canvas></div>
</div>

<div class="stats-section">
  <div class="stats-title">Per-vial values</div>
  <table class="stats-table">
    <thead><tr><th>Vial</th><th>Mean</th><th>Std</th></tr></thead>
    <tbody id="statsBody"></tbody>
  </table>
</div>

{data_tag}

<script>
const VIALS = {vials_json};
const DATASETS = {datasets_json};

{ERROR_BAR_PLUGIN_JS}
{opts_js}

const opts = baseOpts("Vial", {y_label_json});
opts.scales.x.type = "linear";
opts.scales.x.ticks.callback = (v) => VIALS[v] ?? v;
{tick_step_js}
{fa_ylim_js}
{adc_legend_js}

const chart = new Chart(
  document.getElementById("intensityChart").getContext("2d"),
  {{ type: "line", data: {{ datasets: DATASETS }}, options: opts, plugins: [errorBarPlugin] }}
);
document.getElementById("intensityChart").addEventListener("dblclick", () => chart.resetZoom());

{viewer_js}

// Populate stats table
const tbody = document.getElementById("statsBody");
VIALS.forEach((v, j) => {{
  const ds0 = DATASETS[0];
  const mean = ds0.data[j]?.y ?? "";
  const eb = ds0.errorBars?.[j];
  const std = eb ? ((eb.yMax - eb.yMin) / 2).toFixed(4) : "";
  tbody.innerHTML += `<tr><td>${{v}}</td><td>${{typeof mean === "number" ? mean.toFixed(4) : mean}}</td><td>${{std}}</td></tr>`;
}});

</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot vial vs intensity (mean ± std) for 3D or 4D contrasts, "
            "with optional ROI overlay.\n\n"
            "Plot mode (ADC / FA / generic) is detected automatically from "
            "the csv_file filename — no flags required."
        )
    )
    # ---- positional --------------------------------------------------------
    parser.add_argument(
        "csv_file",
        help="CSV file containing mean values (vial, mean[, vol1, vol2...]).",
    )
    parser.add_argument(
        "plot_type",
        choices=["line", "bar", "scatter"],
        help="Type of plot to generate.",
    )
    # ---- standard options --------------------------------------------------
    parser.add_argument(
        "--std_csv",
        help="Optional CSV file containing standard deviations (same shape as mean CSV).",
    )
    parser.add_argument(
        "--roi_image",
        help="Optional PNG image (e.g. mrview screenshot with ROI overlay). PNG mode only.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate points with mean ± std values.",
    )
    parser.add_argument(
        "--output",
        default="vial_subplot.png",
        help="Filename for saving the plot (default: vial_subplot.png).",
    )
    # ---- phantom options (used when ADC mode is auto-detected) -------------
    parser.add_argument(
        "--phantom",
        default=None,
        help=(
            "Phantom name (e.g. SPIRIT).  Used to locate ADC reference data "
            "and to label plot titles.  Required when the csv_file name "
            "contains 'ADC'."
        ),
    )
    parser.add_argument(
        "--template_dir",
        default=None,
        help=(
            "Path to the TemplateData directory.  "
            "ADC reference is read from <template_dir>/<phantom>/adc_reference.json.  "
            "Required when the csv_file name contains 'ADC'."
        ),
    )

    args = parser.parse_args()
    plot_vial_intensity(
        csv_file=args.csv_file,
        plot_type=args.plot_type,
        std_csv=args.std_csv,
        roi_image=args.roi_image,
        annotate=args.annotate,
        output=args.output,
        phantom=args.phantom,
        template_dir=args.template_dir,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_vial_intensity(
    csv_file: str,
    plot_type: str = "scatter",
    std_csv: str | None = None,
    roi_image: str | None = None,
    annotate: bool = False,
    output: str = "vial_subplot.png",
    phantom: str | None = None,
    template_dir: str | None = None,
    output_format: str = "html",
    nifti_image: str | None = None,
    vial_niftis: dict | None = None,
):
    """Plot vial vs intensity. Mode (ADC/FA/generic) auto-detected from csv_file name.

    Can be called programmatically or via the CLI entry point (main()).

    Parameters
    ----------
    csv_file : str
        Path to mean values CSV.
    plot_type : str
        "scatter", "line", or "bar".
    std_csv : str, optional
        Path to standard deviations CSV (same shape as mean CSV).
    roi_image : str, optional
        Path to ROI overlay PNG. Used in PNG mode only.
    annotate : bool
        Annotate points (PNG mode only).
    output : str
        Output file path. Extension is replaced with .html when output_format="html".
    phantom : str, optional
        Phantom name for ADC reference lookup.
    template_dir : str, optional
        Directory containing per-phantom adc_reference.json files.
    output_format : str
        "html" (default) for interactive HTML, or "png" for static matplotlib figure.
    """

    # ---- Auto-detect contrast mode from csv_file filename ------------------
    csv_stem = Path(csv_file).stem
    if re.search(r"ADC", csv_stem, re.IGNORECASE):
        contrast_mode = "adc"
    elif re.search(r"(?<![A-Za-z0-9])FA(?![A-Za-z0-9])", csv_stem):
        contrast_mode = "fa"
    else:
        contrast_mode = "generic"

    print(f"[INFO] Contrast mode detected: {contrast_mode}")

    # ---- Validate: ADC mode needs phantom + template_dir -------------------
    if contrast_mode == "adc":
        if not phantom:
            sys.exit(
                "Error: --phantom is required when the csv_file name contains 'ADC'."
            )
        if not template_dir:
            sys.exit(
                "Error: --template_dir is required when the csv_file name contains 'ADC'."
            )

    # ---- Load mean data -----------------------------------------------------
    if Path(csv_file).suffix.lower() == ".xlsx":
        mean_df = pd.read_excel(csv_file, sheet_name="mean")
    else:
        sep = detect_separator(csv_file)
        mean_df = pd.read_csv(csv_file, sep=sep)
    if mean_df.shape[1] < 2:
        raise ValueError(
            "mean data must have at least two columns (vial + at least one volume)."
        )

    vials = mean_df.iloc[:, 0].astype(str).str.replace(r"\.mif$", "", regex=True)
    mean_values = mean_df.iloc[:, 1:].to_numpy()  # shape (n_vials, n_vols)
    n_vols = mean_values.shape[1]

    # ---- Load std data (from xlsx sheet or separate csv) -------------------
    std_values = None
    if Path(csv_file).suffix.lower() == ".xlsx":
        std_df = pd.read_excel(csv_file, sheet_name="std")
        std_values = std_df.iloc[:, 1:].to_numpy()  # shape (n_vials, n_vols)
    elif std_csv:
        sep_std = detect_separator(std_csv)
        std_df = pd.read_csv(std_csv, sep=sep_std)
        if std_df.shape[1] < 2:
            raise ValueError(
                "std CSV must have at least two columns (vial + at least one volume)."
            )
        std_values = std_df.iloc[:, 1:].to_numpy()  # shape (n_vials, n_vols)

    # ---- Load ADC reference (ADC mode only) --------------------------------
    ref_data = None
    if contrast_mode == "adc":
        ref_data = load_adc_reference(template_dir, phantom)

    # ---- In ADC mode, restrict to vials defined in adc_reference.json ------
    if contrast_mode == "adc" and ref_data is not None:
        adc_vials = [v.upper() for v in ref_data["vials"]]
        mask = vials.str.upper().isin(adc_vials)
        vials = vials[mask].reset_index(drop=True)
        mean_values = mean_values[mask.values]
        if std_values is not None:
            std_values = std_values[mask.values]

    # ---- In ADC mode, scale measured values to ×10⁻³ for display ----------
    display_values = mean_values * 1e3 if contrast_mode == "adc" else mean_values
    display_stds = (
        std_values * 1e3
        if (contrast_mode == "adc" and std_values is not None)
        else std_values
    )

    # ---- Build axis labels and title ---------------------------------------
    phantom_label = f"{phantom} Phantom – " if phantom else ""
    if contrast_mode == "adc":
        y_label = "ADC ×10⁻³ mm²/s"
        title = f"{phantom_label}Measured vs Reference ADC"
    elif contrast_mode == "fa":
        y_label = "Fractional Anisotropy"
        title = f"{phantom_label}Fractional Anisotropy (Mean ± Std)"
    else:
        y_label = "Intensity"
        title = f"{phantom_label}Vial vs Intensity (Mean ± Std)"

    # ========================================================================
    # HTML output path
    # ========================================================================
    if output_format == "html":
        output_file = os.path.abspath(
            re.sub(r"\.(png|html)$", "", output, flags=re.IGNORECASE) + ".html"
        )

        embedded_data = {
            "type": "vial_intensity",
            "contrast_mode": contrast_mode,
            "phantom": phantom,
            "vials": list(vials),
            "means": display_values.tolist(),
            "stds": display_stds.tolist() if display_stds is not None else None,
        }

        html = _build_vial_intensity_html(
            vials=list(vials),
            display_values=display_values,
            display_stds=display_stds,
            n_vols=n_vols,
            contrast_mode=contrast_mode,
            ref_data=ref_data,
            plot_type=plot_type,
            phantom=phantom,
            title=title,
            y_label=y_label,
            embedded_data=embedded_data,
            nifti_image=nifti_image,
            vial_niftis=vial_niftis,
        )

        Path(output_file).write_text(html, encoding="utf-8")
        print(f"[INFO] Interactive HTML saved to: {output_file}")
        return output_file

    # ========================================================================
    # PNG output path (original matplotlib behaviour)
    # ========================================================================
    # ---- Setup figure ------------------------------------------------------
    ncols = 2 if roi_image else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6), squeeze=False)
    axes = axes[0]  # flatten row
    ax = axes[0]

    # ---- Plot each volume --------------------------------------------------
    cmap = plt.get_cmap("tab10")

    for vol_idx in range(n_vols):
        means = display_values[:, vol_idx]
        stds = display_stds[:, vol_idx] if display_stds is not None else None

        if contrast_mode == "adc":
            color = "crimson"
            fmt_line = "-o"
            fmt_scatter = "o"
            marker_kw = dict(
                markersize=7, markerfacecolor="crimson", markeredgecolor="crimson"
            )
            vol_label = f"Vol {vol_idx}" if n_vols > 1 else "Mean (SD) ADC"
        else:
            color = cmap(vol_idx % 10)
            fmt_line = "-o"
            fmt_scatter = "o"
            marker_kw = {}
            vol_label = f"Vol {vol_idx}"

        if plot_type == "line":
            ax.errorbar(
                vials,
                means,
                yerr=stds,
                fmt=fmt_line,
                capsize=5,
                color=color,
                label=vol_label,
                **marker_kw,
            )
        elif plot_type == "bar":
            x = np.arange(len(vials)) + (vol_idx - n_vols / 2) * 0.1
            ax.bar(
                x, means, yerr=stds, capsize=5, color=color, width=0.1, label=vol_label
            )
            ax.set_xticks(np.arange(len(vials)))
            ax.set_xticklabels(vials)
        elif plot_type == "scatter":
            ax.errorbar(
                vials,
                means,
                yerr=stds,
                fmt=fmt_scatter,
                capsize=5,
                color=color,
                label=vol_label,
                **marker_kw,
            )

        if annotate:
            for vial, mean, std in zip(
                vials, means, stds if stds is not None else [0] * len(means)
            ):
                ax.text(
                    vial,
                    mean + (max(means) * 0.02),
                    f"{mean:.3f}±{std:.3f}" if stds is not None else f"{mean:.3f}",
                    ha="center",
                    fontsize=8,
                    color=color,
                )

    # ---- ADC reference overlay (drawn after measured values so crosshairs
    #      are visible even when they coincide with measured data) -----------
    if ref_data is not None:
        overlay_adc_reference(ax, ref_data)

    # ---- Axis labels -------------------------------------------------------
    ax.set_xlabel("Vial", fontsize=12)

    if contrast_mode == "adc":
        ax.set_ylabel("ADC ×10⁻³ mm²/s", fontsize=12)
        ax.set_title(
            f"{phantom_label}Measured vs Reference ADC", fontsize=13, fontweight="bold"
        )
        # Plain numeric ticks (values are already in ×10⁻³)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.2f}"))
    elif contrast_mode == "fa":
        ax.set_ylabel("Fractional Anisotropy", fontsize=12)
        ax.set_title(
            f"{phantom_label}Fractional Anisotropy (Mean ± Std)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_ylim(0, 1)
    else:
        ax.set_ylabel("Intensity", fontsize=12)
        ax.set_title(f"{phantom_label}Vial vs Intensity (Mean ± Std)", fontsize=13)

    ax.grid(True, linestyle="--", alpha=0.6)

    # ---- Legend ------------------------------------------------------------
    if contrast_mode == "adc":
        extra_handles = build_adc_legend(has_measured=True)
        if n_vols > 1:
            vol_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="crimson",
                    linestyle="-" if plot_type == "line" else "None",
                    markersize=7,
                    label=f"Vol {i}",
                    alpha=0.4 + 0.6 * i / max(n_vols - 1, 1),
                )
                for i in range(n_vols)
            ]
            ax.legend(handles=extra_handles + vol_handles, fontsize=10)
        else:
            ax.legend(handles=extra_handles, fontsize=10)
    else:
        ax.legend(title="Volumes", fontsize=10)

    # ---- ROI overlay subplot -----------------------------------------------
    if roi_image:
        ax_img = axes[1]
        img = mpimg.imread(roi_image)
        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title("Contrast with Vial ROIs")

    plt.tight_layout()
    output_file = os.path.abspath(output)
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    print(f"[INFO] Plot saved to: {output_file}")
    plt.close(fig)
    return output_file


@click.command("vial-intensity")
@click.argument("csv_file")
@click.argument("plot_type", default="scatter")
@click.option("--std-csv", default=None)
@click.option("--roi-image", default=None, help="ROI overlay PNG (PNG mode only).")
@click.option("--annotate", is_flag=True, default=False)
@click.option("--output", default="vial_subplot.png")
@click.option("--phantom", default=None)
@click.option("--template-dir", default=None)
@click.option(
    "--format", "output_format",
    default="html",
    type=click.Choice(["html", "png"]),
    show_default=True,
    help="Output format: interactive HTML (default) or static PNG.",
)
def main(csv_file, plot_type, std_csv, roi_image, annotate, output,
         phantom, template_dir, output_format):
    """Plot vial vs intensity (mean ± std) for 3D or 4D contrasts."""
    plot_vial_intensity(
        csv_file=csv_file,
        plot_type=plot_type,
        std_csv=std_csv,
        roi_image=roi_image,
        annotate=annotate,
        output=output,
        phantom=phantom,
        template_dir=template_dir,
        output_format=output_format,
    )


if __name__ == "__main__":
    main()
