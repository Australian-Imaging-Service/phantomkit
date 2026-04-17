#!/usr/bin/env python3
"""
================================================================================
T2 Multi-Echo Spin Echo Plotting Script
================================================================================
This script reads mean and standard deviation data from CSV files for multiple
echo time (TE) contrast images, plots them grouped by vials, and fits a
mono-exponential T2 decay model to estimate T2 relaxation times.

Key outputs:
- Interactive HTML (default) or publication-quality PNG with 3x3 grid plot
- CSV file with fitted T2 values and R² statistics
================================================================================
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; required when plotting runs
# in a background thread (e.g. ThreadPoolExecutor on macOS)

import logging
import os
import re

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def find_csv_file(metric_dir, contrast_name, suffix):
    """
    Find a CSV file in a directory that matches the contrast name and suffix.

    Uses an exact token match: the contrast_name must appear in the filename
    immediately followed by the suffix (no extra characters between them).
    This prevents ambiguous substring hits such as 't2_se_TE_14' matching
    't2_se_TE_140_mean_matrix.csv'.

    Args:
        metric_dir: Directory containing CSV files
        contrast_name: Base name of the contrast to search for
        suffix: File suffix to match (e.g., '_mean_matrix.csv')

    Returns:
        Full path to the matched CSV file

    Raises:
        FileNotFoundError: If no matching file is found
    """
    exact_tail = contrast_name + suffix
    for f in os.listdir(metric_dir):
        if f.endswith(exact_tail):
            return os.path.join(metric_dir, f)
    raise FileNotFoundError(
        f"No CSV file found for contrast '{contrast_name}' with suffix '{suffix}' in {metric_dir}"
    )


def extract_numeric(label):
    """
    Extract the last numeric value from a string label.

    Used to extract echo times from filenames (e.g., 'SE_80' → 80)

    Args:
        label: String containing numbers (e.g., 'contrast_100')

    Returns:
        Last integer found in the string, or None if no numbers found
    """
    numbers = re.findall(r"\d+", label)
    return int(numbers[-1]) if numbers else None


def mono_exp(te, S0, T2):
    """
    Mono-exponential T2 decay model for spin echo MRI data.

    Model: S0 * exp(-TE/T2)

    Args:
        te: Echo time (ms) - can be scalar or array
        S0: Initial signal intensity at TE=0
        T2: Transverse relaxation time (ms)

    Returns:
        Signal intensity at given echo time(s)
    """
    return S0 * np.exp(-te / T2)


def calc_r2(y_true, y_pred):
    """
    Calculate coefficient of determination (R²) for model fit quality.

    R² = 1 - (SS_res / SS_tot)
    where SS_res = sum of squared residuals
          SS_tot = total sum of squares

    Args:
        y_true: Observed data values
        y_pred: Predicted values from model

    Returns:
        R² value (1.0 = perfect fit, 0.0 = no better than mean)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def plot_vial_te_means_std(
    contrast_files,
    metric_dir,
    output_file="vial_summary_pub.png",
    annotate=False,
    roi_image=None,
    output_format: str = "html",
    nifti_image: str | None = None,
    vial_niftis: dict | None = None,
    fits_output: str | None = None,
    relaxometry_reference: dict | None = None,
):
    """
    Create interactive HTML or publication-quality PNG plots of vial intensity
    data with T2 curve fitting.

    Generates a 3x3 grid of subplots showing intensity vs echo time for
    different vial groups, with fitted T2 decay curves overlaid on the measured data.

    Args:
        contrast_files: List of NIfTI file paths for different echo times
        metric_dir: Directory containing mean/std CSV files
        output_file: Output filename for the plot (default: 'vial_summary_pub.png')
        annotate: Whether to annotate points with mean ± std (PNG mode only)
        roi_image: Optional path to ROI overlay image (PNG mode only)
        output_format: "html" (default) for interactive HTML, "png" for static figure
        nifti_image: Path to the background NIfTI file to embed in the HTML viewer.
        vial_niftis: ``{vial_name: path}`` mapping for ROI overlay NIfTIs (all shown in red).

    Returns:
        Absolute path to the saved output file.
    """

    # ========================================================================
    # CONFIGURATION: Vial groupings for subplots
    # ========================================================================
    vial_groups = [
        ["S"],  # Subplot 1: Vial S alone
        ["D", "P"],  # Subplot 2: Vials D and P together
        ["M"],  # Subplot 3: Vial M alone
        ["C", "N"],  # Subplot 4: Vials C and N together
        ["B", "T"],  # Subplot 5: Vials B and T together
        ["A", "R"],  # Subplot 6: Vials A and R together
        ["O"],  # Subplot 7: Vial O alone
        ["Q"],  # Subplot 8: Vial Q alone
        # Subplot 9: Reserved for MRI viewer (HTML) or ROI overlay image (PNG)
    ]

    # ========================================================================
    # DATA LOADING: Read mean and std deviation from CSV files
    # ========================================================================
    vial_labels = None
    contrast_numbers = []
    mean_matrix = []
    std_matrix = []

    for nifti_path in contrast_files:
        base_name = os.path.basename(nifti_path).replace(".nii.gz", "")
        mean_csv = find_csv_file(metric_dir, base_name, "_mean_matrix.csv")
        std_csv = find_csv_file(metric_dir, base_name, "_std_matrix.csv")

        mean_df = pd.read_csv(mean_csv, sep=None, engine="python")
        std_df = pd.read_csv(std_csv, sep=None, engine="python")

        if vial_labels is None:
            vial_labels = mean_df.iloc[:, 0].astype(str).tolist()

        mean_matrix.append(mean_df.iloc[:, 1].to_numpy())
        std_matrix.append(std_df.iloc[:, 1].to_numpy())
        contrast_numbers.append(extract_numeric(base_name))

    mean_matrix = np.array(mean_matrix)
    std_matrix = np.array(std_matrix)

    # ========================================================================
    # DATA ORGANIZATION: Sort by echo time and transpose
    # ========================================================================
    sort_idx = np.argsort(contrast_numbers)
    contrast_numbers = np.array(contrast_numbers)[sort_idx]
    mean_matrix = mean_matrix[sort_idx].T  # rows=vials, cols=TE
    std_matrix = std_matrix[sort_idx].T

    vial_to_idx = {label: i for i, label in enumerate(vial_labels)}
    yticks = [-100, 1000, 2000, 3000, 4000]

    fit_results = []
    subplot_fits = []

    # ========================================================================
    # CURVE FITTING LOOP (shared by both PNG and HTML paths)
    # ========================================================================
    cmap = plt.get_cmap("tab10")

    for g_idx, group in enumerate(vial_groups):
        if len(group) == 1:
            vial = group[0]
            if vial not in vial_to_idx:
                continue
            i = vial_to_idx[vial]

            try:
                popt, pcov = curve_fit(
                    mono_exp,
                    contrast_numbers,
                    mean_matrix[i, :],
                    p0=(mean_matrix[i, 0], 100),
                )
                S0_fit, T2_fit = popt
                fit_signal = mono_exp(contrast_numbers, *popt)
                r2 = calc_r2(mean_matrix[i, :], fit_signal)
                fit_results.append({"Vial": vial, "S0": S0_fit, "T2_ms": T2_fit, "R2": r2})

                x_fit = np.linspace(0, max(contrast_numbers), 200)
                y_fit = mono_exp(x_fit, *popt)
                ci_lower = ci_upper = None
                try:
                    param_samples = np.random.multivariate_normal(popt, pcov, 1000)
                    predictions = np.array([mono_exp(x_fit, *s) for s in param_samples])
                    ci_lower = np.percentile(predictions, 2.5, axis=0)
                    ci_upper = np.percentile(predictions, 97.5, axis=0)
                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.warning(
                        "Could not calculate 95%% CI for vial %s: %s", vial, e
                    )

                subplot_fits.append({
                    "vial": vial,
                    "x_fit": x_fit.tolist(),
                    "y_fit": y_fit.tolist(),
                    "ci_lower": ci_lower.tolist() if ci_lower is not None else None,
                    "ci_upper": ci_upper.tolist() if ci_upper is not None else None,
                })
            except RuntimeError:
                logger.warning("Could not fit T\u2082 for vial %s", vial)

        else:
            for j, vial in enumerate(group):
                if vial not in vial_to_idx:
                    continue
                i = vial_to_idx[vial]

                try:
                    popt, pcov = curve_fit(
                        mono_exp,
                        contrast_numbers,
                        mean_matrix[i, :],
                        p0=(mean_matrix[i, 0], 100),
                    )
                    S0_fit, T2_fit = popt
                    fit_signal = mono_exp(contrast_numbers, *popt)
                    r2 = calc_r2(mean_matrix[i, :], fit_signal)
                    fit_results.append(
                        {"Vial": vial, "S0": S0_fit, "T2_ms": T2_fit, "R2": r2}
                    )

                    x_fit = np.linspace(0, max(contrast_numbers), 200)
                    y_fit = mono_exp(x_fit, *popt)
                    ci_lower = ci_upper = None
                    try:
                        param_samples = np.random.multivariate_normal(popt, pcov, 1000)
                        predictions = np.array([mono_exp(x_fit, *s) for s in param_samples])
                        ci_lower = np.percentile(predictions, 2.5, axis=0)
                        ci_upper = np.percentile(predictions, 97.5, axis=0)
                    except (np.linalg.LinAlgError, ValueError) as e:
                        logger.warning(
                            "Could not calculate 95%% CI for vial %s: %s", vial, e
                        )

                    subplot_fits.append({
                        "vial": vial,
                        "x_fit": x_fit.tolist(),
                        "y_fit": y_fit.tolist(),
                        "ci_lower": ci_lower.tolist() if ci_lower is not None else None,
                        "ci_upper": ci_upper.tolist() if ci_upper is not None else None,
                    })
                except RuntimeError:
                    logger.warning("Could not fit T\u2082 for vial %s", vial)

    # ========================================================================
    # SAVE: Fitted T2 CSV (always, regardless of output format)
    # ========================================================================
    csv_stem = re.sub(r"\.(png|html)$", "", output_file, flags=re.IGNORECASE)
    csv_output = fits_output if fits_output else csv_stem + "_T2_fits.csv"
    pd.DataFrame(fit_results).to_csv(csv_output, index=False)
    logger.info("Fitted parameters saved to %s", csv_output)

    # ========================================================================
    # HTML OUTPUT
    # ========================================================================
    if output_format == "html":
        from phantomkit.plotting._html_common import build_relaxometry_html

        html_file = re.sub(r"\.(png|html)$", "", output_file, flags=re.IGNORECASE) + ".html"

        embedded_data = {
            "type": "maps_te",
            "vial_groups": vial_groups,
            "contrast_numbers": contrast_numbers.tolist(),
            "mean_matrix": mean_matrix.tolist(),
            "std_matrix": std_matrix.tolist(),
            "fit_results": fit_results,
        }

        html = build_relaxometry_html(
            title="T2 Echo Time Mapping",
            subtitle="Intensity vs echo time with T₂ curve fitting",
            x_label="Echo Time (ms)",
            fit_label="T₂ fit",
            fit_key="T2_ms",
            vial_groups=vial_groups,
            contrast_numbers=contrast_numbers,
            mean_matrix=mean_matrix,
            std_matrix=std_matrix,
            vial_labels=vial_labels,
            subplot_fits=subplot_fits,
            embedded_data=embedded_data,
            nifti_image=nifti_image,
            vial_niftis=vial_niftis,
            ref_data=relaxometry_reference,
        )

        from pathlib import Path
        Path(html_file).write_text(html, encoding="utf-8")
        logger.info("Interactive HTML saved to %s", html_file)
        return os.path.abspath(html_file)

    # ========================================================================
    # PNG OUTPUT (original matplotlib behaviour)
    # ========================================================================
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()

    for g_idx, group in enumerate(vial_groups):
        ax = axes[g_idx]

        if len(group) == 1:
            vial = group[0]
            if vial not in vial_to_idx:
                continue
            i = vial_to_idx[vial]

            ax.errorbar(
                contrast_numbers,
                mean_matrix[i, :],
                yerr=std_matrix[i, :],
                fmt="none",
                capsize=5,
                color="black",
                alpha=0.5,
            )
            ax.scatter(
                contrast_numbers,
                mean_matrix[i, :],
                s=50,
                color="black",
                marker="o",
                label=f"Vial {vial}",
                zorder=3,
            )

            fit_entry = next((f for f in subplot_fits if f["vial"] == vial), None)
            if fit_entry:
                x_fit = np.array(fit_entry["x_fit"])
                y_fit = np.array(fit_entry["y_fit"])
                if fit_entry.get("ci_lower") and fit_entry.get("ci_upper"):
                    ax.fill_between(
                        x_fit,
                        fit_entry["ci_lower"],
                        fit_entry["ci_upper"],
                        color="gray",
                        alpha=0.2,
                        zorder=1,
                        label="95% CI",
                    )
                ax.plot(x_fit, y_fit, "--", color="gray", alpha=0.8, zorder=2, label="T₂ fit")

        else:
            for j, vial in enumerate(group):
                if vial not in vial_to_idx:
                    continue
                i = vial_to_idx[vial]

                ax.errorbar(
                    contrast_numbers,
                    mean_matrix[i, :],
                    yerr=std_matrix[i, :],
                    fmt="none",
                    capsize=5,
                    color=cmap(j % 10),
                    alpha=0.5,
                )
                ax.scatter(
                    contrast_numbers,
                    mean_matrix[i, :],
                    s=50,
                    color=cmap(j % 10),
                    marker="o",
                    label=f"Vial {vial}",
                    zorder=3,
                )

                fit_entry = next((f for f in subplot_fits if f["vial"] == vial), None)
                if fit_entry:
                    x_fit = np.array(fit_entry["x_fit"])
                    y_fit = np.array(fit_entry["y_fit"])
                    if fit_entry.get("ci_lower") and fit_entry.get("ci_upper"):
                        ax.fill_between(
                            x_fit,
                            fit_entry["ci_lower"],
                            fit_entry["ci_upper"],
                            color=cmap(j % 10),
                            alpha=0.15,
                            zorder=1,
                        )
                    ax.plot(
                        x_fit, y_fit, "--", color=cmap(j % 10), alpha=0.8, zorder=2
                    )

            ax.legend(loc="upper right", fontsize=8)

        ax.set_title(" & ".join(group), fontsize=10)
        ax.set_ylim(min(yticks), max(yticks))
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        if g_idx % 3 == 0:
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(t) for t in yticks])
            ax.set_ylabel("Intensity", fontsize=9)
        else:
            ax.set_yticks(yticks)
            ax.set_yticklabels([])

    # ROI overlay in last subplot
    ax = axes[-1]
    if roi_image and os.path.exists(roi_image):
        img = plt.imread(roi_image)
        ax.imshow(
            img,
            extent=[
                contrast_numbers[0],
                contrast_numbers[-1],
                min(yticks),
                max(yticks),
            ],
            aspect="auto",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("ROI Overlay", fontsize=10)
    else:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info("Publication-ready plot saved to %s", output_file)
    plt.close(fig)
    return os.path.abspath(output_file)


@click.command("maps-te")
@click.argument("contrast_files", nargs=-1, required=True)
@click.option("-m", "--metric-dir", required=True, help="Directory with mean/std CSVs.")
@click.option("-o", "--output", default="vial_summary_pub.png", show_default=True)
@click.option("--annotate", is_flag=True, default=False)
@click.option("--roi-image", default=None, help="ROI overlay PNG (PNG mode only).")
@click.option(
    "--format", "output_format",
    default="html",
    type=click.Choice(["html", "png"]),
    show_default=True,
    help="Output format: interactive HTML (default) or static PNG.",
)
def main(contrast_files, metric_dir, output, annotate, roi_image, output_format):
    """Plot vial mean ± std for T2 spin-echo with mono-exponential fitting."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    plot_vial_te_means_std(
        list(contrast_files),
        metric_dir=metric_dir,
        output_file=output,
        annotate=annotate,
        roi_image=roi_image,
        output_format=output_format,
    )


if __name__ == "__main__":
    main()
