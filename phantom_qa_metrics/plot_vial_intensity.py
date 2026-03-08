#!/usr/bin/env python3
import csv
import logging
import os
import sys

import click
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def plot_vial_intensity(
    csv_file: str,
    plot_type: str,
    std_csv: str | None = None,
    roi_image: str | None = None,
    annotate: bool = False,
    output: str = "vial_subplot.png",
) -> str:
    """Plot vial vs intensity (mean ± std) for 3D or 4D contrasts, with optional ROI overlay.

    Args:
        csv_file: CSV file containing mean values (vial, mean[, vol1, vol2...]).
        plot_type: Type of plot to generate: 'line', 'bar', or 'scatter'.
        std_csv: Optional CSV file containing standard deviations (same shape as mean).
        roi_image: Optional PNG image (e.g. mrview screenshot with ROI overlay).
        annotate: Annotate points with mean ± std values.
        output: Filename for saving the plot.

    Returns:
        Absolute path to the saved plot file.
    """
    sep = detect_separator(csv_file)
    mean_df = pd.read_csv(csv_file, sep=sep)
    if mean_df.shape[1] < 2:
        raise ValueError(
            "mean CSV must have at least two columns (vial + at least one volume)."
        )

    vials = mean_df.iloc[:, 0].astype(str).str.replace(r"\.mif$", "", regex=True)
    mean_values = mean_df.iloc[:, 1:].to_numpy()
    n_vols = mean_values.shape[1]

    std_values = None
    if std_csv:
        sep_std = detect_separator(std_csv)
        std_df = pd.read_csv(std_csv, sep=sep_std)
        if std_df.shape[1] < 2:
            raise ValueError(
                "std CSV must have at least two columns (vial + at least one volume)."
            )
        std_values = std_df.iloc[:, 1:].to_numpy()

    ncols = 2 if roi_image else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6), squeeze=False)
    axes = axes[0]

    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for vol_idx in range(n_vols):
        means = mean_values[:, vol_idx]
        stds = std_values[:, vol_idx] if std_values is not None else None
        color = cmap(vol_idx % 10)

        if plot_type == "line":
            ax.errorbar(
                vials,
                means,
                yerr=stds,
                fmt="-o",
                capsize=5,
                color=color,
                label=f"Vol {vol_idx}",
            )
        elif plot_type == "bar":
            x = np.arange(len(vials)) + (vol_idx - n_vols / 2) * 0.1
            ax.bar(
                x,
                means,
                yerr=stds,
                capsize=5,
                color=color,
                width=0.1,
                label=f"Vol {vol_idx}",
            )
            ax.set_xticks(np.arange(len(vials)))
            ax.set_xticklabels(vials)
        elif plot_type == "scatter":
            ax.errorbar(
                vials,
                means,
                yerr=stds,
                fmt="o",
                capsize=5,
                color=color,
                label=f"Vol {vol_idx}",
            )

        if annotate:
            for vial, mean, std in zip(
                vials, means, stds if stds is not None else [0] * len(means)
            ):
                ax.text(
                    vial,
                    mean + (max(means) * 0.02),
                    f"{mean:.1f}±{std:.1f}" if stds is not None else f"{mean:.1f}",
                    ha="center",
                    fontsize=8,
                    color=color,
                )

    ax.set_xlabel("Vial")
    ax.set_ylabel("Intensity")
    ax.set_title("Vial vs Intensity (Mean ± Std)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Volumes")

    if roi_image:
        ax_img = axes[1]
        img = mpimg.imread(roi_image)
        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title("Contrast with Vial ROIs")

    plt.tight_layout()
    output_file = os.path.abspath(output)
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    logger.info("Plot saved to: %s", output_file)
    plt.close(fig)
    return output_file


@click.command()
@click.argument("csv_file")
@click.argument("plot_type", type=click.Choice(["line", "bar", "scatter"]))
@click.option(
    "--std_csv", default=None, help="Optional CSV file containing standard deviations."
)
@click.option("--roi_image", default=None, help="Optional PNG image with ROI overlay.")
@click.option(
    "--annotate", is_flag=True, help="Annotate points with mean ± std values."
)
@click.option(
    "--output",
    default="vial_subplot.png",
    show_default=True,
    help="Filename for saving the plot.",
)
def main(csv_file, plot_type, std_csv, roi_image, annotate, output):
    """Plot vial vs intensity (mean ± std) for 3D or 4D contrasts, with optional ROI overlay."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    plot_vial_intensity(
        csv_file=csv_file,
        plot_type=plot_type,
        std_csv=std_csv,
        roi_image=roi_image,
        annotate=annotate,
        output=output,
    )


if __name__ == "__main__":
    main()
