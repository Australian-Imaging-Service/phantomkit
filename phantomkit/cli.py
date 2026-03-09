"""Command-line interface for phantom QA processing."""

import logging
from pathlib import Path

import click

from phantomkit.protocols.gsp_spirit import GspSpiritAnalysis, GspSpiritAnalysisBatch

logger = logging.getLogger(__name__)


@click.group()
def main() -> None:
    """Phantom QA processing workflow."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )


@main.command()
@click.argument("input_image")
@click.option("--template-dir", required=True, help="Template phantom directory.")
@click.option("--output-dir", required=True, help="Output base directory.")
@click.option("--rotation-lib", required=True, help="Rotation library file.")
def single(input_image, template_dir, output_dir, rotation_lib) -> None:
    """Process one session."""
    import pydra

    wf = GspSpiritAnalysis(
        input_image=input_image,
        template_dir=template_dir,
        output_base_dir=output_dir,
        rotation_library_file=rotation_lib,
    )
    with pydra.Submitter(plugin="serial") as sub:
        sub(wf)
    result = wf.result()
    logger.info("Done! Metrics: %s", result.output.metrics_dir)


@main.command()
@click.argument("data_dir")
@click.option("--template-dir", required=True, help="Template phantom directory.")
@click.option("--output-dir", required=True, help="Output base directory.")
@click.option("--rotation-lib", required=True, help="Rotation library file.")
@click.option(
    "--pattern",
    default="*t1*mprage*.nii.gz",
    show_default=True,
    help="Glob pattern for session images.",
)
@click.option(
    "--plugin",
    default="cf",
    show_default=True,
    type=click.Choice(["cf", "serial"]),
    help="Pydra execution plugin.",
)
def batch(data_dir, template_dir, output_dir, rotation_lib, pattern, plugin) -> None:
    """Batch-process multiple sessions."""
    import pydra

    images = sorted(str(img) for img in Path(data_dir).glob(f"*/{pattern}"))
    if not images:
        logger.error("No images found!")
        raise SystemExit(1)

    logger.info("Found %d images:", len(images))
    for img in images:
        logger.info("  %s", img)

    wf = GspSpiritAnalysisBatch(
        input_images=images,
        template_dir=template_dir,
        output_base_dir=output_dir,
        rotation_library_file=rotation_lib,
    )
    with pydra.Submitter(plugin=plugin) as sub:
        sub(wf)
    logger.info("All sessions processed! Output: %s", output_dir)


if __name__ == "__main__":
    main()
