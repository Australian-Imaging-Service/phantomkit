"""
Pydra workflows for per-vial metric extraction.

Transforms vial masks into subject space, extracts per-vial signal
statistics from contrast images, and forward-warps contrasts to
template space.
"""

import logging
import os.path
from pathlib import Path

from pydra.compose import python, workflow
from pydra.tasks.ants.v2.resampling.apply_transforms import ApplyTransforms
from pydra.tasks.mrtrix3.v3_1 import MrConvert, MrGrid, MrInfo, MrStats, MrTransform

from phantom_qa_metrics.registration import ParseMrStatsStdout

logger = logging.getLogger(__name__)


# ============================================================================
# Python Tasks
# ============================================================================


@python.define(outputs=["is_4d", "is_single_slice", "nvols"])
def ParseMrInfoSize(stdout: str) -> tuple[bool, bool, int]:
    """Parse ``mrinfo -size`` stdout into dimensionality flags and volume count."""
    parts = stdout.strip().split()
    is_4d = len(parts) >= 4 and int(parts[3]) > 1
    is_single_slice = len(parts) >= 3 and int(parts[2]) == 1
    nvols = int(parts[3]) if len(parts) >= 4 and int(parts[3]) > 0 else 1
    return is_4d, is_single_slice, nvols


@python.define
def CopyFile(src: str, dst: str) -> str:
    """Copy *src* to *dst* and return *dst*."""
    import shutil

    shutil.copy2(src, dst)
    return dst


@python.define(outputs=["vial_name", "tmp_path", "output_path", "vial_mask_out"])
def PrepVialTransformPaths(
    vial_mask: str, output_vial_dir: str
) -> tuple[str, str, str, str]:
    """Derive per-vial output paths and pass vial_mask through."""
    output_dir = Path(output_vial_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_vial_dir = output_dir.parent / "tmp_vials"
    tmp_vial_dir.mkdir(parents=True, exist_ok=True)
    vial_name = (
        Path(vial_mask).name.replace(".nii.gz", "").replace(".nii", "").split(".")[0]
    )
    return (
        vial_name,
        str(tmp_vial_dir / f"{vial_name}.nii"),
        str(output_dir / f"{vial_name}.nii.gz"),
        vial_mask,
    )


@python.define
def GatherList(items: list[str]) -> list[str]:
    """Identity task used to collect a combined split into a typed list output."""
    return list(items)


# ============================================================================
# Metrics Workflows
# ============================================================================


@workflow.define
def TransformVialsToSubjectSpace(
    vial_masks: list[str],
    reference_image: str,
    transform_matrix: str,
    rotation_matrix_file: str | None,
    iteration: int,
    output_vial_dir: str,
) -> list[str]:
    """
    Transform each vial mask from template space into subject (scanner) space.

    Split over each vial mask:
    1. **PrepVialTransformPaths** derives per-vial output paths.
    2. **ApplyTransforms** applies the inverse affine.
    3. **CopyFile** moves the output to the per-vial tmp path.
    4. **MrTransform** (inverse rotation, iteration > 1) or **MrConvert**
       (copy, iteration == 1) writes the final output.

    A combined **GatherList** collects all per-vial output paths.
    """
    from fileformats.medimage import NiftiGz
    from fileformats.generic import File

    prep = workflow.add(
        PrepVialTransformPaths(
            vial_mask=vial_masks, output_vial_dir=output_vial_dir
        ).split("vial_mask"),
        name="prep",
    )

    at = workflow.add(
        ApplyTransforms(
            input_image=prep.vial_mask_out,
            reference_image=NiftiGz(reference_image),
            transforms=[f"[{transform_matrix}, 1]"],
            interpolation="Linear",
            out_postfix="_warped",
        ),
        name="apply_transforms",
    )

    copy = workflow.add(
        CopyFile(src=at.output_image, dst=prep.tmp_path),
        name="copy",
    )

    if iteration > 1 and rotation_matrix_file:
        final = workflow.add(
            MrTransform(
                in_file=copy.out,
                linear=File(rotation_matrix_file),
                out_file=prep.output_path,
                interp="nearest",
                inverse=True,
                force=True,
            ),
            name="final",
        )
    else:
        final = workflow.add(
            MrConvert(
                in_file=copy.out,
                out_file=prep.output_path,
                quiet=True,
                force=True,
            ),
            name="final",
        )

    gather = workflow.add(
        GatherList(items=final.out_file).combine("prep"),
        name="gather",
    )
    return gather.out


@workflow.define
def ExtractMetricsFromContrasts(
    contrast_files: list[str],
    vial_masks: list[str],
    output_metrics_dir: str,
    session_name: str,
) -> str:
    """
    Extract per-vial statistics (mean, median, std, min, max) from every
    contrast image and write per-metric CSV files.

    For each (contrast, vial) pair:
    - **MrInfo** + **ParseMrInfoSize** detect the volume count.
    - **MrGrid** regrids the vial mask to the contrast image grid.
    - **MrConvert** extracts individual volumes from 4-D series.
    - **MrStats** + **ParseMrStatsStdout** compute per-volume statistics.

    Iterating over ``contrast_files`` triggers pydra's runtime fallback, so
    all node outputs are concrete values available for loop control.

    Returns the metrics directory path.
    """
    import pandas as pd
    from fileformats.medimage import NiftiGz

    metrics_dir = Path(output_metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tmp_vol_dir = metrics_dir.parent / "tmp_vols"
    tmp_vol_dir.mkdir(parents=True, exist_ok=True)

    for contrast_file in contrast_files:
        contrast_path = Path(contrast_file)
        contrast_name = contrast_path.stem
        clean_name = contrast_name.replace(".nii", "").replace(".gz", "")

        info = workflow.add(
            MrInfo(image_=[contrast_file], size=True, quiet=True),
            name=f"info_{clean_name}",
        )
        parse_size = workflow.add(
            ParseMrInfoSize(stdout=info.stdout),
            name=f"parse_size_{clean_name}",
        )
        nvols = parse_size.nvols

        logger.info(
            "Processing %s (%d volume%s)", clean_name, nvols, "s" if nvols > 1 else ""
        )
        metrics_data: dict[str, dict[str, list[float]]] = {
            "mean": {},
            "median": {},
            "std": {},
            "min": {},
            "max": {},
        }

        for vial_mask in vial_masks:
            vial_name = (
                Path(vial_mask)
                .name.replace(".nii.gz", "")
                .replace(".nii", "")
                .split(".")[0]
            )
            for metric in metrics_data:
                metrics_data[metric][vial_name] = []

            grid = workflow.add(
                MrGrid(
                    in_file=NiftiGz(vial_mask),
                    operation="regrid",
                    template=NiftiGz(contrast_file),
                    out_file=str(tmp_vol_dir / f"{clean_name}_{vial_name}.nii.gz"),
                    interp="nearest",
                    force=True,
                ),
                name=f"grid_{clean_name}_{vial_name}",
            )
            regridded_mask = grid.out_file

            for vol_idx in range(nvols):
                tag = f"{clean_name}_{vial_name}_v{vol_idx}"

                if nvols == 1:
                    vol_file = contrast_file
                else:
                    conv = workflow.add(
                        MrConvert(
                            in_file=NiftiGz(contrast_file),
                            coord=[(3, vol_idx)],
                            out_file=str(
                                tmp_vol_dir / f"{clean_name}_vol{vol_idx}.nii.gz"
                            ),
                            quiet=True,
                            force=True,
                        ),
                        name=f"conv_{tag}",
                    )
                    vol_file = conv.out_file

                stats = workflow.add(
                    MrStats(
                        image_=NiftiGz(vol_file),
                        mask=NiftiGz(regridded_mask),
                        output=["mean", "median", "std", "min", "max"],
                        quiet=True,
                    ),
                    name=f"stats_{tag}",
                )
                parse_stats = workflow.add(
                    ParseMrStatsStdout(stdout=stats.stdout),
                    name=f"parse_stats_{tag}",
                )
                values = parse_stats.out
                metrics_data["mean"][vial_name].append(values[0])
                metrics_data["median"][vial_name].append(values[1])
                metrics_data["std"][vial_name].append(values[2])
                metrics_data["min"][vial_name].append(values[3])
                metrics_data["max"][vial_name].append(values[4])

        for metric_name, vial_data in metrics_data.items():
            csv_file = (
                metrics_dir / f"{session_name}_{contrast_name}_{metric_name}_matrix.csv"
            )
            rows = [
                {"vial": vn, **{f"{clean_name}_vol{i}": v for i, v in enumerate(vals)}}
                for vn, vals in vial_data.items()
            ]
            pd.DataFrame(rows).to_csv(csv_file, index=False)
            logger.info("Saved: %s", csv_file.name)

    return str(metrics_dir)


@workflow.define
def TransformContrastsToTemplateSpace(
    contrast_files: list[str],
    transform_matrix: str,
    rotation_matrix_file: str | None,
    iteration: int,
    template_phantom: str,
    tmp_dir: str,
    output_dir: str,
) -> str:
    """
    Forward-transform every contrast image into template space.

    For each contrast:
    1. If ``iteration > 1``, **MrTransform** applies the pre-registration
       rotation (concrete workflow input → evaluated at static-graph-build
       time).
    2. **MrInfo** + **ParseMrInfoSize** detect dimensionality (task outputs
       are concrete at runtime after the for-loop triggers the fallback).
    3. If single-slice, **MrGrid** pads the z-axis.
    4. **ApplyTransforms** applies the forward ANTs affine.
    5. **CopyFile** writes the result to the output directory.

    Returns the template-space output directory path.
    """
    from fileformats.medimage import NiftiGz
    from fileformats.generic import File

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for contrast_file in contrast_files:
        contrast_path = Path(contrast_file)
        contrast_name = contrast_path.stem.replace(".nii", "")
        logger.info("Transforming: %s", contrast_path.name)

        source = contrast_file

        if iteration > 1 and rotation_matrix_file:
            rot = workflow.add(
                MrTransform(
                    in_file=NiftiGz(contrast_file),
                    linear=File(rotation_matrix_file),
                    out_file=f"{contrast_name}_rotated.nii.gz",
                    interp="linear",
                    force=True,
                ),
                name=f"rotate_{contrast_name}",
            )
            source = rot.out_file

        info = workflow.add(
            MrInfo(image_=[source], size=True, quiet=True),
            name=f"info_{contrast_name}",
        )
        parse_size = workflow.add(
            ParseMrInfoSize(stdout=info.stdout),
            name=f"parse_size_{contrast_name}",
        )

        transform_input = source

        if parse_size.is_single_slice:
            pad = workflow.add(
                MrGrid(
                    in_file=source,
                    operation="pad",
                    axis=[(2, (1, 1))],
                    out_file=f"{contrast_name}_padded.nii.gz",
                    force=True,
                ),
                name=f"pad_{contrast_name}",
            )
            transform_input = pad.out_file

        at = workflow.add(
            ApplyTransforms(
                input_image=transform_input,
                reference_image=NiftiGz(template_phantom),
                transforms=[transform_matrix],
                input_image_type=3 if parse_size.is_4d else 0,
                interpolation="Linear",
                out_postfix="_warped",
            ),
            name=f"at_{contrast_name}",
        )

        copy_file = workflow.add(
            CopyFile(
                src=at.output_image,
                dst=str(output_path / contrast_path.name),
            ),
            name=f"copy_{contrast_name}",
        )

        logger.info("%s -> template space", contrast_path.name)
        copied_files.append(copy_file.out)

    @python.define
    def CommonPath(file_paths: list[File]) -> Path:
        return Path(os.path.commonpath(file_paths))

    common_path = workflow.add(CommonPath(copied_files))

    logger.info("All contrasts saved to: %s", output_path)
    return common_path.out
