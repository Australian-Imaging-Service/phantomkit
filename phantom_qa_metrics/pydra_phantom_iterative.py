#!/usr/bin/env python3
"""
Pydra workflow for phantom sub-metrics computation.
Translates compute_sub_metrics_ants_ss.sh into Pydra tasks and workflows.
No Docker required - uses local ANTs and MRtrix3 installations.
"""

import logging
import re
import subprocess
import os.path
from pathlib import Path

from pydra.compose import python, shell, workflow

# Existing pydra shell tasks from pydra-tasks-ants and pydra-tasks-mrtrix3
from pydra.tasks.ants.v2.resampling.apply_transforms import ApplyTransforms
from pydra.tasks.mrtrix3.v3_1.mrconvert import MrConvert
from pydra.tasks.mrtrix3.v3_1.mrgrid import MrGrid
from pydra.tasks.mrtrix3.v3_1.mrinfo import MrInfo
from pydra.tasks.mrtrix3.v3_1.mrmath import MrMath  # noqa: F401 — exported for callers
from pydra.tasks.mrtrix3.v3_1.mrstats import MrStats
from pydra.tasks.mrtrix3.v3_1.mrtransform import MrTransform

logger = logging.getLogger(__name__)


# ============================================================================
# Shell Task: ANTs SyN registration (full SyN — not in pydra-tasks-ants yet)
# ============================================================================


def _syn_warped_callable(output_dir, inputs, stdout, stderr):
    import os

    return os.path.abspath(inputs.output_prefix) + "Warped.nii.gz"


def _syn_inv_warped_callable(output_dir, inputs, stdout, stderr):
    import os

    return os.path.abspath(inputs.output_prefix) + "InverseWarped.nii.gz"


def _syn_matrix_callable(output_dir, inputs, stdout, stderr):
    import os

    return os.path.abspath(inputs.output_prefix) + "0GenericAffine.mat"


@shell.define
class RegistrationSynN(shell.Task["RegistrationSynN.Outputs"]):
    """
    Wrapper for ``antsRegistrationSyN.sh`` (full SyN, not SyNQuick).

    Note: pydra-tasks-ants only provides RegistrationSynQuick; this task
    covers the non-quick variant used for phantom registration.

    Examples
    --------
    >>> task = RegistrationSynN(
    ...     fixed_image="template.nii.gz",
    ...     moving_image="subject.nii.gz",
    ...     output_prefix="/tmp/reg_",
    ... )
    >>> task.cmdline
    'antsRegistrationSyN.sh -d 3 -f template.nii.gz -j 1 -m subject.nii.gz -n 8 -o /tmp/reg_ -t r'
    """

    executable = "antsRegistrationSyN.sh"

    dimension: int = shell.arg(
        help="Image dimension (2 or 3)", argstr="-d {dimension}", default=3
    )
    fixed_image: str = shell.arg(
        help="Fixed (reference/template) image path", argstr="-f {fixed_image}"
    )
    moving_image: str = shell.arg(
        help="Moving (subject) image path", argstr="-m {moving_image}"
    )
    output_prefix: str = shell.arg(
        help="Output filename prefix", argstr="-o {output_prefix}", default="transform"
    )
    transform_type: str = shell.arg(
        help="Transform type: t/r/a/s/sr/b/br",
        argstr="-t {transform_type}",
        default="r",
    )
    num_threads: int = shell.arg(
        help="Number of threads", argstr="-n {num_threads}", default=8
    )
    use_histogram_matching: int = shell.arg(
        help="Use histogram matching (0 or 1)",
        argstr="-j {use_histogram_matching}",
        default=1,
    )

    class Outputs(shell.Outputs):
        warped_image: str = shell.out(
            help="Warped moving image", callable=_syn_warped_callable
        )
        inverse_warped_image: str = shell.out(
            help="Inverse warped image", callable=_syn_inv_warped_callable
        )
        out_matrix: str = shell.out(
            help="Affine transform (.mat)", callable=_syn_matrix_callable
        )


# ============================================================================
# Internal helper functions
# Used inside Python tasks; not exposed as pydra tasks because they contain
# while-loop logic or are tiny utilities.
# ============================================================================


def _load_rotations(rotation_library_file: str) -> list[str]:
    """Load rotation matrix strings from a rotation library file."""
    rotations = []
    with open(rotation_library_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.search(r'"([^"]+)"', line)
            if match:
                rotations.append(match.group(1))
    return rotations


def _create_rotation_matrix_file(rotation_str: str, output_file: str) -> str:
    """Write a 9-element rotation string as a 4×4 affine matrix text file."""
    v = rotation_str.split()
    with open(output_file, "w") as f:
        f.write(f"{v[0]} {v[1]} {v[2]} 0\n")
        f.write(f"{v[3]} {v[4]} {v[5]} 0\n")
        f.write(f"{v[6]} {v[7]} {v[8]} 0\n")
        f.write("0 0 0 1\n")
    return output_file


@python.define
def ParseMrStatsStdout(stdout: str) -> list[float]:
    """Parse the whitespace-separated numbers written by mrstats to stdout."""
    return [float(x) for x in stdout.strip().split()]


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


@python.define(outputs=["vial_name", "regridded_path", "vial_mask_out"])
def PrepVialCheckPaths(vial_mask: str, tmp_dir: str) -> tuple[str, str, str]:
    """Derive vial name and regridded-mask output path; pass vial_mask through."""
    from pathlib import Path

    vial_name = (
        Path(vial_mask)
        .name.replace(".nii.gz", "")
        .replace(".nii", "")
        .replace("Vial", "")
        .replace("vial", "")
        .strip()
    )
    tmp = Path(tmp_dir) / "check_reg"
    tmp.mkdir(parents=True, exist_ok=True)
    return vial_name, str(tmp / f"{vial_name}_regridded.nii.gz"), vial_mask


@python.define(outputs=["vial_name", "tmp_path", "output_path", "vial_mask_out"])
def PrepVialTransformPaths(
    vial_mask: str, output_vial_dir: str
) -> tuple[str, str, str, str]:
    """Derive per-vial output paths and pass vial_mask through."""
    from pathlib import Path

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


@python.define(outputs=["is_valid"])
def AggregateVialCheck(
    vial_names: list[str],
    means_stds: list[list[float]],
) -> bool:
    """
    Given per-vial [mean, std] pairs, check intensity-ranking criteria:
      - No vial std > 50
      - High-intensity vials A, O, Q are in top-5 by mean
      - Low-intensity vials S, D, P are in bottom-5 by mean
    """
    vial_means = {name: vals[0] for name, vals in zip(vial_names, means_stds)}
    high_std = [(n, vals[1]) for n, vals in zip(vial_names, means_stds) if vals[1] > 50]
    failures: list[str] = []

    if high_std:
        failures.append(
            f"High std in {len(high_std)} vial(s): "
            + ", ".join(f"{n}={v:.1f}" for n, v in high_std)
        )

    sorted_vials = sorted(vial_means.items(), key=lambda x: x[1], reverse=True)
    top5 = [v[0] for v in sorted_vials[:5]]
    bottom5 = [v[0] for v in sorted_vials[-5:]]

    for v in ["A", "O", "Q"]:
        if v not in top5:
            failures.append(f"Required high-intensity vial {v!r} not in top 5")
    for v in ["S", "D", "P"]:
        if v not in bottom5:
            failures.append(f"Required low-intensity vial {v!r} not in bottom 5")

    if failures:
        logger.warning("Registration check FAILED:")
        for msg in failures:
            logger.warning("  %s", msg)
        return False

    logger.info("Registration check passed (top5=%s, bottom5=%s)", top5, bottom5)
    return True


# ============================================================================
# Python Task Definitions
# ============================================================================


@python.define(
    outputs=[
        "session_name",
        "output_dir",
        "tmp_dir",
        "vial_dir",
        "metrics_dir",
        "images_template_space_dir",
        "scanner_space_image",
    ]
)
def PrepareSessionPaths(
    input_image: str,
    output_base_dir: str,
) -> tuple[str, str, str, str, str, str, str]:
    """
    Derive all session output paths from the input image location, create the
    directories, and return the path strings for use by downstream tasks.
    """
    from pathlib import Path

    session_name = Path(input_image).parent.name
    output_dir = Path(output_base_dir) / session_name
    tmp_dir = output_dir / "tmp"
    vial_dir = output_dir / "vial_segmentations"
    metrics_dir = output_dir / "metrics"
    images_template_space_dir = output_dir / "images_template_space"
    scanner_space_image = str(output_dir / "TemplatePhantom_ScannerSpace.nii.gz")

    for d in [tmp_dir, vial_dir, metrics_dir, images_template_space_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return (
        session_name,
        str(output_dir),
        str(tmp_dir),
        str(vial_dir),
        str(metrics_dir),
        str(images_template_space_dir),
        scanner_space_image,
    )


@python.define
def GetVialMasks(template_dir: str) -> list[str]:
    """Return sorted list of vial mask paths from the VialsLabelled directory."""
    from pathlib import Path

    return sorted(
        str(m) for m in (Path(template_dir) / "VialsLabelled").glob("*.nii.gz")
    )


@python.define
def GetContrastFiles(input_image: str) -> list[str]:
    """Return sorted list of all NIfTI files in the same directory as input_image."""
    from pathlib import Path

    return sorted(str(f) for f in Path(input_image).parent.glob("*.nii.gz"))


@workflow.define
def CheckRegistration(
    warped_image: str,
    vial_masks: list[str],
    tmp_dir: str,
) -> bool:
    """
    Check whether a registration is correctly oriented.

    Split over each vial mask:
      1. **PrepVialCheckPaths** derives the output path.
      2. **MrGrid** regrids the vial mask to the warped image grid.
      3. **MrStats** + **ParseMrStatsStdout** compute mean and std.

    A combined **AggregateVialCheck** python task verifies intensity ranking
    across all vials.
    """
    from fileformats.medimage import NiftiGz

    prep = workflow.add(
        PrepVialCheckPaths(vial_mask=vial_masks, tmp_dir=tmp_dir).split("vial_mask"),
        name="prep",
    )

    regrid = workflow.add(
        MrGrid(
            in_file=prep.vial_mask_out,
            operation="regrid",
            template=NiftiGz(warped_image),
            out_file=prep.regridded_path,
            interp="nearest",
            quiet=True,
            force=True,
        ),
        name="regrid",
    )

    stats = workflow.add(
        MrStats(
            image_=NiftiGz(warped_image),
            mask=regrid.out_file,
            output=["mean", "std"],
            quiet=True,
        ),
        name="stats",
    )

    parse = workflow.add(
        ParseMrStatsStdout(stdout=stats.stdout),
        name="parse",
    )

    # Combine all per-vial results before aggregation
    agg = workflow.add(
        AggregateVialCheck(
            vial_names=prep.vial_name,
            means_stds=parse.out,
        ).combine("prep"),
        name="aggregate",
    )

    return agg.is_valid


@workflow.define(
    outputs=[
        "warped",
        "transform",
        "inverse_warped",
        "iteration",
        "rotated_input",
        "rotation_matrix_file",
    ]
)
def IterativeRegistration(
    input_image: str,
    template_phantom: str,
    rotation_library_file: str,
    vial_masks: list[str],
    session_name: str,
    tmp_dir: str,
) -> tuple[str, str, str, int, str, str | None]:
    """
    Iteratively register the input image to the template phantom.

    On each iteration:

    1. **RegistrationSynN** performs a rigid registration.
    2. **CheckRegistration** verifies the orientation via vial intensity ranking.
    3. If the check fails and rotations remain, **MrTransform** applies the
       next rotation and retries.

    The while-loop and file-reading trigger pydra's runtime fallback, so
    all node outputs are concrete values available for loop control.
    """
    from pathlib import Path
    from fileformats.medimage import NiftiGz
    from fileformats.generic import File

    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)

    # Reading the rotation library triggers the runtime fallback (rotation_library_file
    # is a lazy proxy at static-graph-build time).
    rotations = _load_rotations(rotation_library_file)

    iteration = 0
    correct_orientation = False
    current_input = input_image
    rotation_matrix_file: str | None = None
    warped: str | None = None
    transform: str | None = None
    inverse_warped: str | None = None

    while not correct_orientation and iteration < len(rotations):
        iteration += 1
        logger.info("=== Iteration %d ===", iteration)

        reg = workflow.add(
            RegistrationSynN(
                fixed_image=template_phantom,
                moving_image=current_input,
                output_prefix=str(tmp / f"{session_name}_Transformed{iteration}_"),
                transform_type="r",
                num_threads=8,
                use_histogram_matching=1,
            ),
            name=f"reg_{iteration}",
        )
        warped = reg.warped_image
        transform = reg.out_matrix
        inverse_warped = reg.inverse_warped_image

        check = workflow.add(
            CheckRegistration(
                warped_image=warped,
                vial_masks=vial_masks,
                tmp_dir=str(tmp / f"check_{iteration}"),
            ),
            name=f"check_{iteration}",
        )
        correct_orientation = check.out

        if correct_orientation:
            logger.info("Correct orientation found at iteration %d", iteration)
            break

        if iteration < len(rotations):
            logger.warning("Registration check failed, applying rotation %d", iteration)
            rotation_str = rotations[iteration]
            rotation_matrix_file = str(tmp / f"rotation_{iteration + 1}.txt")
            _create_rotation_matrix_file(rotation_str, rotation_matrix_file)

            rot = workflow.add(
                MrTransform(
                    in_file=NiftiGz(input_image),
                    linear=File(rotation_matrix_file),
                    out_file=str(
                        tmp / f"{session_name}_iteration{iteration + 1}.nii.gz"
                    ),
                    interp="nearest",
                    force=True,
                ),
                name=f"rotate_{iteration}",
            )
            current_input = rot.out_file

    if not correct_orientation:
        raise RuntimeError(
            f"Failed to find correct orientation after {iteration} attempts"
        )

    return (
        warped,
        transform,
        inverse_warped,
        iteration,
        current_input,
        rotation_matrix_file,
    )


@workflow.define
def SaveTemplateInScannerSpace(
    inverse_warped: str,
    rotation_matrix_file: str | None,
    iteration: int,
    output_path: str,
) -> str:
    """
    Save the template phantom warped into scanner (subject) space.

    Uses **MrConvert** (iteration == 1, no rotation) or **MrTransform** with
    the inverse rotation (iteration > 1).
    """
    from fileformats.medimage import NiftiGz
    from fileformats.generic import File

    if iteration == 1:
        node = workflow.add(
            MrConvert(
                in_file=NiftiGz(inverse_warped),
                out_file=output_path,
                quiet=True,
                force=True,
            ),
            name="convert",
        )
    else:
        node = workflow.add(
            MrTransform(
                in_file=NiftiGz(inverse_warped),
                linear=File(rotation_matrix_file),
                out_file=output_path,
                interp="nearest",
                inverse=True,
                force=True,
            ),
            name="transform",
        )
    return node.out_file


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
    from pathlib import Path
    import pandas as pd
    from fileformats.medimage import NiftiGz

    metrics_dir = Path(output_metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tmp_vol_dir = metrics_dir.parent / "tmp_vols"
    tmp_vol_dir.mkdir(parents=True, exist_ok=True)

    # Iterating over contrast_files (lazy at static-graph-build time) triggers
    # the runtime fallback; from here all values are concrete.
    for contrast_file in contrast_files:
        contrast_path = Path(contrast_file)
        contrast_name = contrast_path.stem
        clean_name = contrast_name.replace(".nii", "").replace(".gz", "")

        # ── MrInfo: detect volume count ───────────────────────────────────────
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

            # ── MrGrid: regrid vial mask to contrast image space ─────────────
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
                    # ── MrConvert: extract one volume from 4-D series ────────
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

                # ── MrStats + parse: per-vial statistics ──────────────────────
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
    from pathlib import Path
    from fileformats.medimage import NiftiGz
    from fileformats.generic import File

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tmp = Path(tmp_dir) / "template_space_contrasts"
    tmp.mkdir(parents=True, exist_ok=True)

    # Iterating over contrast_files (lazy at static-graph-build time) triggers
    # the runtime fallback; from here all values are concrete.
    copied_files = []
    for contrast_file in contrast_files:
        contrast_path = Path(contrast_file)
        contrast_name = contrast_path.stem.replace(".nii", "")
        logger.info("Transforming: %s", contrast_path.name)

        source = contrast_file

        # ── Optional pre-registration rotation (concrete workflow input) ──────
        if iteration > 1 and rotation_matrix_file:
            rot = workflow.add(
                MrTransform(
                    in_file=NiftiGz(contrast_file),
                    linear=File(rotation_matrix_file),
                    out_file=str(tmp / f"{contrast_name}_rotated.nii.gz"),
                    interp="linear",
                    force=True,
                ),
                name=f"rotate_{contrast_name}",
            )
            source = rot.out_file

        # ── Detect dimensionality (task outputs concrete at runtime) ──────────
        info = workflow.add(
            MrInfo(image_=[source], size=True, quiet=True),
            name=f"info_{contrast_name}",
        )
        parse_size = workflow.add(
            ParseMrInfoSize(stdout=info.stdout),
            name=f"parse_size_{contrast_name}",
        )

        transform_input = source

        # ── Pad degenerate single-slice inputs ────────────────────────────────
        if parse_size.is_single_slice:
            pad = workflow.add(
                MrGrid(
                    in_file=source,
                    operation="pad",
                    axis=[(2, (1, 1))],
                    out_file=str(tmp / f"{contrast_name}_padded.nii.gz"),
                    force=True,
                ),
                name=f"pad_{contrast_name}",
            )
            transform_input = pad.out_file

        # ── Forward ANTs affine transform ─────────────────────────────────────
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


@python.define
def GeneratePlots(
    contrast_files: list[str],
    metrics_dir: str,
    vial_dir: str,
    session_name: str,
    template_dir: str,
) -> None:
    """
    Generate per-contrast vial intensity scatter plots and parametric map
    plots (IR/TE) by calling the external Python plotting scripts.

    Vial mask overlays are built by combining masks with **MrGrid**
    (pydra shell task via ``pydra.Submitter``) and then mrcat/mrmath
    (via subprocess, without piping — using intermediate files).
    """
    import pydra
    import re as _re
    from pathlib import Path
    from fileformats.medimage import NiftiGz

    metrics_path = Path(metrics_dir)
    vial_path = Path(vial_dir)
    template_path = Path(template_dir)

    plot_vial_script = template_path.parent / "Functions" / "plot_vial_intensity.py"
    if not plot_vial_script.exists():
        logger.warning("Plotting script not found: %s, skipping", plot_vial_script)
        return

    tmp_vial_dir = vial_path / "tmp"
    tmp_vial_dir.mkdir(exist_ok=True)
    vial_masks_list = sorted(str(m) for m in vial_path.glob("*.nii.gz"))

    def _build_roi_overlay(reference_image: str, prefix: str) -> str:
        """Regrid each vial to the reference image and combine into one mask."""
        regridded: list[str] = []
        for vial_mask in vial_masks_list:
            vial_name = Path(vial_mask).name.replace(".nii.gz", "").replace(".nii", "")
            out = str(tmp_vial_dir / f"{prefix}_{vial_name}.nii.gz")
            grid_task = MrGrid(
                in_file=NiftiGz(vial_mask),
                operation="regrid",
                template=NiftiGz(reference_image),
                out_file=out,
                force=True,
            )
            with pydra.Submitter(plugin="serial") as sub:
                sub(grid_task)
            regridded.append(str(grid_task.result().output.out_file))

        roi_overlay = str(tmp_vial_dir / f"{prefix}_VialsCombined.nii.gz")
        cat_tmp = str(tmp_vial_dir / f"{prefix}_cat_tmp.nii.gz")
        if regridded:
            subprocess.run(
                ["mrcat"] + regridded + [cat_tmp, "-axis", "3", "-force"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["mrmath", cat_tmp, "max", roi_overlay, "-axis", "3", "-force"],
                check=True,
                capture_output=True,
            )
        return roi_overlay

    def _mrview_screenshot(
        contrast_file: str, roi_overlay: str, out_png: str
    ) -> str | None:
        result = subprocess.run(
            [
                "mrview",
                str(contrast_file),
                "-mode",
                "1",
                "-plane",
                "2",
                "-roi.load",
                roi_overlay,
                "-roi.colour",
                "1,0,0",
                "-roi.opacity",
                "1",
                "-comments",
                "0",
                "-noannotations",
                "-fullscreen",
                "-capture.folder",
                str(Path(out_png).parent),
                "-capture.prefix",
                Path(out_png).stem,
                "-capture.grab",
                "-exit",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning("mrview screenshot failed: %s", result.stderr)
            return None
        actual = str(Path(out_png).parent / f"{Path(out_png).stem}0000.png")
        return actual if Path(actual).exists() else None

    # Per-contrast scatter plots
    for contrast_file in contrast_files:
        contrast_path = Path(contrast_file)
        contrast_name = contrast_path.stem
        mean_csv = metrics_path / f"{session_name}_{contrast_name}_mean_matrix.csv"
        std_csv = metrics_path / f"{session_name}_{contrast_name}_std_matrix.csv"
        if not mean_csv.exists():
            continue

        output_plot = metrics_path / f"{session_name}_{contrast_name}_PLOTmeanstd.png"
        roi_overlay = (
            _build_roi_overlay(contrast_file, contrast_name) if vial_masks_list else ""
        )
        screenshot = (
            _mrview_screenshot(
                contrast_file,
                roi_overlay,
                str(tmp_vial_dir / f"{contrast_name}_roi_overlay.png"),
            )
            if roi_overlay
            else None
        )

        cmd = [
            "python",
            str(plot_vial_script),
            str(mean_csv),
            "scatter",
            "--std_csv",
            str(std_csv),
            "--output",
            str(output_plot),
        ]
        if screenshot and Path(screenshot).exists():
            cmd.extend(["--roi_image", screenshot])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Generated plot: %s", output_plot.name)
        else:
            logger.warning("Plot failed for %s: %s", contrast_name, result.stderr)

    # Parametric map plots (IR and TE)
    def _matches(stem: str, token: str) -> bool:
        return bool(_re.search(rf"(?<![a-z0-9]){token}(?![a-z0-9])", stem.lower()))

    for contrast_type, script_name, suffix in [
        ("ir", "plot_maps_ir.py", "ir_map_PLOTmeanstd_TEmapping.png"),
        ("te", "plot_maps_TE.py", "TE_map_PLOTmeanstd_TEmapping.png"),
    ]:
        matching = [f for f in contrast_files if _matches(Path(f).stem, contrast_type)]
        if not matching:
            continue

        plot_script = template_path.parent / script_name
        if not plot_script.exists():
            logger.warning(
                "%s plotting script not found: %s", contrast_type.upper(), plot_script
            )
            continue

        output_plot = metrics_path / f"{session_name}_{suffix}"
        roi_base = str(tmp_vial_dir / f"roi_overlay_{contrast_type}.png")
        roi_file = (
            _build_roi_overlay(matching[0], contrast_type) if vial_masks_list else ""
        )
        screenshot = (
            _mrview_screenshot(matching[0], roi_file, roi_base) if roi_file else None
        )
        roi_image_arg = (
            screenshot if screenshot and Path(screenshot).exists() else roi_base
        )

        cmd = (
            ["python3", str(plot_script)]
            + matching
            + [
                "-m",
                str(metrics_path),
                "-o",
                str(output_plot),
                "--roi_image",
                roi_image_arg,
            ]
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Generated %s map plot", contrast_type.upper())
        else:
            logger.warning(
                "%s map plot failed: %s", contrast_type.upper(), result.stderr
            )


@python.define
def Cleanup(dirs: list[str]) -> None:
    """Remove temporary directories."""
    import shutil
    from pathlib import Path

    for d in dirs:
        p = Path(d)
        if p.exists():
            try:
                shutil.rmtree(p)
                logger.info("Removed: %s", p.name)
            except Exception as e:
                logger.warning("Could not remove %s: %s", p.name, e)


# ============================================================================
# Workflow Definitions
# ============================================================================


@workflow.define(
    outputs=[
        "metrics_dir",
        "vial_dir",
        "images_template_space_dir",
        "scanner_space_image",
    ]
)
def PhantomSessionWorkflow(
    input_image: str,
    template_dir: str,
    output_base_dir: str,
    rotation_library_file: str,
) -> tuple[str, str, str, str]:
    """
    Pydra workflow for processing a single phantom MRI session.

    Workflow steps
    --------------
    1. **PrepareSessionPaths**  — derive output paths and create directories
    2. **GetVialMasks**         — list template vial mask files
    3. **GetContrastFiles**     — list all contrast NIfTI files in the session
    4. **IterativeRegistration** — iterative ANTs registration with orientation
       search (RegistrationSynN + CheckRegistration + MrTransform)
    5. **SaveTemplateInScannerSpace** — warp template back to subject space
       (MrConvert or MrTransform)
    6. **TransformVialsToSubjectSpace** — project vial masks into subject space
       (ApplyTransforms + MrTransform/MrConvert)
    7. **ExtractMetricsFromContrasts** — compute per-vial statistics; writes CSVs
       (MrGrid + MrConvert + MrStats)
    8. **TransformContrastsToTemplateSpace** — forward-warp all contrasts
       (MrTransform + MrGrid + ApplyTransforms)
    9. **GeneratePlots** — scatter and parametric map plots
       (MrGrid for ROI overlays, external Python scripts for plots)
    10. **Cleanup** — remove temporary directories
    """
    template_phantom = str(Path(template_dir) / "ImageTemplate.nii.gz")

    paths = workflow.add(
        PrepareSessionPaths(
            input_image=input_image,
            output_base_dir=output_base_dir,
        ),
        name="prepare_paths",
    )

    vial_masks = workflow.add(
        GetVialMasks(template_dir=template_dir),
        name="get_vial_masks",
    )

    contrast_files = workflow.add(
        GetContrastFiles(input_image=input_image),
        name="get_contrast_files",
    )

    reg = workflow.add(
        IterativeRegistration(
            input_image=input_image,
            template_phantom=template_phantom,
            rotation_library_file=rotation_library_file,
            vial_masks=vial_masks.out,
            session_name=paths.session_name,
            tmp_dir=paths.tmp_dir,
        ),
        name="iterative_registration",
    )

    scanner_template = workflow.add(
        SaveTemplateInScannerSpace(
            inverse_warped=reg.inverse_warped,
            rotation_matrix_file=reg.rotation_matrix_file,
            iteration=reg.iteration,
            output_path=paths.scanner_space_image,
        ),
        name="save_template_scanner_space",
    )

    transform_vials = workflow.add(
        TransformVialsToSubjectSpace(
            vial_masks=vial_masks.out,
            reference_image=input_image,
            transform_matrix=reg.transform,
            rotation_matrix_file=reg.rotation_matrix_file,
            iteration=reg.iteration,
            output_vial_dir=paths.vial_dir,
        ),
        name="transform_vials",
    )

    extract_metrics = workflow.add(
        ExtractMetricsFromContrasts(
            contrast_files=contrast_files.out,
            vial_masks=transform_vials.out,
            output_metrics_dir=paths.metrics_dir,
            session_name=paths.session_name,
        ),
        name="extract_metrics",
    )

    warp_contrasts = workflow.add(
        TransformContrastsToTemplateSpace(
            contrast_files=contrast_files.out,
            transform_matrix=reg.transform,
            rotation_matrix_file=reg.rotation_matrix_file,
            iteration=reg.iteration,
            template_phantom=template_phantom,
            tmp_dir=paths.tmp_dir,
            output_dir=paths.images_template_space_dir,
        ),
        name="warp_contrasts_to_template",
    )

    workflow.add(
        GeneratePlots(
            contrast_files=contrast_files.out,
            metrics_dir=extract_metrics.out,
            vial_dir=paths.vial_dir,
            session_name=paths.session_name,
            template_dir=template_dir,
        ),
        name="generate_plots",
    )

    workflow.add(
        Cleanup(dirs=[paths.tmp_dir]),
        name="cleanup",
    )

    return (
        extract_metrics.out,
        paths.vial_dir,
        warp_contrasts.out,
        scanner_template.out,
    )


@workflow.define(outputs=["results"])
def BatchWorkflow(
    input_images: list[str],
    template_dir: str,
    output_base_dir: str,
    rotation_library_file: str,
) -> list:
    """
    Pydra workflow for batch-processing multiple phantom sessions in parallel.

    Each session is processed by PhantomSessionWorkflow; the sessions are
    split across input_images and results collected back into a list.
    """
    process = workflow.add(
        PhantomSessionWorkflow(
            input_image=input_images,
            template_dir=template_dir,
            output_base_dir=output_base_dir,
            rotation_library_file=rotation_library_file,
        )
        .split("input_image")
        .combine("input_image"),
        name="process_sessions",
    )

    return process.metrics_dir


# ============================================================================
# Command-line Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    import pydra

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Pydra phantom processing workflow (no Docker)"
    )
    subparsers = parser.add_subparsers(dest="command")

    single_parser = subparsers.add_parser("single", help="Process one session")
    single_parser.add_argument("input_image", help="Input NIfTI file")
    single_parser.add_argument("--template-dir", required=True)
    single_parser.add_argument("--output-dir", required=True)
    single_parser.add_argument("--rotation-lib", required=True)

    batch_parser = subparsers.add_parser(
        "batch", help="Batch-process multiple sessions"
    )
    batch_parser.add_argument("data_dir", help="Directory containing session folders")
    batch_parser.add_argument("--template-dir", required=True)
    batch_parser.add_argument("--output-dir", required=True)
    batch_parser.add_argument("--rotation-lib", required=True)
    batch_parser.add_argument("--pattern", default="*t1*mprage*.nii.gz")
    batch_parser.add_argument("--plugin", default="cf", choices=["cf", "serial"])

    args = parser.parse_args()

    if args.command == "single":
        wf = PhantomSessionWorkflow(
            input_image=args.input_image,
            template_dir=args.template_dir,
            output_base_dir=args.output_dir,
            rotation_library_file=args.rotation_lib,
        )
        with pydra.Submitter(plugin="serial") as sub:
            sub(wf)
        result = wf.result()
        logger.info("Done! Metrics: %s", result.output.metrics_dir)

    elif args.command == "batch":
        images = sorted(
            str(img) for img in Path(args.data_dir).glob(f"*/{args.pattern}")
        )
        if not images:
            logger.error("No images found!")
            raise SystemExit(1)

        logger.info("Found %d images:", len(images))
        for img in images:
            logger.info("  %s", img)

        wf = BatchWorkflow(
            input_images=images,
            template_dir=args.template_dir,
            output_base_dir=args.output_dir,
            rotation_library_file=args.rotation_lib,
        )
        with pydra.Submitter(plugin=args.plugin) as sub:
            sub(wf)
        logger.info("All sessions processed! Output: %s", args.output_dir)

    else:
        parser.print_help()
