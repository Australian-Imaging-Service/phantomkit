#!/usr/bin/env python3
"""
Pydra workflow for phantom sub-metrics computation.
Translates compute_sub_metrics_ants_ss.sh into Pydra tasks and workflows.
No Docker required - uses local ANTs and MRtrix3 installations.
"""

import logging
import re
import os.path
from pathlib import Path

import click

from pydra.compose import python, shell, workflow

# Existing pydra shell tasks from pydra-tasks-ants and pydra-tasks-mrtrix3
from pydra.tasks.ants.v2.resampling.apply_transforms import ApplyTransforms
from pydra.tasks.mrtrix3.v3_1 import (
    MrConvert,
    MrGrid,
    MrInfo,
    MrStats,
    MrTransform,
    MrMath,
)  # noqa: F401 — exported for callers

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
def _RegistrationStep(
    image_to_register: str,
    original_image: str,
    applied_rotation: str | None,
    template_phantom: str,
    rotations: list[str],
    vial_masks: list[str],
    session_name: str,
    tmp_dir: str,
    iteration: int,
) -> tuple[str, str, str, int, str, str | None]:
    """
    Single step of the iterative registration loop.

    Runs **RegistrationSynN** then **CheckRegistration**.  If the check
    passes the results are returned directly.  If it fails and rotations
    remain, the next rotation is applied to *original_image* and this
    workflow calls itself recursively with ``iteration + 1``.
    """
    from pathlib import Path
    from fileformats.medimage import NiftiGz
    from fileformats.generic import File

    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    logger.info("=== Iteration %d ===", iteration)

    reg = workflow.add(
        RegistrationSynN(
            fixed_image=template_phantom,
            moving_image=image_to_register,
            output_prefix=str(tmp / f"{session_name}_Transformed{iteration}_"),
            transform_type="r",
            num_threads=8,
            use_histogram_matching=1,
        ),
        name="reg",
    )

    check = workflow.add(
        CheckRegistration(
            warped_image=reg.warped_image,
            vial_masks=vial_masks,
            tmp_dir=str(tmp / "check"),
        ),
        name="check",
    )

    if check.out:
        # Correct orientation found — return this iteration's results.
        logger.info("Correct orientation found at iteration %d", iteration)
        return (
            reg.warped_image,
            reg.out_matrix,
            reg.inverse_warped_image,
            iteration,
            image_to_register,
            applied_rotation,
        )
    elif iteration < len(rotations):
        # Apply the next rotation to the original image and recurse.
        logger.warning("Registration check failed, applying rotation %d", iteration)
        next_rotation_file = str(tmp / f"rotation_{iteration + 1}.txt")
        _create_rotation_matrix_file(rotations[iteration], next_rotation_file)

        rot = workflow.add(
            MrTransform(
                in_file=NiftiGz(original_image),
                linear=File(next_rotation_file),
                out_file=str(tmp / f"{session_name}_iteration{iteration + 1}.nii.gz"),
                interp="nearest",
                force=True,
            ),
            name="rotate",
        )

        next_step = workflow.add(
            _RegistrationStep(
                image_to_register=rot.out_file,
                original_image=original_image,
                applied_rotation=next_rotation_file,
                template_phantom=template_phantom,
                rotations=rotations,
                vial_masks=vial_masks,
                session_name=session_name,
                tmp_dir=str(tmp.parent / f"step_{iteration + 1}"),
                iteration=iteration + 1,
            ),
            name="next_step",
        )
        return (
            next_step.warped,
            next_step.transform,
            next_step.inverse_warped,
            next_step.iteration,
            next_step.rotated_input,
            next_step.rotation_matrix_file,
        )
    else:
        raise RuntimeError(
            f"Failed to find correct orientation after {iteration} attempts"
        )


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

    Loads the rotation library (triggering pydra's runtime fallback), then
    delegates to **_RegistrationStep** which recurses until the orientation
    check passes or all rotations are exhausted.
    """
    from pathlib import Path

    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)

    # Reading the file triggers the runtime fallback so all downstream
    # node outputs are concrete when _RegistrationStep runs.
    rotations = _load_rotations(rotation_library_file)

    step = workflow.add(
        _RegistrationStep(
            image_to_register=input_image,
            original_image=input_image,
            applied_rotation=None,
            template_phantom=template_phantom,
            rotations=rotations,
            vial_masks=vial_masks,
            session_name=session_name,
            tmp_dir=str(tmp / "step_1"),
            iteration=1,
        ),
        name="step",
    )
    return (
        step.warped,
        step.transform,
        step.inverse_warped,
        step.iteration,
        step.rotated_input,
        step.rotation_matrix_file,
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
                    out_file=f"{contrast_name}_rotated.nii.gz",
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
                    out_file=f"{contrast_name}_padded.nii.gz",
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

    # We need to have a node that collates all the output paths
    # to guarantee that they run before the parent directory
    # is returned
    @python.define
    def CommonPath(file_paths: list[File]) -> Path:
        return Path(os.path.commonpath(file_paths))

    common_path = workflow.add(CommonPath(copied_files))

    logger.info("All contrasts saved to: %s", output_path)
    return common_path.out


@shell.define
class MrCat(shell.Task["MrCat.Outputs"]):
    """Concatenate several images into one."""

    executable = "mrcat"

    in_files: list[str] = shell.arg(
        argstr="",
        position=1,
        help="the input image(s).",
    )
    out_file: str = shell.arg(
        argstr="",
        position=2,
        help="the output image.",
    )
    axis: int | None = shell.arg(
        default=None,
        argstr="-axis",
        help="specify axis along which to perform the concatenation.",
    )
    force: bool = shell.arg(
        default=False,
        argstr="-force",
        help="force image output, even if file exists.",
    )

    class Outputs(shell.Outputs):
        out_file: str = shell.outarg(
            argstr="",
            position=2,
            path_template="{out_file}",
            help="the output image.",
        )


@workflow.define
def BuildRoiOverlay(
    vial_masks: list[str], reference_image: str, prefix: str, tmp_dir: str
) -> str:
    """Regrid each vial mask to the reference image space and combine into one overlay."""
    from pathlib import Path
    from fileformats.medimage import NiftiGz

    tmp = Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    cat_tmp = str(tmp / f"{prefix}_cat_tmp.nii.gz")
    roi_overlay = str(tmp / f"{prefix}_VialsCombined.nii.gz")

    regridded: list[str] = []
    for vial_mask in vial_masks:  # triggers runtime fallback
        vial_name = Path(vial_mask).name.replace(".nii.gz", "").replace(".nii", "")
        out = str(tmp / f"{prefix}_{vial_name}.nii.gz")
        grid = workflow.add(
            MrGrid(
                in_file=NiftiGz(vial_mask),
                operation="regrid",
                template=NiftiGz(reference_image),
                out_file=out,
                interp="nearest",
                force=True,
            ),
            name=f"grid_{prefix}_{vial_name}",
        )
        regridded.append(grid.out_file)

    cat = workflow.add(
        MrCat(in_files=regridded, out_file=cat_tmp, axis=3, force=True),
        name=f"cat_{prefix}",
    )
    math = workflow.add(
        MrMath(in_file=[cat.out_file], operation="max", out_file=roi_overlay, axis=3),
        name=f"math_{prefix}",
    )
    return math.out_file


@shell.define
class MrView(shell.Task["MrView.Outputs"]):
    """Capture an mrview screenshot with an optional ROI overlay.

    mrview appends ``0000`` to ``capture_prefix`` when naming the output
    file, so the output is ``{capture_folder}/{capture_prefix}0000.png``.
    """

    executable = "mrview"

    image: str = shell.arg(
        argstr="",
        position=1,
        help="Input image to display.",
    )
    mode: int = shell.arg(
        default=1,
        argstr="-mode",
        help="Display mode (1 = ortho view).",
    )
    plane: int = shell.arg(
        default=2,
        argstr="-plane",
        help="Plane to display (0=sagittal, 1=coronal, 2=axial).",
    )
    roi_load: str | None = shell.arg(
        default=None,
        argstr="-roi.load",
        help="ROI overlay image to load.",
    )
    roi_colour: str | None = shell.arg(
        default=None,
        argstr="-roi.colour",
        help="ROI overlay colour as 'R,G,B'.",
    )
    roi_opacity: float | None = shell.arg(
        default=None,
        argstr="-roi.opacity",
        help="ROI overlay opacity (0–1).",
    )
    comments: int | None = shell.arg(
        default=None,
        argstr="-comments",
        help="Show image comments (0 = hide).",
    )
    noannotations: bool = shell.arg(
        default=False,
        argstr="-noannotations",
        help="Hide image annotations.",
    )
    fullscreen: bool = shell.arg(
        default=False,
        argstr="-fullscreen",
        help="Display fullscreen.",
    )
    capture_folder: str = shell.arg(
        argstr="-capture.folder",
        help="Directory to save the screenshot.",
    )
    capture_prefix: str = shell.arg(
        argstr="-capture.prefix",
        help="Filename prefix for the screenshot.",
    )
    capture_grab: bool = shell.arg(
        default=True,
        argstr="-capture.grab",
        help="Capture a screenshot.",
    )
    exit_after: bool = shell.arg(
        default=True,
        argstr="-exit",
        help="Exit mrview after capturing.",
    )

    class Outputs(shell.Outputs):
        out_png: str = shell.outarg(
            path_template="{capture_folder}/{capture_prefix}0000.png",
            help="Captured screenshot (mrview appends '0000' to the prefix).",
        )


@workflow.define
def GeneratePlots(
    contrast_files: list[str],
    metrics_dir: str,
    vial_dir: str,
    session_name: str,
) -> None:
    """
    Generate per-contrast vial intensity scatter plots and parametric map
    plots (IR/TE) using the plotting API functions directly as pydra tasks.

    Vial mask overlays are built using MrGrid + MrCat + MrMath pydra tasks.
    Screenshots use mrview via MrViewScreenshot.
    """
    import re as _re
    from pathlib import Path
    from phantom_qa_metrics.plot_vial_intensity import plot_vial_intensity
    from phantom_qa_metrics.plot_maps_ir import (
        plot_vial_means_std_pub_from_nifti as plot_ir,
    )
    from phantom_qa_metrics.plot_maps_TE import (
        plot_vial_means_std_pub_from_nifti as plot_te,
    )

    metrics_path = Path(metrics_dir)
    vial_path = Path(vial_dir)
    tmp_vial_dir = str(vial_path / "tmp")

    vial_masks_list = sorted(
        str(m) for m in vial_path.glob("*.nii.gz")
    )  # triggers runtime fallback

    # Per-contrast scatter plots
    for contrast_file in contrast_files:
        contrast_path = Path(contrast_file)
        contrast_name = contrast_path.stem
        mean_csv = str(metrics_path / f"{session_name}_{contrast_name}_mean_matrix.csv")
        std_csv = str(metrics_path / f"{session_name}_{contrast_name}_std_matrix.csv")
        if not Path(mean_csv).exists():
            continue

        output_plot = str(
            metrics_path / f"{session_name}_{contrast_name}_PLOTmeanstd.png"
        )

        roi_image: str | None = None
        if vial_masks_list:
            overlay = workflow.add(
                BuildRoiOverlay(
                    vial_masks=vial_masks_list,
                    reference_image=contrast_file,
                    prefix=contrast_name,
                    tmp_dir=tmp_vial_dir,
                ),
                name=f"overlay_{contrast_name}",
            )
            screenshot = workflow.add(
                MrView(
                    image=contrast_file,
                    roi_load=overlay.out,
                    roi_colour="1,0,0",
                    roi_opacity=1.0,
                    comments=0,
                    noannotations=True,
                    fullscreen=True,
                    capture_folder=tmp_vial_dir,
                    capture_prefix=f"{contrast_name}_roi_overlay",
                ),
                name=f"screenshot_{contrast_name}",
            )
            roi_image = screenshot.out_png

        workflow.add(
            python.define(plot_vial_intensity)(
                csv_file=mean_csv,
                plot_type="scatter",
                std_csv=std_csv,
                roi_image=roi_image,
                output=output_plot,
            ),
            name=f"scatter_{contrast_name}",
        )

    # Parametric map plots (IR and TE)
    def _matches(stem: str, token: str) -> bool:
        return bool(_re.search(rf"(?<![a-z0-9]){token}(?![a-z0-9])", stem.lower()))

    for contrast_type, plot_fn, suffix in [
        ("ir", plot_ir, "ir_map_PLOTmeanstd_TEmapping.png"),
        ("te", plot_te, "TE_map_PLOTmeanstd_TEmapping.png"),
    ]:
        matching = [f for f in contrast_files if _matches(Path(f).stem, contrast_type)]
        if not matching:
            continue

        output_plot = str(metrics_path / f"{session_name}_{suffix}")

        roi_image_arg = str(Path(tmp_vial_dir) / f"roi_overlay_{contrast_type}.png")
        if vial_masks_list:
            overlay = workflow.add(
                BuildRoiOverlay(
                    vial_masks=vial_masks_list,
                    reference_image=matching[0],
                    prefix=contrast_type,
                    tmp_dir=tmp_vial_dir,
                ),
                name=f"overlay_{contrast_type}",
            )
            screenshot = workflow.add(
                MrView(
                    image=matching[0],
                    roi_load=overlay.out,
                    roi_colour="1,0,0",
                    roi_opacity=1.0,
                    comments=0,
                    noannotations=True,
                    fullscreen=True,
                    capture_folder=tmp_vial_dir,
                    capture_prefix=f"roi_overlay_{contrast_type}",
                ),
                name=f"screenshot_{contrast_type}",
            )
            roi_image_arg = screenshot.out_png

        workflow.add(
            python.define(plot_fn)(
                contrast_files=matching,
                metric_dir=metrics_dir,
                output_file=output_plot,
                roi_image=roi_image_arg,
            ),
            name=f"map_plot_{contrast_type}",
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


@click.group()
def main():
    """Pydra phantom processing workflow (no Docker)."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )


@main.command()
@click.argument("input_image")
@click.option("--template-dir", required=True, help="Template phantom directory.")
@click.option("--output-dir", required=True, help="Output base directory.")
@click.option("--rotation-lib", required=True, help="Rotation library file.")
def single(input_image, template_dir, output_dir, rotation_lib):
    """Process one session."""
    import pydra

    wf = PhantomSessionWorkflow(
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
def batch(data_dir, template_dir, output_dir, rotation_lib, pattern, plugin):
    """Batch-process multiple sessions."""
    import pydra

    images = sorted(str(img) for img in Path(data_dir).glob(f"*/{pattern}"))
    if not images:
        logger.error("No images found!")
        raise SystemExit(1)

    logger.info("Found %d images:", len(images))
    for img in images:
        logger.info("  %s", img)

    wf = BatchWorkflow(
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
