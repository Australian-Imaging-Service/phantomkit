"""
Pydra tasks and workflows for phantom template registration.

Registers a phantom MRI scan to a reference template using ANTs SyN,
with an iterative orientation search over a rotation library.
"""

import logging
import re
from pathlib import Path

from fileformats.generic import File
from fileformats.medimage import NiftiGz
from pydra.compose import python, shell, workflow
from pydra.tasks.mrtrix3.v3_1 import MrConvert, MrGrid, MrStats, MrTransform

logger = logging.getLogger(__name__)


# ============================================================================
# Shell Task: ANTs SyN registration (full SyN — not in pydra-tasks-ants yet)
# ============================================================================


@shell.define
class RegistrationSynN(shell.Task["RegistrationSynN.Outputs"]):
    """
    Wrapper for ``antsRegistrationSyN.sh`` (full SyN, not SyNQuick).

    Parameters
    ----------
    fixed_image:
        The fixed (reference/template) image.
    moving_image:
        The moving (subject) image to register.
    output_prefix:
        Prefix for all output files produced by the script.
    transform_type:
        Transform type passed to ``-t``: ``'r'`` (rigid), ``'a'`` (affine),
        ``'s'`` (SyN), etc.
    num_threads:
        Number of ITK threads.
    use_histogram_matching:
        Pass ``-j 1`` to enable histogram matching.
    """

    executable = "antsRegistrationSyN.sh"

    fixed_image: NiftiGz = shell.arg(argstr="-f", help="Fixed image.")
    moving_image: NiftiGz = shell.arg(argstr="-m", help="Moving image.")
    output_prefix: Path = shell.arg(argstr="-o", help="Output prefix.")
    transform_type: str = shell.arg(
        default="s",
        argstr="-t",
        help="Transform type (r=rigid, a=affine, s=SyN).",
    )
    num_threads: int = shell.arg(
        default=1,
        argstr="-n",
        help="Number of ITK threads.",
    )
    use_histogram_matching: int = shell.arg(
        default=0,
        argstr="-j",
        help="Use histogram matching (0 or 1).",
    )

    class Outputs(shell.Outputs):
        warped_image: NiftiGz = shell.outarg(
            path_template="{output_prefix}Warped.nii.gz",
            help="Warped moving image in fixed space.",
        )
        inverse_warped_image: NiftiGz = shell.outarg(
            path_template="{output_prefix}InverseWarped.nii.gz",
            help="Inverse-warped fixed image in moving space.",
        )
        out_matrix: File = shell.outarg(
            path_template="{output_prefix}0GenericAffine.mat",
            help="Generic affine transform matrix (.mat).",
        )


# ============================================================================
# Internal helpers (used inside workflow functions)
# ============================================================================


def _load_rotations(rotation_library_file: File) -> list[str]:
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


def _create_rotation_matrix_file(rotation_str: str, output_file: Path) -> Path:
    """Write a 9-element rotation string as a 4×4 affine matrix text file."""
    v = rotation_str.split()
    with open(output_file, "w") as f:
        f.write(f"{v[0]} {v[1]} {v[2]} 0\n")
        f.write(f"{v[3]} {v[4]} {v[5]} 0\n")
        f.write(f"{v[6]} {v[7]} {v[8]} 0\n")
        f.write("0 0 0 1\n")
    return output_file


# ============================================================================
# Python Tasks
# ============================================================================


@python.define
def ParseMrStatsStdout(stdout: str) -> list[float]:
    """Parse the whitespace-separated numbers written by mrstats to stdout."""
    return [float(x) for x in stdout.strip().split()]


@python.define(outputs=["vial_name", "regridded_path", "vial_mask_out"])
def PrepVialCheckPaths(vial_mask: NiftiGz, tmp_dir: Path) -> tuple[str, Path, NiftiGz]:
    """Derive vial name and regridded-mask output path; pass vial_mask through."""
    from pathlib import Path as _Path

    vial_name = (
        _Path(vial_mask)
        .name.replace(".nii.gz", "")
        .replace(".nii", "")
        .replace("Vial", "")
        .replace("vial", "")
        .strip()
    )
    tmp = _Path(tmp_dir) / "check_reg"
    tmp.mkdir(parents=True, exist_ok=True)
    return vial_name, tmp / f"{vial_name}_regridded.nii.gz", vial_mask


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
# Registration Workflows
# ============================================================================


@workflow.define
def CheckRegistration(
    warped_image: NiftiGz,
    vial_masks: list[NiftiGz],
    tmp_dir: Path,
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
    prep = workflow.add(
        PrepVialCheckPaths(vial_mask=vial_masks, tmp_dir=tmp_dir).split("vial_mask"),
        name="prep",
    )

    regrid = workflow.add(
        MrGrid(
            in_file=prep.vial_mask_out,
            operation="regrid",
            template=warped_image,
            out_file=prep.regridded_path,
            interp="nearest",
            quiet=True,
            force=True,
        ),
        name="regrid",
    )

    stats = workflow.add(
        MrStats(
            image_=warped_image,
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
    image_to_register: NiftiGz,
    original_image: NiftiGz,
    applied_rotation: File | None,
    template_phantom: NiftiGz,
    rotations: list[str],
    vial_masks: list[NiftiGz],
    session_name: str,
    tmp_dir: Path,
    iteration: int,
) -> tuple[NiftiGz, File, NiftiGz, int, NiftiGz, File | None]:
    """
    Single step of the iterative registration loop.

    Runs **RegistrationSynN** then **CheckRegistration**.  If the check
    passes the results are returned directly.  If it fails and rotations
    remain, the next rotation is applied to *original_image* and this
    workflow calls itself recursively with ``iteration + 1``.
    """
    from pathlib import Path as _Path

    tmp = _Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    logger.info("=== Iteration %d ===", iteration)

    reg = workflow.add(
        RegistrationSynN(
            fixed_image=template_phantom,
            moving_image=image_to_register,
            output_prefix=tmp / f"{session_name}_Transformed{iteration}_",
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
            tmp_dir=tmp / "check",
        ),
        name="check",
    )

    if check.out:
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
        logger.warning("Registration check failed, applying rotation %d", iteration)
        next_rotation_file = tmp / f"rotation_{iteration + 1}.txt"
        _create_rotation_matrix_file(rotations[iteration], next_rotation_file)

        rot = workflow.add(
            MrTransform(
                in_file=original_image,
                linear=next_rotation_file,
                out_file=tmp / f"{session_name}_iteration{iteration + 1}.nii.gz",
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
                tmp_dir=tmp.parent / f"step_{iteration + 1}",
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
    input_image: NiftiGz,
    template_phantom: NiftiGz,
    rotation_library_file: File,
    vial_masks: list[NiftiGz],
    session_name: str,
    tmp_dir: Path,
) -> tuple[NiftiGz, File, NiftiGz, int, NiftiGz, File | None]:
    """
    Iteratively register the input image to the template phantom.

    Loads the rotation library (triggering pydra's runtime fallback), then
    delegates to **_RegistrationStep** which recurses until the orientation
    check passes or all rotations are exhausted.
    """
    from pathlib import Path as _Path

    tmp = _Path(tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)

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
            tmp_dir=tmp / "step_1",
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
    inverse_warped: NiftiGz,
    rotation_matrix_file: File | None,
    iteration: int,
    output_path: Path,
) -> NiftiGz:
    """
    Save the template phantom warped into scanner (subject) space.

    Uses **MrConvert** (iteration == 1, no rotation) or **MrTransform** with
    the inverse rotation (iteration > 1).
    """
    if iteration == 1:
        node = workflow.add(
            MrConvert(
                in_file=inverse_warped,
                out_file=output_path,
                quiet=True,
                force=True,
            ),
            name="convert",
        )
    else:
        node = workflow.add(
            MrTransform(
                in_file=inverse_warped,
                linear=rotation_matrix_file,
                out_file=output_path,
                interp="nearest",
                inverse=True,
                force=True,
            ),
            name="transform",
        )
    return node.out_file
