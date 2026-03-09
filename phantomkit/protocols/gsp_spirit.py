"""
Top-level pydra workflows for phantom QA processing.

PhantomSessionWorkflow processes a single session end-to-end;
BatchWorkflow runs multiple sessions in parallel.
"""

import logging
from pathlib import Path

from fileformats.generic import Directory, File
from fileformats.medimage import Nifti
from pydra.compose import python, workflow

from phantomkit.registration import (
    IterativeRegistration,
    SaveTemplateInScannerSpace,
)
from phantomkit.metrics import (
    TransformVialsToSubjectSpace,
    ExtractMetricsFromContrasts,
    TransformContrastsToTemplateSpace,
)
from phantomkit.plotting.visualization import GeneratePlots, Cleanup

logger = logging.getLogger(__name__)


# ============================================================================
# Session preparation tasks
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
    return sorted(
        str(m) for m in (Path(template_dir) / "VialsLabelled").glob("*.nii.gz")
    )


@python.define
def GetContrastFiles(input_image: str) -> list[str]:
    """Return sorted list of all NIfTI files in the same directory as input_image."""
    return sorted(str(f) for f in Path(input_image).parent.glob("*.nii.gz"))


# ============================================================================
# Top-level workflows
# ============================================================================


@workflow.define(
    outputs=[
        "metrics_dir",
        "vial_dir",
        "images_template_space_dir",
        "scanner_space_image",
    ]
)
def GspSpiritAnalysis(
    input_image: Nifti,
    template_dir: Directory,
    rotation_library_file: File,
    output_base_dir: Directory | None = None,
) -> tuple[Directory, str, str, str]:
    """
    Pydra workflow for processing a single GSP SPIRIT phantom MRI session.

    Registers the phantom scan to the GSP SPIRIT template using iterative ANTs
    SyN registration with an orientation search, extracts per-vial signal
    statistics for all contrast images, and generates publication-quality plots.

    Parameters
    ----------
    input_image : str
        Path to the primary NIfTI image for the session (e.g. the T1 MPRAGE).
        All NIfTI files in the same directory are treated as contrast images.
    template_dir : str
        Path to the GSP SPIRIT template directory.  Must contain
        ``ImageTemplate.nii.gz`` and a ``VialsLabelled/`` sub-directory of
        per-vial mask files.
    output_base_dir : str
        Root output directory.  A sub-directory named after the session
        (parent folder of *input_image*) is created automatically.
    rotation_library_file : str
        Path to a text file listing quoted ANTs rotation strings, one per line,
        used during the iterative orientation search.

    Returns
    -------
    metrics_dir : str
        Directory containing per-contrast CSV files of vial signal statistics
        (mean, median, std, min, max).
    vial_dir : str
        Directory containing the vial mask NIfTI files warped into subject
        (scanner) space.
    images_template_space_dir : str
        Directory containing all contrast images warped into template space.
    scanner_space_image : str
        Path to the template phantom image warped back into scanner space.

    Steps
    -----
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
def GspSpiritAnalysisBatch(
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
        GspSpiritAnalysis(
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
