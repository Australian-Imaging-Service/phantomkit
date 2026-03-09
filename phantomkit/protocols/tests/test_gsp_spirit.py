"""Unit tests for phantomkit.workflow."""

from pathlib import Path


from phantomkit.protocols.gsp_spirit import (
    GetContrastFiles,
    GetVialMasks,
    PrepareSessionPaths,
)

# ── PrepareSessionPaths ───────────────────────────────────────────────────────


def test_prepare_session_paths_creates_directories(tmp_path: Path):
    session_dir = tmp_path / "20240101"
    session_dir.mkdir()
    image = session_dir / "t1.nii.gz"
    image.touch()

    out = PrepareSessionPaths(
        input_image=str(image), output_base_dir=str(tmp_path / "output")
    )()

    assert out.session_name == "20240101"
    assert Path(out.tmp_dir).exists()
    assert Path(out.vial_dir).exists()
    assert Path(out.metrics_dir).exists()
    assert Path(out.images_template_space_dir).exists()
    assert out.scanner_space_image.endswith("TemplatePhantom_ScannerSpace.nii.gz")


# ── GetVialMasks ──────────────────────────────────────────────────────────────


def test_get_vial_masks_returns_sorted_list(tmp_path: Path):
    vials_dir = tmp_path / "VialsLabelled"
    vials_dir.mkdir()
    for name in ["VialC.nii.gz", "VialA.nii.gz", "VialB.nii.gz"]:
        (vials_dir / name).touch()

    out = GetVialMasks(template_dir=str(tmp_path))()

    assert [Path(p).name for p in out.out] == [
        "VialA.nii.gz",
        "VialB.nii.gz",
        "VialC.nii.gz",
    ]


def test_get_vial_masks_empty_directory(tmp_path: Path):
    (tmp_path / "VialsLabelled").mkdir()
    out = GetVialMasks(template_dir=str(tmp_path))()
    assert out.out == []


# ── GetContrastFiles ──────────────────────────────────────────────────────────


def test_get_contrast_files_returns_nifti_siblings(tmp_path: Path):
    session_dir = tmp_path / "session01"
    session_dir.mkdir()
    for name in ["t2.nii.gz", "t1.nii.gz", "notes.txt"]:
        (session_dir / name).touch()
    image = session_dir / "t1.nii.gz"

    out = GetContrastFiles(input_image=str(image))()

    assert [Path(p).name for p in out.out] == ["t1.nii.gz", "t2.nii.gz"]


def test_get_contrast_files_excludes_non_nifti(tmp_path: Path):
    session_dir = tmp_path / "session01"
    session_dir.mkdir()
    (session_dir / "scan.nii.gz").touch()
    (session_dir / "report.pdf").touch()
    image = session_dir / "scan.nii.gz"

    out = GetContrastFiles(input_image=str(image))()

    assert len(out.out) == 1
    assert Path(out.out[0]).name == "scan.nii.gz"


# ── Top-level workflow graph construction (no execution) ──────────────────────


def test_gsp_spirit_build():
    from phantomkit.protocols.gsp_spirit import GspSpiritAnalysis

    wf = GspSpiritAnalysis(
        input_image="/fake/session/t1.nii.gz",
        template_dir="/fake/template",
        output_base_dir="/fake/output",
        rotation_library_file="/fake/rotations.txt",
    )
    assert wf is not None


def test_gsp_spirit_batch_build():
    from phantomkit.protocols.gsp_spirit import GspSpiritAnalysisBatch

    wf = GspSpiritAnalysisBatch(
        input_images=["/fake/session1/t1.nii.gz", "/fake/session2/t1.nii.gz"],
        template_dir="/fake/template",
        output_base_dir="/fake/output",
        rotation_library_file="/fake/rotations.txt",
    )
    assert wf is not None
