"""Unit tests for phantomkit.metrics."""

from pathlib import Path


from phantomkit.metrics import (
    CopyFile,
    GatherList,
    ParseMrInfoSize,
    PrepVialTransformPaths,
    TransformContrastsToTemplateSpace,
    TransformVialsToSubjectSpace,
    ExtractMetricsFromContrasts,
)

# ── ParseMrInfoSize ───────────────────────────────────────────────────────────


def test_parse_mrinfo_size_3d():
    out = ParseMrInfoSize(stdout="256 256 50")()
    assert out.is_4d is False
    assert out.is_single_slice is False
    assert out.nvols == 1


def test_parse_mrinfo_size_4d():
    out = ParseMrInfoSize(stdout="256 256 50 10")()
    assert out.is_4d is True
    assert out.is_single_slice is False
    assert out.nvols == 10


def test_parse_mrinfo_size_single_slice():
    out = ParseMrInfoSize(stdout="256 256 1")()
    assert out.is_4d is False
    assert out.is_single_slice is True


# ── GatherList ────────────────────────────────────────────────────────────────


def test_gather_list_preserves_order():
    items = ["/a/b.nii.gz", "/c/d.nii.gz", "/e/f.nii.gz"]
    out = GatherList(items=items)()
    assert out.out == items


def test_gather_list_empty():
    out = GatherList(items=[])()
    assert out.out == []


# ── CopyFile ──────────────────────────────────────────────────────────────────


def test_copy_file(tmp_path: Path):
    src = tmp_path / "source.txt"
    src.write_text("hello")
    dst = tmp_path / "dest.txt"
    out = CopyFile(src=str(src), dst=str(dst))()
    assert out.out == str(dst)
    assert dst.read_text() == "hello"


# ── PrepVialTransformPaths ────────────────────────────────────────────────────


def test_prep_vial_transform_paths(tmp_path: Path):
    mask = tmp_path / "VialA.nii.gz"
    mask.touch()
    vial_dir = tmp_path / "vials"
    out = PrepVialTransformPaths(vial_mask=str(mask), output_vial_dir=str(vial_dir))()
    assert out.vial_name == "VialA"
    assert out.vial_mask_out == str(mask)
    assert out.output_path.endswith("VialA.nii.gz")
    assert Path(vial_dir).exists()


# ── Workflow graph construction (no execution) ────────────────────────────────


def test_transform_vials_to_subject_space_builds():
    wf = TransformVialsToSubjectSpace(
        vial_masks=["/fake/VialA.nii.gz"],
        reference_image="/fake/ref.nii.gz",
        transform_matrix="/fake/affine.mat",
        rotation_matrix_file=None,
        iteration=1,
        output_vial_dir="/fake/vials",
    )
    assert wf is not None


def test_extract_metrics_from_contrasts_builds_empty():
    # Empty contrast list — loop body never runs; graph trivially builds
    wf = ExtractMetricsFromContrasts(
        contrast_files=[],
        vial_masks=[],
        output_metrics_dir="/fake/metrics",
        session_name="test_session",
    )
    assert wf is not None


def test_transform_contrasts_to_template_space_builds_empty():
    wf = TransformContrastsToTemplateSpace(
        contrast_files=[],
        transform_matrix="/fake/affine.mat",
        rotation_matrix_file=None,
        iteration=1,
        template_phantom="/fake/template.nii.gz",
        tmp_dir="/fake/tmp",
        output_dir="/fake/output",
    )
    assert wf is not None
