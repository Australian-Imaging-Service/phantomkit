"""Unit tests for phantomkit.cli."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from fileformats.medimage import NiftiGz

from phantomkit.cli import main


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Top-level help
# ---------------------------------------------------------------------------


def test_main_help(runner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "plot" in result.output
    assert "list" in result.output


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_shows_gsp_spirit(runner) -> None:
    result = runner.invoke(main, ["list"])
    assert result.exit_code == 0
    assert "gsp-spirit" in result.output


def test_list_shows_batch_supported(runner) -> None:
    result = runner.invoke(main, ["list"])
    assert "batch supported" in result.output


# ---------------------------------------------------------------------------
# run group structure
# ---------------------------------------------------------------------------


def test_run_help(runner) -> None:
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "gsp-spirit" in result.output


def test_run_gsp_spirit_help(runner) -> None:
    result = runner.invoke(main, ["run", "gsp-spirit", "--help"])
    assert result.exit_code == 0
    assert "--template-dir" in result.output
    assert "--rotation-library-file" in result.output
    assert "--output-base-dir" in result.output
    assert "--plugin" in result.output
    assert "--pattern" in result.output


def test_run_gsp_spirit_missing_required_options(runner) -> None:
    result = runner.invoke(main, ["run", "gsp-spirit", "/fake/image.nii.gz"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# run — single-session mode (pydra stubbed out)
# ---------------------------------------------------------------------------


def test_run_gsp_spirit_single(runner, tmp_path) -> None:
    image = NiftiGz.sample(dest_dir=tmp_path, stem="t1")
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    rot_lib = tmp_path / "rotations.txt"
    rot_lib.touch()

    mock_submitter = MagicMock()
    mock_submitter.__enter__ = MagicMock(return_value=mock_submitter)
    mock_submitter.__exit__ = MagicMock(return_value=False)

    # single_cls is captured in the CLI closure at build time, so we patch
    # pydra.engine.Submitter to confirm pydra was invoked without running it.
    with patch("pydra.engine.Submitter", return_value=mock_submitter) as mock_sub:
        result = runner.invoke(
            main,
            [
                "run",
                "gsp-spirit",
                str(image),
                "--template-dir",
                str(template_dir),
                "--rotation-library-file",
                str(rot_lib),
                "--output-base-dir",
                str(tmp_path / "output"),
                "--plugin",
                "serial",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_sub.assert_called_once_with(plugin="serial")
    mock_submitter.assert_called_once()  # sub(wf)


# ---------------------------------------------------------------------------
# run — batch mode (directory input)
# ---------------------------------------------------------------------------


def test_run_gsp_spirit_batch_from_dir(runner, tmp_path) -> None:
    # Create two session dirs with NIfTI files
    for i in range(2):
        NiftiGz.sample(dest_dir=tmp_path / f"session{i}", stem="t1")

    template_dir = tmp_path / "template"
    template_dir.mkdir()
    rot_lib = tmp_path / "rotations.txt"
    rot_lib.touch()

    mock_submitter = MagicMock()
    mock_submitter.__enter__ = MagicMock(return_value=mock_submitter)
    mock_submitter.__exit__ = MagicMock(return_value=False)

    with patch("pydra.engine.Submitter", return_value=mock_submitter) as mock_sub:
        result = runner.invoke(
            main,
            [
                "run",
                "gsp-spirit",
                str(tmp_path),
                "--template-dir",
                str(template_dir),
                "--rotation-library-file",
                str(rot_lib),
                "--output-base-dir",
                str(tmp_path / "output"),
                "--plugin",
                "serial",
                "--pattern",
                "*.nii.gz",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_sub.assert_called_once_with(plugin="serial")
    mock_submitter.assert_called_once()  # sub(wf)


def test_run_batch_no_matching_files(runner, tmp_path) -> None:
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    rot_lib = tmp_path / "rotations.txt"
    rot_lib.touch()

    result = runner.invoke(
        main,
        [
            "run",
            "gsp-spirit",
            str(tmp_path),
            "--template-dir",
            str(template_dir),
            "--rotation-library-file",
            str(rot_lib),
            "--pattern",
            "*.nii.gz",
        ],
    )
    assert result.exit_code != 0
    assert "No files" in result.output


# ---------------------------------------------------------------------------
# plot group structure
# ---------------------------------------------------------------------------


def test_plot_help(runner) -> None:
    result = runner.invoke(main, ["plot", "--help"])
    assert result.exit_code == 0
    assert "vial-intensity" in result.output
    assert "maps-ir" in result.output
    assert "maps-te" in result.output


def test_plot_vial_intensity_help(runner) -> None:
    result = runner.invoke(main, ["plot", "vial-intensity", "--help"])
    assert result.exit_code == 0
    assert "CSV_FILE" in result.output
    assert "--output" in result.output


def test_plot_maps_ir_help(runner) -> None:
    result = runner.invoke(main, ["plot", "maps-ir", "--help"])
    assert result.exit_code == 0
    assert "CONTRAST_FILES" in result.output
    assert "--metric_dir" in result.output


def test_plot_maps_te_help(runner) -> None:
    result = runner.invoke(main, ["plot", "maps-te", "--help"])
    assert result.exit_code == 0
    assert "CONTRAST_FILES" in result.output
    assert "--metric_dir" in result.output
