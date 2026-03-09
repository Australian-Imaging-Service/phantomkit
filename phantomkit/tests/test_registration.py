"""Unit tests for phantomkit.registration."""

from pathlib import Path

import pytest
from fileformats.medimage import NiftiGz

from phantomkit.registration import (
    AggregateVialCheck,
    ParseMrStatsStdout,
    PrepVialCheckPaths,
    _create_rotation_matrix_file,
    _load_rotations,
)

# ── ParseMrStatsStdout ────────────────────────────────────────────────────────


def test_parse_mrstats_stdout_multiple_values() -> None:
    out = ParseMrStatsStdout(stdout="1.5 2.3 0.8")()
    assert out.out == pytest.approx([1.5, 2.3, 0.8])


def test_parse_mrstats_stdout_single_value() -> None:
    out = ParseMrStatsStdout(stdout="42.0")()
    assert out.out == pytest.approx([42.0])


# ── AggregateVialCheck ────────────────────────────────────────────────────────

# 10 vials: A, O, Q in top-3; S, D, P in bottom-3 — should pass
_PASSING_NAMES = ["A", "O", "Q", "B", "C", "D", "P", "S", "E", "F"]
_PASSING_STATS = [
    [10.0, 1.0],
    [9.0, 1.0],
    [8.0, 1.0],
    [7.0, 1.0],
    [6.0, 1.0],
    [5.0, 1.0],
    [4.0, 1.0],
    [3.0, 1.0],
    [2.0, 1.0],
    [1.0, 1.0],
]


def test_aggregate_vial_check_passes() -> None:
    out = AggregateVialCheck(vial_names=_PASSING_NAMES, means_stds=_PASSING_STATS)()
    assert out.is_valid is True


def test_aggregate_vial_check_fails_high_std() -> None:
    stats = [list(s) for s in _PASSING_STATS]
    stats[0][1] = 100.0  # vial A has std > 50
    out = AggregateVialCheck(vial_names=_PASSING_NAMES, means_stds=stats)()
    assert out.is_valid is False


def test_aggregate_vial_check_fails_ranking() -> None:
    # Put A at the bottom — should fail because A must be in top-5
    names = ["F", "E", "B", "C", "G", "D", "P", "S", "O", "A"]
    stats = [[float(10 - i), 1.0] for i in range(10)]
    out = AggregateVialCheck(vial_names=names, means_stds=stats)()
    assert out.is_valid is False


# ── _load_rotations ───────────────────────────────────────────────────────────


def test_load_rotations(tmp_path: Path) -> None:
    lib = tmp_path / "rotations.txt"
    lib.write_text(
        "# comment\n" 'rotation "1 0 0 0 1 0 0 0 1"\n' 'rotation "0 1 0 -1 0 0 0 0 1"\n'
    )
    rotations = _load_rotations(str(lib))
    assert rotations == ["1 0 0 0 1 0 0 0 1", "0 1 0 -1 0 0 0 0 1"]


def test_load_rotations_skips_blank_lines(tmp_path: Path) -> None:
    lib = tmp_path / "rotations.txt"
    lib.write_text('\nrotation "1 0 0 0 1 0 0 0 1"\n\n')
    assert len(_load_rotations(str(lib))) == 1


# ── _create_rotation_matrix_file ─────────────────────────────────────────────


def test_create_rotation_matrix_file(tmp_path: Path) -> None:
    out_file = tmp_path / "rot.txt"
    result = _create_rotation_matrix_file("1 0 0 0 1 0 0 0 1", out_file)
    assert result == out_file
    lines = out_file.read_text().splitlines()
    assert lines == ["1 0 0 0", "0 1 0 0", "0 0 1 0", "0 0 0 1"]


# ── PrepVialCheckPaths ────────────────────────────────────────────────────────


def test_prep_vial_check_paths(tmp_path: Path) -> None:
    mask = NiftiGz.sample(dest_dir=tmp_path, stem="VialA")
    out = PrepVialCheckPaths(vial_mask=mask, tmp_dir=tmp_path)()
    assert out.vial_name == "A"
    assert out.vial_mask_out.fspath == mask.fspath
    assert str(out.regridded_path).endswith("A_regridded.nii.gz")
