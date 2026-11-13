"""Unit tests for phantomkit.dicom_scan."""

import subprocess
from pathlib import Path

import pytest

from phantomkit.dicom_scan import (
    DicomSeriesEntry,
    convert_dicom_series,
    list_dicom_series,
    select_best_per_description,
    stage_dicom_tree,
)

# Real mrinfo output, captured verbatim from a dataset where a scanner/PACS
# export split one logical DWI series across many single-instance
# subdirectories (series numbers 5001-5036) in addition to the properly
# grouped original (series number 5) -- the exact scenario this module
# exists to handle correctly where per-subdirectory, folder-name-based
# discovery cannot.
REAL_MRINFO_LISTING = """\
mrinfo: scanning folder "/data/dicom" for DICOM data...
mrinfo: [WARNING] mismatched series number and UID - this may cause problems with series grouping
.....done
Select series ('q' to abort):
   0 -    1 MR images 15:15:40 SEQ-A_NIF_SAG_MPRAGE (?) [4] ORIGINAL PRIMARY M NONE
   1 -   38 MR images 15:17:24 SEQ-B_NIF_AX_DWI_b1000_AP (?) [5] ORIGINAL PRIMARY DIFFUSION NONE
   2 -    1 MR images 15:22:09 SEQ-B_NIF_AX_DWI_b1000_AP_ADC (?) [6] DERIVED PRIMARY DIFFUSION ADC
   3 -    1 MR images 15:22:09 SEQ-B_NIF_AX_DWI_b1000_AP_TRACEW (?) [7] DERIVED PRIMARY DIFFUSION TRACEW
   4 -    1 MR images 15:22:09 SEQ-B_NIF_AX_DWI_b1000_AP_FA (?) [8] DERIVED PRIMARY DIFFUSION FA
   5 -    1 MR images 15:22:09 SEQ-B_NIF_AX_DWI_b1000_AP_ColFA (?) [9] DERIVED PRIMARY DIFFUSION FA
   6 -   38 MR images 15:22:43 SEQ-B_NIF_AX_DWI_b1000_PA (?) [10] ORIGINAL PRIMARY DIFFUSION NONE
  11 -  192 MR images 15:15:40 SEQ-A_NIF_SAG_MPRAGE (*tfl3d1_224ns) [4001] ORIGINAL PRIMARY M NORM DIS2D MFSPLIT
  12 - 2204 MR images 15:17:24 SEQ-B_NIF_AX_DWI_b1000_AP (*ep_b0) [5036] ORIGINAL PRIMARY DIFFUSION NONE ND NORM MFSPLIT
  17 - 2204 MR images 15:22:43 SEQ-B_NIF_AX_DWI_b1000_PA (*ep_b1000#10) [10011] ORIGINAL PRIMARY DIFFUSION NONE ND NORM MFSPLIT
? mrinfo: [ERROR] operation cancelled by user
"""


def test_list_dicom_series_parses_real_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0, stdout=REAL_MRINFO_LISTING, stderr=""
        ),
    )

    entries = list_dicom_series(Path("/data/dicom"))

    assert len(entries) == 10
    mprage = next(e for e in entries if e.index == 0)
    assert mprage.description == "SEQ-A_NIF_SAG_MPRAGE"
    assert mprage.count == 1
    assert mprage.series_number == 4
    assert mprage.flags == ["ORIGINAL", "PRIMARY", "M", "NONE"]
    assert mprage.is_derived is False
    assert mprage.is_mfsplit is False
    assert mprage.is_diffusion is False

    dwi_ap = next(e for e in entries if e.index == 1)
    assert dwi_ap.description == "SEQ-B_NIF_AX_DWI_b1000_AP"
    assert dwi_ap.count == 38
    assert dwi_ap.is_diffusion is True
    assert dwi_ap.is_derived is False
    assert dwi_ap.is_mfsplit is False

    adc = next(e for e in entries if e.index == 2)
    assert adc.description == "SEQ-B_NIF_AX_DWI_b1000_AP_ADC"
    assert adc.is_derived is True

    mfsplit_ap = next(e for e in entries if e.index == 12)
    assert mfsplit_ap.description == "SEQ-B_NIF_AX_DWI_b1000_AP"
    assert mfsplit_ap.count == 2204
    assert mfsplit_ap.is_mfsplit is True
    assert mfsplit_ap.is_derived is False


def test_list_dicom_series_returns_empty_for_no_dicom_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 1, stdout="", stderr="mrinfo: [ERROR] no supported images found\n"
        ),
    )
    assert list_dicom_series(Path("/empty")) == []


def test_list_dicom_series_handles_missing_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*a, **k):
        raise FileNotFoundError("mrinfo not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert list_dicom_series(Path("/data/dicom")) == []


def test_select_best_per_description_excludes_derived_regardless_of_name() -> None:
    entries = [
        DicomSeriesEntry(0, 1, "15:00", "SEQ-A", "?", 4, ["ORIGINAL", "PRIMARY", "M", "NONE"]),
        DicomSeriesEntry(1, 38, "15:17", "SEQ-B_AP", "?", 5, ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE"]),
        DicomSeriesEntry(2, 1, "15:22", "SEQ-B_AP_ADC", "?", 6, ["DERIVED", "PRIMARY", "DIFFUSION", "ADC"]),
        DicomSeriesEntry(3, 1, "15:22", "SEQ-B_AP_TRACEW", "?", 7, ["DERIVED", "PRIMARY", "DIFFUSION", "TRACEW"]),
        DicomSeriesEntry(4, 1, "15:22", "SEQ-B_AP_ColFA", "?", 8, ["DERIVED", "PRIMARY", "DIFFUSION", "FA"]),
    ]
    selected = select_best_per_description(entries)
    assert set(selected) == {"SEQ-A", "SEQ-B_AP"}


def test_select_best_per_description_prefers_non_mfsplit() -> None:
    entries = [
        DicomSeriesEntry(1, 38, "15:17", "SEQ-B_AP", "?", 5, ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE"]),
        DicomSeriesEntry(
            12, 2204, "15:17", "SEQ-B_AP", "?", 5036,
            ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE", "ND", "NORM", "MFSPLIT"],
        ),
    ]
    selected = select_best_per_description(entries)
    assert selected["SEQ-B_AP"].index == 1
    assert selected["SEQ-B_AP"].count == 38


def test_select_best_per_description_falls_back_to_mfsplit_if_thats_all_there_is() -> None:
    entries = [
        DicomSeriesEntry(
            12, 2204, "15:17", "SEQ-B_AP", "?", 5036,
            ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE", "MFSPLIT"],
        ),
    ]
    selected = select_best_per_description(entries)
    assert selected["SEQ-B_AP"].index == 12


def test_select_best_per_description_breaks_ties_by_lowest_series_number() -> None:
    entries = [
        DicomSeriesEntry(0, 1, "15:00", "SEQ-A", "?", 9, ["ORIGINAL", "PRIMARY", "M", "NONE"]),
        DicomSeriesEntry(1, 1, "15:00", "SEQ-A", "?", 4, ["ORIGINAL", "PRIMARY", "M", "NONE"]),
    ]
    selected = select_best_per_description(entries)
    assert selected["SEQ-A"].index == 1
    assert selected["SEQ-A"].series_number == 4


def test_stage_dicom_tree_returns_none_when_no_series_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "phantomkit.dicom_scan.list_dicom_series", lambda input_dir: []
    )
    assert stage_dicom_tree(tmp_path / "in", tmp_path / "out") is None


def test_convert_dicom_series_requests_grad_export_only_for_diffusion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = {}

    def _fake_run(cmd, input, text, check, capture_output, timeout):
        captured["cmd"] = cmd
        captured["input"] = input

    monkeypatch.setattr(subprocess, "run", _fake_run)

    anatomical = DicomSeriesEntry(0, 1, "15:00", "SEQ-A", "?", 4, ["ORIGINAL", "PRIMARY", "M", "NONE"])
    convert_dicom_series(tmp_path, anatomical, tmp_path / "out" / "SEQ-A.nii.gz")
    assert "-export_grad_fsl" not in captured["cmd"]
    assert captured["input"] == "0\n"

    dwi = DicomSeriesEntry(1, 38, "15:17", "SEQ-B", "?", 5, ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE"])
    convert_dicom_series(tmp_path, dwi, tmp_path / "out" / "SEQ-B.nii.gz")
    assert "-export_grad_fsl" in captured["cmd"]
    assert captured["input"] == "1\n"
