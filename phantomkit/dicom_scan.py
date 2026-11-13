#!/usr/bin/env python3
"""
MRtrix3-based DICOM series discovery and extraction.

Replaces dcm2niix for turning a raw DICOM input tree into a flat pile of
correctly-grouped, SeriesDescription-named NIfTI files (plus sidecars).

Why: dcm2niix's own series-grouping can be unreliable on some real-world
exports — e.g. a scanner/PACS export that splits one logical series across
many subdirectories (one per instance/volume), or where SeriesNumber and
SeriesInstanceUID disagree. mrinfo groups by the DICOM data itself (series
UID), independent of however the files happen to be laid out on disk, so
querying it once over the whole input tree and converting by its own
series index is more robust than per-subdirectory, name-based discovery.

``mrinfo``/``mrconvert``, when pointed at a directory containing more than
one DICOM series, print a numbered listing and block on a "Select series"
stdin prompt. That listing is the authoritative, UID-grouped series list —
this module parses it directly (never a dcm2niix output) and drives
``mrconvert`` by piping the wanted index to that same prompt.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional


class DicomSeriesEntry(NamedTuple):
    index: int
    count: int
    time: str
    description: str
    comment: str
    series_number: int
    flags: list  # e.g. ["ORIGINAL", "PRIMARY", "DIFFUSION", "NONE", "MFSPLIT"]

    @property
    def is_derived(self) -> bool:
        return "DERIVED" in self.flags

    @property
    def is_mfsplit(self) -> bool:
        return "MFSPLIT" in self.flags

    @property
    def is_diffusion(self) -> bool:
        return "DIFFUSION" in self.flags


# Matches a line like:
#   11 -  192 MR images 15:15:40 SEQ-A_NIF_SAG_MPRAGE (*tfl3d1_224ns) [4001] ORIGINAL PRIMARY M NORM DIS2D MFSPLIT
_SERIES_LINE_RE = re.compile(
    r"^\s*(?P<index>\d+)\s*-\s*(?P<count>\d+)\s+\S+\s+images?\s+"
    r"(?P<time>\S+)\s+(?P<description>.+?)\s+"
    r"\((?P<comment>[^)]*)\)\s+\[(?P<series_number>\d+)\]\s*(?P<flags>.*)$"
)


def list_dicom_series(input_dir: Path) -> list:
    """Return every DICOM series mrinfo finds anywhere under input_dir.

    Runs a single ``mrinfo <input_dir>`` (mrinfo recurses on its own),
    captures the numbered series listing it prints when more than one
    series is present, then aborts out of its interactive prompt with "q"
    — this call never actually selects/converts anything, it's read-only
    discovery. Returns [] if input_dir contains no DICOM data at all (or
    exactly one series silently accepted with no prompt — see note below).
    """
    try:
        result = subprocess.run(
            ["mrinfo", str(input_dir)],
            input="q\n",
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    output = result.stdout + "\n" + result.stderr
    entries = []
    for line in output.splitlines():
        m = _SERIES_LINE_RE.match(line)
        if not m:
            continue
        entries.append(
            DicomSeriesEntry(
                index=int(m["index"]),
                count=int(m["count"]),
                time=m["time"],
                description=m["description"].strip(),
                comment=m["comment"].strip(),
                series_number=int(m["series_number"]),
                flags=m["flags"].split(),
            )
        )
    return entries


def select_best_per_description(entries: list) -> dict:
    """Group entries by SeriesDescription, keeping one representation each.

    DERIVED series (ADC/TRACEW/FA/ColFA-style post-processed maps) are
    excluded entirely, regardless of description — the pipeline re-derives
    these itself from the raw DWI series, and unlike ADC/FA they don't all
    share a name ending the rest of the codebase's "_(ADC|FA)$" skip
    convention already recognizes (e.g. "..._ColFA", "..._TRACEW" don't
    match it), so they'd otherwise slip through as bogus DWI candidates.
    Among the remaining ORIGINAL entries sharing one description, prefer
    non-MFSPLIT over MFSPLIT (the split-into-single-frame duplicate of the
    same acquisition). Ties broken by lowest series_number, matching
    acquisition order.
    """
    by_description: dict = {}
    for entry in entries:
        if entry.is_derived:
            continue
        by_description.setdefault(entry.description, []).append(entry)

    selected = {}
    for description, candidates in by_description.items():
        non_split = [e for e in candidates if not e.is_mfsplit]
        pool = non_split if non_split else candidates
        selected[description] = min(pool, key=lambda e: e.series_number)
    return selected


def convert_dicom_series(
    input_dir: Path, entry: DicomSeriesEntry, out_nii: Path
) -> dict:
    """Extract one series (by its mrinfo index) directly to NIfTI + sidecars.

    Always exports a JSON sidecar; additionally exports FSL-style
    bvec/bval files when the series' own DICOM ImageType flags mark it as
    DIFFUSION, so diffusion series get the same gradient-table sidecars
    dcm2niix would have produced, without guessing from the data itself.
    """
    out_nii.parent.mkdir(parents=True, exist_ok=True)
    stem = out_nii.name
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    json_path = out_nii.parent / f"{stem}.json"

    cmd = [
        "mrconvert", str(input_dir), str(out_nii),
        "-json_export", str(json_path), "-force",
    ]
    result = {"nii": str(out_nii), "json": str(json_path), "bvec": None, "bval": None}
    if entry.is_diffusion:
        bvec_path = out_nii.parent / f"{stem}.bvec"
        bval_path = out_nii.parent / f"{stem}.bval"
        cmd += ["-export_grad_fsl", str(bvec_path), str(bval_path)]
        result["bvec"] = str(bvec_path)
        result["bval"] = str(bval_path)

    subprocess.run(
        cmd, input=f"{entry.index}\n", text=True,
        check=True, capture_output=True, timeout=600,
    )
    return result


def stage_dicom_tree(input_dir: Path, out_dir: Path) -> Optional[Path]:
    """Convert every distinct DICOM series under input_dir into a flat pile
    of SeriesDescription-named NIfTI files (+ sidecars) in out_dir.

    Returns out_dir if at least one series was converted, else None (e.g.
    input_dir has no DICOM data at all — caller should fall back to
    whatever non-DICOM handling already applies).
    """
    entries = list_dicom_series(input_dir)
    if not entries:
        return None

    selected = select_best_per_description(entries)
    out_dir.mkdir(parents=True, exist_ok=True)

    converted_any = False
    for description, entry in sorted(selected.items()):
        safe_name = re.sub(r"[^\w.\-]", "_", description)
        out_nii = out_dir / f"{safe_name}.nii.gz"
        try:
            convert_dicom_series(input_dir, entry, out_nii)
            converted_any = True
        except subprocess.CalledProcessError as e:
            detail = e.stderr.decode(errors="replace").strip() if e.stderr else ""
            print(f"  WARNING: mrconvert failed for series '{description}': {detail}")

    return out_dir if converted_any else None
