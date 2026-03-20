<img src="https://raw.githubusercontent.com/australian-imaging-service/phantomkit/main/docs/source/_static/logo.svg" alt="PhantomKit" width="600"/>

[![CI/CD](https://github.com/australian-imaging-service/phantomkit/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/australian-imaging-service/phantomkit/actions/workflows/ci-cd.yml)
[![Codecov](https://codecov.io/gh/australian-imaging-service/phantomkit/branch/main/graph/badge.svg?token=UIS0OGPST7)](https://codecov.io/gh/australian-imaging-service/phantomkit)
[![PyPI version](https://img.shields.io/pypi/v/phantomkit.svg)](https://pypi.python.org/pypi/phantomkit/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://australian-imaging-service.github.io/phantomkit/)

# PhantomKit

**PhantomKit** is a Python toolkit for automated quality assurance (QA) of MRI scanners using physical phantoms. It provides pydra-based workflows that register phantom scans to a reference template, extract per-vial signal statistics across multiple contrast types, and generate publication-quality plots — with full support for DWI preprocessing and diffusion metric (ADC, FA) extraction.

## Features

- **End-to-end pipeline** — single command processes a raw DICOM session directory through DWI preprocessing, phantom QC in DWI space, and native contrast QC
- **Automatic series classification** — DWI, reverse phase-encode, T1, IR, and TE series are detected and paired automatically from folder names and DICOM sidecar metadata; no manual configuration required
- **DWI preprocessing** — FSL `dwifslpreproc` with automatic phase-encoding correction mode selection (`rpe_none`, `rpe_pair`, `rpe_all`, `rpe_split`), optional denoising/Gibbs correction, tensor fitting, and T1-to-DWI coregistration via FLIRT
- **Template-based registration** — iterative ANTs rigid registration with automatic orientation search across a rotation library; vial masks propagated to subject space via inverse transform
- **Vial metric extraction** — per-vial mean and std across all contrast images (T1, IR, TE, ADC, FA), written to CSV
- **Plotting** — ADC/FA scatter plots with SPIRIT reference values, T1/IR and T2/TE parametric map plots with mrview ROI overlays
- **Checkpoint-based resumption** — re-running the pipeline skips stages whose outputs already exist, avoiding redundant reprocessing of expensive steps
- **Parallel batch processing** — pydra-native splitting and combining for multi-session datasets

## Installation

```bash
python -m pip install phantomkit
```

### External dependencies

The DWI pipeline requires FSL, MRtrix3, and ANTs to be available on `PATH`:

- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) — `dwifslpreproc`, `flirt`, `convert_xfm`
- [MRtrix3](https://www.mrtrix.org/) — `mrconvert`, `dwi2tensor`, `tensor2metric`, `dwidenoise`, `mrdegibbs`, `mrstats`
- [ANTs](http://stnava.github.io/ANTs/) — `antsRegistrationSyN.sh`, `antsApplyTransforms`
- [dcm2niix](https://github.com/rordenlab/dcm2niix) — DICOM to NIfTI conversion

## Basic usage

### End-to-end pipeline

Point the pipeline at a session directory containing DICOM subdirectories. It automatically classifies the series, selects the appropriate preprocessing mode, and runs all three stages:

```bash
phantomkit pipeline     --input-dir  /data/session01     --output-dir /results/session01     --phantom    SPIRIT
```

Optional flags:

```
--denoise-degibbs    Apply dwidenoise + mrdegibbs before preprocessing
--gradcheck          Run dwigradcheck to verify gradient orientations
--nocleanup          Keep intermediate tmp/ directories after completion
--readout-time       Override TotalReadoutTime (seconds) for dwifslpreproc
--eddy-options       Override FSL eddy options string
--dry-run            Print the planned workflow without executing
```

Output structure:

```
/results/session01/
  <DWI_series>/
    DWI_preproc_biascorr.mif.gz
    ADC.nii.gz
    FA.nii.gz
    T1_in_DWI_space.nii.gz
    metrics/                      ← per-vial CSVs and QA plots
    vial_segmentations/
  native_contrasts_staging/
    metrics/                      ← per-vial CSVs and parametric map plots
    vial_segmentations/
    images_template_space/
```

### Native contrast QC only (no DWI)

If the input directory contains only T1, IR, and/or TE series (no DWI), the pipeline runs Stage 3 only:

```bash
phantomkit pipeline     --input-dir  /data/session01     --output-dir /results/session01     --phantom    SPIRIT
```

### Protocol workflows (pydra API)

```python
from phantomkit.analyses.vial_signal import VialSignalAnalysis

wf = VialSignalAnalysis(
    input_image="/data/session01/t1_mprage.nii.gz",
    template_dir="/templates/SPIRIT",
    rotation_library_file="/templates/SPIRIT/rotations.txt",
)
outputs = wf(cache_root="/data/cache-root")
```

Or via the command line:

```bash
# Single session
phantomkit run vial-signal /data/session01/t1_mprage.nii.gz     --template-dir /templates/SPIRIT     --rotation-library-file /templates/SPIRIT/rotations.txt     --output-base-dir /results

# Batch — process every matching image found under /data/
phantomkit run vial-signal /data/     --template-dir /templates/SPIRIT     --rotation-library-file /templates/SPIRIT/rotations.txt     --output-base-dir /results     --pattern "*t1*mprage*.nii.gz"

# List available protocols
phantomkit list
```

### Plotting

Generate QA plots from existing CSV metric files:

```bash
# ADC/FA scatter plot
phantomkit plot vial-intensity     /results/session01/metrics/session01_ADC_mean_matrix.csv scatter     --std-csv  /results/session01/metrics/session01_ADC_std_matrix.csv     --phantom  SPIRIT     --template-dir /templates     --output   /results/session01/metrics/session01_ADC_PLOTmeanstd.png

# T1 inversion-recovery parametric map plot
phantomkit plot maps-ir     /results/session01/images_template_space/ir_*.nii.gz     --metric-dir /results/session01/metrics     --output     /results/session01/metrics/session01_T1map_plot.png

# T2 spin-echo parametric map plot
phantomkit plot maps-te     /results/session01/images_template_space/te_*.nii.gz     --metric-dir /results/session01/metrics     --output     /results/session01/metrics/session01_T2map_plot.png
```

See the [CLI documentation](https://australian-imaging-service.github.io/phantomkit/cli.html) for the full option reference.

## License

Copyright 2026 Australian Imaging Service. Released under the [Apache License 2.0](LICENSE).
