<img src="https://raw.githubusercontent.com/australian-imaging-service/phantomkit/main/docs/source/_static/logo.svg" alt="PhantomKit" width="600"/>

[![CI/CD](https://github.com/australian-imaging-service/phantomkit/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/australian-imaging-service/phantomkit/actions/workflows/ci-cd.yml)
[![Codecov](https://codecov.io/gh/australian-imaging-service/phantomkit/branch/main/graph/badge.svg?token=UIS0OGPST7)](https://codecov.io/gh/australian-imaging-service/phantomkit)
[![PyPI version](https://img.shields.io/pypi/v/phantomkit.svg)](https://pypi.python.org/pypi/phantomkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/phantomkit.svg)](https://pypi.python.org/pypi/phantomkit/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://australian-imaging-service.github.io/phantomkit/)

# PhantomKit

**PhantomKit** is a Python toolkit for automated quality assurance (QA) of medical imaging scanners using physical phantoms. It provides pydra-based workflows that register phantom scans to a reference template, extract per-vial signal statistics across multiple contrast types, and generate publication-quality plots — supporting both MRI and PET phantom protocols.

## Features

- **Template-based registration** — iterative ANTs SyN registration with automatic orientation search across a rotation library
- **Vial metric extraction** — per-vial mean, median, std, min and max across all contrast images, written to CSV
- **Plotting** — scatter plots of vial intensity and parametric map plots (T1/IR, T2/TE) with mrview ROI overlays
- **Protocol support** — extensible `protocols` sub-package for phantom- and project-specific workflow configurations
- **Parallel batch processing** — pydra-native splitting and combining for multi-session datasets

## Installation

```bash
python -m pip install phantomkit
```

## Basic usage

```python
from phantomkit.protocols.gsp_spirit import GspSpiritAnalysis

wf = GspSpiritAnalysis(
    input_image="/data/session01/t1_mprage.nii.gz",
    template_dir="/templates/gsp_spirit",
    rotation_library_file="/templates/gsp_spirit/rotations.txt",
)
outputs = wf(cache_root="/data/cache-root")
```

Or via the command line:

```bash
phantomkit run gsp-spirit /data/session01/t1_mprage.nii.gz \
    --template-dir /templates/gsp_spirit \
    --output-dir /results \
    --rotation-lib /templates/gsp_spirit/rotations.txt
```

## License

Copyright 2026 Australian Imaging Service. Released under the [Apache License 2.0](LICENSE).
