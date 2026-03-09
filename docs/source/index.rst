.. _home:

PhantomKit
==========

.. image:: https://github.com/australian-imaging-service/phantomkit/actions/workflows/ci-cd.yml/badge.svg
   :target: https://github.com/australian-imaging-service/phantomkit/actions/workflows/ci-cd.yml
.. image:: https://codecov.io/gh/australian-imaging-service/phantomkit/branch/main/graph/badge.svg?token=UIS0OGPST7
   :target: https://codecov.io/gh/australian-imaging-service/phantomkit
.. image:: https://img.shields.io/pypi/pyversions/phantomkit.svg
   :target: https://pypi.python.org/pypi/phantomkit/
   :alt: Supported Python versions
.. image:: https://img.shields.io/pypi/v/phantomkit.svg
   :target: https://pypi.python.org/pypi/phantomkit/
   :alt: Latest Version

*PhantomKit* is a Python toolkit for automated quality assurance (QA) of medical imaging
scanners using physical phantoms. It provides pydra-based workflows that register phantom
scans to a reference template, extract per-vial signal statistics across multiple contrast
types, and generate publication-quality plots — supporting both MRI and PET phantom
protocols.

Key features:

- **Template-based registration** — iterative ANTs SyN registration with automatic
  orientation search across a rotation library
- **Vial metric extraction** — per-vial mean, median, std, min and max across all
  contrast images, written to CSV
- **Plotting** — scatter plots of vial intensity and parametric map plots (T1/IR, T2/TE)
  with mrview ROI overlays
- **Protocol support** — extensible ``protocols`` sub-package for phantom- and
  project-specific workflow configurations
- **Parallel batch processing** — pydra-native splitting and combining for multi-session
  datasets


Installation
------------

*PhantomKit* can be installed for Python >=3.11 using *pip*:

.. code-block:: console

    $ python3 -m pip install phantomkit


License
-------

*PhantomKit* is released under the
`Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.

Copyright 2026 Australian Imaging Service.


.. toctree::
    :maxdepth: 2
    :hidden:

    quick_start
    api
