Command-line interface
======================

PhantomKit installs the ``phantom-process`` command, which groups two
subcommand families:

- ``phantom-process run`` — execute a full QA protocol workflow
- ``phantom-process plot`` — generate individual QA plots from existing CSVs

Use ``--help`` at any level for a full option listing::

    phantom-process --help
    phantom-process run --help
    phantom-process run gsp-spirit --help
    phantom-process plot --help
    phantom-process plot vial-intensity --help


Discovering available protocols
--------------------------------

.. code-block:: console

    $ phantom-process list
    Available protocols:
      gsp-spirit (batch supported)
        Pydra workflow for processing a single GSP SPIRIT phantom MRI session.


Running a protocol
------------------

Single session
~~~~~~~~~~~~~~

Pass a path to a NIfTI file as ``INPUT`` to process one session:

.. code-block:: console

    $ phantom-process run gsp-spirit /data/session01/t1_mprage.nii.gz \
          --template-dir  /templates/gsp_spirit \
          --rotation-library-file /templates/gsp_spirit/rotations.txt \
          --output-base-dir /results

The session name is taken from the parent directory of the image
(``session01`` in the example above).  All ``*.nii.gz`` files in the same
directory are treated as additional contrast images.

Batch mode
~~~~~~~~~~

Pass a *directory* as ``INPUT`` to process every matching image found
recursively inside it:

.. code-block:: console

    $ phantom-process run gsp-spirit /data \
          --template-dir  /templates/gsp_spirit \
          --rotation-library-file /templates/gsp_spirit/rotations.txt \
          --output-base-dir /results \
          --pattern "*t1*mprage*.nii.gz" \
          --plugin cf

``--pattern`` controls which files are selected (default ``*.nii.gz``).
``--plugin`` selects the pydra execution backend: ``cf`` (concurrent
futures, default) for parallel execution or ``serial`` for sequential.

Protocol options
~~~~~~~~~~~~~~~~

All parameters of the underlying workflow are exposed as options.  Run
``phantom-process run <protocol> --help`` to see the full list.  For
``gsp-spirit``:

.. code-block:: text

    Options:
      --template-dir PATH             Path to the GSP SPIRIT template directory.
                                      [required]
      --rotation-library-file PATH    Path to the rotation library text file.
                                      [required]
      --output-base-dir PATH          Root output directory (default: cwd).
      --plugin [cf|serial]            Pydra execution plugin.  [default: cf]
      --pattern TEXT                  Glob pattern for batch mode.
                                      [default: *.nii.gz]


Generating plots
----------------

The ``plot`` subcommands generate standalone QA figures from CSV metric
files already written to disk (e.g. after a completed ``run``).

Vial intensity scatter plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots per-vial mean ± std as a scatter, line, or bar chart for a single
contrast:

.. code-block:: console

    $ phantom-process plot vial-intensity \
          /results/session01/metrics/session01_t1_mprage_mean_matrix.csv \
          scatter \
          --std_csv /results/session01/metrics/session01_t1_mprage_std_matrix.csv \
          --output  /results/session01/metrics/session01_t1_PLOTmeanstd.png

Options:

.. code-block:: text

    Arguments:
      CSV_FILE              Mean-value CSV produced by ExtractMetricsFromContrasts.
      {line|bar|scatter}    Plot style.

    Options:
      --std_csv TEXT    CSV file containing standard deviations.
      --roi_image TEXT  PNG screenshot of the contrast with vial ROI overlay.
      --annotate        Annotate each point with mean ± std.
      --output TEXT     Output filename.  [default: vial_subplot.png]

Inversion-recovery (T1) map plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots vial mean ± std across multiple IR contrasts with T₁ curve fitting:

.. code-block:: console

    $ phantom-process plot maps-ir \
          /results/session01/images_template_space/ir_*.nii.gz \
          --metric_dir /results/session01/metrics \
          --output     /results/session01/metrics/session01_T1map_plot.png

Options:

.. code-block:: text

    Arguments:
      CONTRAST_FILES...   IR NIfTI images in template space.

    Options:
      -m, --metric_dir TEXT   Directory containing mean/std CSV files.  [required]
      -o, --output TEXT       Output filename.  [default: vial_summary_T1.png]
      --annotate              Annotate each point with mean ± std.
      --roi_image TEXT        PNG ROI overlay for the extra subplot.

Spin-echo (T2/TE) map plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots vial mean ± std across multiple TE contrasts with mono-exponential
T₂ fitting:

.. code-block:: console

    $ phantom-process plot maps-te \
          /results/session01/images_template_space/te_*.nii.gz \
          --metric_dir /results/session01/metrics \
          --output     /results/session01/metrics/session01_T2map_plot.png

Options are identical to ``maps-ir``.
