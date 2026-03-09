Quick start
===========

PhantomKit analyses are implemented as `Pydra <https://nipype.io/pydra>`__ workflows. To
execute them, you first parameterise the workflow by instantiate the Pydra workflow class,
then call the workflow object with execution args::

.. code-block:: python

   >>> from pydra.utils import get_fields
   >>> from phantomkit.analyses.vial_signal import VialSignalAnalysis
   >>> vial_signal_wf = VialSignalAnalysis(
      input_image="/path/to/input/image.nii.gz",
      template_dir="/path/to/template/dir",
      rotation_library_file="/path/to/rotation/library/file.txt"
   )
   >>> outputs = vial_signal_wf(cache_root="/path/to/cache/root")
   >>> print("\n".join(f.name for f in outputs.metrics_dir.contents))
