from fileformats.medimage import DicomSeries
from pydra.compose import python


@python.define(outputs=["a_metric"])
def GradientAnalysis(
    dmri: DicomSeries,
) -> float:
    """Analyses a DICOM series of a Gold Standard Phantom DWI scan to compute the gradient metric.

    Parameters
    ----------
    dmri : DicomSeries
        The DICOM series of a Gold Standard Phantom DWI scan to be analysed.

    Returns
    -------
    float
        The computed gradient metric.

    """
    raise NotImplementedError(
        "This function is a placeholder and needs to be implemented."
    )
