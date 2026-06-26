"""RFID (reconstruction FID) via torch-fidelity. For stage-2 generation FID,
use the fd_evaluator path via `evaluate_image_set`."""

from torch_fidelity import calculate_metrics

from .utils import ImgArrDataset


def calculate_rfid(
    arr1,
    arr2=None,
    bs=64,
    device="cuda",
    fid_statistics_file=None,
):
    arr1_ds = ImgArrDataset(arr1)

    if fid_statistics_file is not None:
        metrics_kwargs = dict(
            input1=arr1_ds,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            batch_size=bs,
            fid=True,
            cuda=(device == "cuda"),
        )
    else:
        if arr2 is None:
            raise ValueError("Either arr2 or fid_statistics_file must be provided.")
        arr2_ds = ImgArrDataset(arr2)
        metrics_kwargs = dict(
            input1=arr1_ds,
            input2=arr2_ds,
            batch_size=bs,
            fid=True,
            cuda=(device == "cuda"),
        )

    metrics = calculate_metrics(**metrics_kwargs)
    return metrics["frechet_inception_distance"]


def calculate_fid_isc(arr1, arr2, bs=64, device="cuda"):
    """Return (FID, Inception-Score-mean). FID is arr1-vs-arr2; IS is computed on
    arr1 (the generated set). Single torch-fidelity pass."""
    metrics = calculate_metrics(
        input1=ImgArrDataset(arr1),
        input2=ImgArrDataset(arr2),
        batch_size=bs,
        fid=True,
        isc=True,
        cuda=(device == "cuda"),
    )
    return metrics["frechet_inception_distance"], metrics["inception_score_mean"]
