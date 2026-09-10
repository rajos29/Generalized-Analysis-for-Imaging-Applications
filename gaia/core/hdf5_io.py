from __future__ import annotations

from pathlib import Path
import re

import numpy as np
try:
    import h5py
except ImportError:
    h5py = None

from gaia.core.image_ops import ensure_rgb

def list_hdf5_image_datasets(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    if h5py is None:
        raise RuntimeError("HDF5 support requires h5py. Install it with: python -m pip install h5py")

    datasets: list[tuple[str, tuple[int, ...], str]] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, node) -> None:
            if not isinstance(node, h5py.Dataset):
                return
            if len(node.shape) < 2:
                return
            if not np.issubdtype(node.dtype, np.number):
                return
            datasets.append((name, tuple(int(value) for value in node.shape), str(node.dtype)))

        handle.visititems(visitor)
    return datasets


def sample_sort_key(name: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in name if not ch.isdigit())
    digits = "".join(ch for ch in name if ch.isdigit())
    return prefix, int(digits) if digits else -1


def hdf5_dataset_catalog(path: Path) -> dict[str, object]:
    datasets = list_hdf5_image_datasets(path)
    groups: dict[str, dict[str, dict[str, object]]] = {}
    for dataset_name, shape, dtype in datasets:
        parts = dataset_name.split("/")
        if len(parts) != 2:
            continue
        group, sample = parts
        groups.setdefault(group, {})[sample] = {
            "dataset": dataset_name,
            "shape": shape,
            "dtype": dtype,
        }
    if not groups:
        raise ValueError(f"No paired group/sample datasets were found in {path.name}.")

    sample_sets = [set(samples) for samples in groups.values()]
    common_samples = sorted(set.intersection(*sample_sets), key=sample_sort_key) if sample_sets else []
    all_samples = sorted(set().union(*sample_sets), key=sample_sort_key) if sample_sets else []
    return {"groups": groups, "common_samples": common_samples, "all_samples": all_samples}


def frame_from_array(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array)
    data = np.squeeze(data)
    if data.ndim < 2:
        raise ValueError(f"Expected at least 2D image data, got shape {data.shape}.")

    while data.ndim > 3:
        data = data[data.shape[0] // 2]

    if data.ndim == 3 and data.shape[-1] not in (3, 4):
        data = data[data.shape[0] // 2]

    if data.ndim == 2:
        return ensure_rgb(normalize_to_uint8(data))
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        return ensure_rgb(data, source_order="RGB")
    raise ValueError(f"Could not convert HDF5 array shape {data.shape} to a display frame.")


def frame_from_dataset_slice(dataset, z_index: int | None = None) -> np.ndarray:
    shape = tuple(int(value) for value in dataset.shape)
    if len(shape) == 2:
        data = dataset[()]
    elif len(shape) == 3 and shape[-1] not in (3, 4):
        z = min(max(0, int(z_index if z_index is not None else shape[0] // 2)), shape[0] - 1)
        data = dataset[z]
    else:
        data = dataset[()]
    return frame_from_array(data)


def load_hdf5_frame(path: Path, dataset_name: str) -> np.ndarray:
    if h5py is None:
        raise RuntimeError("HDF5 support requires h5py. Install it with: python -m pip install h5py")
    with h5py.File(path, "r") as handle:
        data = handle[dataset_name][()]
    return frame_from_array(data)


def load_hdf5_sample_frame(path: Path, group: str, sample: str, z_index: int | None = None) -> np.ndarray:
    if h5py is None:
        raise RuntimeError("HDF5 support requires h5py. Install it with: python -m pip install h5py")
    with h5py.File(path, "r") as handle:
        dataset_name = f"{group}/{sample}"
        if dataset_name not in handle:
            raise ValueError(f"{dataset_name} was not found in {path.name}.")
        return frame_from_dataset_slice(handle[dataset_name], z_index)

