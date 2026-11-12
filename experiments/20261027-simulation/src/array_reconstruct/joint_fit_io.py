"""Load only public training and internal-validation members of virtual cases.

The manifest is inspected before opening the array archive.  A sealed session's
array prefix is never indexed, even though its descriptor remains part of the
original source manifest.  Full-field scoring keeps its separate explicit
loader in ``joint_virtual``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .joint_virtual import VirtualFitData, VirtualSession


_ARRAY_FIELDS = (
    "samples_xy_m", "steering", "observed_field", "train_indices",
    "calibration_indices", "evaluation_indices",
)
_KNOWN_SPLITS = frozenset(("train", "internal", "sealed"))


def load_virtual_training_data(directory: Path | str) -> VirtualFitData:
    """Return original ordered train/internal sessions without reading sealed data.

    This function opens exactly ``manifest.json`` and ``fit_data.npz``.  It
    never opens, probes, hashes or follows a truth artifact or truth reference.
    Source-byte identity and reserved-case authorization belong to the caller.
    Metadata is preserved exactly as in the existing public loader, excluding
    the session descriptors that are represented by the returned dataclasses.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("synthetic") is not True:
        raise ValueError("only labelled synthetic observations are accepted")
    descriptions = manifest.get("sessions")
    if not isinstance(descriptions, list):
        raise ValueError("public manifest must contain session descriptors")
    names, prefixes, selected = set(), set(), []
    required = {"name", "frequency_hz", "angle_deg", "grid_side", "split", "array_prefix"}
    for item in descriptions:
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError("invalid public session descriptor")
        if not isinstance(item["split"], str) or item["split"] not in _KNOWN_SPLITS:
            raise ValueError(f"unknown session split: {item['split']!r}")
        name, prefix = item["name"], item["array_prefix"]
        if not isinstance(name, str) or not name or not isinstance(prefix, str) or not prefix:
            raise ValueError("session names and array prefixes must be nonempty strings")
        if name in names or prefix in prefixes:
            raise ValueError("duplicate session name or array prefix")
        names.add(name)
        prefixes.add(prefix)
        if item["split"] in ("train", "internal"):
            selected.append(item)
    if not any(item["split"] == "train" for item in selected):
        raise ValueError("training-only input must contain a training session")

    # Filtering above is deliberately before np.load and before every member
    # access below.  Listing archive names does not materialize array members.
    with np.load(directory / "fit_data.npz", allow_pickle=False) as values:
        sessions = tuple(
            VirtualSession(
                item["name"], item["frequency_hz"], item["angle_deg"],
                *[values[item["array_prefix"] + key].copy() for key in _ARRAY_FIELDS],
                item["grid_side"], item["split"],
            )
            for item in selected
        )
        calibration = {
            key.removeprefix("calibration_"): values[key].copy()
            for key in values.files if key.startswith("calibration_")
        }
        elements = values["elements_m"].copy()
    metadata = {key: value for key, value in manifest.items() if key != "sessions"}
    return VirtualFitData(elements, sessions, calibration, metadata)
