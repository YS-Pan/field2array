"""I/O boundary tests: no physical solve and no CUDA use."""
import json
from pathlib import Path

import numpy as np
import pytest

from array_reconstruct import joint_fit_io
from array_reconstruct.joint_virtual import (
    VirtualFitData, VirtualSession, load_virtual_fit_data, save_virtual_fit_data,
)


@pytest.fixture
def public_case(tmp_path):
    sessions = []
    for index, split in enumerate(["train"] * 6 + ["internal", "sealed", "sealed"]):
        sessions.append(VirtualSession(
            f"session_{index}", 12.7e9 if index < 7 else 12.1e9,
            float(index), np.arange(8, dtype=float).reshape(4, 2) + index,
            np.arange(4, dtype=float).astype(complex) + 1j * index,
            np.arange(4, dtype=float).astype(complex) + index + 1j,
            np.array([0, 2]), np.array([0]), np.array([1, 2, 3]), 2, split,
        ))
    data = VirtualFitData(np.arange(8, dtype=float).reshape(4, 2), tuple(sessions),
                          {"f12.7": np.arange(4, dtype=float) + 1},
                          {"synthetic": True, "seed": 43101, "identity": {"same": [1, 2]}})
    target = tmp_path / "public"
    save_virtual_fit_data(target, data)
    # Existence alone must never tempt the fit loader to inspect latent bytes.
    (target / "truth_for_scoring_only.npz").write_bytes(b"must-never-be-opened")
    return target


def test_member_access_filters_sealed_before_materialization(public_case, monkeypatch):
    original_load = joint_fit_io.np.load
    original_open = Path.open
    accessed = []
    def path_open(path, *args, **kwargs):
        if "truth" in path.name or "clean" in path.name:
            raise AssertionError(f"latent file opened: {path}")
        return original_open(path, *args, **kwargs)
    class MemberTrap:
        def __init__(self, archive):
            self.archive = archive
            self.files = archive.files
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.archive.close()
        def __getitem__(self, key):
            accessed.append(key)
            if key.startswith(("session7_", "session8_", "truth", "clean")):
                raise AssertionError(f"forbidden member was materialized: {key}")
            return self.archive[key]
    def guarded_load(path, **kwargs):
        assert path.name == "fit_data.npz"
        assert kwargs == {"allow_pickle": False}
        return MemberTrap(original_load(path, **kwargs))
    monkeypatch.setattr(joint_fit_io.np, "load", guarded_load)
    monkeypatch.setattr(Path, "open", path_open)
    loaded = joint_fit_io.load_virtual_training_data(public_case)
    assert len(loaded.sessions) == 7
    assert [s.split for s in loaded.sessions] == ["train"] * 6 + ["internal"]
    assert not any(k.startswith(("session7_", "session8_")) for k in accessed)
    assert len(accessed) == 7 * 6 + 2


def test_excluded_sealed_object_array_is_never_decoded(public_case):
    archive = public_case / "fit_data.npz"
    with np.load(archive, allow_pickle=False) as previous:
        arrays = {key: previous[key] for key in previous.files}
    arrays["session7_observed_field"] = np.array([{"latent": "forbidden object"}], dtype=object)
    np.savez_compressed(archive, **arrays)
    loaded = joint_fit_io.load_virtual_training_data(public_case)
    assert len(loaded.sessions) == 7


def test_every_used_array_exactly_matches_filtered_original_loader(public_case):
    old = load_virtual_fit_data(public_case)
    new = joint_fit_io.load_virtual_training_data(public_case)
    assert new.metadata == old.metadata
    np.testing.assert_array_equal(new.elements_m, old.elements_m)
    assert new.calibration_relative_amplitude.keys() == old.calibration_relative_amplitude.keys()
    for key in old.calibration_relative_amplitude:
        np.testing.assert_array_equal(new.calibration_relative_amplitude[key], old.calibration_relative_amplitude[key])
    expected = [s for s in old.sessions if s.split in ("train", "internal")]
    assert [s.name for s in new.sessions] == [s.name for s in expected]
    for actual, previous in zip(new.sessions, expected):
        assert (actual.name, actual.frequency_hz, actual.angle_deg, actual.grid_side, actual.split) == (
            previous.name, previous.frequency_hz, previous.angle_deg, previous.grid_side, previous.split)
        for field in joint_fit_io._ARRAY_FIELDS:
            current, prior = getattr(actual, field), getattr(previous, field)
            np.testing.assert_array_equal(current, prior)
            assert current.dtype == prior.dtype


@pytest.mark.parametrize("change", ["unknown_split", "synthetic_false", "duplicate_prefix"])
def test_invalid_manifest_rejected_before_archive_open(public_case, monkeypatch, change):
    path = public_case / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if change == "unknown_split":
        manifest["sessions"][-1]["split"] = "mystery"
    elif change == "synthetic_false":
        manifest["synthetic"] = False
    else:
        manifest["sessions"][-1]["array_prefix"] = manifest["sessions"][0]["array_prefix"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    def forbidden_open(*args, **kwargs):
        raise AssertionError("archive opened before manifest rejection")
    monkeypatch.setattr(joint_fit_io.np, "load", forbidden_open)
    with pytest.raises(ValueError):
        joint_fit_io.load_virtual_training_data(public_case)
