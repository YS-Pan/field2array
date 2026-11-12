import numpy as np
import pytest

from array_reconstruct.joint_virtual import (reference_elements, scalar_regions, scan_grid,
    generate_virtual_case, frequency_key, make_problem, save_virtual_case,
    load_virtual_fit_data, load_virtual_truth_for_scoring, with_training_stride,
    save_virtual_fit_data, build_truth_reference_binding, verify_truth_reference_binding)


def test_registered_full_geometry_order_and_pitch():
    xy, provenance = reference_elements(32)
    assert xy.shape == (1024, 2)
    assert not provenance["formula_only"]
    assert len(provenance["sha256"]) == 64
    assert np.allclose(xy[0] - xy[1], [.0055, .011])
    assert np.allclose((xy.max(axis=0)+xy.min(axis=0))/2, 0)


def test_real_pitch_full_window_regions_are_disjoint_complete():
    for side in (101, 134):
        xy = scan_grid(side)
        assert np.isclose(xy[1, 0]-xy[0, 0], .003)
        assert np.isclose(np.ptp(xy[:, 0]), (side-1)*.003)
        calibration, evaluation = scalar_regions(side)
        assert len(np.intersect1d(calibration, evaluation)) == 0
        assert len(calibration)+len(evaluation) == side*side


def test_rng_streams_and_fit_truth_separation():
    data, truth = generate_virtual_case(43101, 2, snr_db=40)
    quiet, qtruth = generate_virtual_case(43101, 2, snr_db=None)
    assert not hasattr(data, "geometry_mm")
    assert not hasattr(data, "clean_fields")
    assert data.metadata["random_streams"]["noise"]["spawn_key"] != data.metadata["random_streams"]["weights"]["spawn_key"]
    assert np.array_equal(truth.geometry_mm, qtruth.geometry_mm)
    for key in truth.weights:
        assert np.array_equal(truth.weights[key], qtruth.weights[key])
    for ss, qs in zip(data.sessions, quiet.sessions):
        assert np.array_equal(truth.clean_fields[ss.name], qtruth.clean_fields[qs.name])
        assert not np.array_equal(ss.observed_field, qs.observed_field)
    assert {ss.angle_deg for ss in data.sessions if ss.split == "train" and ss.frequency_hz == 12.7e9} == {0, 30, -30, -15}
    assert {ss.angle_deg for ss in data.sessions if ss.split == "internal"} == {15}
    assert {ss.angle_deg for ss in data.sessions if ss.split == "sealed"} == {-45, 45}
    assert len(data.sessions) == 9


def test_confirmation_is_sealed():
    with pytest.raises(ValueError, match="sealed"):
        generate_virtual_case(88201, 2)


def test_shared_problem_uses_only_training_and_independent_frequency_b():
    data, truth = generate_virtual_case(43101, 2)
    problem = make_problem(data, 1e-4, backend="qr")
    assert len(problem.groups) == 3
    assert sorted(len(group.sessions) for group in problem.groups) == [1, 1, 4]
    names = {ss.name for group in problem.groups for ss in group.sessions}
    assert not any(ss.name in names for ss in data.sessions if ss.split != "train")
    assert len(truth.weights) == 3


def test_generator_and_optimizer_forward_agree_without_shared_kernel():
    data, truth = generate_virtual_case(43101, 2, snr_db=None)
    problem = make_problem(data, 1e-4, backend="qr")
    from array_reconstruct.joint_geometry import Session
    for ss in data.sessions:
        ii = np.arange(0, len(ss.samples_xy_m), 73)
        target = Session(ss.name, ss.frequency_hz, ss.samples_xy_m[ii], ss.steering, ss.observed_field[ii])
        predicted = problem.predict(target, truth.geometry_mm, truth.weights[frequency_key(ss.frequency_hz)],
                                    g=truth.session_scalars[ss.name])
        assert np.allclose(predicted, truth.clean_fields[ss.name][ii], rtol=2e-13, atol=2e-11)


def test_independent_response_control_keeps_identical_observation_weights():
    data, _ = generate_virtual_case(43101, 2)
    shared = make_problem(data, 1e-4, model="M2", backend="qr")
    independent = make_problem(data, 1e-4, model="M1", backend="qr")
    shared_weights = {ss.name: ss.weight for group in shared.groups for ss in group.sessions}
    independent_weights = {ss.name: ss.weight for group in independent.groups for ss in group.sessions}
    assert shared_weights.keys() == independent_weights.keys()
    for name in shared_weights:
        assert np.array_equal(shared_weights[name], independent_weights[name])


def test_serialized_fit_artifact_contains_no_truth(tmp_path):
    data, truth = generate_virtual_case(43101, 2)
    folder = tmp_path / "case"
    save_virtual_case(folder, data, truth)
    with np.load(folder / "fit_data.npz") as public:
        assert not any("geometry" in key or "weights" in key or "clean" in key or "scalar" in key for key in public.files)
    with np.load(folder / "truth_for_scoring_only.npz") as hidden:
        assert "geometry_mm" in hidden.files
        assert any("clean_field" in key for key in hidden.files)
    with pytest.raises(FileExistsError):
        save_virtual_case(folder, data, truth)
    restored = load_virtual_fit_data(folder)
    assert not hasattr(restored, "geometry_mm")
    assert restored.metadata == data.metadata
    for before, after in zip(data.sessions, restored.sessions):
        assert before.name == after.name
        assert np.array_equal(before.observed_field, after.observed_field)
        assert np.array_equal(before.samples_xy_m, after.samples_xy_m)
    restored_truth = load_virtual_truth_for_scoring(folder)
    assert np.array_equal(restored_truth.geometry_mm, truth.geometry_mm)
    for key, bb in truth.weights.items():
        assert np.array_equal(bb, restored_truth.weights[key])


def test_dense_matched_sampling_changes_only_training_indices(tmp_path):
    import json
    data, truth = generate_virtual_case(43101, 2, train_stride=5)
    dense = with_training_stride(data, 2)
    assert data.metadata["train_stride"] == 5
    assert dense.metadata["train_stride"] == 2
    assert dense.elements_m is data.elements_m
    assert dense.calibration_relative_amplitude is data.calibration_relative_amplitude
    for original, changed in zip(data.sessions, dense.sessions):
        for name in ("observed_field", "samples_xy_m", "steering", "calibration_indices", "evaluation_indices"):
            assert getattr(original, name) is getattr(changed, name)
        if original.split == "train":
            assert len(changed.train_indices) > len(original.train_indices)
        else:
            assert changed is original
    original_folder, derived_folder = tmp_path / "original", tmp_path / "dense"
    save_virtual_case(original_folder, data, truth)
    save_virtual_fit_data(derived_folder, dense)
    assert not (derived_folder / "truth_for_scoring_only.npz").exists()
    (derived_folder / "truth_reference.json").write_text(json.dumps({"dataset_directory": str(original_folder)}))
    public = load_virtual_fit_data(derived_folder)
    assert public.metadata["train_stride"] == 2
    assert not hasattr(public, "geometry_mm")
    with pytest.raises(ValueError, match="unbound"):
        load_virtual_truth_for_scoring(derived_folder)
    binding = build_truth_reference_binding(derived_folder, original_folder)
    assert binding["truth_array_values_parsed"] is False
    assert verify_truth_reference_binding(binding)["verified"]
    assert np.array_equal(load_virtual_truth_for_scoring(derived_folder, supplementary_binding=binding).geometry_mm, truth.geometry_mm)
    with (original_folder / "manifest.json").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="stale"):
        load_virtual_truth_for_scoring(derived_folder, supplementary_binding=binding)
