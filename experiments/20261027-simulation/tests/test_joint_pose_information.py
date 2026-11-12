"""Literal2/3 nuisance-column tests; no production data, profile or GPU calls."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from array_reconstruct import joint_pose_information as info


class PoseInformationLiteralTests(unittest.TestCase):
    def project(self, N, Z):
        work = info._work()
        result, report = info._project(N, Z, 1e-10, work)
        return result, report, work

    def test_known_complex_span_rank_and_null(self):
        N = np.array([[1, 0], [0, 1], [0, 0]], complex)
        Z = np.array([[1, 2, 3, 4, 5, 6], [2, 1, 0, 1, 2, 3], [1, 0, 0, 0, 0, 0]], complex)
        projected, report, work = self.project(N, Z)
        expected = np.zeros_like(Z); expected[2] = Z[2]
        np.testing.assert_allclose(projected, expected, atol=1e-13)
        self.assertEqual(report["nuisance_rank_complex"], 2)
        self.assertEqual(work["nuisance_qr_calls"], 1)
        self.assertEqual(work["nuisance_R_svd_calls"], 1)
        zero, report, _ = self.project(np.eye(3, dtype=complex), Z)
        np.testing.assert_allclose(zero, 0, atol=1e-13)
        self.assertEqual(report["nuisance_rank_complex"], 3)

    def test_complex_gain_includes_plus_and_minus_i(self):
        column = np.array([1, 1j, 0], complex)
        N = np.column_stack((column, np.zeros(3)))
        Z = np.zeros((3, 6), complex)
        Z[:, :3] = np.column_stack((column, 1j*column, -1j*column))
        projected, report, _ = self.project(N, Z)
        np.testing.assert_allclose(projected, 0, atol=1e-13)
        self.assertEqual(report["nuisance_rank_complex"], 1)
        self.assertEqual(report["nuisance_rank_real"], 2)
        self.assertEqual(report["zero_nuisance_columns"], 1)

    def test_nuisance_units_do_not_change_projection(self):
        N = np.array([[1, 0], [1j, 1], [0, 1j]], complex)
        Z = np.arange(18).reshape(3, 6)+1j*np.arange(18, 36).reshape(3, 6)
        first, report1, _ = self.project(N, Z)
        second, report2, _ = self.project(N*np.array([1e-8*(1+1j), 1e8*(2-3j)]), Z)
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)
        self.assertEqual(report1["nuisance_rank_complex"], report2["nuisance_rank_complex"])

    def test_three_blocks_stack_six_real_directions_and_units(self):
        N = np.array([[1, 0], [0, 1], [0, 0]], complex)
        work, blocks, gram = info._work(), [], np.zeros((6, 6))
        for block in range(3):
            Z = np.zeros((3, 6), complex)
            Z[2, 2*block], Z[2, 2*block+1] = 1, 1j
            projected, _ = info._project(N, Z, 1e-10, work)
            blocks.append(projected); gram += (Z.conj().T@Z).real
        with patch.object(info.linalg, "svd", wraps=info.linalg.svd) as svd:
            result = info._summary(blocks, gram, 1e-10, work, "literal_data_only")
        self.assertTrue(all(call.kwargs["full_matrices"] is False for call in svd.call_args_list))
        self.assertEqual(result["normalized_u6"]["rank"], 6)
        np.testing.assert_allclose(result["normalized_u6"]["singular_values"], np.ones(6))
        np.testing.assert_allclose(result["per_mm_per_degree"]["singular_values"], [1, 1, 1/3, .1, .1, .1])
        self.assertIsNone(result["precision_or_identifiability_gate"])
        self.assertFalse(result["normalized_u6"]["singular_vector_basis_uniqueness_claim"])

    def test_cancellation_noise_does_not_become_information(self):
        N = np.array([[1, 0], [0, 1], [1, 1]], complex)
        Z = N@np.arange(12).reshape(2, 6)
        work = info._work()
        projected, _ = info._project(N, Z, 1e-10, work)
        result = info._summary([projected], (Z.conj().T@Z).real, 1e-10, work, "literal_data_only")
        self.assertEqual(result["normalized_u6"]["rank"], 0)
        self.assertEqual(result["normalized_u6"]["status"], "rank_deficient_unknown")

    def test_zero_rank_and_short_six_column_spectrum(self):
        work = info._work()
        P, report = info._project(np.zeros((2, 2)), np.zeros((2, 6)), 1e-10, work)
        result = info._summary([P], np.zeros((6, 6)), 1e-10, work, "zero_literal")
        self.assertEqual(report["nuisance_rank_complex"], 0)
        self.assertEqual(result["normalized_u6"]["rank"], 0)
        self.assertEqual(len(result["normalized_u6"]["below_tolerance_directions"]), 6)
        self.assertEqual(work["nuisance_qr_calls"], 0)

    def test_template_augmented_sensitivity_is_distinct(self):
        N = np.array([[1, 0], [0, 1], [0, 0]], complex)
        Z = np.zeros((3, 6), complex); Z[:2, 0], Z[:2, 1] = [1, 1], [1, -1]
        class Group(SimpleNamespace):
            @property
            def b0(self): raise AssertionError("b0 must not enter geometry derivatives")
        group = Group(lambda_reg=0., template_strength=.5, template_prior=np.ones(2), amplitude_strength=0.)
        penalty = info._penalty(group, 2)
        pure, _, _ = self.project(N, Z)
        aug, _, _ = self.project(np.vstack((N, penalty)), np.vstack((Z, np.zeros((len(penalty), 6)))))
        np.testing.assert_allclose(pure, 0, atol=1e-13)
        np.testing.assert_allclose(aug[:, 0], 0, atol=1e-13)
        self.assertGreater(np.linalg.norm(aug[:, 1]), .1)

    def test_tiny_pose_kernel_uses_no_profile_or_gpu(self):
        xy = np.array([[-.006, -.004], [.002, .003], [.008, .001]])
        sessions = tuple(SimpleNamespace(name=f"s{i}", frequency_hz=12.7e9, samples_xy_m=xy,
            steering=np.array([1, np.exp(.2j*i)]), weight=np.array([.2, .3, .4])) for i in range(2))
        group = SimpleNamespace(name="f12.7", sessions=sessions, lambda_reg=.01,
            template_strength=.02, template_prior=np.ones(2), amplitude_strength=0.)
        def forbidden(*args, **kwargs): raise AssertionError("no evaluate/predict/profile allowed")
        marker = object()
        problem = SimpleNamespace(elements_m=np.array([[-.005, 0.], [.005, 0.]]), groups=(group,),
            q_array=2., q_probe=0., sign=-1, evaluate=forbidden, predict=forbidden,
            profile_gradient_hessian=forbidden, counts={"evaluations": 123}, _last=marker)
        result = info.projected_pose_information(problem, [60., 0., 0.], [.2, -.3, .6],
            {"f12.7": np.array([1.+0j, .9+.1j])}, {"s0": 1.+0j, "s1": .9+.1j})
        self.assertEqual(problem.counts, {"evaluations": 123})
        self.assertIs(problem._last, marker)
        self.assertEqual(result["analysis_work"]["profile_evaluations"], 0)
        self.assertEqual(result["analysis_work"]["kernel_first_bundle_calls"], 2)
        self.assertEqual(result["analysis_work"]["kernel_first_derivative_matrices"], 12)
        self.assertEqual(result["analysis_work"]["nuisance_qr_calls"], 2)
        self.assertEqual(result["groups"][0]["data"]["nuisance_columns_complex"], 3)
        self.assertEqual(result["groups"][0]["augmented"]["penalty_rows"], 4)
        self.assertFalse(result["W_is_noise_precision"])
        self.assertFalse(result["old_profile_J_or_H_used"])
        self.assertIn("prior_assisted", result["augmented_prior_assisted"]["residual_kind"])
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
