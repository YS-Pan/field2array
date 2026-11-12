"""Literal three-element CPU fixtures only; no persisted scientific arrays."""
import unittest
import numpy as np

from array_reconstruct.joint_geometry import Session, FrequencyGroup, JointProblem
from array_reconstruct.joint_newton import profile_gradient_hessian as old_curvature
from array_reconstruct.joint_pose_cpu import PoseJointProblem, PoseFitResult
from array_reconstruct.joint_pose_newton import profile_gradient_hessian, outer_bounds, polish_profile


def fixture():
    elements = np.array([[-.012, -.007, 0.], [.010, -.004, 0.], [.003, .011, 0.]])
    xy = np.array([[-.022, -.018], [-.013, .012], [-.004, -.009], [.006, .019], [.017, -.012], [.024, .006], [.001, .003]])
    field = np.array([.7+.2j, .3-.4j, -.2+.6j, .4+.3j, -.5-.1j, .2+.1j, .8-.3j])
    groups = []
    for k, f in enumerate([12.1e9, 12.5e9, 12.7e9]):
        sessions = tuple(Session(f"f{k}s{j}", f, xy+.0002*j,
            np.array([1., np.exp(.13j*(j+1)), np.exp(-.09j*(j+1))]),
            field*(1+.07*k+.03j*j), weight=.08+.01*k) for j in range(2))
        groups.append(FrequencyGroup(f"f{k}", sessions, .07,
            template_prior=np.array([1., .9, 1.15], complex), template_strength=.11))
    return elements, tuple(groups)


def relative(a, b):
    return np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30)


class PoseCPUFixtureTests(unittest.TestCase):
    def test_zero_pose_reproduces_original_nine_parameter_subspace(self):
        elements, groups = fixture()
        old, new = JointProblem(elements, groups, q=2, backend="qr"), PoseJointProblem(elements, groups)
        gains = {"f0s1": .97+.04j, "f1s1": 1.02-.03j, "f2s1": .99+.02j}
        x9 = old.initial_parameters((59.4, .2, -.3), gains)
        x12 = new.initial_parameters((59.4, .2, -.3), gains, angles_deg=(0., 0., 0.))
        keep = [0, 1, 2, 6, 7, 8, 9, 10, 11]
        a, b = old.evaluate(x9), new.evaluate(x12)
        self.assertEqual(new.parameter_count, 12)
        self.assertEqual(list(new._gain_indices.values()), [6, 8, 10])
        np.testing.assert_allclose(b.pose_u6[3:], 0, atol=0)
        np.testing.assert_allclose(a.residual, b.residual, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(a.jacobian, b.jacobian[:, keep], rtol=2e-10, atol=2e-12)
        for f in a.b:
            np.testing.assert_allclose(a.b[f], b.b[f], rtol=2e-12, atol=2e-13)
            np.testing.assert_allclose(a.db[f], b.db[f][:, keep], rtol=2e-10, atol=2e-12)
        ha, hb = old_curvature(old, x9), profile_gradient_hessian(new, x12)
        np.testing.assert_allclose(ha["gradient"], hb["gradient"][keep], rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(ha["hessian"], hb["hessian"][np.ix_(keep, keep)], rtol=2e-9, atol=2e-11)

    def test_all_twelve_columns_at_nonzero_pose_and_gain_cross_terms(self):
        elements, groups = fixture()
        problem = PoseJointProblem(elements, groups)
        x = problem.initial_parameters((59.4, .2, -.3),
            {"f0s1": .97+.04j, "f1s1": 1.02-.03j, "f2s1": .99+.02j}, angles_deg=(.31, -.27, .83))
        point = profile_gradient_hessian(problem, x)
        value = point["evaluation"]
        np.testing.assert_allclose(point["gradient"], value.jacobian.T@value.residual, rtol=1e-10, atol=1e-12)
        h = 1e-5
        J = np.empty_like(value.jacobian); H = np.empty((12, 12)); g = np.empty(12)
        for j in range(12):
            delta = np.zeros(12); delta[j] = h
            plus, minus = profile_gradient_hessian(problem, x+delta), profile_gradient_hessian(problem, x-delta)
            J[:, j] = (plus["evaluation"].residual-minus["evaluation"].residual)/(2*h)
            H[:, j] = (plus["gradient"]-minus["gradient"])/(2*h)
            g[j] = (plus["evaluation"].cost-minus["evaluation"].cost)/(2*h)
        self.assertLess(relative(J, value.jacobian), 1e-5)
        self.assertLess(relative(H, point["hessian"]), 1e-5)
        self.assertLess(relative(g, point["gradient"]), 1e-5)
        for j in range(12):
            self.assertLess(relative(J[:, j], value.jacobian[:, j]), 1e-5, f"J column {j}")
            self.assertLess(relative(H[:, j], point["hessian"][:, j]), 1e-5, f"H column {j}")
        self.assertGreater(np.linalg.norm(point["hessian"][3:6, 6:]), 1e-8)
        self.assertLess(point["diagnostics"]["hessian_relative_asymmetry_before_symmetrizing"], 1e-10)

    def test_explicit_pose_predictions_validation_and_rejections(self):
        elements, groups = fixture()
        problem = PoseJointProblem(elements, groups)
        x = problem.initial_parameters(angles_deg=(.31, -.27, .83))
        point = problem.evaluate(x)
        session = groups[0].sessions[0]
        with self.assertRaises(TypeError):
            problem.predict(session, point.geometry_mm, point.b["f0"])
        pred = problem.predict(session, point.geometry_mm, point.b["f0"], angles_deg=point.angles_deg)
        zero = problem.predict(session, point.geometry_mm, point.b["f0"], angles_deg=(0., 0., 0.))
        self.assertGreater(np.linalg.norm(pred-zero), 1e-5)
        result = problem.validation_evaluate(x, {"f0": (session,)})
        np.testing.assert_allclose(result["predictions"][session.name], pred, atol=1e-13)
        h = 1e-5; delta = np.zeros(12); delta[4] = h
        vplus = problem.validation_evaluate(x+delta, {"f0": (session,)})["residual"]
        vminus = problem.validation_evaluate(x-delta, {"f0": (session,)})["residual"]
        np.testing.assert_allclose((vplus-vminus)/(2*h), result["jacobian"][:, 4], rtol=2e-6, atol=2e-9)
        lower, upper = outer_bounds(problem)
        np.testing.assert_equal(lower[:6], -1); np.testing.assert_equal(upper[:6], 1)
        self.assertTrue(np.isneginf(lower[7]) and np.isposinf(upper[7]))
        with self.assertRaises(ValueError):
            problem.fit(start_parameters=x, fixed_geometry=[True]*3)
        with self.assertRaises(ValueError):
            problem.initial_parameters(angles_deg=(1.01, 0, 0))
        bad = FrequencyGroup("bad", groups[0].sessions, .07, amplitude_prior=np.ones(3), amplitude_strength=.1)
        with self.assertRaises(NotImplementedError):
            PoseJointProblem(elements, (bad,))
        with self.assertRaises(ValueError):
            PoseJointProblem(elements, groups, backend="normal")

    def test_tiny_trf_and_newton_keep_angles_and_checkpoint_schema(self):
        elements, groups = fixture()
        problem = PoseJointProblem(elements, groups[:1])
        x = problem.initial_parameters(angles_deg=(.2, -.1, .6))
        trace = []
        fit = problem.fit(start_parameters=x, fixed_geometry=[True]*6, max_nfev=2,
                          checkpoint_callback=trace.append)
        self.assertIsInstance(fit, PoseFitResult)
        np.testing.assert_allclose(fit.angles_deg, [.2, -.1, .6], atol=1e-15)
        self.assertTrue(all(len(row["pose_u6"]) == 6 and len(row["angles_deg"]) == 3 for row in trace))
        polished = polish_profile(problem, x, fixed_geometry=[True]*6, maxiter=1)
        np.testing.assert_allclose(polished.angles_deg, [.2, -.1, .6], atol=1e-15)
        self.assertEqual(polished.evaluation.jacobian.shape[1], 8)
        bound_x = x.copy(); bound_x[3] = 1.
        polished = polish_profile(problem, bound_x, maxiter=1)
        self.assertAlmostEqual(polished.diagnostics["startup_nudge"][3], -1e-6)
        self.assertLessEqual(np.max(np.abs(polished.pose_u6)), 1.)


if __name__ == "__main__":
    unittest.main()
