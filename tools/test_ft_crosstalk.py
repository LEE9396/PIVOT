"""간섭 복원, 원본 밀도 행렬, 정보량 부족과 독립 검증 실패를 확인한다."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from calibrate_ft_crosstalk import (calibrate_manifest, correct_difference, evaluate,
                                   fit_crosstalk, load_cycle)
import density_id_drake as alg
import design_core as dc


class CrosstalkChecks(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[0., 1., -2.], [.3, 0., 1.5], [-10., .7, 0.]])
        rng = np.random.default_rng(94)
        self.g = rng.normal(size=(18, 3))
        self.g /= np.linalg.norm(self.g, axis=1)[:, None]
        self.force = alg.G_ACC * 1.332 * self.g
        self.torque = np.cross([.05, -.07, .15], self.force)
        self.ideal = np.column_stack([self.force, self.torque])
        self.measured = self.ideal.copy()
        self.measured[:, :3] += self.torque @ self.k.T

    def test_calibration_preserves_torque_and_original_density_equation(self):
        fit = fit_crosstalk(self.measured[:9], self.force[:9])
        self.assertTrue(fit['identifiable'])
        np.testing.assert_allclose(fit['matrix'], self.k, atol=1e-12)
        result = evaluate(self.measured[9:], self.force[9:], fit['matrix'])
        np.testing.assert_allclose(result['corrected_wrench'], self.ideal[9:], atol=1e-12)
        np.testing.assert_allclose(result['norm_mass_kg'], 1.332, atol=1e-12)
        ratio = .541 / 1.332
        other_mass = evaluate(self.measured[9:]*ratio, self.force[9:]*ratio, fit['matrix'])
        np.testing.assert_allclose(other_mass['norm_mass_kg'], .541, atol=1e-12)
        # 서로 다른 다섯 부품 밀도와 물체 관절각을 사용해 원본 센서 생성 경로부터 검산한다.
        configurations = ([0., 0.], [.5, -.7], [-.6, 1.1], [1., .3])
        matrix = np.vstack([alg.regressor(theta, alg.G_DIRS) for theta in configurations])
        rho = alg.TRUE_RHO.copy()
        self.assertEqual(np.linalg.matrix_rank(matrix), len(rho))
        with patch.object(alg, 'R_EPS_DIAG', np.zeros(6)):
            ideal = np.concatenate([alg.measure(theta, alg.G_DIRS)
                                    for theta in configurations]).reshape(-1, 6)
        corrupted = ideal.copy()
        corrupted[:, :3] += ideal[:, 3:] @ self.k.T
        corrected = correct_difference(corrupted, fit['matrix'])
        np.testing.assert_allclose(corrected.ravel(), matrix @ rho, atol=1e-10)
        def estimate(wrench):
            return dc.wls_map([(matrix, wrench.ravel(), np.eye(len(matrix))*1e-8)],
                              np.full(len(rho), 1000.), np.eye(len(rho))*1e10,
                              alg.RHO_BOUNDS)
        np.testing.assert_allclose(estimate(corrected), rho, atol=1e-4)
        # 총중량이 정확해도 토크가 틀리면 부품 밀도는 틀릴 수 있다.
        wrong_torque = ideal.copy()
        wrong_torque[:, 3] += .02
        np.testing.assert_array_equal(wrong_torque[:, :3], ideal[:, :3])
        self.assertGreater(np.max(np.abs(estimate(wrong_torque) - rho)), 1.)

    def test_rank_noise_and_unmodelled_force_are_not_hidden(self):
        repeated = np.tile(self.measured[0], (100, 1))
        fit = fit_crosstalk(repeated, np.tile(self.force[0], (100, 1)))
        self.assertFalse(fit['identifiable'])
        weak = self.measured.copy()
        weak[:, 3:] *= 1e-4
        self.assertFalse(fit_crosstalk(weak, self.force)['identifiable'])
        held_out = self.measured[9:].copy()
        held_out[:, 2] += 2.
        self.assertGreater(evaluate(held_out, self.force[9:], self.k)['max_force_residual_n'], 1.9)
        wrong_torque = self.ideal.copy()
        wrong_torque[:, 3:] += .18*self.g
        check = evaluate(wrong_torque, self.force, np.zeros((3, 3)))
        self.assertLess(check['max_force_residual_n'], 1e-12)
        self.assertAlmostEqual(check['max_parallel_torque_nm'], .18)
        with self.assertRaises(ValueError):
            correct_difference(self.measured, np.eye(3))

    def write_cycle(self, directory, index):
        item = {'reference_mass_kg': 1.332}
        for j, role in enumerate(('empty_before', 'loaded', 'empty_after')):
            w = np.array([1., 2., 40., .2, -.1, .3])
            if j == 1:
                w += self.measured[index]
            blocks = []
            for b in range(3):
                q = [float(index)] * 6
                blocks.append(dict(wrench_mean=w.tolist(), wrench_samples=[w.tolist()]*2,
                                   achieved_g_hat=self.g[index].tolist(),
                                   joint_before_deg=q, joint_after_deg=q,
                                   sample_time_s=[1000*index+10*j+2*b, 1000*index+10*j+2*b+1],
                                   gripper_before={'position': 147}, gripper_after={'position': 147}))
            data = dict(status='recorded', load_state='loaded' if j == 1 else 'empty',
                        wrench_frame='ft_mount', sensor_sha256='sensor', kinematics_sha256='fk',
                        ft_session_id='test', configuration_id=str(index), blocks=blocks)
            name = f'{index}_{role}.json'
            (directory / name).write_text(json.dumps(data))
            item[role] = name
        return item

    def test_manifest_and_cycle_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cycles = [self.write_cycle(directory, i) for i in range(18)]
            manifest = dict(train=cycles[:9], validation=cycles[9:])
            result = calibrate_manifest(manifest, directory)
            self.assertTrue(result['force_validation_passed'])
            self.assertFalse(result['calibration_valid'])
            self.assertIsNone(result['estimator_reply'])
            bad = deepcopy(manifest)
            bad['validation'][0] = cycles[0]
            with self.assertRaises(ValueError):
                calibrate_manifest(bad, directory)
            path = directory / cycles[0]['loaded']
            data = json.loads(path.read_text())
            for b in data['blocks']:
                b['gripper_before']['position'] = b['gripper_after']['position'] = 3
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, '개구'):
                load_cycle(cycles[0], directory)
            self.write_cycle(directory, 0)
            path = directory / cycles[9]['loaded']
            data = json.loads(path.read_text())
            for b in data['blocks']:
                b['wrench_mean'][2] += 2.
                for sample in b['wrench_samples']:
                    sample[2] += 2.
            path.write_text(json.dumps(data))
            rejected = calibrate_manifest(manifest, directory)
            self.assertFalse(rejected['force_validation_passed'])
            np.testing.assert_allclose(rejected['fit']['matrix'], result['fit']['matrix'])


if __name__ == '__main__':
    unittest.main()
