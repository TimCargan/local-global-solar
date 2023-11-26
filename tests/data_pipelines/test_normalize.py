import numpy as np
import numpy.testing as np_testing
from absl.testing import absltest, parameterized
from unittest.mock import patch

from zephyrus.data_pipelines.transformers.normalize import Normalizer
from zephyrus.utils.runner import Runner


class TestNormalizer(parameterized.TestCase):

    @classmethod
    @patch.multiple(Runner, __abstractmethods__=set())
    def setUpClass(cls):
        data = Runner().make_plant_index_data_set(1)() # Read raw data from plant 0
        cls.normalizer: Normalizer = Normalizer()
        cls._dataset = data[0]

    def setUp(self):
        self.raw_data = self._dataset.as_numpy_iterator().next()
        self.norm_data = Normalizer()(self._dataset).as_numpy_iterator().next()

    def test_ensure_shapes(self):
        """
        Make sure the shape and num of cols match
        """
        nx, ny = self.norm_data
        rx, ry = self.raw_data
        self.assertSetEqual(set(rx.keys()), set(nx.keys())) # Ensure they have the same col
        # Make sure shapes are the same
        for col in rx.keys():
            self.assertEqual(rx[col].shape, nx[col].shape)
        self.assertEqual(ry.shape, ny.shape)

    def test_zscore_cols(self):
        """
        - Assert values are in the correct scale
        """
        target_min = -20
        target_max = 20
        x, y = self.norm_data
        for n, m, s in Normalizer.zscore:
            v = x[n]
            with self.subTest(n):
                mean = np.mean(v) * 1e-2 
                self.assertAlmostEqual(0.0, mean, places=1)
                self.assertLessEqual(target_min, v.min())
                self.assertGreaterEqual(target_max, v.max())

    def test_irrad_holds(self):
        irrad = self.raw_data[0]["irradiance"]
        n_irrad = Normalizer._norm_irrad(irrad, clip=True)
        mean = np.mean(n_irrad)
        self.assertBetween(mean, -1, 1)

    @parameterized.parameters((False,), (True,))
    def test_norm_unnorm_identity(self, clip):
        """
        Assert x == unnorm(norm(x))
        Check for both clip and not clip norm
        """
        irrad = self.raw_data[0]["irradiance"]
        norm_data = Normalizer._norm_irrad(irrad, clip)
        unnorm = Normalizer._unnorm_irrad(norm_data, clip)

        irrad = irrad if not clip else np.clip(irrad, a_min=np.expm1(3), a_max=np.expm1(9))
        np_testing.assert_allclose(irrad, unnorm, rtol=1e-5) # Avoid issues with float


if __name__ == '__main__':
    absltest.main()
