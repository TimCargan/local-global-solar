import numpy.testing as np_testing
from absl import flags
from absl.testing import absltest, flagsaver, parameterized
from unittest.mock import MagicMock, patch

from zephyrus.model_runners.irrad_with_images import WithImages

FLAGS = flags.FLAGS



class Test_WithImages_Transform(parameterized.TestCase):

    @classmethod
    @flagsaver.flagsaver(img_lookup_min_date="2018-08-01", img_lookup_max_date="2018-09-01",
                         img_lookup_initial_num_buckets=2**8)
    @patch.multiple(WithImages, __abstractmethods__=set())
    def setUpClass(cls):
        """
        Set up a dummy version of
        """
        FLAGS(["Test"])
        runner = WithImages(plant_index=1)
        mock = MagicMock(name="runner")
        runner._run_all = mock
        runner.run()
        data = mock.call_args.args[0]
        first_el = data[0].as_numpy_iterator().next()
        cls._mock_result = mock
        cls._first_el = first_el

    def setUp(self):
        self.first_el = self._first_el

    def test_mock_ok(self):
        """
        Make sure the mock was called ok
        """
        self._mock_result.assert_called()


    def test_irradin_matches_irrad(self):
        x, y = self.first_el
        irrad_in = x["irradiance_in"] #[Batch, Time]
        irrad_pred = y["pred"] # [Batch, Time]
        np_testing.assert_allclose(irrad_in, irrad_pred[:, :irrad_in.shape[-1]], rtol=1e-5)


if __name__ == '__main__':
    absltest.main()
