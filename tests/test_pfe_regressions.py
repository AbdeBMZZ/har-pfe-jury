import unittest
import numpy as np
from src.data.preprocessing import preprocess_signal, G
from src.data.har_dataset import HARDataset
from src.data.anticipation_dataset import AnticipationDataset, build_anticipation_datasets
from src.data.homogenization import build_unified_dataset
from src.evaluation.metrics import ContinualResultsMatrix


class NumericalRegressions(unittest.TestCase):
    def test_gravity_and_units_never_modify_gyro(self):
        x = np.ones((200, 6))
        result = preprocess_signal(x, 50, unit='g', has_gravity=True)
        np.testing.assert_allclose(result[:, 3:], x[:, 3:])
        self.assertLess(np.abs(result[:, :3]).max(), .01)

    def test_no_gravity_flag_preserves_acceleration(self):
        x = np.ones((200, 6))
        result = preprocess_signal(x, 50, unit='g', has_gravity=False)
        np.testing.assert_allclose(result[:, :3], G)
        np.testing.assert_allclose(result[:, 3:], 1)

    def test_full_corpus_normalization_rejected(self):
        with self.assertRaisesRegex(ValueError, 'leaks'):
            build_unified_dataset('unused', normalize=True)

    def test_single_subject_split_purges_overlapping_windows(self):
        # Real windows overlap 50%; their sample IDs make leakage detectable.
        x = np.array([np.arange(i * 75, i * 75 + 150) for i in range(20)])[:, :, None]
        ds = HARDataset(x, np.ones(20), np.ones(20), np.array(['hapt'] * 20))
        train, test = ds.train_test_split()
        self.assertEqual(len(set(train.X.ravel()) & set(test.X.ravel())), 0)
        self.assertLess(train.X.max(), test.X.min())

    def test_last_anticipation_target_is_not_dropped(self):
        x = np.arange(4 * 6).reshape(4, 6, 1)
        ds = AnticipationDataset(x, np.arange(4), np.ones(4), seq_len=3)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds.y[0], 3)

    def test_missing_subjects_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Subject IDs'):
            build_anticipation_datasets(np.zeros((20, 6, 1)), np.ones(20))

    def test_missing_fwt_and_joint_reference_are_not_fake_scores(self):
        matrix = ContinualResultsMatrix(3)
        matrix.R[:] = [[.7, .4, np.nan], [.6, .8, .5], [.5, .6, .9]]
        self.assertTrue(np.isnan(matrix.forward_transfer()))
        self.assertTrue(np.isnan(matrix.intransigence()))
        self.assertAlmostEqual(matrix.forward_transfer(np.array([.2, .2, .2])), .25)
        self.assertAlmostEqual(matrix.intransigence(np.array([.8, .9, 1.])), .1)

    def test_forgetting_is_not_always_minus_bwt(self):
        matrix = ContinualResultsMatrix(3)
        matrix.R[:] = [[.7, np.nan, np.nan], [.9, .8, np.nan], [.6, .7, .8]]
        self.assertAlmostEqual(matrix.backward_transfer(), -.1)
        self.assertAlmostEqual(matrix.forgetting(), .2)


try:
    import torch
except (ImportError, OSError):
    torch = None


@unittest.skipIf(torch is None, 'PyTorch absent; no costly installation for this audit')
class TorchRegressions(unittest.TestCase):
    def test_uncertainty_restores_modes_and_reports_entropy(self):
        from src.models.uncertainty_replay import UncertaintyWeightedReplayBuffer
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(6, 2), torch.nn.Dropout(.1))
        with torch.no_grad():
            model[1].weight.zero_()
            model[1].bias.zero_()
        model.train()
        model[2].eval()
        modes = [m.training for m in model.modules()]
        buf = UncertaintyWeightedReplayBuffer(capacity=4, n_classes=2, window_size=2, n_channels=3)
        buf.add_batch(np.zeros((4, 2, 3), dtype=np.float32), np.array([0, 1, 0, 1]))
        buf.sample_uncertain(model, 2, torch.device('cpu'))
        self.assertEqual([m.training for m in model.modules()], modes)
        self.assertAlmostEqual(buf.mean_uncertainty(), np.log(2), places=6)
        buf.add_batch(np.zeros((1, 2, 3), dtype=np.float32), np.array([0]))
        self.assertIsNone(buf._scores)


if __name__ == '__main__':
    unittest.main()
