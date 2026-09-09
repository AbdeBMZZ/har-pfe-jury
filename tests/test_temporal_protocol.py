import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.data.temporal_protocol import (causal_imu, recording_windows, subject_partition,
    save_protocol, load_protocol, fit_scaler, validate_checkpoint, TemporalAnticipationDataset)


class TemporalProtocolTests(unittest.TestCase):
    def test_future_cannot_change_past(self):
        a = np.random.default_rng(7).normal(size=(100, 6))
        b = a.copy(); b[60:] += 100
        np.testing.assert_array_equal(causal_imu(a, 'g')[:60], causal_imu(b, 'g')[:60])
        np.testing.assert_allclose(causal_imu(a, 'g')[:, 3:], a[:, 3:], rtol=1e-6)

    def test_unlabelled_samples_are_not_compressed(self):
        y = np.ones(40, dtype=int); y[12:20] = 0
        _, _, starts = recording_windows(np.ones((40, 6)), y, 'ms2', window=8, stride=4, warmup=0)
        self.assertEqual(starts.tolist(), [0, 4, 20, 24, 28, 32])

    def test_prefix_and_strict_future_same_targets(self):
        X = np.arange(20 * 8 * 6).reshape(20, 8, 6)
        kwargs = dict(X=X, y=np.arange(20), subjects=np.ones(20), sessions=np.repeat(['a', 'b'], 10),
                      starts=np.tile(np.arange(10)*4, 2), stride=4, seq_len=2, horizon_samples=2)
        full = TemporalAnticipationDataset(**kwargs, obs_ratio=1)
        half = TemporalAnticipationDataset(**kwargs, obs_ratio=.5)
        np.testing.assert_array_equal(full.target_indices, half.target_indices)
        np.testing.assert_array_equal(half.X, full.X[:, :, :4])
        self.assertTrue(np.all(full.actual_horizon_samples >= 2))
        self.assertTrue(np.all(full.target_indices // 10 == full.last_context_indices // 10))
        np.testing.assert_array_equal(half.actual_horizon_samples, full.actual_horizon_samples + 4)

    def test_missing_window_blocks_sequence(self):
        ds = TemporalAnticipationDataset(np.ones((4, 8, 6)), np.ones(4), np.ones(4),
             np.repeat('a', 4), np.array([0, 4, 40, 44]), stride=4, seq_len=2, horizon_samples=2)
        self.assertEqual(len(ds), 0)

    def test_subject_splits_and_scaler(self):
        split = subject_partition(np.arange(10))
        self.assertEqual(split, subject_partition(np.arange(10)))
        self.assertFalse(set(split['train']) & set(split['test']))
        self.assertFalse(set(split['validation']) & set(split['test']))
        self.assertEqual(sum(map(len, split.values())), 10)
        mean, std = fit_scaler(np.full((2, 8, 6), 3.))
        np.testing.assert_array_equal(mean, np.full(6, 3.))
        self.assertTrue(np.all(std > 0))

    def test_manifest_tamper_and_legacy_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = save_protocol(folder, np.ones((5, 8, 6)), np.ones(5), np.arange(5),
                                     np.arange(5).astype(str), np.zeros(5), {'seed': 42})
            _, loaded = load_protocol(folder)
            self.assertEqual(manifest, loaded)
            with self.assertRaises(ValueError):
                validate_checkpoint({}, manifest)
            validate_checkpoint(dict(protocol_id=manifest['protocol_id'], train_subjects=manifest['splits']['train']), manifest)
            np.save(Path(folder)/'y.npy', np.zeros(5))
            with self.assertRaises(ValueError):
                load_protocol(folder)


if __name__ == '__main__':
    unittest.main()
