"""Strict HAPT protocol: recording provenance, causal features and fixed splits.

Separate from legacy merged arrays, whose recording boundaries cannot be recovered.
All sample intervals use [start, end), at 50 Hz.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
from .anticipation_dataset import AnticipationDataset


def causal_imu(raw, acceleration_unit):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6 or not np.isfinite(raw).all():
        raise ValueError('Expected finite (samples, 6) raw IMU data')
    if acceleration_unit not in {'g', 'ms2'}:
        raise ValueError('Specify the unit in the downloaded HAPT README: g or ms2')
    result = raw.copy()
    if acceleration_unit == 'g':
        result[:, :3] *= 9.80665
    # Forward-only filter: changes in future samples cannot change past outputs.
    gravity = sosfilt(butter(5, .5, fs=50, output='sos'), result[:, :3], axis=0)
    result[:, :3] -= gravity
    return result.astype(np.float32)


def recording_windows(raw, labels, unit, window=150, stride=75, warmup=500):
    if window < 1 or stride < 1 or warmup < 0:
        raise ValueError('Invalid window, stride or warmup')
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (len(raw),):
        raise ValueError('One label per raw sample is required')
    signal = causal_imu(raw, unit)
    starts = np.arange(warmup, max(warmup, len(signal) - window + 1), stride, dtype=np.int64)
    # Do not compress the raw stream or bridge unlabelled intervals.
    starts = np.array([s for s in starts if np.all(labels[s:s + window] > 0)], dtype=np.int64)
    if len(starts) == 0:
        return np.empty((0, window, 6), np.float32), np.empty(0, np.int64), starts
    X = np.stack([signal[s:s + window] for s in starts])
    y = np.array([np.bincount(labels[s:s + window]).argmax() for s in starts])
    return X, y, starts


def subject_partition(subjects, seed=42):
    unique = np.unique(subjects)
    if len(unique) < 5:
        raise ValueError('At least five subjects are required for train/validation/test')
    unique = np.random.default_rng(seed).permutation(unique)
    holdout = max(1, len(unique) // 5)
    return {'train': sorted(unique[2 * holdout:].tolist()),
            'validation': sorted(unique[holdout:2 * holdout].tolist()),
            'test': sorted(unique[:holdout].tolist())}


def save_protocol(folder, X, y, subjects, sessions, starts, config):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an experiment's immutable data definition.
    if any(folder.iterdir()):
        raise ValueError('Protocol output directory must be empty; use a new directory')
    arrays = dict(X=X, y=y, subjects=subjects, sessions=sessions, starts=starts)
    if any(len(a) != len(X) for a in arrays.values()) or len(X) == 0:
        raise ValueError('Aligned nonempty arrays are required')
    splits = subject_partition(subjects, config['seed'])
    checksums = {}
    for name, value in arrays.items():
        file = folder / (name + '.npy')
        np.save(file, value, allow_pickle=False)
        checksums[file.name] = file_hash(file)
    manifest = dict(schema_version=1, dataset='HAPT-RawData', config=config,
                    splits=splits, checksums=checksums, n_windows=len(X))
    manifest['protocol_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (folder / 'protocol.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def file_hash(file):
    digest = hashlib.sha256()
    with open(file, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'protocol.json').read_text())
    identity = {k: v for k, v in manifest.items() if k != 'protocol_id'}
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if expected != manifest['protocol_id']:
        raise ValueError('Manifest has changed; create a new protocol')
    for name, checksum in manifest['checksums'].items():
        if file_hash(folder / name) != checksum:
            raise ValueError(f'Protocol array changed: {name}')
    data = {name: np.load(folder / (name + '.npy'), allow_pickle=False, mmap_mode='r')
            for name in ['X', 'y', 'subjects', 'sessions', 'starts']}
    return data, manifest


def fit_scaler(X):
    # Call only on the training subjects. No statistics from validation/test.
    mean = np.mean(X, axis=(0, 1), dtype=np.float64)
    std = np.std(X, axis=(0, 1), dtype=np.float64)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def validate_checkpoint(checkpoint, manifest):
    if checkpoint.get('protocol_id') != manifest['protocol_id']:
        raise ValueError('Checkpoint must come from this exact strict protocol; legacy checkpoint refused')
    if checkpoint.get('train_subjects') != manifest['splits']['train']:
        raise ValueError('Checkpoint training subjects do not match the protocol')


class TemporalAnticipationDataset(AnticipationDataset):
    """Predict the label of a future full window; not time-to-event prediction.

    Context windows must be contiguous within one recording. The same context
    and target indices are used at every observation ratio. Earlier decisions
    therefore have longer actual horizons, which are exported explicitly.
    """
    def __init__(self, X, y, subjects, sessions, starts, *, stride=75,
                 obs_ratio=.5, seq_len=5, horizon_samples=50, majority_label=None):
        if not 0 < obs_ratio <= 1 or seq_len < 1 or horizon_samples < 1 or stride < 1:
            raise ValueError('Require positive sequence/stride/horizon and 0 < ratio <= 1')
        if X.ndim != 3 or len({len(X), len(y), len(subjects), len(sessions), len(starts)}) != 1:
            raise ValueError('Aligned window provenance is required')
        self.obs_ratio, self.seq_len = obs_ratio, seq_len
        self.truncate_from = 'start'
        self.obs_len = max(1, int(X.shape[1] * obs_ratio))
        self.majority_label = majority_label
        contexts, targets, target_indices, last_indices, gaps = [], [], [], [], []
        for session in np.unique(sessions):
            idx = np.flatnonzero(sessions == session)
            if len(np.unique(subjects[idx])) != 1:
                raise ValueError('A recording must belong to exactly one subject')
            idx = idx[np.argsort(starts[idx], kind='stable')]
            ss = starts[idx]
            if np.any(np.diff(ss) <= 0):
                raise ValueError('Window starts must be unique within each recording')
            for j in range(seq_len - 1, len(idx)):
                context_idx = idx[j - seq_len + 1:j + 1]
                if np.any(np.diff(starts[context_idx]) != stride):
                    continue
                context_full_end = int(ss[j]) + X.shape[1]
                k = int(np.searchsorted(ss, context_full_end + horizon_samples))
                if k >= len(idx):
                    continue
                # Do not jump across missing windows or an unlabelled gap.
                if np.any(np.diff(ss[j:k + 1]) != stride):
                    continue
                target_idx = idx[k]
                observed_end = int(ss[j]) + self.obs_len
                contexts.append(X[context_idx, :self.obs_len, :])
                targets.append(y[target_idx])
                target_indices.append(target_idx)
                last_indices.append(idx[j])
                gaps.append(int(starts[target_idx]) - observed_end)
        self.X = (np.asarray(contexts, np.float32) if contexts else
                  np.empty((0, seq_len, self.obs_len, X.shape[-1]), np.float32))
        self.y = np.asarray(targets, np.int64)
        self.target_indices = np.asarray(target_indices, np.int64)
        self.last_context_indices = np.asarray(last_indices, np.int64)
        self.actual_horizon_samples = np.asarray(gaps, np.int64)

    def majority_baseline(self):
        if self.majority_label is None:
            raise ValueError('The constant baseline must be selected on training targets')
        from ..evaluation.metrics import macro_f1
        return macro_f1(self.y, np.full(len(self.y), self.majority_label)) if len(self.y) else float('nan')
