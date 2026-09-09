"""Create a new strict protocol from original HAPT RawData; never from merged arrays."""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.temporal_protocol import recording_windows, save_protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True, help='HAPT RawData directory containing labels.txt')
    p.add_argument('--out', required=True, help='New empty directory')
    p.add_argument('--acc-unit', required=True, choices=['g', 'ms2'], help='Check the original data README')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    root = Path(args.raw)
    annotations = np.loadtxt(root / 'labels.txt', dtype=np.int64, ndmin=2)
    if annotations.shape[1] != 5:
        raise ValueError('Expected experiment, user, activity, start, end')
    chunks = {k: [] for k in ['X', 'y', 'subjects', 'sessions', 'starts']}
    for exp in np.unique(annotations[:, 0]):
        rows = annotations[annotations[:, 0] == exp]
        users = np.unique(rows[:, 1])
        if len(users) != 1:
            raise ValueError('Multiple subjects in an experiment')
        user = int(users[0])
        tag = f'exp{exp:02d}_user{user:02d}'
        acc = np.loadtxt(root / f'acc_{tag}.txt', ndmin=2)
        gyro = np.loadtxt(root / f'gyro_{tag}.txt', ndmin=2)
        if acc.shape != gyro.shape or acc.shape[1] != 3:
            raise ValueError(f'Invalid IMU shape for {tag}')
        raw = np.concatenate([acc, gyro], axis=1)
        labels = np.zeros(len(raw), np.int64)
        for _, _, label, start, end in rows:
            if not (1 <= start <= end <= len(raw) and 1 <= label <= 12):
                raise ValueError(f'Invalid annotation in {tag}')
            if np.any(labels[start - 1:end] != 0):
                raise ValueError(f'Overlapping annotations in {tag}')
            labels[start - 1:end] = label
        X, y, starts = recording_windows(raw, labels, args.acc_unit)
        if not len(X):
            continue
        for key, value in dict(X=X, y=y, starts=starts,
                               subjects=np.full(len(X), user),
                               sessions=np.full(len(X), tag)).items():
            chunks[key].append(value)
    if not chunks['X']:
        raise ValueError('No valid windows')
    arrays = {k: np.concatenate(v) for k, v in chunks.items()}
    manifest = save_protocol(args.out, **arrays, config=dict(
        seed=args.seed, acceleration_unit=args.acc_unit, hz=50, window=150,
        stride=75, warmup_samples=500, filter='causal Butterworth order=5 cutoff=0.5Hz',
        label_space='original HAPT labels 1..12; no merging', normalization='train-only'))
    print(f"Created {manifest['n_windows']} windows; protocol {manifest['protocol_id']}")
    print(manifest['splits'])


if __name__ == '__main__':
    main()
