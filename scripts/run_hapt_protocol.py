"""Supervised recognition then temporal anticipation, with a separate test command."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.temporal_protocol import load_protocol, fit_scaler, validate_checkpoint, TemporalAnticipationDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True)
    p.add_argument('--stage', choices=['recognition', 'anticipation', 'test'], required=True)
    p.add_argument('--checkpoint', help='Recognition checkpoint for anticipation; anticipation checkpoint for test')
    p.add_argument('--out', required=True, help='New checkpoint or test JSON; existing files refused')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--ratio', type=float, default=.5)
    p.add_argument('--seq-len', type=int, default=5)
    p.add_argument('--horizon-samples', type=int, default=50)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if Path(args.out).exists():
        p.error('Output already exists; use a new path')
    if args.epochs < 1 or args.batch_size < 1:
        p.error('Epochs and batch size must be positive')
    if args.stage != 'recognition' and not args.checkpoint:
        p.error('--checkpoint is required for anticipation/test')
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import f1_score, accuracy_score
    from src.models.har_model import build_model
    data, manifest = load_protocol(args.protocol)
    seed = manifest['config']['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    masks = {k: np.isin(data['subjects'], v) for k, v in manifest['splits'].items()}
    model = build_model(n_classes=13).to(args.device)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        validate_checkpoint(checkpoint, manifest)
        expected = 'recognition' if args.stage == 'anticipation' else 'anticipation'
        if checkpoint['stage'] != expected:
            raise ValueError(f'Expected a {expected} checkpoint')
        model.load_state_dict(checkpoint['model'])
        mean, std = np.array(checkpoint['mean']), np.array(checkpoint['std'])
    else:
        mean, std = fit_scaler(data['X'][masks['train']])
    temporal = checkpoint['temporal'] if args.stage == 'test' else dict(
        obs_ratio=args.ratio, seq_len=args.seq_len, horizon_samples=args.horizon_samples,
        stride=manifest['config']['stride'])
    datasets = {}
    for split in (['test'] if args.stage == 'test' else ['train', 'validation']):
        mask = masks[split]
        X = ((data['X'][mask] - mean) / std).astype(np.float32)
        y = data['y'][mask]
        if args.stage == 'recognition':
            datasets[split] = (X, y)
        else:
            ds = TemporalAnticipationDataset(X, y, data['subjects'][mask],
                 data['sessions'][mask], data['starts'][mask], **temporal)
            datasets[split] = (ds.X, ds.y)
            if split == 'test':
                horizons = ds.actual_horizon_samples / manifest['config']['hz']
        if len(datasets[split][1]) == 0:
            raise ValueError(f'No usable examples in {split}; inspect the protocol')
    def loader(split, shuffle=False):
        X, y = datasets[split]
        return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))),
                          batch_size=args.batch_size, shuffle=shuffle)
    forward = model if args.stage == 'recognition' else model.anticipate
    def evaluate(split):
        model.eval()
        predictions = []
        with torch.no_grad():
            for X, _ in loader(split):
                predictions.extend(forward(X.to(args.device)).argmax(-1).cpu().tolist())
        y = datasets[split][1]
        return dict(accuracy=float(accuracy_score(y, predictions)),
                    macro_f1=float(f1_score(y, predictions, labels=list(range(1, 13)), average='macro', zero_division=0)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.stage == 'test':
        scores = evaluate('test')
        y = datasets['test'][1]
        scores.update(protocol_id=manifest['protocol_id'], temporal=temporal, n_examples=len(y),
            macro_f1_labels=list(range(1, 13)),
            majority_macro_f1=float(f1_score(y, np.full(len(y), checkpoint['majority_label']),
                labels=list(range(1, 13)), average='macro', zero_division=0)),
            actual_horizon_seconds_min=float(horizons.min()), actual_horizon_seconds_max=float(horizons.max()))
        Path(args.out).write_text(json.dumps(scores, indent=2))
        print(json.dumps(scores, indent=2)); return
    if args.stage == 'anticipation':
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.anticipation_head.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-3)
    best_score, best_state, history = -1, None, []
    for epoch in range(args.epochs):
        model.train()
        if args.stage == 'anticipation':
            model.backbone.eval()
        for X, y in loader('train', True):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(forward(X.to(args.device)), y.to(args.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        score = evaluate('validation')
        history.append(score)
        print(f'Epoch {epoch + 1}: {score}', flush=True)
        if score['macro_f1'] > best_score:
            best_score, best_state = score['macro_f1'], copy.deepcopy(model.state_dict())
    torch.save(dict(model=best_state, stage=args.stage, protocol_id=manifest['protocol_id'],
        train_subjects=manifest['splits']['train'], mean=mean.tolist(), std=std.tolist(),
        temporal=temporal, history=history, arguments=vars(args),
        majority_label=int(np.bincount(datasets['train'][1]).argmax())), args.out)


if __name__ == '__main__':
    main()
