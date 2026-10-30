"""Identify the ordered datasets addressed by persisted replay indices.

Hash metadata and manifest contents, without reading/augmenting image tensors.
Image payloads themselves are not hashed. Subsets keep both their ordered index
selection and their parent identity so changes to the source list are detected.
"""
import hashlib
import json
from pathlib import Path

from torch.utils.data import Subset


def _hash_records(records):
    digest = hashlib.sha256()
    for record in records:
        encoded = json.dumps(record, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':')).encode('utf-8')
        digest.update(encoded)
        digest.update(b'\n')
    return digest.hexdigest()


def _path(value):
    return Path(value).expanduser().resolve().as_posix()


def _manifest_identity(dataset, records):
    folder = type(dataset).__module__.rsplit('.', 1)[-1]
    root = Path(dataset.root).expanduser().resolve()
    names = sorted({record['dataset'] for record in records})
    manifests = []
    for name in names:
        directory = root / folder
        if folder == 'vsod_dataset':
            directory /= dataset.mode
        path = directory / name / f'{dataset.mode}.txt'
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(65536), b''):
                digest.update(chunk)
        manifests.append({'path': _path(path), 'sha256': digest.hexdigest()})
    return manifests


def _dataset_identity(dataset):
    kind = f'{type(dataset).__module__}.{type(dataset).__qualname__}'
    identity = {'kind': kind, 'length': len(dataset)}
    if isinstance(dataset, Subset):
        identity['indices_sha256'] = _hash_records(int(index) for index in dataset.indices)
        identity['parent'] = _dataset_identity(dataset.dataset)
    elif hasattr(dataset, 'datas_id'):
        records = dataset.datas_id
        identity['samples_sha256'] = _hash_records(
            {key: _path(value) if key.endswith('_path') else value
             for key, value in record.items()} for record in records)
        identity['manifests'] = _manifest_identity(dataset, records)
    elif hasattr(dataset, 'samples'):
        # Preserve enumeration order: changing it invalidates the replay mapping.
        identity['samples_sha256'] = _hash_records(
            [_path(value) for value in sample] for sample in dataset.samples)
    elif hasattr(dataset, 'dataset'):
        # IndexedDataset adds sample_index but leaves its parent's mapping intact.
        identity['parent'] = _dataset_identity(dataset.dataset)
    else:
        raise ValueError(f'Cannot establish replay data identity for {kind}')
    return identity


def build_training_data_identity(loaders, config):
    return {
        'version': 1,
        'max_train_samples': getattr(config, 'max_train_samples', 0),
        'datasets': {name: _dataset_identity(loader.dataset)
                     for name, loader in sorted(loaders.items())},
    }


def validate_training_data_identity(saved, current):
    if saved is None:
        raise ValueError('Checkpoint has no replay data identity; safe --resume is unavailable. '
                         'Use --model_path for a weights-only warm start with new replay buffers.')
    if saved.get('version') != current['version']:
        raise ValueError('Replay data identity version differs from this run')
    if saved.get('max_train_samples') != current['max_train_samples']:
        raise ValueError('Replay data identity mismatch: max_train_samples changed')
    previous = saved.get('datasets', {})
    if set(previous) != set(current['datasets']):
        raise ValueError('Replay data identity mismatch: selected training datasets changed')
    for name, identity in current['datasets'].items():
        if previous[name] != identity:
            raise ValueError(f'Replay data identity mismatch for {name}: ordered manifests, '
                             'sample paths, dataset length or subset indices changed. '
                             'Restore the original data mapping before --resume.')
