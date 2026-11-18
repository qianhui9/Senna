"""E2 visual-feature experiment. Reuses E1's index, IO and scene bootstrap."""
import argparse
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
import yaml

from experiments.motivation_exp1.utils.data_utils import (
    ROOT, CAMERAS, digest, read_csv, write_csv, write_json,
)
from .drift_metrics import stable_key

LOG = logging.getLogger(__name__)


def configuration(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(Path(__file__).with_name('feature_drift_config.yaml')))
    p.add_argument('--mode', choices=['all', 'extract', 'analyze', 'plot', 'report'], default='all')
    for key in ['exp1-directory', 'output-directory', 'checkpoint-path', 'device']:
        p.add_argument('--' + key)
    for key in ['max-scenes', 'max-pairs', 'seed', 'bootstrap-iterations', 'profile-samples', 'warmup-iterations']:
        p.add_argument('--' + key, type=int)
    p.add_argument('--stable-thresholds', type=float, nargs='+')
    p.add_argument('--companion-tolerance-ms', type=float)
    p.add_argument('--existing', choices=['resume', 'overwrite'], default='resume')
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    for key in cfg:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    for key in ['exp1_directory', 'output_directory']:
        cfg[key] = str((ROOT / Path(cfg[key]).expanduser()).resolve())
    if cfg['exp1_directory'] == cfg['output_directory']:
        raise ValueError('E2 output must differ from E1 input directory')
    if args.smoke_test:
        cfg['max_scenes'] = min(cfg['max_scenes'], 2)
        cfg['max_pairs'] = min(cfg['max_pairs'], 36)
    for key in ['max_scenes', 'max_pairs', 'bootstrap_iterations', 'warmup_iterations', 'profile_samples']:
        if cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    thresholds = cfg['stable_thresholds']
    if (len(thresholds) < 2 or len(set(thresholds)) != len(thresholds)
            or any(not np.isfinite(t) or not 0 <= t <= 2 for t in thresholds)
            or len({stable_key(t) for t in thresholds}) != len(thresholds)):
        raise ValueError('Supply at least two distinct finite cosine thresholds in [0, 2]')
    cfg['stable_thresholds'] = sorted(thresholds)
    if not np.isfinite(cfg['companion_tolerance_ms']) or cfg['companion_tolerance_ms'] < 0:
        raise ValueError('Invalid companion tolerance')
    if not cfg['device'].startswith('cuda:'):
        raise ValueError('Original Senna fp16 extraction requires an explicit CUDA device')
    return cfg, args


def ensure_index(cfg):
    source = Path(cfg['exp1_directory'])
    if not all((source / name).is_file() for name in ['camera_manifest.csv', 'temporal_pairs.csv']):
        # Delegate to E1; never scan nuScenes or construct temporal pairs here.
        from experiments.motivation_exp1.build_nuscenes_manifest import run
        from experiments.motivation_exp1.utils.data_utils import parser, configuration as e1_configuration
        config = source / 'resolved_config.yaml'
        if not config.is_file():
            config = ROOT / 'experiments/motivation_exp1/config.yaml'
        index_args = parser('index').parse_args(['--config', str(config), '--mode', 'index',
                    '--output-dir', str(source), '--existing', 'overwrite'])
        index_cfg = e1_configuration(index_args)
        source.mkdir(parents=True, exist_ok=True)
        # Invoke only the existing index stage, even if stale index_status says
        # complete. Do not run E1 initialization or clear unrelated E1 results.
        run(index_cfg, index_args, source)
        if not (source / 'resolved_config.yaml').exists():
            (source / 'resolved_config.yaml').write_text(yaml.safe_dump(index_cfg, sort_keys=False))
    prior = source / 'resolved_config.yaml'
    e1 = yaml.safe_load(prior.read_text()) if prior.exists() else {}
    cfg['checkpoint_path'] = (cfg['checkpoint_path'] or os.environ.get('SENNA_MODEL_PATH')
                              or e1.get('senna', {}).get('checkpoint_path', ''))
    if not cfg['checkpoint_path']:
        raise ValueError('Supply --checkpoint-path or SENNA_MODEL_PATH')
    cfg['checkpoint_path'] = str((ROOT / Path(cfg['checkpoint_path']).expanduser()).resolve())


class ManifestViews:
    """In-memory lookup into E1's manifest, not a new dataset index."""
    def __init__(self, frames, tolerance_ms):
        self.tokens, self.groups, self.times = {}, defaultdict(list), {}
        self.tolerance_us = tolerance_ms * 1000
        for row in frames:
            token = row['sample_data_token']
            if token in self.tokens:
                raise ValueError(f'Duplicate manifest token: {token}')
            self.tokens[token] = row
            self.groups[row['scene_token'], row['camera_channel']].append(row)
        for key, seq in self.groups.items():
            seq.sort(key=lambda r: (int(r['timestamp_us']), r['sample_data_token']))
            self.times[key] = [int(r['timestamp_us']) for r in seq]

    def validate_pair(self, pair):
        if pair['camera_channel'] not in CAMERAS:
            raise ValueError('Unknown camera')
        for side in ['previous', 'current']:
            row = self.tokens[pair[f'{side}_sample_data_token']]
            for key in ['scene_token', 'scene_name', 'camera_channel']:
                if row[key] != pair[key]:
                    raise ValueError(f'{side} manifest {key} mismatch')
            for key in ['image_path', 'timestamp_us', 'sample_token']:
                if row[key] != pair[f'{side}_{key}']:
                    raise ValueError(f'{side} manifest {key} mismatch')
        dt = (int(pair['current_timestamp_us']) - int(pair['previous_timestamp_us'])) / 1000
        if dt <= 0 or abs(dt - float(pair['actual_delta_t_ms'])) > 1e-6:
            raise ValueError('Nonpositive or inconsistent time interval')

    def views(self, pair, side):
        self.validate_pair(pair)
        anchor = self.tokens[pair[f'{side}_sample_data_token']]
        timestamp = int(anchor['timestamp_us'])
        result = []
        for camera in CAMERAS:
            if camera == pair['camera_channel']:
                row = anchor
            else:
                key = pair['scene_token'], camera
                seq, times = self.groups[key], self.times[key]
                k = bisect_left(times, timestamp)
                choices = [i for i in [k - 1, k] if 0 <= i < len(seq)]
                if not choices:
                    raise ValueError(f'Missing companion camera {camera}')
                row = seq[min(choices, key=lambda i: (abs(times[i] - timestamp), times[i]))]
            if abs(int(row['timestamp_us']) - timestamp) > self.tolerance_us:
                raise ValueError(f'Companion exposure too far from anchor: {camera}')
            if not Path(row['image_path']).is_file():
                raise FileNotFoundError(row['image_path'])
            result.append(row)
        return result


def select_pairs(pairs, cfg):
    if len({r['pair_id'] for r in pairs}) != len(pairs):
        raise ValueError('Duplicate pair IDs in E1 index')
    scenes = sorted({r['scene_token'] for r in pairs}, key=lambda s: hashlib.sha256(f"{cfg['seed']}:{s}".encode()).hexdigest())[:cfg['max_scenes']]
    strata = defaultdict(list)
    for row in pairs:
        if row['scene_token'] in scenes:
            strata[row['scene_token'], row['camera_channel'], row['gap_group']].append(row)
    for seq in strata.values():
        seq.sort(key=lambda r: hashlib.sha256(f"{cfg['seed']}:{r['pair_id']}".encode()).hexdigest())
    keys = sorted(strata, key=lambda k: hashlib.sha256(f"{cfg['seed']}:{k}".encode()).hexdigest())
    selected, offset = [], 0
    while len(selected) < cfg['max_pairs']:
        batch = [strata[key][offset] for key in keys if offset < len(strata[key])]
        if not batch:
            break
        selected.extend(batch[:cfg['max_pairs'] - len(selected)])
        offset += 1
    return selected


def file_inventory(directory):
    return {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
            for p in sorted(Path(directory).iterdir()) if p.is_file()}


def initialize(cfg, args, selected):
    from importlib.metadata import version
    out = Path(cfg['output_directory'])
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(cfg['checkpoint_path'])
    model_cfg = json.loads((checkpoint / 'config.json').read_text())
    sources = [p for p in Path(__file__).parent.glob('*drift*.py')]
    sources += [ROOT / p for p in ['data_tools/senna_qa_utils.py', 'llava/mm_utils.py',
        'llava/model/builder.py', 'llava/senna/senna_llava_llama.py',
        'llava/senna/senna_llava_arch.py', 'llava/model/multimodal_encoder/clip_encoder.py',
        'experiments/motivation_exp1/utils/data_utils.py', 'experiments/motivation_exp1/utils/stats_utils.py',
        'experiments/motivation_exp1/utils/profiling_utils.py']]
    identity = {'config': cfg, 'sources': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'inputs': {name: digest(Path(cfg['exp1_directory']) / name) for name in ['camera_manifest.csv', 'temporal_pairs.csv']},
        'model_files': file_inventory(checkpoint), 'vision_files': file_inventory(model_cfg['mm_vision_tower']),
        'packages': {p: version(p) for p in ['torch', 'transformers', 'numpy', 'matplotlib', 'PyYAML', 'Pillow']},
        'selected_pair_ids': [r['pair_id'] for r in selected]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    old_path = out / 'run_metadata.json'
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    if old and old.get('experiment') != 'spatial_visual_feature_drift':
        raise ValueError('Output has unrelated run_metadata.json; choose a different output directory')
    if old.get('fingerprint') and old['fingerprint'] != fingerprint and args.existing != 'overwrite':
        raise ValueError('Code/config/index/weights changed; use a new output directory or --existing overwrite')
    if args.existing == 'overwrite' and (out / 'pair_results').exists():
        # Preserve earlier measured data instead of erasing experiment history.
        archive = out / 'previous_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f_feature_drift')
        archive.mkdir(parents=True)
        (out / 'pair_results').rename(archive / 'pair_results')
        owned = ['run_metadata.json', 'aggregate_summary.json', 'extraction_status.json', 'model_metadata.json',
                 'feature_drift_samples.csv', 'feature_drift_summary.csv', 'camera_drift_summary.csv',
                 'temporal_gap_summary.csv', 'spatial_drift_maps.csv', 'experiment_report.md',
                 'selected_temporal_pairs.csv', 'feature_issues.json', 'pair_artifacts.json',
                 'cdf_curves.csv', 'scene_drift_summary.csv', 'region_drift_summary.csv', 'latency_provenance.json']
        for name in owned:
            if (out / name).exists():
                (out / name).rename(archive / name)
    write_json(old_path, {**identity, 'fingerprint': fingerprint, 'experiment': 'spatial_visual_feature_drift',
        'argv': sys.argv, 'python': sys.version, 'camera_order': CAMERAS,
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'comparison': 'same scene, camera and patch index; no physical motion compensation',
        'retained_data': 'pairwise drift arrays only; no stored feature history or cross-request feature reuse'})
    write_csv(out / 'selected_temporal_pairs.csv', selected)
    (out / 'resolved_feature_drift_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main(argv=None):
    cfg, args = configuration(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    ensure_index(cfg)
    pairs = select_pairs(read_csv(Path(cfg['exp1_directory']) / 'temporal_pairs.csv'), cfg)
    if not pairs:
        raise ValueError('No E1 temporal pairs selected')
    if args.dry_run:
        views = ManifestViews(read_csv(Path(cfg['exp1_directory']) / 'camera_manifest.csv'), cfg['companion_tolerance_ms'])
        failures = []
        for pair in pairs:
            try:
                views.views(pair, 'previous'); views.views(pair, 'current')
            except (ValueError, KeyError, OSError) as exc:
                failures.append({'pair_id': pair['pair_id'], 'error': str(exc)})
        print(json.dumps({'config': cfg, 'pairs': len(pairs), 'scenes': len({r['scene_token'] for r in pairs}),
                          'pair_validation_errors': failures}, indent=2))
        return int(bool(failures))
    out = initialize(cfg, args, pairs)
    try:
        if args.mode in ['all', 'extract']:
            from .extract_feature_drift import extract
            extract(cfg, args, out, pairs)
        if args.mode in ['all', 'analyze']:
            from .analyze_feature_drift import analyze
            analyze(cfg, out, pairs)
        if args.mode in ['all', 'plot']:
            from .plot_feature_drift import plot
            plot(cfg, out)
        if args.mode in ['all', 'report']:
            from .analyze_feature_drift import report
            report(cfg, out)
    except Exception as exc:
        write_json(out / 'invocation_feature_drift_status.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
    status = json.loads((out / 'extraction_status.json').read_text()) if (out / 'extraction_status.json').exists() else {}
    failed = status.get('failed_pairs', 0) > 0
    write_json(out / 'invocation_feature_drift_status.json', {'status': 'partial' if failed else 'complete', 'mode': args.mode})
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
