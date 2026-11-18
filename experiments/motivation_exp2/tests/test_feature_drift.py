"""Analytical unit checks; these fixtures never supply experiment features."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.motivation_exp1.utils.data_utils import CAMERAS, read_csv
from experiments.motivation_exp1.utils.stats_utils import scene_bootstrap
from experiments.motivation_exp2.drift_metrics import token_distances, drift_statistics, padded_content_fraction
from experiments.motivation_exp2.extract_feature_drift import load_artifacts
from experiments.motivation_exp2.run_feature_drift import ManifestViews, configuration, select_pairs, ensure_index


class MetricTests(unittest.TestCase):
    def test_padding_overlap_matches_original_geometry(self):
        fractions = padded_content_fraction((1600, 900), 24, 24)
        self.assertAlmostEqual(float(fractions.mean()), 900 / 1600)
        self.assertEqual(int((fractions >= .5).sum()), 336)
        self.assertEqual(int((fractions == 0).sum()), 240)
        np.testing.assert_allclose(padded_content_fraction((900, 900), 24, 24), 1)

    def test_known_angles_and_normalization(self):
        a = [[1, 0], [2, 0], [0, 3]]
        b = [[4, 0], [0, 5], [0, -2]]
        d, l2 = token_distances(a, b)
        np.testing.assert_allclose(d, [0, 1, 2])
        np.testing.assert_allclose(l2, [0, np.sqrt(2), 2])

    def test_degenerate_tokens_reject_pair(self):
        for b in [[[0, 0]], [[float('nan'), 1]], [[float('inf'), 1]]]:
            with self.assertRaises(ValueError):
                token_distances([[1, 0]], b)
        with self.assertRaises(ValueError):
            token_distances([[1, 0]], [[1, 0], [0, 1]])

    def test_uniform_and_concentrated_drift(self):
        uniform = drift_statistics(np.ones(576), [0.1, 1])
        self.assertAlmostEqual(uniform['gini'], 0)
        self.assertEqual(uniform['top10_tokens'], 58)
        self.assertAlmostEqual(uniform['top10_contribution'], 58 / 576)
        spike = drift_statistics([0] * 9 + [1], [0, .5, 1])
        self.assertAlmostEqual(spike['gini'], .9)
        self.assertAlmostEqual(spike['top10_contribution'], 1)
        self.assertEqual(spike['stable_ratio_0.5'], .9)
        self.assertEqual(spike['stable_ratio_1'], 1)
        zero = drift_statistics(np.zeros(576), [0, .1])
        self.assertEqual(zero['gini'], 0)
        self.assertEqual(zero['top10_contribution'], 0)

    def test_scene_weight_and_ci(self):
        records = [{'scene_token': 'a', 'x': 0}] * 20 + [{'scene_token': 'b', 'x': 1}]
        result = scene_bootstrap(records, 'x', iterations=1000)
        self.assertEqual(result['scene_equal_weight_mean'], .5)
        self.assertEqual(result['scenes'], 2)
        self.assertIsNotNone(result['ci95'])
        self.assertIsNone(scene_bootstrap(records[:1], 'x')['ci95'])


class PairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.frames = []
        for camera in CAMERAS:
            for t in [1000000, 1083000]:
                path = self.root / f'{camera}_{t}.jpg'
                path.touch()
                self.frames.append({'sample_data_token': f'{camera}_{t}', 'sample_token': 'sample',
                    'scene_name': 'scene', 'scene_token': 's', 'camera_channel': camera,
                    'timestamp_us': str(t), 'image_path': str(path)})
        self.pair = {'pair_id': 'p', 'scene_name': 'scene', 'scene_token': 's', 'camera_channel': CAMERAS[0],
                     'gap_group': 'adjacent', 'actual_delta_t_ms': '83'}
        for side, t in [('previous', 1000000), ('current', 1083000)]:
            self.pair.update({f'{side}_sample_data_token': f'{CAMERAS[0]}_{t}',
                f'{side}_sample_token': 'sample', f'{side}_timestamp_us': str(t),
                f'{side}_image_path': str(self.root / f'{CAMERAS[0]}_{t}.jpg')})

    def tearDown(self):
        self.tmp.cleanup()

    def test_camera_order_and_exact_anchor(self):
        views = ManifestViews(self.frames, 60).views(self.pair, 'previous')
        self.assertEqual([r['camera_channel'] for r in views], list(CAMERAS))
        self.assertEqual(views[0]['sample_data_token'], self.pair['previous_sample_data_token'])

    def test_cross_scene_camera_time_and_path_mismatch(self):
        manifest = ManifestViews(self.frames, 60)
        for key, bad in [('scene_token', 'other'), ('camera_channel', CAMERAS[1]),
                         ('current_timestamp_us', '900000'), ('previous_image_path', 'bad')]:
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError)):
                manifest.views({**self.pair, key: bad}, 'previous')

    def test_no_cross_scene_companion_or_unbounded_offset(self):
        frames = [r for r in self.frames if r['camera_channel'] != CAMERAS[-1]]
        with self.assertRaises((KeyError, ValueError)):
            ManifestViews(frames, 60).views(self.pair, 'previous')
        frames = [dict(r) for r in self.frames]
        for r in frames:
            if r['camera_channel'] == CAMERAS[-1]:
                r['timestamp_us'] = str(int(r['timestamp_us']) + 1000000)
        with self.assertRaises(ValueError):
            ManifestViews(frames, 60).views(self.pair, 'previous')

    def test_checksum_failure_is_reported(self):
        folder = self.root / 'pair_results'
        folder.mkdir()
        (folder / 'p.npz').write_bytes(b'broken')
        (self.root / 'pair_artifacts.json').write_text(json.dumps({'p': 'bad_hash'}))
        records, _, issues = load_artifacts(self.root, [self.pair])
        self.assertFalse(records)
        self.assertIn('checksum', issues[0]['error'])

    def test_balanced_selection_and_config(self):
        pairs = [{**self.pair, 'scene_token': s, 'camera_channel': c, 'gap_group': g,
                  'pair_id': f'{s}-{c}-{g}-{i}'} for s in ['a', 'b'] for c in CAMERAS
                 for g in ['adjacent', '0.25s', '0.5s'] for i in range(3)]
        selected = select_pairs(pairs, {'max_scenes': 2, 'max_pairs': 36, 'seed': 42})
        self.assertEqual(len({(r['scene_token'], r['camera_channel'], r['gap_group']) for r in selected}), 36)
        with self.assertRaises(ValueError):
            configuration(['--stable-thresholds', '.1', '.1'])

    def test_missing_index_delegates_to_e1_despite_stale_complete_status(self):
        (self.root / 'index_status.json').write_text('{"status":"complete"}')
        cfg = {'exp1_directory': str(self.root), 'checkpoint_path': '/tmp/test-checkpoint'}
        def generate(index_cfg, args, out):
            self.assertEqual(args.existing, 'overwrite')
            self.assertEqual(Path(index_cfg['output']['directory']), self.root)
            (out / 'camera_manifest.csv').touch()
            (out / 'temporal_pairs.csv').touch()
        with patch('experiments.motivation_exp1.build_nuscenes_manifest.run', side_effect=generate) as delegated:
            ensure_index(cfg)
            delegated.assert_called_once()
        self.assertEqual(json.loads((self.root / 'index_status.json').read_text())['status'], 'complete')


@unittest.skipUnless(os.environ.get('SENNA_FEATURE_DRIFT_OUTPUT'), 'Set SENNA_FEATURE_DRIFT_OUTPUT for real-run checks')
class RealRunTests(unittest.TestCase):
    def test_raw_distances_and_scene_aggregation(self):
        out = Path(os.environ['SENNA_FEATURE_DRIFT_OUTPUT'])
        pairs = read_csv(out / 'selected_temporal_pairs.csv')
        rows, arrays, issues = load_artifacts(out, pairs)
        self.assertFalse(issues)
        self.assertEqual(len(rows), len(pairs))
        for row in rows:
            d = arrays[row['pair_id']].astype(float)
            # Independent Gini definition, rather than repeating the sorted formula.
            gini = np.abs(d[:, None] - d[None, :]).sum() / (2 * len(d) * d.sum()) if d.sum() else 0
            self.assertAlmostEqual(gini, row['gini'], places=7)
            with np.load(out / 'pair_results' / (row['pair_id'] + '.npz'), allow_pickle=False) as z:
                self.assertEqual(set(z.files), {'cosine', 'normalized_l2', 'row'})
                np.testing.assert_allclose(z['normalized_l2'] ** 2, 2 * z['cosine'], atol=1e-6)
        data = json.loads((out / 'aggregate_summary.json').read_text())
        overall = next(g for g in data['groups'] if g['dimension'] == 'all')
        scenes = {r['scene_token'] for r in rows}
        expected = np.mean([np.mean([arrays[r['pair_id']].mean(dtype=float) for r in rows if r['scene_token'] == s]) for s in scenes])
        self.assertAlmostEqual(expected, overall['metrics']['mean_token_drift']['scene_equal_weight_mean'], places=7)

    def test_spatial_cdf_and_export_consistency(self):
        from PIL import Image
        out = Path(os.environ['SENNA_FEATURE_DRIFT_OUTPUT'])
        data = json.loads((out / 'aggregate_summary.json').read_text())
        maps = read_csv(out / 'spatial_drift_maps.csv')
        curves = read_csv(out / 'cdf_curves.csv')
        for group in data['groups']:
            if group['dimension'] == 'camera_gap':
                spatial = [float(r['mean_token_drift']) for r in maps if r['scope'] == 'scene_equal_mean' and r['camera_channel'] + '|' + r['gap_group'] == group['group']]
                self.assertEqual(len(spatial), data['layout']['patch_tokens'])
                self.assertAlmostEqual(float(np.mean(spatial)), group['metrics']['mean_token_drift']['scene_equal_weight_mean'], places=7)
            if group['dimension'] == 'gap':
                for t in data['thresholds']:
                    cdf = next(float(r['cdf']) for r in curves if r['dimension'] == 'gap' and r['group'] == group['group'] and float(r['distance']) == t)
                    self.assertAlmostEqual(cdf, group['metrics'][f'stable_ratio_{t:g}']['scene_equal_weight_mean'], places=7)
        figures = json.loads((out / 'feature_figure_metadata.json').read_text())['figures']
        for name in figures:
            with Image.open(out / 'figures' / (name + '.png')) as im:
                self.assertTrue(all(abs(v - 300) < 1 for v in im.info['dpi']))
            self.assertTrue((out / 'figures' / (name + '.pdf')).read_bytes().startswith(b'%PDF'))

    def test_real_outputs(self):
        out = Path(os.environ['SENNA_FEATURE_DRIFT_OUTPUT'])
        data = json.loads((out / 'aggregate_summary.json').read_text())
        model = json.loads((out / 'model_metadata.json').read_text())
        self.assertEqual(data['failed_pairs'], 0)
        self.assertEqual(model['class'], 'SennaLlavaLlamaForCausalLM')
        self.assertTrue(model['original_helper_equivalence']['passed'])
        self.assertEqual(data['layout']['tower_output_shape'], [6, 576, 1024])
        self.assertEqual(data['layout']['patch_tokens'], 24 * 24)
        self.assertEqual(len([g for g in data['groups'] if g['dimension'] == 'camera']), 6)
        self.assertEqual(len([g for g in data['groups'] if g['dimension'] == 'gap']), 3)
        for group in data['groups']:
            metrics = group['metrics']
            stable = [metrics[f'stable_ratio_{t:g}']['scene_equal_weight_mean'] for t in data['thresholds']]
            self.assertEqual(stable, sorted(stable))
            self.assertTrue(0 <= metrics['gini']['scene_equal_weight_mean'] <= 1)
            self.assertLessEqual(metrics['top10_contribution']['scene_equal_weight_mean'], metrics['top20_contribution']['scene_equal_weight_mean'] + 1e-7)
        for name in ['feature_drift_samples.csv', 'feature_drift_summary.csv', 'camera_drift_summary.csv',
                     'temporal_gap_summary.csv', 'spatial_drift_maps.csv', 'run_metadata.json', 'experiment_report.md']:
            self.assertTrue((out / name).stat().st_size > 0, name)
        figures = json.loads((out / 'feature_figure_metadata.json').read_text())['figures']
        self.assertGreaterEqual(len(figures), 6)
        for name in figures:
            for suffix in ['png', 'pdf']:
                self.assertGreater((out / 'figures' / f'{name}.{suffix}').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
