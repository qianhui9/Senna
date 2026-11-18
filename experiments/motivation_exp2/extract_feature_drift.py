"""Transparent hooks on the real loaded Senna multimodal preparation path."""
import argparse
import inspect
import json
import logging
from pathlib import Path
import time

import numpy as np

from experiments.motivation_exp1.utils.data_utils import CAMERAS, ROOT, digest, read_csv, write_csv, write_json
from experiments.motivation_exp1.utils.profiling_utils import ModuleHooks, CudaTimeline, original_request, tensor_digest
from .drift_metrics import drift_statistics, token_distances, padded_content_fraction
from .run_feature_drift import ManifestViews, file_inventory

LOG = logging.getLogger(__name__)


class SennaFeatures:
    def __init__(self, cfg):
        import torch
        import transformers
        from data_tools.senna_inputs import check_model
        from llava.model.builder import load_senna_pretrained_model
        from llava.senna.senna_llava_llama import SennaLlavaLlamaForCausalLM
        from llava.constants import IMAGE_TOKEN_INDEX
        self.torch, self.cfg = torch, cfg
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; real feature extraction cannot proceed')
        check_model(cfg['checkpoint_path'])
        torch.cuda.set_device(cfg['device'])
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        self.tokenizer, self.model, self.processor, _ = load_senna_pretrained_model(
            cfg['checkpoint_path'], None, model_name='llava', device_map=cfg['device'])
        if not isinstance(self.model, SennaLlavaLlamaForCausalLM):
            raise TypeError('Loader did not return the original Senna model')
        self.model.eval()
        devices = {str(p.device) for p in self.model.parameters()}
        if devices != {cfg['device']}:
            raise ValueError(f'Expected single-device original fp16 model: {devices}')
        self.tower = self.model.get_vision_tower()
        if type(self.tower).__name__ != 'CLIPVisionTower':
            raise ValueError('Only the verified single-scale CLIP token layout is supported')
        self.patch = self.tower.vision_tower.vision_model.embeddings.patch_embedding
        embedding = self.tower.vision_tower.vision_model.embeddings
        embedding_source = inspect.getsource(type(embedding).forward)
        if 'flatten(2).transpose(1, 2)' not in embedding_source:
            raise ValueError('Installed CLIP embedding spatial order requires verification')
        self.ids = torch.tensor([[self.tokenizer.bos_token_id] + [IMAGE_TOKEN_INDEX] * len(CAMERAS)], device=cfg['device'])
        self.modules = {'vision_tower': self.tower, 'projector': self.model.get_model().mm_projector,
                        'img_adapter': self.model.get_model().img_adapter, 'patch_embedding': self.patch}
        self.metadata = {
            'class': type(self.model).__name__, 'class_module': type(self.model).__module__,
            'extraction_path': 'load_senna_pretrained_model -> SennaLlavaLlamaForCausalLM.prepare_inputs_labels_for_multimodal -> Senna.encode_images -> loaded get_vision_tower(); Senna.mm_project executes unchanged',
            'preprocessing': 'data_tools.senna_qa_utils.load_images/process_images; fp16; unsqueeze(0); original six-camera order',
            'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda_version': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(), 'device': cfg['device'], 'precision': 'fp16',
            'gpu_uuid': str(getattr(torch.cuda.get_device_properties(cfg['device']), 'uuid', 'unavailable')),
            'model_config': self.model.config.to_dict(), 'vision_config': self.tower.config.to_dict(),
            'processor': self.processor.to_dict(), 'camera_order': CAMERAS,
            'selected_hidden_state': self.tower.select_layer, 'selected_feature': self.tower.select_feature,
            'embedding_forward_source': embedding_source,
            'weight_files': file_inventory(cfg['checkpoint_path']),
            'vision_files': file_inventory(self.model.config.mm_vision_tower),
            'layer_semantics': 'Required final Tower OUTPUT as consumed by Senna. mm_vision_select_layer may select the penultimate CLIP state; this is not relabeled as the last CLIP block.',
        }
        self.layout = None

    def _layout(self, features, calls, raw):
        shape = list(features.shape)
        patch_shape = calls['patch_embedding'][0]['output_shape']
        image_shape = calls['vision_tower'][0]['input_shape'][0]
        rows, cols = patch_shape[-2:]
        n = rows * cols
        selection = self.tower.select_feature
        expected = n if selection == 'patch' else n + 1 if selection == 'cls_patch' else None
        if shape != [len(CAMERAS), expected, self.tower.config.hidden_size]:
            raise ValueError(f'Unsupported actual Tower output {shape}')
        if raw['selected_shape'] != [len(CAMERAS), n + 1, self.tower.config.hidden_size]:
            raise ValueError('Unexpected CLIP special-token structure')
        if image_shape[-2] // self.patch.kernel_size[0] != rows or image_shape[-1] // self.patch.kernel_size[1] != cols:
            raise ValueError('Patch convolution and input resolution disagree')
        if len(calls['vision_tower']) != 1 or len(calls['projector']) != 6 or len(calls['img_adapter']) != 6:
            raise ValueError('Expected one six-view Tower call and six per-view projections/adapters')
        layout = {'grid_rows': rows, 'grid_cols': cols, 'patch_tokens': n, 'feature_dimension': shape[-1],
                  'raw_clip_tokens': n + 1, 'cls_removed_by_tower': selection == 'patch',
                  'analysis_removes_cls': selection == 'cls_patch', 'tower_output_shape': shape,
                  'raw_input_shape': [1, *image_shape], 'module_shapes': calls,
                  'clip_hidden_states': raw['hidden_states'], 'selected_hidden_state_index': raw['selected_index'],
                  'spatial_order': 'CLIP Conv2d -> flatten(2).transpose(1,2), row-major; prefix CLS excluded; coordinates refer to padded/resized model input'}
        if self.layout and self.layout != layout:
            raise ValueError('Runtime feature layout changed between requests')
        self.layout = layout
        return features[:, 1:] if selection == 'cls_patch' else features

    def extract(self, views, target=None, original=False, capture_input=False, measure=False):
        from data_tools import senna_qa_utils as qa
        torch = self.torch
        captured, raw = {}, {}
        def tower_hook(module, inputs, output):
            captured['features'] = output.detach()
        def clip_hook(module, inputs, output):
            selected = output.hidden_states[self.tower.select_layer]
            raw.update(selected_shape=list(selected.shape), hidden_states=len(output.hidden_states),
                       selected_index=self.tower.select_layer % len(output.hidden_states))
        timeline = CudaTimeline(torch, self.cfg['device']) if measure else None
        paths = [r['image_path'] for r in views]
        with ModuleHooks(self.modules, timeline) as hooks:
            handles = [self.tower.register_forward_hook(tower_hook),
                       self.tower.vision_tower.register_forward_hook(clip_hook)]
            try:
                if original:
                    request = argparse.Namespace(model_path=self.cfg['checkpoint_path'], model_base=None,
                        query='\n'.join(['<image>'] * 6) + '\nDescribe the driving scene.',
                        conv_mode='llava_v1', image_file=paths, sep=',', temperature=0,
                        top_p=None, num_beams=1, max_new_tokens=1)
                    _, _, snapshot = original_request(request, self.tokenizer, self.model, self.processor, True)
                else:
                    images = qa.load_images(paths)
                    try:
                        tensor = qa.process_images(images, self.processor, self.model.config).to(self.model.device, dtype=torch.float16).unsqueeze(0)
                        snapshot = {'original_image_sizes': [im.size for im in images]}
                        if capture_input:
                            snapshot['images_sha256'] = tensor_digest(tensor)
                        with torch.inference_mode():
                            result = self.model.prepare_inputs_labels_for_multimodal(
                                self.ids, None, None, None, None, tensor, image_sizes=[im.size for im in images])
                        del result, tensor
                    finally:
                        for im in images:
                            im.close()
                totals = timeline.totals() if timeline else {}
                features = self._layout(captured.pop('features'), hooks.calls, raw)
                # Only the target camera is copied to CPU in the experiment loop.
                features = (features if target is None else features[target]).float().cpu().numpy()
                return features, totals, snapshot
            finally:
                for handle in handles:
                    handle.remove()
                captured.clear()

    def verify(self, views):
        original, _, snapshot = self.extract(views, original=True)
        prepared, _, prepared_input = self.extract(views, capture_input=True)
        equal = np.array_equal(original, prepared)
        same_input = snapshot['images_sha256'] == prepared_input['images_sha256']
        self.metadata['original_helper_equivalence'] = {
            'passed': equal and same_input, 'all_six_tower_features_bitwise_equal': equal,
            'preprocessed_image_sha256_equal': same_input, 'original_image_shape': snapshot['image_shape'],
            'paths': [r['image_path'] for r in views],
            'original_helper': 'data_tools.senna_qa_utils.eval_multi_img_model_wo_init',
            'reason_visual_only': 'Prompt tokens do not enter encode_images. Direct original multimodal preparation omits LLM generation; all six camera Tower outputs match the original generation helper exactly.'}
        del original, prepared
        if not equal or not same_input:
            raise RuntimeError('Original Senna helper and visual-only preparation disagree')


def reused_latency(cfg, extractor):
    """Conservative reuse, with explicit evidence for every available comparison."""
    source = Path(cfg['exp1_directory'])
    try:
        old = json.loads((source / 'model_metadata.json').read_text())
        run = json.loads((source / 'run_metadata.json').read_text())
        metadata = extractor.metadata
        checks = {key: old.get(key) == json.loads(json.dumps(metadata.get(key), default=str)) for key in
                  ['class', 'gpu', 'torch', 'transformers', 'cuda_version', 'vision_config', 'processor', 'weight_files']}
        checks['model_config'] = old['model_config'] == json.loads(json.dumps(metadata['model_config'], default=str))
        checks['vision_files'] = run.get('vision_files') == metadata['vision_files']
        checks['device'] = run['config']['senna']['device'] == cfg['device']
        checks['precision'] = run['config']['senna']['precision'] == 'fp16'
        for name in ['llava/model/builder.py', 'llava/senna/senna_llava_llama.py',
                     'llava/senna/senna_llava_arch.py', 'llava/model/multimodal_encoder/clip_encoder.py',
                     'data_tools/senna_qa_utils.py']:
            checks[name] = run['sources'].get(name) == digest(ROOT / name)
        rows = [r for r in read_csv(source / 'profiling_samples.csv')
                if r['measurement_mode'] == 'timing' and r['timing_decomposition_valid'].lower() == 'true']
        actual_shape = extractor.layout['tower_output_shape']
        checks['all_observed_shapes'] = bool(rows) and all(
            json.loads(r['input_shape']) == extractor.layout['raw_input_shape'] and
            json.loads(r['module_shapes'])['vision_tower'][0]['output_shape'] == actual_shape and
            int(r['vision_tower_calls']) == 1 for r in rows)
        if not all(checks.values()):
            return {'source': 'new_profiling', 'reuse_checks': checks, 'reason': 'E1 identity checks did not all pass'}
        values = [float(r['vision_tower_ms']) for r in rows]
        return {'source': 'experiment1', 'reuse_checks': checks, 'n': len(values),
                'vision_tower_mean_ms': float(np.mean(values)), 'vision_tower_p95_ms': float(np.percentile(values, 95)),
                'six_view_complete_encoding_mean_ms': float(np.mean(values)),
                'source_path': str(source / 'profiling_samples.csv'), 'source_sha256': digest(source / 'profiling_samples.csv'),
                'scope': 'One Tower call encodes all six views. Do not multiply by six.',
                'hardware_identity_limit': 'E1 recorded GPU model and device ordinal, not GPU serial/UUID; comparison is at that recorded granularity.'}
    except (OSError, ValueError, KeyError) as exc:
        return {'source': 'new_profiling', 'reason': str(exc)}


def load_artifacts(out, pairs):
    """Read atomic per-pair CPU drift products, validating declared checksums."""
    index_path = out / 'pair_artifacts.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    records, arrays, issues = [], {}, []
    for pair in pairs:
        pid = pair['pair_id']
        path = out / 'pair_results' / (pid + '.npz')
        if not path.exists():
            continue
        try:
            if pid in index and index[pid] != digest(path):
                raise ValueError('Pair checkpoint checksum mismatch')
            with np.load(path, allow_pickle=False) as data:
                row = json.loads(str(data['row']))
                d, l2 = data['cosine'].copy(), data['normalized_l2'].copy()
            if row['pair_id'] != pid or not np.isfinite(d).all() or d.ndim != 1 or len(d) != row['patch_tokens']:
                raise ValueError('Invalid pair checkpoint contents')
            if l2.shape != d.shape or not np.isfinite(l2).all():
                raise ValueError('Invalid auxiliary distances')
            records.append(row); arrays[pid] = d
        except (OSError, ValueError, KeyError) as exc:
            issues.append({'pair_id': pid, 'stage': 'resume', 'error': str(exc)})
    return records, arrays, issues


def extract(cfg, args, out, pairs):
    started = time.monotonic()
    artifact_dir = out / 'pair_results'
    artifact_dir.mkdir(exist_ok=True)
    records, _, issues = load_artifacts(out, pairs)
    done = {r['pair_id'] for r in records}
    manifest = ManifestViews(read_csv(Path(cfg['exp1_directory']) / 'camera_manifest.csv'), cfg['companion_tolerance_ms'])
    if len(done) == len(pairs) and (out / 'model_metadata.json').is_file() and (out / 'latency_provenance.json').is_file():
        LOG.info('Reusing %d checksum-validated measured pairs', len(done))
        write_json(out / 'extraction_status.json', {'status': 'complete', 'successful_pairs': len(done), 'failed_pairs': 0})
        return
    extractor = SennaFeatures(cfg)
    # Find a valid endpoint without silently losing any failed pairs below.
    valid_views = None
    for pair in pairs:
        try:
            valid_views = manifest.views(pair, 'previous')
            break
        except (ValueError, KeyError, OSError):
            continue
    if valid_views is None:
        raise ValueError('No valid six-view endpoint; run --dry-run for pair errors')
    extractor.verify(valid_views)
    for _ in range(cfg['warmup_iterations']):
        extractor.extract(valid_views, target=0)
    extractor.metadata['layout'] = extractor.layout
    write_json(out / 'model_metadata.json', extractor.metadata)
    latency = reused_latency(cfg, extractor)
    new_timings = []
    artifact_index = {r['pair_id']: digest(artifact_dir / (r['pair_id'] + '.npz')) for r in records}
    write_json(out / 'extraction_status.json', {'status': 'running', 'successful_pairs': len(done), 'selected_pairs': len(pairs)})
    for i, pair in enumerate(pairs):
        if pair['pair_id'] in done:
            continue
        try:
            previous_views, current_views = manifest.views(pair, 'previous'), manifest.views(pair, 'current')
            target = CAMERAS.index(pair['camera_channel'])
            measure = latency['source'] == 'new_profiling' and len(new_timings) < cfg['profile_samples']
            previous, timings, previous_info = extractor.extract(previous_views, target=target, measure=measure)
            if measure:
                new_timings.append(timings['vision_tower'])
            current, _, current_info = extractor.extract(current_views, target=target)
            cosine, l2 = token_distances(previous, current)
            del previous, current
            row = {**pair, 'status': 'ok', 'patch_tokens': len(cosine), 'feature_dimension': extractor.layout['feature_dimension'],
                   'grid_rows': extractor.layout['grid_rows'], 'grid_cols': extractor.layout['grid_cols'],
                   **drift_statistics(cosine, cfg['stable_thresholds']), 'normalized_l2_mean': float(l2.mean())}
            if extractor.model.config.image_aspect_ratio != 'pad':
                raise ValueError('Content-region sensitivity currently verifies original square padding only')
            fractions = [padded_content_fraction(info['original_image_sizes'][target], row['grid_rows'], row['grid_cols'])
                         for info in [previous_info, current_info]]
            content = np.minimum(*fractions) >= .5
            padded = np.maximum(*fractions) <= 1e-9
            if not content.any():
                raise ValueError('No predominantly image-content patches')
            row.update({'content_' + key: value for key, value in drift_statistics(cosine[content], cfg['stable_thresholds']).items()})
            row['content_patch_tokens'] = int(content.sum())
            row['fully_padded_patch_tokens'] = int(padded.sum())
            row['fully_padded_mean_token_drift'] = float(cosine[padded].mean()) if padded.any() else None
            row['previous_original_size'] = previous_info['original_image_sizes'][target]
            row['current_original_size'] = current_info['original_image_sizes'][target]
            for side, views in [('previous', previous_views), ('current', current_views)]:
                row[side + '_six_view_tokens'] = [r['sample_data_token'] for r in views]
                row[side + '_six_view_paths'] = [r['image_path'] for r in views]
                row[side + '_companion_offset_ms'] = [(int(r['timestamp_us']) - int(pair[side + '_timestamp_us'])) / 1000 for r in views]
            path = artifact_dir / (pair['pair_id'] + '.npz')
            tmp = path.with_suffix('.npz.tmp')
            with tmp.open('wb') as handle:
                np.savez_compressed(handle, cosine=cosine, normalized_l2=l2, row=json.dumps(row))
            tmp.replace(path)
            artifact_index[pair['pair_id']] = digest(path)
            write_json(out / 'pair_artifacts.json', artifact_index)
            records.append(row)
            LOG.info('Pair %d/%d %s %s: mean=%.4f Gini=%.3f', i + 1, len(pairs), pair['camera_channel'], pair['gap_group'], row['mean_token_drift'], row['gini'])
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            issues.append({'pair_id': pair['pair_id'], 'scene_token': pair['scene_token'],
                           'camera_channel': pair['camera_channel'], 'stage': 'extract', 'error': f'{type(exc).__name__}: {exc}'})
            write_json(out / 'feature_issues.json', issues)
            LOG.exception('Failed pair %s', pair['pair_id'])
            if isinstance(exc, RuntimeError):
                # CUDA errors may invalidate the context; preserve completed work.
                raise
    if latency['source'] == 'new_profiling':
        # A resumed run may have too few remaining pairs; independently profile
        # the verified real endpoint to finish the requested small measurement.
        while len(new_timings) < cfg['profile_samples']:
            _, timing, _ = extractor.extract(valid_views, target=0, measure=True)
            new_timings.append(timing['vision_tower'])
        latency.update(n=len(new_timings), vision_tower_mean_ms=float(np.mean(new_timings)),
                       vision_tower_p95_ms=float(np.percentile(new_timings, 95)),
                       six_view_complete_encoding_mean_ms=float(np.mean(new_timings)),
                       samples_ms=new_timings, warmup_iterations=cfg['warmup_iterations'],
                       scope='CUDA events around the original six-view Tower; feature copy occurs after end event')
    write_json(out / 'latency_provenance.json', latency)
    successful = {r['pair_id'] for r in records}
    missing = set(r['pair_id'] for r in pairs) - successful
    write_json(out / 'feature_issues.json', issues)
    write_json(out / 'extraction_status.json', {'status': 'partial' if missing else 'complete',
               'successful_pairs': len(successful), 'failed_pairs': len(missing), 'failed_pair_ids': sorted(missing),
               'seconds': time.monotonic() - started})
    del extractor
