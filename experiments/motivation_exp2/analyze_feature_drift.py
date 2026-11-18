"""Scene-equal aggregation, scene bootstrap intervals, and grounded reporting."""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from experiments.motivation_exp1.utils.data_utils import CAMERAS, write_csv, write_json
from experiments.motivation_exp1.utils.stats_utils import scene_bootstrap
from .drift_metrics import stable_key
from .extract_feature_drift import load_artifacts


def analyze(cfg, out, pairs):
    records, arrays, issues = load_artifacts(out, pairs)
    existing = json.loads((out / 'feature_issues.json').read_text()) if (out / 'feature_issues.json').exists() else []
    successful = {r['pair_id'] for r in records}
    missing = [r for r in pairs if r['pair_id'] not in successful]
    for pair in missing:
        if not any(i['pair_id'] == pair['pair_id'] for i in existing + issues):
            issues.append({'pair_id': pair['pair_id'], 'stage': 'analyze', 'error': 'Missing measured pair artifact'})
    write_json(out / 'feature_issues.json', existing + issues)
    if not records:
        raise ValueError('No real measured feature pairs to analyze')
    metrics = ['mean_token_drift', 'median_token_drift', 'p90_token_drift', 'p95_token_drift',
               'normalized_l2_mean', 'top10_contribution', 'top20_contribution', 'gini', 'actual_delta_t_ms']
    metrics += [stable_key(t) for t in cfg['stable_thresholds']]
    metrics += ['content_' + m for m in ['mean_token_drift', 'median_token_drift', 'p90_token_drift', 'p95_token_drift',
                 'top10_contribution', 'top20_contribution', 'gini', *[stable_key(t) for t in cfg['stable_thresholds']]]]
    metrics += ['fully_padded_mean_token_drift', 'content_patch_tokens', 'fully_padded_patch_tokens']
    samples = []
    for row in records:
        samples.append({key: json.dumps(v) if isinstance(v, list) else v for key, v in row.items()})
    # Explicit failure rows, rather than filtering the samples table silently.
    fields = list(samples[0]) + ['error']
    for pair in missing:
        errors = [i['error'] for i in existing + issues if i['pair_id'] == pair['pair_id']]
        samples.append({**pair, 'status': 'failed', 'error': '; '.join(errors)})
    write_csv(out / 'feature_drift_samples.csv', samples, fields)
    groups = defaultdict(list)
    for row in records:
        for key in [('all', 'all'), ('gap', row['gap_group']), ('camera', row['camera_channel']),
                    ('camera_gap', row['camera_channel'] + '|' + row['gap_group'])]:
            groups[key].append(row)
    flat, summaries, scene_rows = [], [], []
    for (dimension, group), rows in sorted(groups.items()):
        stats = {m: scene_bootstrap(rows, m, seed=cfg['seed'], iterations=cfg['bootstrap_iterations']) for m in metrics}
        summary = {'dimension': dimension, 'group': group, 'pairs': len(rows), 'scenes': stats['mean_token_drift']['scenes'], 'metrics': stats}
        summaries.append(summary)
        for metric, stat in stats.items():
            ci = stat['ci95'] or [None, None]
            flat.append({'dimension': dimension, 'group': group, 'metric': metric,
                         'scene_equal_weight_mean': stat['scene_equal_weight_mean'], 'ci95_low': ci[0], 'ci95_high': ci[1],
                         'scenes': stat['scenes'], 'pairs': len(rows), 'bootstrap_unit': 'scene',
                         'bootstrap_iterations': cfg['bootstrap_iterations']})
        scene_tokens = sorted({r['scene_token'] for r in rows})
        for scene in scene_tokens:
            subset = [r for r in rows if r['scene_token'] == scene]
            scene_rows.append({'dimension': dimension, 'group': group, 'scene_token': scene,
                               'scene_name': subset[0]['scene_name'], 'pairs': len(subset),
                               **{m: stats[m]['scene_means'].get(scene) for m in metrics}})
    write_csv(out / 'feature_drift_summary.csv', flat)
    write_csv(out / 'temporal_gap_summary.csv', [r for r in flat if r['dimension'] == 'gap'])
    write_csv(out / 'camera_drift_summary.csv', [r for r in flat if r['dimension'] in ['camera', 'camera_gap']])
    write_csv(out / 'scene_drift_summary.csv', scene_rows)

    # Exact per-pair CDF evaluated on a fixed distance grid; average pairs within
    # each scene before averaging / bootstrapping scenes. No token IID bootstrap.
    xs = np.unique(np.concatenate([np.linspace(0, .5, 501), np.linspace(.5, 2, 151), cfg['stable_thresholds']]))
    curves = []
    for (dimension, group), rows in sorted(groups.items()):
        if dimension not in ['gap', 'camera']:
            continue
        per_scene = defaultdict(list)
        for row in rows:
            d = np.sort(arrays[row['pair_id']])
            per_scene[row['scene_token']].append(np.searchsorted(d, xs, side='right') / len(d))
        matrix = np.stack([np.mean(per_scene[s], axis=0) for s in sorted(per_scene)])
        mean = matrix.mean(axis=0)
        low = high = None
        if len(matrix) >= 2:
            rng = np.random.default_rng(cfg['seed'])
            # Multinomial scene weights avoid allocating iterations*scenes*bins.
            weights = rng.multinomial(len(matrix), np.full(len(matrix), 1 / len(matrix)), size=cfg['bootstrap_iterations']) / len(matrix)
            low, high = np.percentile(weights @ matrix, [2.5, 97.5], axis=0)
        curves.extend({'dimension': dimension, 'group': group, 'distance': float(x), 'cdf': float(mean[i]),
                       'ci95_low': float(low[i]) if low is not None else None,
                       'ci95_high': float(high[i]) if high is not None else None, 'scenes': len(matrix)} for i, x in enumerate(xs))
    write_csv(out / 'cdf_curves.csv', curves)

    # Preserve scene-level maps in addition to equal-scene aggregate maps.
    map_groups = defaultdict(list)
    for row in records:
        map_groups[row['scene_token'], row['camera_channel'], row['gap_group']].append(row)
    maps, region_rows = [], []
    aggregate_maps = defaultdict(list)
    thresholds = cfg['stable_thresholds']
    for (scene, camera, gap), rows in sorted(map_groups.items()):
        matrix = np.stack([arrays[r['pair_id']] for r in rows])
        nr, nc = int(rows[0]['grid_rows']), int(rows[0]['grid_cols'])
        if any((r['grid_rows'], r['grid_cols']) != (nr, nc) for r in rows):
            raise ValueError('Grid layout changed within scene')
        values = {'mean_token_drift': matrix.mean(axis=0)}
        for t in thresholds:
            values[stable_key(t)] = (matrix <= t).mean(axis=0)
        aggregate_maps[camera, gap].append((scene, values, nr, nc))
        for u in range(nr * nc):
            maps.append({'scope': 'scene', 'scene_token': scene, 'scene_name': rows[0]['scene_name'],
                         'camera_channel': camera, 'gap_group': gap, 'row': u // nc, 'col': u % nc,
                         'pairs': len(rows), 'scenes': 1, **{m: float(v[u]) for m, v in values.items()}})
        # 3x3 coarse regions in model coordinates; means and fractions use all
        # real pair tokens in each region. Their CIs are bootstrapped by scene.
        indices = np.arange(nr * nc).reshape(nr, nc)
        for ri, region_r in enumerate(np.array_split(np.arange(nr), 3)):
            for ci, region_c in enumerate(np.array_split(np.arange(nc), 3)):
                selected = indices[np.ix_(region_r, region_c)].ravel()
                d = matrix[:, selected]
                region_rows.append({'scene_token': scene, 'scene_name': rows[0]['scene_name'],
                    'camera_channel': camera, 'gap_group': gap, 'region_row': ri, 'region_col': ci,
                    'mean_token_drift': float(d.mean()),
                    **{stable_key(t): float((d <= t).mean()) for t in thresholds}})
    for (camera, gap), items in sorted(aggregate_maps.items()):
        nr, nc = items[0][2:]
        values = {m: np.mean([v[m] for _, v, _, _ in items], axis=0) for m in items[0][1]}
        for u in range(nr * nc):
            maps.append({'scope': 'scene_equal_mean', 'scene_token': 'ALL', 'scene_name': 'ALL',
                         'camera_channel': camera, 'gap_group': gap, 'row': u // nc, 'col': u % nc,
                         'pairs': sum(len(r) for (s, c, g), r in map_groups.items() if c == camera and g == gap),
                         'scenes': len(items), **{m: float(v[u]) for m, v in values.items()}})
    write_csv(out / 'spatial_drift_maps.csv', maps)
    regions = defaultdict(list)
    for r in region_rows:
        regions[r['camera_channel'], r['gap_group'], r['region_row'], r['region_col']].append(r)
    region_summary = []
    for (camera, gap, ri, ci), rows in sorted(regions.items()):
        for metric in ['mean_token_drift', *[stable_key(t) for t in thresholds]]:
            stat = scene_bootstrap(rows, metric, cfg['seed'], cfg['bootstrap_iterations'])
            low, high = stat['ci95'] or [None, None]
            region_summary.append({'camera_channel': camera, 'gap_group': gap, 'region_row': ri, 'region_col': ci,
                'metric': metric, 'scene_equal_weight_mean': stat['scene_equal_weight_mean'],
                'ci95_low': low, 'ci95_high': high, 'scenes': stat['scenes']})
    write_csv(out / 'region_drift_summary.csv', region_summary)
    layout = json.loads((out / 'model_metadata.json').read_text())['layout']
    write_json(out / 'aggregate_summary.json', {
        'experiment': 'spatial_visual_feature_drift', 'data_kind': 'real_senna_vision_tower_features',
        'successful_pairs': len(records), 'selected_pairs': len(pairs), 'failed_pairs': len(missing),
        'scenes': len({r['scene_token'] for r in records}), 'tokens_analyzed': sum(len(d) for d in arrays.values()),
        'layout': layout, 'thresholds': thresholds, 'groups': summaries,
        'latency': json.loads((out / 'latency_provenance.json').read_text()),
        'definitions': {
            'cosine': '1 - cosine(F_t[u], F_previous[u]); fp32 arithmetic on real fp16 model output, clipped to [0,2]',
            'normalized_l2': 'L2 distance of individually unit-normalized tokens; equals sqrt(2*cosine) up to roundoff',
            'stable': 'cosine distance <= threshold; all configured thresholds reported',
            'topk': 'sum of largest ceil(N*k) distances / total distance; zero-total contribution defined as 0',
            'gini': '2*sum(i*d_sorted_i)/(N*sum(d))-(N+1)/N; zero-total Gini=0; measures unequal magnitude, not spatial adjacency',
            'summary': 'Compute metrics within each pair; mean pairs within scene, then equal-weight scenes. Median/P90/P95 summaries are averages of per-pair token quantiles, not pooled token quantiles.',
            'ci': '95% percentile bootstrap of scene means; no CI with fewer than 2 scenes; pointwise intervals for CDF/threshold curves, not simultaneous bands',
            'cdf': 'Pair token empirical CDF -> within-scene mean -> equal-scene mean and bootstrap',
            'coordinates': 'Corresponding model input patch positions; padded input regions remain part of the actual Tower grid; no physical motion compensation',
            'content_sensitivity': 'Auxiliary subset: patches with >=50% geometric overlap with real image pixels in BOTH endpoints after original expand2square; interpolation at boundaries ignored. No input or model changes. Fully padded subset has zero overlap at both endpoints. Attention can propagate content into padded tokens.',
            'opportunity': 'Stable tokens describe theoretical reuse opportunity only; no mapping to saved FLOPs or measured latency/speedup.'}})
    write_json(out / 'extraction_status.json', {'status': 'partial' if missing else 'complete', 'successful_pairs': len(records), 'failed_pairs': len(missing)})


def report(cfg, out):
    data = json.loads((out / 'aggregate_summary.json').read_text())
    model = json.loads((out / 'model_metadata.json').read_text())
    def fmt(v):
        return 'N/A' if v is None else f'{v:.4f}'
    def stat(group, metric):
        s = group['metrics'][metric]
        ci = s['ci95']
        return fmt(s['scene_equal_weight_mean']) + (f' [{fmt(ci[0])}, {fmt(ci[1])}]' if ci else ' [CI unavailable]')
    layout = data['layout']
    lines = ['# Experiment 2 — Spatially Non-uniform Visual Feature Changes', '',
        f"真实 Senna Vision Tower 特征：{data['successful_pairs']}/{data['selected_pairs']} 对，{data['scenes']} 个 scene，{data['tokens_analyzed']} 个位置比较；失败 {data['failed_pairs']} 对。", '',
        '## 特征提取与布局核验', '',
        f"- 模型：`{model['class_module']}.{model['class']}`；原始 `load_senna_pretrained_model` 加载。",
        '- 调用：`Senna.prepare_inputs_labels_for_multimodal → Senna.encode_images → get_vision_tower()`；Projector 和 Image Adapter 原样执行，省略后续 LLM 生成。',
        '- 预处理复用 `senna_qa_utils.load_images/process_images`，顺序为 ' + ', '.join(CAMERAS) + '。',
        f"- 原始输入 `{layout['raw_input_shape']}`，Tower `{layout['tower_output_shape']}`；每摄像头 {layout['patch_tokens']} 个 Patch Token × {layout['feature_dimension']} 维，网格 {layout['grid_rows']}×{layout['grid_cols']}。",
        f"- Projector 实测首个输出 `{layout['module_shapes']['projector'][0]['output_shape']}`；Image Adapter `{layout['module_shapes']['img_adapter'][0]['output_shape']}`；两者各执行六次。",
        f"- 原始 CLIP {layout['raw_clip_tokens']} 个 Token；CLS 由 Tower 移除：{layout['cls_removed_by_tower']}；分析额外移除 CLS：{layout['analysis_removes_cls']}。",
        f"- Senna 选用 `hidden_states[{model['selected_hidden_state']}]`，实测索引 {layout['selected_hidden_state_index']} / {layout['clip_hidden_states']} 个 hidden states。这里的最终输出指 **Senna 实际消费的 Tower 输出**，不将倒数第二层标记成 CLIP 最后一层。",
        f"- 与原始生成 helper 的六摄像头 Tower 特征逐元素完全相等：{model['original_helper_equivalence']['all_six_tower_features_bitwise_equal']}；预处理输入 SHA256 相等：{model['original_helper_equivalence']['preprocessed_image_sha256_equal']}。",
        '- 直接读取 E1 `temporal_pairs.csv`；同 scene、同 camera、同位置配对。其他五路仅按 manifest 中最近曝光补齐原始六路输入，允许的偏差和每对六路时间偏移均已记录。',
        '- 空间位置对应经过 padding/resize 后的模型输入；包括模型实际看到的 padding 区域。未进行物理对象运动补偿。', '',
        '## 时间间隔与六视角统计', '',
        '所有数值为 scene 等权均值，方括号为 scene Bootstrap 95% CI。Median/P90/P95 是每对 Token 分位数的 scene 平均值。', '',
        '| 分组 | Scenes / Pairs | 实际间隔 ms | Mean | Median | P90 | P95 | Top10% | Top20% | Gini |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    wanted = [g for g in data['groups'] if g['dimension'] in ['gap', 'camera']]
    for g in wanted:
        lines.append('| ' + g['group'] + f" | {g['scenes']} / {g['pairs']} | " + ' | '.join(stat(g, m) for m in ['actual_delta_t_ms', 'mean_token_drift', 'median_token_drift', 'p90_token_drift', 'p95_token_drift', 'top10_contribution', 'top20_contribution', 'gini']) + ' |')
    lines += ['', '## Stable Token Ratio 阈值敏感性', '',
              '| 时间间隔 | ' + ' | '.join(f'd ≤ {t:g}' for t in cfg['stable_thresholds']) + ' |',
              '|---|' + '---:|' * len(cfg['stable_thresholds'])]
    for g in wanted:
        if g['dimension'] == 'gap':
            lines.append('| ' + g['group'] + ' | ' + ' | '.join(stat(g, stable_key(t)) for t in cfg['stable_thresholds']) + ' |')
    overall = next(g for g in data['groups'] if g['dimension'] == 'all')
    lines += ['', '## 空间非均匀性与解释边界', '',
        f"总体 Gini = {stat(overall, 'gini')}；Top10% 贡献 = {stat(overall, 'top10_contribution')}；Top20% 贡献 = {stat(overall, 'top20_contribution')}。",
        f"Top-K 使用向上取整：10% 选 {int(np.ceil(layout['patch_tokens'] * .1))} 个，20% 选 {int(np.ceil(layout['patch_tokens'] * .2))} 个；应与实际 k/N 的均匀基线比较。零总漂移的 Gini 和贡献率定义为 0。",
        'Gini 和 Top-K 描述变化幅度的非均匀性，不能单独证明高漂移位置在空间上连续成团。热力图、稳定位置图和 3×3 区域统计补充位置证据；不把少量高漂移 Token 自动描述为贡献了“大部分”变化。',
        '应结合全部阈值和时间间隔判断证据，结果不预设稳定 Token 必然占多数。本实验不对特征稳定的原因（静态物体、padding、注意力混合等）作因果归因。', '',
        '## Padding 敏感性检查', '',
        '全网格主分析包含真实输入的 padding。下表另取前后两帧均有 ≥50% 几何面积来自原始图像的 Patch 重新计算，避免把 padding 造成的低变化全部归因于道路场景稳定性。边界插值未精确建模；padding Token 仍可通过全局注意力受到图像内容影响。', '',
        '| 时间间隔 | 内容 Patch 数 | 内容区 Mean | 内容区 Gini | 内容区 Top10% | 内容区 Top20% | 全 padding 区 Mean |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in wanted:
        if g['dimension'] == 'gap':
            lines.append('| ' + g['group'] + ' | ' + ' | '.join(stat(g, m) for m in
                ['content_patch_tokens', 'content_mean_token_drift', 'content_gini', 'content_top10_contribution',
                 'content_top20_contribution', 'fully_padded_mean_token_drift']) + ' |')
    lines += ['', '| 内容区域 Stable 比例 | ' + ' | '.join(f'd ≤ {t:g}' for t in cfg['stable_thresholds']) + ' |',
              '|---|' + '---:|' * len(cfg['stable_thresholds'])]
    for g in wanted:
        if g['dimension'] == 'gap':
            lines.append('| ' + g['group'] + ' | ' + ' | '.join(stat(g, 'content_' + stable_key(t)) for t in cfg['stable_thresholds']) + ' |')
    lines += ['', '## 真实视觉计算成本', '']
    latency = data['latency']
    lines += [f"- 来源：`{latency['source']}`，{latency['n']} 个测量；六视角完整 Vision Tower Mean {latency['vision_tower_mean_ms']:.3f} ms，P95 {latency['vision_tower_p95_ms']:.3f} ms。",
        '- 该成本已经包含六个摄像头，不能再乘六。复用判据或补测信息见 `latency_provenance.json`。',
        '- 稳定 Token 比例仅表示理论复用机会，不能等同于 FLOPs 节省或真实时延收益；本实验没有实现缓存、增量执行、跳窗或 Token Pruning，也没有测量加速比。', '',
        '## 完整性、统计与复现', '',
        f"- scene Bootstrap 重采样 {cfg['bootstrap_iterations']} 次，随机种子 {cfg['seed']}；少于两个 scene 不输出 CI。曲线 CI 是逐点区间。",
        '- 所有失败/缺失样本见 `feature_issues.json` 和 samples CSV 的 failed 行；零范数/非有限特征导致整对拒绝。',
        '- `pair_results/` 仅存 CPU 距离数组和配对元数据，支持逐对原子断点恢复；未保存完整视觉特征。',
        '- 原始索引、代码、依赖、权重文件大小/mtime 和逐对结果 SHA256 记录在 metadata 中；改变配置/代码需新目录或显式 overwrite（旧测量归档）。',
        '- 配对选择不依赖图像内容或测得漂移。完整命令：', '', '```bash',
        f"conda run --no-capture-output -n Senna python -m experiments.motivation_exp2.run_feature_drift --config experiments/motivation_exp2/feature_drift_config.yaml --exp1-directory {cfg['exp1_directory']} --checkpoint-path {cfg['checkpoint_path']} --device {cfg['device']} --output-directory {cfg['output_directory']} --existing resume", '```', '',
        '图：`figures/E2_*.png`（300 DPI）及同名矢量 PDF。代表性配对按 scene、camera 和时间顺序选择，不按漂移强弱筛选。', '',
        '测试结果若已运行，见 `verification_results.json`；本报告统计来自真实模型输出，不使用合成特征。']
    path = out / 'experiment_report.md'
    if path.exists() and not path.read_text().startswith('# Experiment 2 — Spatially Non-uniform'):
        archive = out / 'previous_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_geometry_report')
        archive.mkdir(parents=True, exist_ok=True)
        path.rename(archive / 'experiment_report.md')
    path.write_text('\n'.join(lines) + '\n')
