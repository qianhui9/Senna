"""E1 plotting style: 300 DPI PNG and vector PDF, with scene-level evidence."""
from collections import defaultdict
import json
import os

import numpy as np

from experiments.motivation_exp1.utils.data_utils import CAMERAS, read_csv, write_json
from .drift_metrics import stable_key
from .extract_feature_drift import load_artifacts


def plot(cfg, out):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/senna_feature_drift_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    data = json.loads((out / 'aggregate_summary.json').read_text())
    groups = data['groups']
    gaps = sorted([g for g in groups if g['dimension'] == 'gap'],
                  key=lambda g: g['metrics']['actual_delta_t_ms']['scene_equal_weight_mean'])
    gap_names = [g['group'] for g in gaps]
    curves = read_csv(out / 'cdf_curves.csv')
    folder = out / 'figures'
    folder.mkdir(exist_ok=True)
    metadata = {'style_source': 'experiments/motivation_exp1/plot_results.py', 'figures': {}}
    def save(fig, name, description):
        fig.savefig(folder / (name + '.png'), dpi=300, bbox_inches='tight')
        fig.savefig(folder / (name + '.pdf'), bbox_inches='tight')
        plt.close(fig)
        metadata['figures'][name] = {'png_dpi': 300, 'pdf': 'vector', 'description': description}
    colors = plt.get_cmap('tab10')
    def cdf(ax, dimension, names):
        xmax = 0.1
        for i, name in enumerate(names):
            rows = [r for r in curves if r['dimension'] == dimension and r['group'] == name]
            if not rows:
                continue
            xs = np.array([float(r['distance']) for r in rows])
            ys = np.array([float(r['cdf']) for r in rows])
            xmax = max(xmax, xs[min(np.searchsorted(ys, .995), len(xs) - 1)])
            ax.plot(xs, ys, label=name.removeprefix('CAM_'), color=colors(i))
            if rows[0]['ci95_low']:
                ax.fill_between(xs, [float(r['ci95_low']) for r in rows], [float(r['ci95_high']) for r in rows], color=colors(i), alpha=.13)
        ax.set(xlabel='Token cosine distance', ylabel='Scene-equal mean CDF', xlim=(0, min(2, xmax * 1.05)), ylim=(0, 1.01))
        ax.legend(fontsize=8)
    fig, ax = plt.subplots(figsize=(6.5, 3.7), layout='constrained')
    cdf(ax, 'gap', gap_names)
    ax.set_title('Token Drift by Temporal Gap (pointwise scene bootstrap CI)')
    save(fig, 'E2_drift_cdf', 'Exact empirical token CDF; pairs averaged within scenes, then scenes equally')

    fig, ax = plt.subplots(figsize=(6.5, 3.7), layout='constrained')
    for i, g in enumerate(gaps):
        stats = [g['metrics'][stable_key(t)] for t in cfg['stable_thresholds']]
        ax.plot(cfg['stable_thresholds'], [s['scene_equal_weight_mean'] for s in stats], 'o-', label=g['group'], color=colors(i))
        if all(s['ci95'] for s in stats):
            ax.fill_between(cfg['stable_thresholds'], [s['ci95'][0] for s in stats], [s['ci95'][1] for s in stats], color=colors(i), alpha=.15)
    ax.set(xlabel='Cosine distance threshold', ylabel='Stable token ratio (d ≤ threshold)', ylim=(0, 1.02), title='Threshold Sensitivity (pointwise scene bootstrap CI)')
    ax.legend()
    save(fig, 'E2_stable_thresholds', 'All configured thresholds; no threshold chosen after observing results')

    fig, ax = plt.subplots(figsize=(7, 4), layout='constrained')
    cdf(ax, 'camera', CAMERAS)
    ax.set_title('Six-camera Token Drift Distributions (gaps combined)')
    save(fig, 'E2_camera_distributions', 'Six camera CDFs with equal scene weighting')

    fig, ax = plt.subplots(figsize=(6.5, 3.7), layout='constrained')
    n = data['layout']['patch_tokens']
    for j, pct in enumerate([10, 20]):
        xs = np.arange(len(gaps)) + (j - .5) * .3
        stats = [g['metrics'][f'top{pct}_contribution'] for g in gaps]
        ax.bar(xs, [s['scene_equal_weight_mean'] for s in stats], width=.28, label=f'Top {pct}%')
        for x, s in zip(xs, stats):
            if s['ci95']:
                ax.vlines(x, *s['ci95'], color='black', linewidth=1)
        ax.axhline(np.ceil(n * pct / 100) / n, linestyle='--', linewidth=.8, color=colors(j), label=f'{pct}% uniform reference (ceil K)')
    ax.set_xticks(range(len(gaps)), gap_names)
    ax.set(ylabel='Fraction of total drift', ylim=(0, 1), title='Top-K Token Drift Contribution (scene bootstrap CI)')
    ax.legend(fontsize=7, ncol=2)
    save(fig, 'E2_topk_contribution', 'K=ceil(N*fraction), actual uniform k/N baseline shown')

    scenes = [r for r in read_csv(out / 'scene_drift_summary.csv') if r['dimension'] == 'gap']
    scene_names = sorted({r['scene_name'] for r in scenes})
    fig, ax = plt.subplots(figsize=(max(7, len(scene_names) * .65), 4), layout='constrained')
    for i, gap in enumerate(gap_names):
        lookup = {r['scene_name']: r for r in scenes if r['group'] == gap}
        ys = [float(lookup[s]['gini']) if s in lookup else np.nan for s in scene_names]
        ax.plot(range(len(scene_names)), ys, 'o-', label=gap)
    ax.set_xticks(range(len(scene_names)), scene_names, rotation=35, ha='right')
    ax.set(ylabel='Mean within-pair token Gini', ylim=(0, 1), title='Drift Magnitude Concentration Across Scenes')
    ax.legend()
    save(fig, 'E2_scene_concentration', 'Scene means; magnitude inequality is not a claim of spatial contiguity')

    maps = [r for r in read_csv(out / 'spatial_drift_maps.csv') if r['scope'] == 'scene_equal_mean']
    nr, nc = data['layout']['grid_rows'], data['layout']['grid_cols']
    shape = (nr, nc)
    def map_grid(metric, name, title, upper=None):
        fig, axes = plt.subplots(len(gap_names), 6, figsize=(15, 2.7 * len(gap_names)), squeeze=False, layout='constrained')
        vmax = upper or max(float(r[metric]) for r in maps)
        for i, gap in enumerate(gap_names):
            for j, camera in enumerate(CAMERAS):
                ax = axes[i, j]
                rows = [r for r in maps if r['camera_channel'] == camera and r['gap_group'] == gap]
                if not rows:
                    ax.text(.5, .5, 'Unavailable', ha='center', transform=ax.transAxes)
                    continue
                matrix = np.full(shape, np.nan)
                for r in rows:
                    matrix[int(r['row']), int(r['col'])] = float(r[metric])
                im = ax.pcolormesh(matrix, vmin=0, vmax=vmax, cmap='viridis', shading='flat')
                ax.invert_yaxis()
                ax.set_aspect('equal')
                ax.set_title(camera.removeprefix('CAM_'), fontsize=8)
                ax.set_ylabel(gap + '\nPatch row' if j == 0 else '')
                ax.set_xlabel('Patch col')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7, label=metric)
        fig.suptitle(title)
        save(fig, name, 'Equal-scene average; model padded/resized coordinates; shared color scale')
    map_grid('mean_token_drift', 'E2_mean_spatial_maps', 'Six-camera Mean Spatial Token Drift by Gap')
    # Maps retain every threshold in CSV. One explicitly labeled example figure.
    threshold = cfg['stable_thresholds'][min(2, len(cfg['stable_thresholds']) - 1)]
    map_grid(stable_key(threshold), 'E2_stable_spatial_maps', f'Stable Token Spatial Frequency (cosine distance ≤ {threshold:g})', upper=1)

    pairs = read_csv(out / 'selected_temporal_pairs.csv')
    records, arrays, _ = load_artifacts(out, pairs)
    # Content-independent representative: first successful scene, then the gap
    # nearest 250 ms; earliest temporal pair per camera.
    scene = min(r['scene_name'] for r in records)
    gap = min(gaps, key=lambda g: abs(g['metrics']['actual_delta_t_ms']['scene_equal_weight_mean'] - 250))['group']
    selected = []
    for camera in CAMERAS:
        candidates = [r for r in records if r['scene_name'] == scene and r['gap_group'] == gap and r['camera_channel'] == camera]
        selected.append(min(candidates, key=lambda r: int(r['previous_timestamp_us'])) if candidates else None)
    vmax = max(float(arrays[r['pair_id']].max()) for r in selected if r)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')
    for ax, camera, row in zip(axes.ravel(), CAMERAS, selected):
        ax.set_title(camera.removeprefix('CAM_'))
        if row is None:
            ax.text(.5, .5, 'Unavailable', ha='center', transform=ax.transAxes)
            continue
        matrix = arrays[row['pair_id']].reshape(shape)
        im = ax.pcolormesh(matrix, vmin=0, vmax=vmax, shading='flat', cmap='viridis')
        ax.invert_yaxis(); ax.set_aspect('equal')
        ax.set(xlabel='Patch col', ylabel='Patch row')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.8, label='Cosine distance')
    fig.suptitle(f'Representative Measured Pairs: {scene}, {gap}, {nr}×{nc}')
    save(fig, 'E2_representative_heatmaps', 'Earliest measured pair per camera in first scene; not synchronized across cameras')
    metadata['representative_pair_ids'] = [r['pair_id'] if r else None for r in selected]
    write_json(out / 'feature_figure_metadata.json', metadata)
