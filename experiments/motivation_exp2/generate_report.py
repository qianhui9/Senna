"""Evidence-derived report with explicit unsupported claims and exclusions."""
import json

import pandas as pd

from .utils.data_utils import read_table, standalone


def fmt(value, digits=4):
    return "NaN / unavailable" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n" + "\n".join("| "+" | ".join(str(v) for v in r)+" |" for r in rows)+"\n"


def run(cfg, out):
    s = json.loads((out / "statistics" / "summary.json").read_text())
    scenes = json.loads((out / "metadata" / "scene_list.json").read_text())
    index = json.loads((out / "metadata" / "index_summary.json").read_text())
    projection = json.loads((out / "metadata" / "projection_summary.json").read_text())
    frames = read_table(out, "camera_frames")
    figures_path = out / "metadata" / "figure_metadata.json"
    figures = json.loads(figures_path.read_text()) if figures_path.is_file() else {}
    lines = ["# Experiment 2: Physical Continuity vs. Image-Space Displacement", "",
       "## Goal and hypotheses", "",
       "Measure whether official annotated physical instances persist across driving keyframes, and whether their camera bounding rectangles remain at fixed image locations. All evidence below is from configured real nuScenes data; no VLM, GPU, training, feature reuse, state alignment, or scheduling is executed.", "",
       "H1: annotated instance identity often persists. H2: persistent instances can move in image coordinates and fixed grids. H3: observed ego-motion and image migration are statistically associated. Effect sizes and scene-level intervals are reported; no universal significance threshold or ego-only causal claim is assumed.", "",
       "## Dataset and indexing", "", f"- nuScenes version: `{cfg['nuscenes']['version']}`.",
       f"- Scenes: {index['scenes']}; annotated keyframes: {index['keyframes']}; keyframe pairs: {index['pairs']}; camera frames: {index['camera_frames']}.",
       f"- Reused and verified E1 camera-manifest keyframe rows: {index['e1_keyframe_rows_verified']}.",
       "- All configured sample.next gaps are retained within each scene; no sweeps or cross-scene pairs are used. Sample and per-camera timestamp gaps are separate.",
       f"- Gaps: {cfg['pairs']['keyframe_gaps']}; cameras: {cfg['dataset']['cameras']}.", "",
       table(["Scene", "Token", "Description"], [[r['name'], r['token'], r['description']] for r in scenes]),
       "## Geometry and annotation timing", "",
       "Official `get_sample_data(..., BoxVisibility.NONE)` provides all camera-frame boxes; each Box token is checked against the sample's annotation tokens. Identity is the annotation's `instance_token`, with category resolved via the official devkit instance/category records.", "",
       "Column-vector convention: ego poses map Ego→Global, calibrated sensor transforms map Camera→Ego. World→Camera equals inverse(Camera→Ego) × inverse(Ego→Global). Actual camera exposure ego poses and intrinsics are used. Pixel origin is top left. Official `view_points` projects the 3D corners; the image-clipped 2D rectangle supplies the center and grid region.", "",
       "Sample-level keyframe boxes are used without interpolation at camera exposure time, following the official keyframe API. This is a time-alignment assumption, not sweep-time ground truth. Signed offset = (camera timestamp − sample timestamp).", "",
       f"- Camera/annotation offsets: min {frames.annotation_camera_time_offset_ms.min():.3f} ms, median {frames.annotation_camera_time_offset_ms.median():.3f} ms, max {frames.annotation_camera_time_offset_ms.max():.3f} ms.",
       f"- Near plane: {cfg['projection']['near_plane_m']} m. Crossings are explicitly excluded, not approximately clipped.",
       f"- Minimum image-clipped box size: {cfg['projection']['min_bbox_width_px']} × {cfg['projection']['min_bbox_height_px']} pixels; geometric rectangle area ratio ≥ {cfg['projection']['min_visible_area_ratio']}.",
       "- Projected-in-view objects are not guaranteed to be unoccluded or fully visible. The geometric area ratio is clipped/raw bounding-rectangle area, not exact object visibility.",
       "- Numerical independent-transform comparisons and real official-wireframe overlays are saved in `metadata/projection_regression.json` and `visualizations/projection_regression_*.png`.", "",
       "## Counts and exclusions", "",
       table(["Quantity", "Count"], list(s['counts'].items())),
       "Global persistent pair-instance counts repeat instances over time; matched camera-pair counts can also repeat a physical instance across cameras. Neither count is an independent sample size.", "",
       table(["Projection status", "Count", "Fraction of attempts"], [[k,v,fmt(v/projection['attempts'])] for k,v in projection['status_counts'].items()]),
       f"Geometrically clipped attempts: {projection['geometric_clipped']}. Valid fraction: {fmt(projection['valid']/projection['attempts'])}. "
       "The denominator includes all annotated instances in each camera, including objects behind that camera; this fraction is not a detector recall.", "",
       f"Index anomalies/missing data: {len(index['issues'])}. Full issue records: `metadata/index_summary.json` and `metadata/displacement_summary.json`.",
       "Zero initial instance sets produce NaN, not 0%; empty grid unions produce NaN, and empty old-grid sets have undefined retention.", "",
       "## Statistical units and uncertainty", "",
       f"All mean CIs resample scene means ({cfg['analysis']['bootstrap_iterations']} draws, seed {cfg['analysis']['seed']}); one-scene CIs are unavailable. Instance-observation weighted means and scene-equal means are separate. Persistence additionally reports the pooled instance-weighted count ratio. Distribution quantiles pool observations and are descriptive. Cross-camera projections and repeated frames are never used as independent bootstrap units.", "",
       "Spearman correlations on individual observations are descriptive, with no IID p-values. Scene-level correlation compares scene medians, and its interval bootstraps scenes. Time gap, camera direction, object distance and approximate object motion strata are retained in `statistics/summary.json`.", "",
       "## Physical persistence", ""]
    rows = []
    if len(scenes) <= 2:
        lines += ["This is a small-scene functional workload. With at most two scene clusters, bootstrap intervals are smoke-test diagnostics and do not support publication-level precision or representative-driving conclusions.", ""]
    for kind in ["global_persistence", "camera_persistence"]:
        for group in s[kind]:
            m = group['metrics']['persistent_ratio']
            rows.append([group['gap_steps'], group.get('camera_channel','GLOBAL'), m['n'], fmt(m['instance_weighted_ratio']), fmt(m['scene_equal_weight_mean']), str(m['ci95'])])
    lines += [table(["Gap", "Set", "Valid pairs", "Pooled instance ratio", "Scene mean", "Scene CI95"], rows), "## Image displacement", ""]
    rows = []
    for group in s['displacement']:
        m, px, iou = [group['metrics'][k] for k in ['normalized_displacement','center_displacement_px','bbox_iou']]
        rows.append([group['gap_steps'],m['n'],fmt(px['p50'],1),fmt(m['observation_weighted_mean']),fmt(m['p50']),fmt(m['p75']),fmt(m['p90']),fmt(m['p95']),fmt(m['scene_equal_weight_mean']),str(m['ci95']),fmt(iou['p50'])])
    lines += [table(["Gap","Camera-pair instances","Median px","Pooled norm mean","Norm median","P75","P90","P95","Scene norm mean","CI95","Median bbox IoU"],rows),
       "Image displacement is Euclidean clipped-rectangle center displacement divided by the actual image diagonal. IoU compares two rectangles at the same camera pixel coordinates across time; it is neither matching accuracy nor 3D overlap. Scale ratio is area(t1)/area(t0). Boundary clipping can affect these measurements.", "",
       "## Ego-motion and object-motion strata", "",
       "Relative ego transform = inverse(T_global_ego0) × T_global_ego1. Translation magnitude is global Euclidean distance; components are in the initial ego frame. Rotation uses the shortest normalized-quaternion angle; signed yaw differences wrap to [−π, π]. Speed/yaw rate use actual camera gaps. Object speed uses global annotation-center distance divided by the actual sample gap.", "",
       f"Approximately static threshold: ≤ {cfg['analysis']['static_speed_threshold_mps']} m/s. This subset excludes much unannotated background (roads, buildings) and can include annotation noise or slowly moving objects.", "",
       table(["Gap","Object motion","n","Median normalized displacement","Scene mean","CI95"], [[g['gap_steps'],g['motion_group'],g['rows'],fmt(g['metrics']['normalized_displacement']['p50']),fmt(g['metrics']['normalized_displacement']['scene_equal_weight_mean']),str(g['metrics']['normalized_displacement']['ci95'])] for g in s['static_dynamic']]),
       table(["Gap","Object motion","Variable","n","Observation ρ","Scene-median ρ","Scene CI95"], [[g['gap_steps'],g['motion_group'],g['x'],g['n'],fmt(g['observation_spearman']),fmt(g['scene_median_spearman']),str(g['scene_median_ci95'])] for g in s['correlations']]),
       "Distance changes, object motion, camera direction, clipping, occlusion and 3D annotation error are possible contributors. These correlations do not isolate ego-motion as the cause. A stricter ego-only causal decomposition remains future analysis.", "",
       "## Fixed image grid diagnostics", "",
       f"Main cell rule: `{cfg['analysis']['primary_grid_rule']}`; sensitivity rules: {cfg['analysis']['grid_rules']}; coverage rule minimum cell area ratio: {cfg['analysis']['grid_min_coverage_ratio']}.",
       "J = |G0∩G1|/|G0∪G1|; R = |G0∩G1|/|G0|. Rules use positive-area intersection, a minimum intersection/cell-area fraction, or cell-center inclusion. Boundary-only contact is excluded. Grid displacement measures center shift in cell-width/height units. These diagnostic grids are not Senna/CLIP token grids or cache hit rates.", "",
       table(["Gap","Grid","Rule","Valid J n","Scene J mean","J CI95","Scene old-cell retention"], [[g['gap_steps'],f"{g['grid_rows']}×{g['grid_cols']}",g['grid_rule'],g['metrics']['grid_overlap']['n'],fmt(g['metrics']['grid_overlap']['scene_equal_weight_mean']),str(g['metrics']['grid_overlap']['ci95']),fmt(g['metrics']['old_grid_retention']['scene_equal_weight_mean'])] for g in s['grid']]),
       "## Sensitivity analyses", "",
       table(["Geometric area ratio threshold","Gap","Valid projections","Matched camera observations","Norm median","Scene norm mean"], [[g['geometric_area_threshold'],g['gap_steps'],g['valid_projection_count'],g['matched_observations'],fmt(g['displacement']['p50']),fmt(g['displacement']['scene_equal_weight_mean'])] for g in s['projection_sensitivity']]),
       table(["Static threshold m/s","Gap","Motion group","n","Norm median","Scene norm mean"], [[g['threshold_mps'],g['gap_steps'],g['motion_group'],g['rows'],fmt(g['metrics']['normalized_displacement']['p50']),fmt(g['metrics']['normalized_displacement']['scene_equal_weight_mean'])] for g in s['static_speed_sensitivity']]),
       "Grid-rule sensitivity is reported above. Near-plane, minimum-size and bounding-rectangle limitations remain even when the geometric area threshold changes.", "",
       "## Figures and representative cases", ""]
    for name, metadata in figures.items():
        lines += [f"![{name}](figures/{name}.png)", "", metadata['caption'], ""]
    examples = out / "metadata" / "qualitative_examples.json"
    if examples.exists():
        lines += [table(["Example","Official instance","Motion group","Measurements"], [[f"[{r['file']}]({r['file']})",r['instance_token'],r['motion_group'],r['caption']] for r in json.loads(examples.read_text())]), ""]
    lines += ["## A. Observation Supported by Evidence", ""]
    if not s['counts']['matched_camera_pair_instances']:
        lines += ["No valid same-instance camera pairs were obtained; H2/H3 cannot be assessed from this run."]
    for group in s['displacement']:
        gap, m = group['gap_steps'], group['metrics']['normalized_displacement']
        gp = next(g for g in s['global_persistence'] if g['gap_steps'] == gap)['metrics']['persistent_ratio']
        main = [g for g in s['grid'] if g['gap_steps'] == gap and g['grid_rule'] == cfg['analysis']['primary_grid_rule']]
        lines.append(f"- Gap {gap}: global scene-equal annotated persistence is {fmt(gp['scene_equal_weight_mean'])}; "
                     f"among {m['n']} persistent camera-pair observations, normalized displacement median/P90 is {fmt(m['p50'])}/{fmt(m['p90'])}. "
                     f"Fixed-grid scene-equal Jaccard means are {', '.join(str(g['grid_rows'])+'×'+str(g['grid_cols'])+': '+fmt(g['metrics']['grid_overlap']['scene_equal_weight_mean']) for g in main)}.")
        if m['p50'] is not None and m['p50'] > 0:
            lines.append("  Identity persistence does not imply an unchanged image coordinate in this measured subset. The measured magnitude and interval, rather than a universal significance claim, quantify the mismatch.")
        elif m['p50'] is not None:
            lines.append("  At least half of these observations have zero center displacement; this run does not establish a substantial typical shift for this gap.")
    lines += ["", "Ego associations are reported in the correlation table and controlled strata; their direction and strength may differ across groups. The data establish measured relations in the selected workload, not a sole causal explanation or universal claim about all driving scenes.", "",
       "## B. Claims Not Established", "",
       "This experiment does **not** establish:", "",
       "- World Alignment outperforming other association methods.",
       "- Arbitrary visual features being reusable.",
       "- VLM caches necessarily failing or fixed-grid overlap equalling cache hit rate.",
       "- CLIP patch features permitting skipped computation.",
       "- WorldStream improving accuracy or inference speed.",
       "- Incremental GPU runtime necessarily yielding benefits.",
       "- Deadline scheduling being proven necessary.", "",
       "The evidence is limited to annotated instance persistence, projected bounding rectangles, fixed diagnostic grids, and observed ego-motion associations in the selected real dataset.", "",
       "## Verification", ""]
    verification = out / "metadata" / "verification_results.json"
    if verification.exists():
        checked = json.loads(verification.read_text())
        lines += [f"Actual automated tests: {checked.get('tests_run')} run; {checked.get('failures')} failures; {checked.get('errors')} errors; {checked.get('skipped')} skipped. Details and command are saved in `metadata/verification_results.json`.", ""]
    else:
        lines += ["Verification tests are available under `experiments/motivation_exp2/tests`; this report does not assume they have been executed. Run them and retain their output.", ""]
    lines += [
       "## Reproduction and provenance", "",
       "Retain `metadata/experiment_config.yaml`, `metadata/run_metadata.json`, raw CSVs and plotting scripts. Config/source/table identity is checked before stage reuse. Run from the Senna repository root:", "",
       "```bash", f"python experiments/motivation_exp2/run_exp2.py --config {out / 'metadata/experiment_config.yaml'} --run-all", "```", "",
       "Official schema/API references: [nuScenes tutorial](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html), [devkit implementation](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py).", ""]
    (out / "experiment_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    standalone("report")
