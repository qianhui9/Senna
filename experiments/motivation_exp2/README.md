# WorldStream Motivation Experiment 2

Offline nuScenes analysis of physical instance continuity versus image-coordinate
movement. It loads no Senna/CLIP weights, executes no VLM, and needs no CUDA.
All commands run from the Senna repository root in `conda activate Senna`.

## Configuration and reuse

The E1 official devkit loader, camera channel protocol, scene selector, atomic
CSV/JSON writers and scene-bootstrap implementation are reused. By default the
E1 `manifest_summary.json` supplies the exact same ten scenes and
`camera_manifest.csv` verifies keyframe exposure timestamps and scene/camera
identity. E1 sweep pairs and similarity results are never used as annotated
object pairs, and no E1 outputs are modified.

Set `NUSCENES_ROOT` or `--dataroot`. If neither is supplied, the existing E1
resolved configuration supplies its dataroot. Paths are relative to the repository
root; no dataset absolute path is hardcoded. Missing configured manifests are an
error. To use E1's seeded selection without existing outputs, set
`--set 'dataset.scene_manifest=""' --set 'dataset.frame_manifest=""'`.

Every YAML parameter can be overridden with repeated `--set section.key=YAML`.
Convenience flags include `--max-scenes`, `--cameras`, `--keyframe-gaps`,
`--seed`, `--bootstrap-iterations`, `--output-dir`, and `--visualization-samples`.
Example: `--set 'analysis.grid_shapes=[[8,8],[16,16]]'`.

## Run

```bash
python experiments/motivation_exp2/run_exp2.py --config experiments/motivation_exp2/config.yaml --dry-run
python experiments/motivation_exp2/run_exp2.py --config experiments/motivation_exp2/config.yaml --max-scenes 2 --smoke-test --output-dir outputs/motivation_exp2_smoke
python experiments/motivation_exp2/run_exp2.py --config experiments/motivation_exp2/config.yaml --run-all
```

The default invocation runs all stages. Smoke caps scenes at two and qualitative
examples at six, while retaining all six cameras, all gaps, 1,000 scene bootstrap
draws, and all keyframes in those scenes. Use separate directories for smoke and
full runs. Without an explicit output override, smoke mode automatically uses
the `_smoke` output directory. `--existing reuse` (default) verifies config/source/dependency/data
identity, then skips complete stages whose required files still exist.
`--existing overwrite` recomputes requested stages with the same identity and
invalidates downstream stage statuses. A changed scientific configuration or
source requires a new output directory or `--run-all --existing overwrite`,
which preserves previous experiment-owned products under `previous_runs/`
before recomputing all stages. Partial-stage overwrite cannot change identity.
Do not run concurrent writers into one
output directory. Stage writes are atomic at the table level; an interrupted
stage restarts from the beginning.

Independent stages:

```bash
python experiments/motivation_exp2/run_exp2.py --stage build-pairs
python experiments/motivation_exp2/run_exp2.py --stage project-boxes
python experiments/motivation_exp2/run_exp2.py --stage persistence
python experiments/motivation_exp2/run_exp2.py --stage displacement
python experiments/motivation_exp2/run_exp2.py --stage grid-overlap
python experiments/motivation_exp2/run_exp2.py --stage ego-motion
python experiments/motivation_exp2/run_exp2.py --stage aggregate
python experiments/motivation_exp2/run_exp2.py --stage plot
python experiments/motivation_exp2/run_exp2.py --stage report
```

Module entry points also work, e.g.
`python -m experiments.motivation_exp2.analyze_grid_overlap`.
After `project-boxes`, remaining stages read CSV/JSON and do not load the devkit.
Full metadata loading can require substantial CPU RAM even for two scenes.

## Measurement definitions

- Pairs follow `sample.next` within one scene for gaps 1/2/4; actual sample and
  per-camera time gaps are retained separately. No annotated sweeps are assumed.
- Only identical official `instance_token` values are matched. Annotation tokens
  differ over time. Category names come from official instance/category lookup.
- `get_sample_data(..., BoxVisibility.NONE)` returns all camera-frame Boxes;
  Box tokens are checked against annotation tokens. Each frame uses its own
  camera ego pose, calibration, intrinsics and actual image dimensions.
- Camera-forward corners are projected with official `view_points`. Fully
  behind/near-plane boxes and near-plane crossings have explicit rejection
  statuses. Crossing boxes are excluded rather than inaccurately clipped.
  Image-boundary rectangles are clipped while preserving raw extents. Geometric
  in-image rectangle area ratio is **not occlusion visibility**. Minimum width,
  height and area-ratio filters define **Projected-in-view Objects**.
- Sample-level keyframe annotations are used without interpolation at camera
  exposure time, as in the official keyframe API. Signed sample/camera offsets
  are retained; this timing assumption is documented in the report.
- Global persistence = intersection of annotated instance sets / old set size.
  Camera persistence applies the same definition to valid projected instances.
  Empty denominators are NaN, not zero.
- Displacement uses the clipped rectangle center and actual image diagonal;
  IoU compares time-separated rectangles in the same camera pixel coordinates.
  Scale ratio is bbox area(t1)/area(t0). This is not 3D object overlap.
- Relative ego transform uses column vectors:
  `inverse(T_global_ego0) @ T_global_ego1`. Translation components are in the
  initial ego frame; quaternion rotation angle is shortest angle, signed yaw
  wraps correctly through ±π. Rates use actual **camera** gaps.
- Object center speed uses **sample** time gaps. Approximately static ≤0.5m/s
  is configurable, with 0.2/0.5/1.0m/s sensitivity. This is not all background.
- Fixed diagnostic grids use any positive-area intersection (main), minimum
  intersection/cell-area fraction, and cell-center inclusion. Jaccard overlap
  and old-cell retention are separate; empty unions/old sets return NaN.
  These are not CLIP token grids or model-cache hit rates.
- Means/95% CIs use scene replication; no independent-frame/object confidence
  claim is made. Observation-weighted summaries and scene-equal means are both
  reported; persistence additionally includes pooled instance-count ratios.
  Correlation p-values are not calculated under a false IID assumption.
- Category, camera, actual time gap, distance, object-motion, ego-motion bins and
  scene strata are in `statistics/summary.json` and aggregate CSVs. Bounding-box
  clipping and annotation errors remain possible confounders.

## Outputs and figures

`metadata/` holds resolved config, scene list, provenance, stage status,
exclusions and real projection regressions. `data/` contains the seven required
tables plus independently readable annotation/frame/ego indexes and aggregate
statistics. `statistics/summary.json` includes all distributions and sensitivity
analyses. M2(a–e) are exported as PNG 300 DPI and vector PDF (photos embedded as
raster). `visualizations/` includes real paired images and official 3D-wireframe
projection overlays. `experiment_report.md` separates evidence-supported
observations from claims that this experiment cannot establish.

The five `gen_M2*.py` scripts independently reproduce individual figures:

```bash
python -m experiments.motivation_exp2.gen_M2b --output-dir outputs/motivation_exp2
```

Scatter sampling is seeded and affects rendering only; statistics use all rows.
Qualitative galleries use three paired-image rows per page, up to twelve cases
across separate PNG/PDF pages; the requested individual examples are all saved.
M2c/d use 6.8-inch canvases; hollow markers mean singleton-scene bins without a
CI. Exact per-bin scene/observation counts are saved in figure metadata.
Small two-scene runs provide functional
diagnostics, not publication-level uncertainty.
Qualitative cases use seeded scene/camera/gap/motion strata and 5–95% quantile
targets, including both small and large shifts. Figure captions and LaTeX include
snippets are saved with the outputs. No predetermined result is fabricated.

## Tests

```bash
python -m unittest discover -s experiments/motivation_exp2/tests -v
SENNA_EXP2_REAL_OUTPUT=outputs/motivation_exp2 python -m unittest experiments.motivation_exp2.tests.test_real_nuscenes_regression -v
```

Synthetic fixtures test identity, sample links, poses, official transform
agreement, projection boundaries, near-plane policy, ±π wrapping, IoU, grid
rules, undefined denominators and scene weighting. They never supply experiment
evidence. Opt-in real tests reload the official devkit, compare real sample
transform/projection results and inspect rendered regression artifact integrity.
Human visual inspection of regression overlays still matters; these are not
pixel-perfect golden photographs.
