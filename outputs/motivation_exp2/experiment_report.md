# Experiment 2: Physical Continuity vs. Image-Space Displacement

## Goal and hypotheses

Measure whether official annotated physical instances persist across driving keyframes, and whether their camera bounding rectangles remain at fixed image locations. All evidence below is from configured real nuScenes data; no VLM, GPU, training, feature reuse, state alignment, or scheduling is executed.

H1: annotated instance identity often persists. H2: persistent instances can move in image coordinates and fixed grids. H3: observed ego-motion and image migration are statistically associated. Effect sizes and scene-level intervals are reported; no universal significance threshold or ego-only causal claim is assumed.

## Dataset and indexing

- nuScenes version: `v1.0-trainval`.
- Scenes: 10; annotated keyframes: 403; keyframe pairs: 1139; camera frames: 2418.
- Reused and verified E1 camera-manifest keyframe rows: 2418.
- All configured sample.next gaps are retained within each scene; no sweeps or cross-scene pairs are used. Sample and per-camera timestamp gaps are separate.
- Gaps: [1, 2, 4]; cameras: ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK'].

| Scene | Token | Description |
| --- | --- | --- |
| scene-0274 | 9709626638f5406f9f773348150d02fd | Peds, construction zone, construction workers, barriers, cars, jaywalker |
| scene-0275 | 8377abf77f464a9cb62eacf63f383422 | Peds, bus, barriers, stopped at intersection |
| scene-0276 | 201b7c65a61f4bc1a2333ea90ba9a932 | Intersection, peds, cars, scooter, parked bicycle |
| scene-0557 | 3f90afe9f7dc49399347ae1626502aed | Parked cars, pedestrian, stopped at intersection, oncoming vehicle |
| scene-0559 | 0e7ede02718341558414865d5c604745 | Passing intersection, pedestrian, parked cars, construction zone, construction vehicles |
| scene-0632 | b789de07180846cc972118ee6d1fb027 | Rain, industrial, turn right, turn right, parked cars |
| scene-0913 | 55b3a17359014f398b6bbd90e94a8e1b | Rain, parked cars, semi trailers, barriers, construction vehicle, parking lot, parked buses |
| scene-0967 | 6776b91389394ff18e57b269863b4dbf | Peds on sidewalk, parked cars, parked motorcycle |
| scene-0971 | 64bfc5edd71147858ce7446892d7f864 | Cross T junction, peds, parked motorcycles, turn right, jaywalker |
| scene-0972 | e005041f659c47e194cd5b18ea6fc346 | Parked bicycles, peds, ped walk along the road, parked cars, motorcycle, bus |

## Geometry and annotation timing

Official `get_sample_data(..., BoxVisibility.NONE)` provides all camera-frame boxes; each Box token is checked against the sample's annotation tokens. Identity is the annotation's `instance_token`, with category resolved via the official devkit instance/category records.

Column-vector convention: ego poses map Ego→Global, calibrated sensor transforms map Camera→Ego. World→Camera equals inverse(Camera→Ego) × inverse(Ego→Global). Actual camera exposure ego poses and intrinsics are used. Pixel origin is top left. Official `view_points` projects the 3D corners; the image-clipped 2D rectangle supplies the center and grid region.

Sample-level keyframe boxes are used without interpolation at camera exposure time, following the official keyframe API. This is a time-alignment assumption, not sweep-time ground truth. Signed offset = (camera timestamp − sample timestamp).

- Camera/annotation offsets: min -47.151 ms, median -24.383 ms, max 2.106 ms.
- Near plane: 0.1 m. Crossings are explicitly excluded, not approximately clipped.
- Minimum image-clipped box size: 8 × 8 pixels; geometric rectangle area ratio ≥ 0.2.
- Projected-in-view objects are not guaranteed to be unoccluded or fully visible. The geometric area ratio is clipped/raw bounding-rectangle area, not exact object visibility.
- Numerical independent-transform comparisons and real official-wireframe overlays are saved in `metadata/projection_regression.json` and `visualizations/projection_regression_*.png`.

## Counts and exclusions

| Quantity | Count |
| --- | --- |
| keyframe_pairs | 1139 |
| projection_attempts | 102678 |
| valid_projections | 19390 |
| global_persistent_pair_instances | 43911 |
| matched_camera_pair_instances | 40043 |
| unique_matched_physical_instances | 1033 |
| matched_scenes | 10 |
| grid_diagnostic_rows | 360387 |

Global persistent pair-instance counts repeat instances over time; matched camera-pair counts can also repeat a physical instance across cameras. Neither count is an independent sample size.

| Projection status | Count | Fraction of attempts |
| --- | --- | --- |
| behind_or_inside_near_plane | 50364 | 0.4905 |
| outside_image | 27704 | 0.2698 |
| valid | 19390 | 0.1888 |
| near_plane_crossing_excluded | 4059 | 0.0395 |
| low_in_image_area_ratio | 962 | 0.0094 |
| too_small | 199 | 0.0019 |

Geometrically clipped attempts: 31831. Valid fraction: 0.1888. The denominator includes all annotated instances in each camera, including objects behind that camera; this fraction is not a detector recall.

Index anomalies/missing data: 0. Full issue records: `metadata/index_summary.json` and `metadata/displacement_summary.json`.
Zero initial instance sets produce NaN, not 0%; empty grid unions produce NaN, and empty old-grid sets have undefined retention.

## Statistical units and uncertainty

All mean CIs resample scene means (1000 draws, seed 42); one-scene CIs are unavailable. Instance-observation weighted means and scene-equal means are separate. Persistence additionally reports the pooled instance-weighted count ratio. Distribution quantiles pool observations and are descriptive. Cross-camera projections and repeated frames are never used as independent bootstrap units.

Spearman correlations on individual observations are descriptive, with no IID p-values. Scene-level correlation compares scene medians, and its interval bootstraps scenes. Time gap, camera direction, object distance and approximate object motion strata are retained in `statistics/summary.json`.

## Physical persistence

| Gap | Set | Valid pairs | Pooled instance ratio | Scene mean | Scene CI95 |
| --- | --- | --- | --- | --- | --- |
| 1 | GLOBAL | 393 | 0.9528 | 0.9531 | [0.9371756837571054, 0.9678870902047585] |
| 2 | GLOBAL | 383 | 0.9114 | 0.9141 | [0.8935973629891089, 0.9365325536859382] |
| 4 | GLOBAL | 363 | 0.8296 | 0.8399 | [0.808919610316618, 0.8773126716257896] |
| 1 | CAM_BACK | 393 | 0.9017 | 0.9204 | [0.9014512241914254, 0.9402102623670757] |
| 1 | CAM_BACK_LEFT | 259 | 0.8300 | 0.7642 | [0.7330426808660764, 0.812397158074709] |
| 1 | CAM_BACK_RIGHT | 298 | 0.8257 | 0.7997 | [0.7479509762699768, 0.8515941883048769] |
| 1 | CAM_FRONT | 365 | 0.8765 | 0.8793 | [0.8553806473211955, 0.9033740923155844] |
| 1 | CAM_FRONT_LEFT | 311 | 0.8469 | 0.8011 | [0.7549862084820186, 0.8516469186288418] |
| 1 | CAM_FRONT_RIGHT | 350 | 0.8527 | 0.8553 | [0.8074643403736608, 0.9040237114253344] |
| 2 | CAM_BACK | 383 | 0.8116 | 0.8481 | [0.8159372902970435, 0.8829847561954953] |
| 2 | CAM_BACK_LEFT | 252 | 0.6791 | 0.5733 | [0.5035773021568783, 0.6686597560845482] |
| 2 | CAM_BACK_RIGHT | 290 | 0.6671 | 0.6321 | [0.5444059501883826, 0.7197492525916657] |
| 2 | CAM_FRONT | 356 | 0.7656 | 0.7769 | [0.7276391733065409, 0.8241254411209326] |
| 2 | CAM_FRONT_LEFT | 302 | 0.7154 | 0.6452 | [0.5712496192425565, 0.7297251100933586] |
| 2 | CAM_FRONT_RIGHT | 340 | 0.7250 | 0.7397 | [0.6660275826395827, 0.821389791494238] |
| 4 | CAM_BACK | 363 | 0.6489 | 0.7155 | [0.655757027196852, 0.7796041759242109] |
| 4 | CAM_BACK_LEFT | 240 | 0.4695 | 0.3055 | [0.19133229499032195, 0.44642624209986403] |
| 4 | CAM_BACK_RIGHT | 276 | 0.4104 | 0.3915 | [0.2790287373962649, 0.5157571995639416] |
| 4 | CAM_FRONT | 338 | 0.5700 | 0.6190 | [0.5446537687671722, 0.696924589967478] |
| 4 | CAM_FRONT_LEFT | 285 | 0.5105 | 0.4098 | [0.2950576610727446, 0.535350760032322] |
| 4 | CAM_FRONT_RIGHT | 320 | 0.5145 | 0.5471 | [0.43178912311469014, 0.676612967469251] |

## Image displacement

| Gap | Camera-pair instances | Median px | Pooled norm mean | Norm median | P75 | P90 | P95 | Scene norm mean | CI95 | Median bbox IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16453 | 64.3 | 0.0622 | 0.0350 | 0.0842 | 0.1534 | 0.2143 | 0.0578 | [0.04505824017608725, 0.0706179909209242] | 0.1317 |
| 2 | 13871 | 111.1 | 0.1034 | 0.0605 | 0.1461 | 0.2590 | 0.3501 | 0.0999 | [0.07785188898166971, 0.12111230085879608] | 0.0000 |
| 4 | 9719 | 162.9 | 0.1451 | 0.0887 | 0.2106 | 0.3680 | 0.4873 | 0.1467 | [0.11739784696890891, 0.17483235999973848] | 0.0000 |

Image displacement is Euclidean clipped-rectangle center displacement divided by the actual image diagonal. IoU compares two rectangles at the same camera pixel coordinates across time; it is neither matching accuracy nor 3D overlap. Scale ratio is area(t1)/area(t0). Boundary clipping can affect these measurements.

## Ego-motion and object-motion strata

Relative ego transform = inverse(T_global_ego0) × T_global_ego1. Translation magnitude is global Euclidean distance; components are in the initial ego frame. Rotation uses the shortest normalized-quaternion angle; signed yaw differences wrap to [−π, π]. Speed/yaw rate use actual camera gaps. Object speed uses global annotation-center distance divided by the actual sample gap.

Approximately static threshold: ≤ 0.5 m/s. This subset excludes much unannotated background (roads, buildings) and can include annotation noise or slowly moving objects.

| Gap | Object motion | n | Median normalized displacement | Scene mean | CI95 |
| --- | --- | --- | --- | --- | --- |
| 1 | approximately_static | 11597 | 0.0413 | 0.0617 | [0.04794183406002679, 0.07574117618694719] |
| 1 | moving | 4856 | 0.0240 | 0.0475 | [0.03764657226674964, 0.05772893990619206] |
| 2 | approximately_static | 9766 | 0.0728 | 0.1080 | [0.08454092156950929, 0.12984336311067282] |
| 2 | moving | 4105 | 0.0406 | 0.0782 | [0.05926458291274409, 0.09905999260210654] |
| 4 | approximately_static | 6650 | 0.1069 | 0.1601 | [0.12819665019873216, 0.18750045578936536] |
| 4 | moving | 3069 | 0.0593 | 0.1125 | [0.08529212850053106, 0.14432088385849418] |

| Gap | Object motion | Variable | n | Observation ρ | Scene-median ρ | Scene CI95 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | approximately_static | ego_translation_m | 11597 | 0.3504 | 0.7091 | [0.05831447963800908, 1.0] |
| 1 | approximately_static | ego_rotation_deg | 11597 | 0.1967 | 0.1273 | [-0.8680307384730725, 0.7902097902097902] |
| 1 | approximately_static | ego_yaw_change_deg | 11597 | -0.0606 | -0.4545 | [-0.8846153846153846, 0.3595911949685529] |
| 1 | moving | ego_translation_m | 4856 | 0.2322 | 0.2364 | [-0.5346639150943396, 0.8888888888888888] |
| 1 | moving | ego_rotation_deg | 4856 | 0.2172 | 0.7212 | [0.17487028301886792, 0.9627329192546584] |
| 1 | moving | ego_yaw_change_deg | 4856 | -0.1157 | -0.6121 | [-0.9874213836477987, 0.039402107326635225] |
| 2 | approximately_static | ego_translation_m | 9766 | 0.3440 | 0.6000 | [-0.08767099056603772, 0.9352994227994226] |
| 2 | approximately_static | ego_rotation_deg | 9766 | 0.2041 | -0.0182 | [-0.8472054140127389, 0.6784018529241459] |
| 2 | approximately_static | ego_yaw_change_deg | 9766 | -0.0723 | -0.6000 | [-0.9499433962264151, 0.1111111111111111] |
| 2 | moving | ego_translation_m | 4105 | 0.1893 | 0.0909 | [-0.6153846153846154, 0.8000617283950617] |
| 2 | moving | ego_rotation_deg | 4105 | 0.2232 | 0.6848 | [0.13712541620421756, 0.9627329192546584] |
| 2 | moving | ego_yaw_change_deg | 4105 | -0.1231 | -0.7455 | [-0.9610514485514485, -0.20253164556962025] |
| 4 | approximately_static | ego_translation_m | 6650 | 0.3183 | 0.4182 | [-0.4002941176470588, 0.8493336851827415] |
| 4 | approximately_static | ego_rotation_deg | 6650 | 0.2284 | 0.2485 | [-0.577639751552795, 0.8125] |
| 4 | approximately_static | ego_yaw_change_deg | 6650 | -0.0849 | -0.5636 | [-0.8955075701166543, 0.15098039215686215] |
| 4 | moving | ego_translation_m | 3069 | 0.1204 | -0.0545 | [-0.7083922955974843, 0.8693022663610899] |
| 4 | moving | ego_rotation_deg | 3069 | 0.2388 | 0.7939 | [0.3291139240506329, 0.9627329192546584] |
| 4 | moving | ego_yaw_change_deg | 3069 | -0.1232 | -0.4182 | [-0.8742138364779874, 0.311914971820632] |

Distance changes, object motion, camera direction, clipping, occlusion and 3D annotation error are possible contributors. These correlations do not isolate ego-motion as the cause. A stricter ego-only causal decomposition remains future analysis.

## Fixed image grid diagnostics

Main cell rule: `any`; sensitivity rules: ['any', 'coverage', 'center']; coverage rule minimum cell area ratio: 0.25.
J = |G0∩G1|/|G0∪G1|; R = |G0∩G1|/|G0|. Rules use positive-area intersection, a minimum intersection/cell-area fraction, or cell-center inclusion. Boundary-only contact is excluded. Grid displacement measures center shift in cell-width/height units. These diagnostic grids are not Senna/CLIP token grids or cache hit rates.

| Gap | Grid | Rule | Valid J n | Scene J mean | J CI95 | Scene old-cell retention |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8×8 | any | 16453 | 0.6317 | [0.5686826808535219, 0.6941824976261647] | 0.7303 |
| 1 | 8×8 | center | 8723 | 0.3400 | [0.25719030581140423, 0.42142262962205984] | 0.4589 |
| 1 | 8×8 | coverage | 8262 | 0.3947 | [0.3094263537696683, 0.4716354397735087] | 0.5089 |
| 1 | 16×16 | any | 16453 | 0.5138 | [0.441376377265997, 0.5859708671508064] | 0.6203 |
| 1 | 16×16 | center | 12638 | 0.3145 | [0.2343019820680111, 0.39267402304788757] | 0.4206 |
| 1 | 16×16 | coverage | 13142 | 0.3800 | [0.294181755131136, 0.4646403195813445] | 0.4759 |
| 1 | 32×32 | any | 16453 | 0.4312 | [0.3559850973412793, 0.5052525049937173] | 0.5361 |
| 1 | 32×32 | center | 15316 | 0.3139 | [0.23635025520790337, 0.38799987066021974] | 0.4080 |
| 1 | 32×32 | coverage | 15748 | 0.3625 | [0.28173025046378913, 0.44109961527702135] | 0.4544 |
| 2 | 8×8 | any | 13871 | 0.4782 | [0.4064797356477251, 0.5531444221030407] | 0.5708 |
| 2 | 8×8 | center | 7475 | 0.2099 | [0.15035326674044808, 0.27092173590974294] | 0.3042 |
| 2 | 8×8 | coverage | 7024 | 0.2484 | [0.1868330737546846, 0.3071798756879077] | 0.3469 |
| 2 | 16×16 | any | 13871 | 0.3640 | [0.2936478124389912, 0.43795805415860567] | 0.4537 |
| 2 | 16×16 | center | 10626 | 0.2045 | [0.14475877499131737, 0.26549492932370333] | 0.2836 |
| 2 | 16×16 | coverage | 11140 | 0.2486 | [0.17476337837080788, 0.3221750185092375] | 0.3242 |
| 2 | 32×32 | any | 13871 | 0.2910 | [0.22554544720110753, 0.36047786194659953] | 0.3721 |
| 2 | 32×32 | center | 12901 | 0.2017 | [0.14145443549060244, 0.2643360299330929] | 0.2706 |
| 2 | 32×32 | coverage | 13307 | 0.2339 | [0.16801605503333208, 0.3032701885101631] | 0.3016 |
| 4 | 8×8 | any | 9719 | 0.3673 | [0.2955509084386922, 0.44482584690448157] | 0.4425 |
| 4 | 8×8 | center | 5409 | 0.1367 | [0.08672597403917846, 0.190177708645832] | 0.2101 |
| 4 | 8×8 | coverage | 5096 | 0.1537 | [0.09875182173645626, 0.21473777149408943] | 0.2364 |
| 4 | 16×16 | any | 9719 | 0.2630 | [0.1950737115905317, 0.335870898928853] | 0.3335 |
| 4 | 16×16 | center | 7485 | 0.1426 | [0.09205539067866081, 0.2029974849015838] | 0.2041 |
| 4 | 16×16 | coverage | 7915 | 0.1740 | [0.11378466095858174, 0.2459417496242044] | 0.2376 |
| 4 | 32×32 | any | 9719 | 0.2009 | [0.14091253564348233, 0.26865133732865026] | 0.2637 |
| 4 | 32×32 | center | 9042 | 0.1377 | [0.08408887912920447, 0.1974760845321722] | 0.1916 |
| 4 | 32×32 | coverage | 9355 | 0.1599 | [0.1036726970923419, 0.22368596810130054] | 0.2149 |

## Sensitivity analyses

| Geometric area ratio threshold | Gap | Valid projections | Matched camera observations | Norm median | Scene norm mean |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 1 | 20352 | 17361 | 0.0362 | 0.0607 |
| 0.0 | 2 | 20352 | 14674 | 0.0634 | 0.1052 |
| 0.0 | 4 | 20352 | 10330 | 0.0934 | 0.1559 |
| 0.2 | 1 | 19390 | 16453 | 0.0350 | 0.0578 |
| 0.2 | 2 | 19390 | 13871 | 0.0605 | 0.0999 |
| 0.2 | 4 | 19390 | 9719 | 0.0887 | 0.1467 |
| 0.5 | 1 | 18189 | 15363 | 0.0332 | 0.0557 |
| 0.5 | 2 | 18189 | 12908 | 0.0559 | 0.0944 |
| 0.5 | 4 | 18189 | 9021 | 0.0822 | 0.1365 |

| Static threshold m/s | Gap | Motion group | n | Norm median | Scene norm mean |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 1 | approximately_static | 9579 | 0.0407 | 0.0604 |
| 0.2 | 1 | moving | 6874 | 0.0285 | 0.0544 |
| 0.2 | 2 | approximately_static | 8347 | 0.0726 | 0.1063 |
| 0.2 | 2 | moving | 5524 | 0.0462 | 0.0931 |
| 0.2 | 4 | approximately_static | 5858 | 0.1079 | 0.1588 |
| 0.2 | 4 | moving | 3861 | 0.0672 | 0.1326 |
| 0.5 | 1 | approximately_static | 11597 | 0.0413 | 0.0617 |
| 0.5 | 1 | moving | 4856 | 0.0240 | 0.0475 |
| 0.5 | 2 | approximately_static | 9766 | 0.0728 | 0.1080 |
| 0.5 | 2 | moving | 4105 | 0.0406 | 0.0782 |
| 0.5 | 4 | approximately_static | 6650 | 0.1069 | 0.1601 |
| 0.5 | 4 | moving | 3069 | 0.0593 | 0.1125 |
| 1.0 | 1 | approximately_static | 12604 | 0.0403 | 0.0613 |
| 1.0 | 1 | moving | 3849 | 0.0208 | 0.0414 |
| 1.0 | 2 | approximately_static | 10527 | 0.0709 | 0.1069 |
| 1.0 | 2 | moving | 3344 | 0.0360 | 0.0718 |
| 1.0 | 4 | approximately_static | 7161 | 0.1043 | 0.1593 |
| 1.0 | 4 | moving | 2558 | 0.0534 | 0.1084 |

Grid-rule sensitivity is reported above. Near-plane, minimum-size and bounding-rectangle limitations remain even when the geometric area threshold changes.

## Figures and representative cases

![M2a_physical_persistence](figures/M2a_physical_persistence.png)

Global annotated persistence and camera projected-in-view persistence use separate sets and different denominators; the curve gap is not a decomposition. Scene-equal pair means; 95% scene-bootstrap intervals; n counts pairs with a nonempty initial set, summed over gaps in each legend. Actual mean sample gaps are shown; camera exposure gaps are recorded separately.

![M2b_image_displacement](figures/M2b_image_displacement.png)

Same official instance ID, same camera and scene, valid projections at both times. Boxes: pooled 25–75%, median; whiskers: 5–95%. Orange diamonds: scene-equal means with 95% scene-bootstrap intervals. n is camera-pair instance observations; repeated temporal pairs, repeated instances and cross-camera projections are dependent.

![M2c_ego_motion_displacement](figures/M2c_ego_motion_displacement.png)

Time-gap strata; approximately static means global annotation-center speed ≤0.5 m/s over actual sample timestamps. Scatter is seeded and independently capped at 1,200 observations per group, so it does not encode group frequency; binned summaries use all rows. X medians count each unique camera pair once; y is scene-equal displacement mean with 95% scene-bootstrap intervals. Open markers denote one-scene bins without CIs; exact per-bin observation and scene counts are in metadata/figure_bin_counts.json. Dashed segments connect bin summaries, not a fitted causal response. Associations do not establish ego-only causation.

![M2d_grid_overlap](figures/M2d_grid_overlap.png)

Approximately static annotated objects only: global annotation-center speed ≤0.5 m/s over actual sample timestamps. Main rule any: positive-area box/cell intersection. Grids are diagnostic image regions, not CLIP tokens. X medians count unique camera pairs once; y is scene-equal Jaccard mean with 95% scene-bootstrap intervals. Open markers denote singleton-scene bins without CIs; exact per-bin counts are in metadata/figure_bin_counts.json. Grid variants share the same underlying observations and are not independent objects. Dashed segments connect bin summaries.

![M2e_qualitative_examples](figures/M2e_qualitative_examples.png)

The two gold rectangles within each row mark the matched official instance; labels show ten-character token prefixes (full IDs in metadata/CSV). Fixed [16, 16] grid, any overlap. Seeded scene/camera/gap/motion round-robin strata with 5–95% quantile targets include both small and large displacement; cases are illustrative, not statistical evidence. Page 1/4; 30 individual examples saved.

![M2e_qualitative_examples_page2](figures/M2e_qualitative_examples_page2.png)

The two gold rectangles within each row mark the matched official instance; labels show ten-character token prefixes (full IDs in metadata/CSV). Fixed [16, 16] grid, any overlap. Seeded scene/camera/gap/motion round-robin strata with 5–95% quantile targets include both small and large displacement; cases are illustrative, not statistical evidence. Page 2/4; 30 individual examples saved.

![M2e_qualitative_examples_page3](figures/M2e_qualitative_examples_page3.png)

The two gold rectangles within each row mark the matched official instance; labels show ten-character token prefixes (full IDs in metadata/CSV). Fixed [16, 16] grid, any overlap. Seeded scene/camera/gap/motion round-robin strata with 5–95% quantile targets include both small and large displacement; cases are illustrative, not statistical evidence. Page 3/4; 30 individual examples saved.

![M2e_qualitative_examples_page4](figures/M2e_qualitative_examples_page4.png)

The two gold rectangles within each row mark the matched official instance; labels show ten-character token prefixes (full IDs in metadata/CSV). Fixed [16, 16] grid, any overlap. Seeded scene/camera/gap/motion round-robin strata with 5–95% quantile targets include both small and large displacement; cases are illustrative, not statistical evidence. Page 4/4; 30 individual examples saved.

| Example | Official instance | Motion group | Measurements |
| --- | --- | --- | --- |
| [visualizations/example_00.png](visualizations/example_00.png) | 970c58eec6d8438fabe45ef0e007af9b | moving | scene-0557 | CAM_FRONT | Δt=2.100s | ego 0.01m, 0.08° | displacement 56.3px | J=0.800 | moving |
| [visualizations/example_01.png](visualizations/example_01.png) | cdfe58028a85437a9a674b3e40b0de3c | approximately_static | scene-0276 | CAM_FRONT_RIGHT | Δt=0.400s | ego 1.50m, 0.59° | displacement 18.2px | J=0.400 | approximately_static |
| [visualizations/example_02.png](visualizations/example_02.png) | 699aa864ecd84cf28251ed1c62899204 | approximately_static | scene-0274 | CAM_BACK | Δt=0.400s | ego 2.70m, 0.22° | displacement 7.0px | J=1.000 | approximately_static |
| [visualizations/example_03.png](visualizations/example_03.png) | 0fb867083804484c9a2716ac81655381 | approximately_static | scene-0913 | CAM_BACK_LEFT | Δt=0.500s | ego 5.05m, 0.19° | displacement 160.0px | J=0.250 | approximately_static |
| [visualizations/example_04.png](visualizations/example_04.png) | 6691fe985beb40d0a32510f180c256cb | approximately_static | scene-0274 | CAM_BACK_LEFT | Δt=1.900s | ego 15.54m, 0.53° | displacement 548.6px | J=0.000 | approximately_static |
| [visualizations/example_05.png](visualizations/example_05.png) | 7af0c8c5882a413c86ebf90c514496eb | approximately_static | scene-0632 | CAM_FRONT | Δt=1.000s | ego 5.08m, 2.38° | displacement 68.5px | J=0.667 | approximately_static |
| [visualizations/example_06.png](visualizations/example_06.png) | cdbc121bfba1476eacd2c2d4c49e05e2 | moving | scene-0274 | CAM_FRONT_LEFT | Δt=1.000s | ego 5.23m, 0.31° | displacement 116.9px | J=0.000 | moving |
| [visualizations/example_07.png](visualizations/example_07.png) | 290bac5f558344ada78024abf2c7ff28 | moving | scene-0971 | CAM_BACK_RIGHT | Δt=1.000s | ego 4.49m, 7.52° | displacement 92.3px | J=0.000 | moving |
| [visualizations/example_08.png](visualizations/example_08.png) | 03e5bc118a29476980e7a86af24ba303 | moving | scene-0559 | CAM_BACK | Δt=1.000s | ego 6.42m, 1.01° | displacement 12.6px | J=1.000 | moving |
| [visualizations/example_09.png](visualizations/example_09.png) | 8826721c6075476392484f500ea392b0 | approximately_static | scene-0972 | CAM_BACK | Δt=2.000s | ego 9.93m, 2.96° | displacement 49.7px | J=1.000 | approximately_static |
| [visualizations/example_10.png](visualizations/example_10.png) | f1165d52bf4545df90998a1194152be4 | approximately_static | scene-0971 | CAM_BACK | Δt=1.000s | ego 6.34m, 0.87° | displacement 56.3px | J=1.000 | approximately_static |
| [visualizations/example_11.png](visualizations/example_11.png) | a379f897aab0458fb57dc09e0d2d8224 | moving | scene-0275 | CAM_FRONT_RIGHT | Δt=1.900s | ego 10.92m, 0.69° | displacement 114.8px | J=0.333 | moving |
| [visualizations/example_12.png](visualizations/example_12.png) | f496699e85a746daa91479d936815f14 | moving | scene-0971 | CAM_FRONT_RIGHT | Δt=1.900s | ego 7.72m, 37.74° | displacement 10.3px | J=0.400 | moving |
| [visualizations/example_13.png](visualizations/example_13.png) | 65e7eaec45764e91b69e2db131576699 | approximately_static | scene-0632 | CAM_FRONT_RIGHT | Δt=0.500s | ego 2.39m, 3.98° | displacement 168.4px | J=0.474 | approximately_static |
| [visualizations/example_14.png](visualizations/example_14.png) | c5a84da0d2354f73982d26ef313284d6 | approximately_static | scene-0274 | CAM_BACK_LEFT | Δt=0.900s | ego 6.43m, 0.54° | displacement 589.7px | J=0.000 | approximately_static |
| [visualizations/example_15.png](visualizations/example_15.png) | b643c8ce661646eb97c6023e6fed9b98 | approximately_static | scene-0913 | CAM_BACK_RIGHT | Δt=0.500s | ego 4.98m, 0.27° | displacement 248.8px | J=0.200 | approximately_static |
| [visualizations/example_16.png](visualizations/example_16.png) | c7a83adc61e345b2af2c58128f0dbdc7 | approximately_static | scene-0632 | CAM_FRONT_LEFT | Δt=1.000s | ego 5.27m, 1.06° | displacement 221.0px | J=0.167 | approximately_static |
| [visualizations/example_17.png](visualizations/example_17.png) | 0203eacab8144a22bb601b6e46006312 | moving | scene-0274 | CAM_BACK_RIGHT | Δt=0.500s | ego 3.64m, 0.26° | displacement 169.0px | J=0.000 | moving |
| [visualizations/example_18.png](visualizations/example_18.png) | df70e364489546038f245c482871ce1c | moving | scene-0967 | CAM_FRONT_RIGHT | Δt=0.450s | ego 2.13m, 0.62° | displacement 70.9px | J=0.500 | moving |
| [visualizations/example_19.png](visualizations/example_19.png) | e7dbd95082e7419298da2c5c0afaaef9 | approximately_static | scene-0274 | CAM_BACK | Δt=0.900s | ego 6.85m, 1.98° | displacement 102.8px | J=0.333 | approximately_static |
| [visualizations/example_20.png](visualizations/example_20.png) | 021a2284e7204c9fb3b2b35008a54d0c | approximately_static | scene-0275 | CAM_BACK_RIGHT | Δt=1.950s | ego 2.77m, 5.00° | displacement 458.7px | J=0.400 | approximately_static |
| [visualizations/example_21.png](visualizations/example_21.png) | 49f72147fff1454db945bed9332b6f6b | approximately_static | scene-0967 | CAM_BACK_RIGHT | Δt=1.000s | ego 2.59m, 0.38° | displacement 325.3px | J=0.324 | approximately_static |
| [visualizations/example_22.png](visualizations/example_22.png) | 074cfece001948c9a352d44332c4a0c0 | approximately_static | scene-0559 | CAM_BACK_RIGHT | Δt=1.000s | ego 5.56m, 0.69° | displacement 807.7px | J=0.000 | approximately_static |
| [visualizations/example_23.png](visualizations/example_23.png) | c8072d7b1c114a28b620aa0630ddde46 | approximately_static | scene-0274 | CAM_FRONT_RIGHT | Δt=1.900s | ego 13.59m, 2.07° | displacement 1077.9px | J=0.000 | approximately_static |
| [visualizations/example_24.png](visualizations/example_24.png) | 290bac5f558344ada78024abf2c7ff28 | moving | scene-0971 | CAM_BACK | Δt=2.000s | ego 4.93m, 21.05° | displacement 304.4px | J=0.000 | moving |
| [visualizations/example_25.png](visualizations/example_25.png) | 074cfece001948c9a352d44332c4a0c0 | approximately_static | scene-0559 | CAM_FRONT_RIGHT | Δt=1.002s | ego 5.96m, 0.51° | displacement 727.5px | J=0.050 | approximately_static |
| [visualizations/example_26.png](visualizations/example_26.png) | 4a79e5b45798495e8c3166c8f4ca3f05 | moving | scene-0559 | CAM_BACK_RIGHT | Δt=1.900s | ego 13.30m, 2.54° | displacement 197.4px | J=0.722 | moving |
| [visualizations/example_27.png](visualizations/example_27.png) | e37ef9a447354bf393a5c39c35037a0a | approximately_static | scene-0276 | CAM_FRONT | Δt=2.000s | ego 3.95m, 7.76° | displacement 267.9px | J=0.000 | approximately_static |
| [visualizations/example_28.png](visualizations/example_28.png) | 30741c62103349c48a1dd3e28e47b0c8 | moving | scene-0971 | CAM_FRONT | Δt=0.500s | ego 2.72m, 0.88° | displacement 35.7px | J=0.500 | moving |
| [visualizations/example_29.png](visualizations/example_29.png) | 98ce48a112174c6d9b7455d10c8d98bc | approximately_static | scene-0972 | CAM_BACK | Δt=1.000s | ego 7.35m, 1.10° | displacement 254.0px | J=0.273 | approximately_static |


## A. Observation Supported by Evidence

- Gap 1: global scene-equal annotated persistence is 0.9531; among 16453 persistent camera-pair observations, normalized displacement median/P90 is 0.0350/0.1534. Fixed-grid scene-equal Jaccard means are 8×8: 0.6317, 16×16: 0.5138, 32×32: 0.4312.
  Identity persistence does not imply an unchanged image coordinate in this measured subset. The measured magnitude and interval, rather than a universal significance claim, quantify the mismatch.
- Gap 2: global scene-equal annotated persistence is 0.9141; among 13871 persistent camera-pair observations, normalized displacement median/P90 is 0.0605/0.2590. Fixed-grid scene-equal Jaccard means are 8×8: 0.4782, 16×16: 0.3640, 32×32: 0.2910.
  Identity persistence does not imply an unchanged image coordinate in this measured subset. The measured magnitude and interval, rather than a universal significance claim, quantify the mismatch.
- Gap 4: global scene-equal annotated persistence is 0.8399; among 9719 persistent camera-pair observations, normalized displacement median/P90 is 0.0887/0.3680. Fixed-grid scene-equal Jaccard means are 8×8: 0.3673, 16×16: 0.2630, 32×32: 0.2009.
  Identity persistence does not imply an unchanged image coordinate in this measured subset. The measured magnitude and interval, rather than a universal significance claim, quantify the mismatch.

Ego associations are reported in the correlation table and controlled strata; their direction and strength may differ across groups. The data establish measured relations in the selected workload, not a sole causal explanation or universal claim about all driving scenes.

## B. Claims Not Established

This experiment does **not** establish:

- World Alignment outperforming other association methods.
- Arbitrary visual features being reusable.
- VLM caches necessarily failing or fixed-grid overlap equalling cache hit rate.
- CLIP patch features permitting skipped computation.
- WorldStream improving accuracy or inference speed.
- Incremental GPU runtime necessarily yielding benefits.
- Deadline scheduling being proven necessary.

The evidence is limited to annotated instance persistence, projected bounding rectangles, fixed diagnostic grids, and observed ego-motion associations in the selected real dataset.

## Verification

Actual automated tests: 42 run; 0 failures; 0 errors; 0 skipped. Details and command are saved in `metadata/verification_results.json`.

## Reproduction and provenance

Retain `metadata/experiment_config.yaml`, `metadata/run_metadata.json`, raw CSVs and plotting scripts. Config/source/table identity is checked before stage reuse. Run from the Senna repository root:

```bash
python experiments/motivation_exp2/run_exp2.py --config /data/qh/phdCode/work4/Senna/outputs/motivation_exp2/metadata/experiment_config.yaml --run-all
```

Official schema/API references: [nuScenes tutorial](https://www.nuscenes.org/tutorials/nuscenes_tutorial.html), [devkit implementation](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py).
