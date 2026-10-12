# Experiment 3: Physical Identity vs. Representation Stability



## Research question and scope



Test whether official physical identity implies invariant intermediate visual representations across continuous camera observations. Q1–Q5 cover same-ID variation, observation conditions, CLIP layers, approximate motion strata and matched different-ID baselines. All images undergo original full-image visual forwards. This is a motivation observation, with no WorldStream implementation or task-accuracy evaluation.



## Reproducibility and original model



- Senna Git commit: `8428d6689aefd25ed0d1b629d54fc34d1e75061b`; source/input hashes and weight file identities: `metadata/run_metadata.json`.

- Checkpoint: `/data/qh/phdCode/work4/models/Senna`; loaded class `SennaLlavaLlamaForCausalLM`.

- Tower: `CLIPVisionTower`, `/data/qh/phdCode/work4/models/clip-vit-large-patch14-336`; 24 transformer blocks, hidden dimension 1024, patch size 14, image size 336.

- Actual device/precision: `cuda:0` / `torch.float16`; GPU `NVIDIA GeForce RTX 3090`.

- Selected feature `patch`, configured layer -2, resolved hidden-state index 23.

- Camera order: FRONT, FRONT LEFT, FRONT RIGHT, BACK LEFT, BACK RIGHT, BACK. Each camera is encoded independently before original per-view token compression.



```text

Sequential(
  (0): Linear(in_features=1024, out_features=4096, bias=True)
  (1): GELU(approximate='none')
  (2): Linear(in_features=4096, out_features=4096, bias=True)
)

img_adapter(
  (adapter): Sequential(
    (0): Linear(in_features=576, out_features=128, bias=True)
    (1): GELU(approximate='none')
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): GELU(approximate='none')
    (4): Linear(in_features=128, out_features=128, bias=True)
  )
)

```



Learned token-axis Linear/GELU network; globally mixes 576 into 128 tokens. Only global analysis permitted.

Projector descriptors are collected before the img_adapter. The adapter output is summarized globally in `data/global_stability.csv`; it never receives bounding-box patch indices.



## Original image preprocessing and exact spatial mapping



Model aspect-ratio strategy: `pad`. Original `llava.mm_utils.process_images` performs mean-color square padding, processor resize, center crop, rescale and normalization. Processor configuration:



```json

{
  "_processor_class": null,
  "feature_extractor_type": "CLIPFeatureExtractor",
  "do_resize": true,
  "size": {
    "shortest_edge": 336
  },
  "resample": 3,
  "do_center_crop": true,
  "crop_size": {
    "height": 336,
    "width": 336
  },
  "do_rescale": true,
  "rescale_factor": 0.00392156862745098,
  "do_normalize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "do_convert_rgb": true,
  "image_processor_type": "CLIPImageProcessor"
}

```



Every image records its original dimensions, integer padding/crop offsets and rounded resize dimensions in its feature shard and `object_patch_mapping.csv`. Runtime input dimensions are compared with the recorded transform. Boxes are clipped to the actual input region; completely removed, low-retention and insufficient-patch objects are excluded and counted.

Area pooling requires intersection/patch area ≥ 0.25, at least 2 patches, retained bbox area ≥ 0.5. Row-major patch indices exclude CLS. Mean pooling uses the same qualifying patches; center pooling uses contained centers. Grid [24, 24] is derived from the actual vision configuration and validated against tensor token counts.



## Layer indexing and actual output shapes



CLIP hidden_states[0] is embedding+position after pre_layernorm, before transformer block 1



| layer | shape | dtype | spatial_mapping |
| --- | --- | --- | --- |
| layer_0 | [1, 577, 1024] | torch.float16 | True |
| layer_6 | [1, 577, 1024] | torch.float16 | True |
| layer_12 | [1, 577, 1024] | torch.float16 | True |
| layer_18 | [1, 577, 1024] | torch.float16 | True |
| layer_23 | [1, 577, 1024] | torch.float16 | True |
| layer_24 | [1, 577, 1024] | torch.float16 | True |
| projector | [1, 576, 4096] | torch.float16 | True |
| adapter_global | [1, 128, 4096] | torch.float16 | False |



Configured early/middle/late/terminal hidden states and the resolved Senna-selected state are extracted with `output_hidden_states=True` through transparent forward hooks. Hooks return None. All-layer comparisons use an identical valid comparison cohort, separately by aggregation method and control group.



## Forward correctness and numerical control



3 batches compared original and instrumented Vision Tower + compressed visual outputs; all passed: True. Absolute tolerance 0.001, relative tolerance 0.001. Full shape/dtype/max-error checks and repeated object-descriptor metrics: `metadata/output_equivalence.json`.

Repeated-input selected-layer descriptor controls: n=16, maximum cosine distance=5.960464477539063e-08. Aggregation and distances use FP32; saved descriptors use FP16. Nonfinite and near-zero norms are explicitly labeled, not replaced with zero.



## Official identity pairs, selection and missingness



E2 matched records: 40043; selected-scene candidates 40043; quality-filtered 2257 (5.64%); eligible 37786; selected 3000; sampling omissions 34786; unique physical instances 709.

Actual scenes 10, cameras ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'], camera gaps {'min': 0.399996, 'max': 2.2, 'median': 1.0}. Scene list: `metadata/scene_list.json`.

seeded round robin over scene/camera/category/gap/motion/area/displacement/ego strata; descriptive selected cohort, not population estimate

Per-scene, camera, category, gap and motion counts before/after selection and all quality exclusions are saved in `metadata/pair_selection.json`. Sample selection is not representative of the whole nuScenes population.

Identity is checked against E2 projected official annotations at both endpoints with exact scene/camera/instance/sample-data tokens, category, bbox and actual image existence. Similarity time uses camera exposures; approximate object speed uses annotation-center displacement over sample timestamp gaps.



| aggregation | mapping_status | n |
| --- | --- | --- |
| area | insufficient_patches | 5550 |
| area | valid | 11763 |
| center | insufficient_patches | 7492 |
| center | valid | 9821 |
| mean | insufficient_patches | 5550 |
| mean | valid | 11763 |



## Same identity and matched comparison baselines



B1 compares same official instance over time. B2 changes identity within the exact category; B3 changes category. Both use the same anchor, scene, camera and exact target frame, with target area and reliable depth ratios bounded by predeclared constraints. Unmatched anchors are explicitly recorded, and controls can be reused across anchors. B4 compares predeclared fixed spatial patch centers in two frames, excluding padded-content centers; summary first averages sampled patches per frame pair.

| comparison_group | metric | n | mean | p50 | std | p25 | p75 | p90 | p95 | scene_equal_weight_mean | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1_same_instance | cosine_similarity | 2011 | 0.75453 | 0.79184 | 0.1562 | 0.6812 | 0.86414 | 0.91762 | 0.93779 | 0.75821 | 0.74397 | 0.77251 |
| B1_same_instance | cosine_distance | 2011 | 0.24547 | 0.20816 | 0.1562 | 0.13586 | 0.3188 | 0.46883 | 0.55832 | 0.24179 | 0.22749 | 0.25603 |
| B1_same_instance | relative_l2_error | 2011 | 0.69059 | 0.66295 | 0.23283 | 0.53314 | 0.82256 | 1.0143 | 1.1291 | 0.68749 | 0.66346 | 0.71114 |
| B2_different_instance_same_category | cosine_similarity | 921 | 0.68103 | 0.7049 | 0.16569 | 0.58665 | 0.80408 | 0.87081 | 0.9027 | 0.68813 | 0.66903 | 0.70512 |
| B2_different_instance_same_category | cosine_distance | 921 | 0.31897 | 0.2951 | 0.16569 | 0.19592 | 0.41335 | 0.55518 | 0.64021 | 0.31187 | 0.29488 | 0.33097 |
| B2_different_instance_same_category | relative_l2_error | 921 | 0.79596 | 0.77232 | 0.23237 | 0.62825 | 0.93208 | 1.105 | 1.2156 | 0.78622 | 0.76321 | 0.81133 |
| B3_different_category | cosine_similarity | 852 | 0.52185 | 0.52647 | 0.18633 | 0.3795 | 0.67004 | 0.7725 | 0.8183 | 0.53434 | 0.5012 | 0.56929 |
| B3_different_category | cosine_distance | 852 | 0.47815 | 0.47353 | 0.18633 | 0.32996 | 0.6205 | 0.73638 | 0.79398 | 0.46566 | 0.43071 | 0.4988 |
| B3_different_category | relative_l2_error | 852 | 0.97758 | 0.96561 | 0.233 | 0.81566 | 1.1161 | 1.2535 | 1.3633 | 0.96495 | 0.92568 | 1.0028 |



Matched same-minus-different similarity margins use the same anchors, not subtraction of unrelated population means:



| control_group | n | mean | p50 | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- |
| B2_different_instance_same_category | 921 | 0.068408 | 0.037079 | 0.049756 | 0.08516 |
| B3_different_category | 852 | 0.2092 | 0.17246 | 0.16605 | 0.2373 |



Matched-cohort overlap and descriptive AUC (no fitted classifier):



| layer_id | aggregation | control_group | n | histogram_overlap | descriptive_auc |
| --- | --- | --- | --- | --- | --- |
| layer_0 | area | B2_different_instance_same_category | 921 | 0.83931 | 0.57709 |
| layer_0 | area | B3_different_category | 852 | 0.78521 | 0.62592 |
| layer_0 | center | B2_different_instance_same_category | 697 | 0.84505 | 0.5687 |
| layer_0 | center | B3_different_category | 599 | 0.80634 | 0.60046 |
| layer_0 | mean | B2_different_instance_same_category | 921 | 0.84908 | 0.57779 |
| layer_0 | mean | B3_different_category | 852 | 0.7723 | 0.62361 |
| layer_12 | area | B2_different_instance_same_category | 921 | 0.73616 | 0.66533 |
| layer_12 | area | B3_different_category | 852 | 0.49648 | 0.82099 |
| layer_12 | center | B2_different_instance_same_category | 697 | 0.7604 | 0.65575 |
| layer_12 | center | B3_different_category | 599 | 0.54257 | 0.79852 |
| layer_12 | mean | B2_different_instance_same_category | 921 | 0.7481 | 0.66006 |
| layer_12 | mean | B3_different_category | 852 | 0.51291 | 0.81253 |
| layer_18 | area | B2_different_instance_same_category | 921 | 0.75461 | 0.65416 |
| layer_18 | area | B3_different_category | 852 | 0.46948 | 0.83426 |
| layer_18 | center | B2_different_instance_same_category | 697 | 0.75036 | 0.65152 |
| layer_18 | center | B3_different_category | 599 | 0.47746 | 0.82066 |
| layer_18 | mean | B2_different_instance_same_category | 921 | 0.7709 | 0.64721 |
| layer_18 | mean | B3_different_category | 852 | 0.49178 | 0.82236 |
| layer_23 | area | B2_different_instance_same_category | 921 | 0.76764 | 0.63494 |
| layer_23 | area | B3_different_category | 852 | 0.51761 | 0.81022 |
| layer_23 | center | B2_different_instance_same_category | 697 | 0.77044 | 0.63596 |
| layer_23 | center | B3_different_category | 599 | 0.52421 | 0.79624 |
| layer_23 | mean | B2_different_instance_same_category | 921 | 0.78936 | 0.63001 |
| layer_23 | mean | B3_different_category | 852 | 0.53873 | 0.79596 |
| layer_24 | area | B2_different_instance_same_category | 921 | 0.79805 | 0.63213 |
| layer_24 | area | B3_different_category | 852 | 0.54577 | 0.80101 |
| layer_24 | center | B2_different_instance_same_category | 697 | 0.77188 | 0.633 |
| layer_24 | center | B3_different_category | 599 | 0.5576 | 0.78543 |
| layer_24 | mean | B2_different_instance_same_category | 921 | 0.80456 | 0.62555 |
| layer_24 | mean | B3_different_category | 852 | 0.57981 | 0.78423 |
| layer_6 | area | B2_different_instance_same_category | 921 | 0.75679 | 0.65396 |
| layer_6 | area | B3_different_category | 852 | 0.59859 | 0.77497 |
| layer_6 | center | B2_different_instance_same_category | 697 | 0.77044 | 0.63826 |
| layer_6 | center | B3_different_category | 599 | 0.63272 | 0.74637 |
| layer_6 | mean | B2_different_instance_same_category | 921 | 0.7633 | 0.6488 |
| layer_6 | mean | B3_different_category | 852 | 0.60681 | 0.76521 |
| projector | area | B2_different_instance_same_category | 921 | 0.81542 | 0.61409 |
| projector | area | B3_different_category | 852 | 0.58803 | 0.77081 |
| projector | center | B2_different_instance_same_category | 697 | 0.81779 | 0.60648 |
| projector | center | B3_different_category | 599 | 0.61102 | 0.75386 |

Showing 40/42 rows; full values are in the accompanying CSV/JSON.



## Actual time-gap distributions



| time_gap_group | metric | n | mean | p50 | p95 | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | cosine_similarity | 842 | 0.79911 | 0.83499 | 0.94434 | 0.78937 | 0.81786 |
| 0.5 | cosine_distance | 842 | 0.20089 | 0.16501 | 0.48897 | 0.18214 | 0.21063 |
| 0.5 | relative_l2_error | 842 | 0.61793 | 0.58491 | 1.0329 | 0.58662 | 0.63525 |
| 1.0 | cosine_similarity | 675 | 0.73972 | 0.77635 | 0.92556 | 0.72714 | 0.76334 |
| 1.0 | cosine_distance | 675 | 0.26028 | 0.22365 | 0.56916 | 0.23666 | 0.27286 |
| 1.0 | relative_l2_error | 675 | 0.71685 | 0.68967 | 1.1348 | 0.68053 | 0.74113 |
| 2.0 | cosine_similarity | 469 | 0.6972 | 0.72146 | 0.91913 | 0.67393 | 0.71517 |
| 2.0 | cosine_distance | 469 | 0.3028 | 0.27854 | 0.60886 | 0.28483 | 0.32607 |
| 2.0 | relative_l2_error | 469 | 0.78027 | 0.76349 | 1.1843 | 0.75603 | 0.82746 |
| other | cosine_similarity | 25 | 0.72786 | 0.78875 | 0.89037 | 0.59854 | 0.82237 |
| other | cosine_distance | 25 | 0.27214 | 0.21125 | 0.52402 | 0.17763 | 0.40146 |
| other | relative_l2_error | 25 | 0.7464 | 0.70193 | 1.1807 | 0.59944 | 0.91317 |

Different gaps also involve different viewpoints, projected scales and scene content; this does not isolate a pure temporal-decay effect.



## Image displacement, ego motion and observation conditions



Normalized displacement is E2 bbox-center Euclidean migration divided by original image diagonal. Displacement, translation, rotation, yaw, depth, area, scale ratio and geometric projection quality are retained. `observation_condition_stability.csv` stratifies category, motion and time; `condition_correlations.json` reports descriptive observation Spearman and scene-median bootstrap intervals.

| category_name | motion_group | time_gap_group | variable | n | scenes | observation_spearman | scene_median_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| human.pedestrian.adult | approximately_static | 0.5 | normalized_displacement | 57 | 5 | 0.50058 | 0.6 |
| human.pedestrian.adult | approximately_static | 0.5 | ego_translation_m | 57 | 5 | 0.54622 | 0.5 |
| human.pedestrian.adult | approximately_static | 0.5 | ego_rotation_deg | 57 | 5 | 0.17097 | 0.3 |
| human.pedestrian.adult | approximately_static | 0.5 | ego_yaw_change_deg | 57 | 5 | -0.24402 | -0.5 |
| human.pedestrian.adult | approximately_static | 0.5 | bbox_scale_ratio | 57 | 5 | -0.2279 | 0.2 |
| human.pedestrian.adult | approximately_static | 1.0 | normalized_displacement | 43 | 4 | 0.58185 | -0.2 |
| human.pedestrian.adult | approximately_static | 1.0 | ego_translation_m | 43 | 4 | 0.52069 | -0.2 |
| human.pedestrian.adult | approximately_static | 1.0 | ego_rotation_deg | 43 | 4 | -0.13051 | -0.4 |
| human.pedestrian.adult | approximately_static | 1.0 | ego_yaw_change_deg | 43 | 4 | -0.52205 | 0 |
| human.pedestrian.adult | approximately_static | 1.0 | bbox_scale_ratio | 43 | 4 | -0.14195 | 0 |
| human.pedestrian.adult | approximately_static | 2.0 | normalized_displacement | 32 | 4 | 0.55315 | 1 |
| human.pedestrian.adult | approximately_static | 2.0 | ego_translation_m | 32 | 4 | 0.43275 | 1 |
| human.pedestrian.adult | approximately_static | 2.0 | ego_rotation_deg | 32 | 4 | 0.17823 | 0.2 |
| human.pedestrian.adult | approximately_static | 2.0 | ego_yaw_change_deg | 32 | 4 | -0.043642 | 0.2 |
| human.pedestrian.adult | approximately_static | 2.0 | bbox_scale_ratio | 32 | 4 | 0.041056 | 0.4 |
| human.pedestrian.adult | approximately_static | other | normalized_displacement | 1 | 1 | unavailable | unavailable |
| human.pedestrian.adult | approximately_static | other | ego_translation_m | 1 | 1 | unavailable | unavailable |
| human.pedestrian.adult | approximately_static | other | ego_rotation_deg | 1 | 1 | unavailable | unavailable |
| human.pedestrian.adult | approximately_static | other | ego_yaw_change_deg | 1 | 1 | unavailable | unavailable |
| human.pedestrian.adult | approximately_static | other | bbox_scale_ratio | 1 | 1 | unavailable | unavailable |
| human.pedestrian.adult | moving | 0.5 | normalized_displacement | 89 | 8 | 0.24144 | 0.30952 |
| human.pedestrian.adult | moving | 0.5 | ego_translation_m | 89 | 8 | 0.36706 | 0.59524 |
| human.pedestrian.adult | moving | 0.5 | ego_rotation_deg | 89 | 8 | 0.28401 | 0.69048 |
| human.pedestrian.adult | moving | 0.5 | ego_yaw_change_deg | 89 | 8 | 0.14745 | -0.2619 |
| human.pedestrian.adult | moving | 0.5 | bbox_scale_ratio | 89 | 8 | -0.14534 | -0.47619 |
| human.pedestrian.adult | moving | 1.0 | normalized_displacement | 69 | 8 | 0.25623 | 0.71429 |
| human.pedestrian.adult | moving | 1.0 | ego_translation_m | 69 | 8 | 0.38946 | 0.85714 |
| human.pedestrian.adult | moving | 1.0 | ego_rotation_deg | 69 | 8 | 0.40638 | 0.64286 |
| human.pedestrian.adult | moving | 1.0 | ego_yaw_change_deg | 69 | 8 | -0.11935 | -0.2619 |
| human.pedestrian.adult | moving | 1.0 | bbox_scale_ratio | 69 | 8 | -0.26182 | -0.64286 |
| human.pedestrian.adult | moving | 2.0 | normalized_displacement | 57 | 8 | 0.13663 | 0.47619 |
| human.pedestrian.adult | moving | 2.0 | ego_translation_m | 57 | 8 | 0.31187 | 0.71429 |
| human.pedestrian.adult | moving | 2.0 | ego_rotation_deg | 57 | 8 | 0.062906 | 0.2619 |
| human.pedestrian.adult | moving | 2.0 | ego_yaw_change_deg | 57 | 8 | -0.098686 | 0.47619 |
| human.pedestrian.adult | moving | 2.0 | bbox_scale_ratio | 57 | 8 | -0.08705 | 0.095238 |
| human.pedestrian.adult | moving | other | normalized_displacement | 1 | 1 | unavailable | unavailable |
| human.pedestrian.adult | moving | other | ego_translation_m | 1 | 1 | unavailable | unavailable |
| human.pedestrian.adult | moving | other | ego_rotation_deg | 1 | 1 | unavailable | unavailable |
| human.pedestrian.adult | moving | other | ego_yaw_change_deg | 1 | 1 | unavailable | unavailable |
| human.pedestrian.adult | moving | other | bbox_scale_ratio | 1 | 1 | unavailable | unavailable |

Showing 40/420 rows; full values are in the accompanying CSV/JSON.

These associations do not identify causal effects of position embeddings, ego motion or elapsed time. Geometric image-clipping ratios are not occlusion rates; official visibility labels are not precise masks.



## Across-layer stability



| layer_id | comparison_group | n | p50 | mean | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| layer_0 | B1_same_instance | 2011 | 0.39714 | 0.41265 | 0.3681 | 0.44562 |
| layer_0 | B2_different_instance_same_category | 921 | 0.51019 | 0.50584 | 0.4656 | 0.55594 |
| layer_0 | B3_different_category | 852 | 0.56424 | 0.55162 | 0.47426 | 0.56662 |
| layer_12 | B1_same_instance | 2011 | 0.22745 | 0.25602 | 0.22839 | 0.27371 |
| layer_12 | B2_different_instance_same_category | 921 | 0.34715 | 0.35885 | 0.32909 | 0.36897 |
| layer_12 | B3_different_category | 852 | 0.49106 | 0.48688 | 0.42203 | 0.50265 |
| layer_18 | B1_same_instance | 2011 | 0.21765 | 0.25823 | 0.2362 | 0.26945 |
| layer_18 | B2_different_instance_same_category | 921 | 0.31992 | 0.35081 | 0.31857 | 0.35922 |
| layer_18 | B3_different_category | 852 | 0.53659 | 0.53504 | 0.4692 | 0.55605 |
| layer_23 | B1_same_instance | 2011 | 0.20816 | 0.24547 | 0.22749 | 0.25603 |
| layer_23 | B2_different_instance_same_category | 921 | 0.2951 | 0.31897 | 0.29488 | 0.33097 |
| layer_23 | B3_different_category | 852 | 0.47353 | 0.47815 | 0.43071 | 0.4988 |
| layer_24 | B1_same_instance | 2011 | 0.15773 | 0.18645 | 0.17069 | 0.19396 |
| layer_24 | B2_different_instance_same_category | 921 | 0.22144 | 0.24393 | 0.2254 | 0.25791 |
| layer_24 | B3_different_category | 852 | 0.35694 | 0.36831 | 0.32353 | 0.38452 |
| layer_6 | B1_same_instance | 2011 | 0.28145 | 0.28375 | 0.25736 | 0.30462 |
| layer_6 | B2_different_instance_same_category | 921 | 0.39309 | 0.38156 | 0.3511 | 0.40156 |
| layer_6 | B3_different_category | 852 | 0.47413 | 0.46709 | 0.41239 | 0.47632 |
| projector | B1_same_instance | 2011 | 0.17998 | 0.24927 | 0.23208 | 0.26027 |
| projector | B2_different_instance_same_category | 921 | 0.258 | 0.32533 | 0.30295 | 0.34162 |
| projector | B3_different_category | 852 | 0.49876 | 0.52742 | 0.46179 | 0.56964 |



## Approximately static and moving objects



Main threshold ≤ 0.5 m/s; sensitivity thresholds [0.2, 0.5, 1.0]. Nonfinite speed is Unknown. NuScenes annotation-center velocities may contain noise and do not identify unannotated roads/buildings as static instances.

| category_name | motion_group | time_gap_group | n | mean | p50 | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| human.pedestrian.adult | approximately_static | 0.5 | 57 | 0.23263 | 0.20984 | 0.15987 | 0.24952 |
| human.pedestrian.adult | approximately_static | 1.0 | 43 | 0.25476 | 0.23869 | 0.22705 | 0.2772 |
| human.pedestrian.adult | approximately_static | 2.0 | 32 | 0.2863 | 0.25113 | 0.25582 | 0.34281 |
| human.pedestrian.adult | approximately_static | other | 1 | 0.26504 | 0.26504 | unavailable | unavailable |
| human.pedestrian.adult | moving | 0.5 | 89 | 0.25184 | 0.20428 | 0.18823 | 0.30437 |
| human.pedestrian.adult | moving | 1.0 | 69 | 0.30483 | 0.26324 | 0.24474 | 0.37945 |
| human.pedestrian.adult | moving | 2.0 | 57 | 0.34861 | 0.3392 | 0.28452 | 0.45349 |
| human.pedestrian.adult | moving | other | 1 | 0.32037 | 0.32037 | unavailable | unavailable |
| human.pedestrian.construction_worker | approximately_static | 0.5 | 13 | 0.21606 | 0.19108 | 0.1598 | 0.42084 |
| human.pedestrian.construction_worker | approximately_static | 1.0 | 2 | 0.20009 | 0.20009 | 0.18915 | 0.21103 |
| human.pedestrian.construction_worker | approximately_static | 2.0 | 1 | 0.55834 | 0.55834 | unavailable | unavailable |
| human.pedestrian.construction_worker | moving | 0.5 | 17 | 0.20554 | 0.17464 | 0.14505 | 0.35768 |
| human.pedestrian.construction_worker | moving | 1.0 | 12 | 0.22197 | 0.17969 | 0.18382 | 0.42331 |
| human.pedestrian.construction_worker | moving | 2.0 | 7 | 0.39097 | 0.3246 | 0.29319 | 0.46431 |
| human.pedestrian.police_officer | approximately_static | 0.5 | 1 | 0.15177 | 0.15177 | unavailable | unavailable |
| human.pedestrian.police_officer | approximately_static | 1.0 | 1 | 0.39691 | 0.39691 | unavailable | unavailable |
| human.pedestrian.police_officer | approximately_static | 2.0 | 1 | 0.52845 | 0.52845 | unavailable | unavailable |
| human.pedestrian.police_officer | moving | 1.0 | 1 | 0.33812 | 0.33812 | unavailable | unavailable |
| movable_object.barrier | approximately_static | 0.5 | 86 | 0.19678 | 0.15353 | 0.18472 | 0.31982 |
| movable_object.barrier | approximately_static | 1.0 | 78 | 0.29837 | 0.24588 | 0.27316 | 0.36447 |
| movable_object.barrier | approximately_static | 2.0 | 36 | 0.26925 | 0.2062 | 0.25872 | 0.40671 |
| movable_object.barrier | approximately_static | other | 4 | 0.33253 | 0.23517 | 0.20801 | 0.70609 |
| movable_object.barrier | moving | 0.5 | 5 | 0.23612 | 0.28174 | 0.11087 | 0.31961 |
| movable_object.barrier | moving | 1.0 | 3 | 0.62434 | 0.70941 | 0.26941 | 0.8018 |
| movable_object.pushable_pullable | approximately_static | 0.5 | 14 | 0.15443 | 0.15736 | 0.096044 | 0.15892 |
| movable_object.pushable_pullable | approximately_static | 1.0 | 13 | 0.2458 | 0.20994 | 0.22718 | 0.46924 |
| movable_object.pushable_pullable | approximately_static | 2.0 | 7 | 0.27629 | 0.25259 | 0.27188 | 0.30271 |
| movable_object.trafficcone | approximately_static | 0.5 | 37 | 0.18597 | 0.1492 | 0.14332 | 0.25174 |
| movable_object.trafficcone | approximately_static | 1.0 | 27 | 0.19529 | 0.17816 | 0.15888 | 0.22893 |
| movable_object.trafficcone | approximately_static | 2.0 | 17 | 0.24416 | 0.24932 | 0.22107 | 0.37314 |
| movable_object.trafficcone | moving | 0.5 | 4 | 0.27606 | 0.16603 | 0.17808 | 0.30872 |
| static_object.bicycle_rack | approximately_static | 0.5 | 4 | 0.44622 | 0.47115 | 0.30393 | 0.58852 |
| static_object.bicycle_rack | approximately_static | 1.0 | 3 | 0.38902 | 0.27234 | unavailable | unavailable |
| static_object.bicycle_rack | approximately_static | 2.0 | 2 | 0.5133 | 0.5133 | unavailable | unavailable |
| static_object.bicycle_rack | approximately_static | other | 1 | 0.4164 | 0.4164 | unavailable | unavailable |
| static_object.bicycle_rack | moving | 0.5 | 1 | 0.19868 | 0.19868 | unavailable | unavailable |
| static_object.bicycle_rack | moving | 1.0 | 2 | 0.21683 | 0.21683 | unavailable | unavailable |
| static_object.bicycle_rack | moving | 2.0 | 1 | 0.34428 | 0.34428 | unavailable | unavailable |
| vehicle.bicycle | approximately_static | 0.5 | 19 | 0.28898 | 0.25822 | 0.21452 | 0.40461 |
| vehicle.bicycle | approximately_static | 1.0 | 12 | 0.31296 | 0.31593 | 0.23518 | 0.36053 |

Showing 40/84 rows; full values are in the accompanying CSV/JSON.

Motion comparisons are within category and actual gap groups; remaining scale/depth/viewpoint and annotation differences prevent a causal interpretation.



## Object-scale and patch-aggregation sensitivity



| aggregation | minimum_patch_count | n | mean | p50 | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| area | 2 | 2011 | 0.24547 | 0.20816 | 0.22749 | 0.25603 |
| center | 2 | 1669 | 0.25609 | 0.21863 | 0.2402 | 0.26719 |
| mean | 2 | 2011 | 0.24374 | 0.20989 | 0.22701 | 0.25393 |
| area | 4 | 1317 | 0.20842 | 0.17694 | 0.1911 | 0.22184 |
| center | 4 | 962 | 0.20991 | 0.17776 | 0.19632 | 0.22569 |
| mean | 4 | 1317 | 0.20614 | 0.17733 | 0.19041 | 0.21846 |
| area | 8 | 765 | 0.18126 | 0.15139 | 0.16733 | 0.19953 |
| center | 8 | 651 | 0.18258 | 0.15591 | 0.17039 | 0.20376 |
| mean | 8 | 765 | 0.17656 | 0.1512 | 0.16162 | 0.19439 |

Full area-bin summaries: `data/area_sensitivity.csv`; pooling/patch-count summaries: `data/patch_sensitivity.csv`. Minimum-patch sensitivity only tightens the primary filter; samples rejected by the configured primary patch minimum are not silently reintroduced.



## Statistical uncertainty and limitations



Bootstrap scene-equal means, not independent objects/frames. A single-scene CI is unavailable. Quantiles and AUC are descriptive. Condition associations are not causal.

Repeated objects, shared frames, cross-camera observations and reused negatives are correlated. The number of scenes limits inference. Axis-aligned projected 3D-box rectangles include background and do not establish unobstructed visibility. CLIP patch states depend on global self-attention. Offline pooled descriptors are not native object tokens or pure object semantics. Conclusions apply to the sampled checkpoint, precision and cohort.



## Figures and qualitative examples



Qualitative selection uses lower/upper feature-distance quartiles and large/small displacement quartiles, choosing cases nearest each within-condition median; remaining examples are seeded references. Missing conditions are reported rather than manufactured.

Unavailable qualitative conditions: [].


![figures/M3a_same_vs_different_similarity.png](figures/M3a_same_vs_different_similarity.png)


![figures/M3b_temporal_feature_variation.png](figures/M3b_temporal_feature_variation.png)


![figures/M3c_layer_stability.png](figures/M3c_layer_stability.png)


![figures/M3d_displacement_feature_variation.png](figures/M3d_displacement_feature_variation.png)


![figures/M3e_static_dynamic_variation.png](figures/M3e_static_dynamic_variation.png)


![figures/M3f_qualitative_examples.png](figures/M3f_qualitative_examples.png)



## Reproduction



```bash

/home/qh/miniconda3/envs/Senna/bin/python experiments/motivation_exp3/run_exp3.py \

  --config experiments/motivation_exp3/config.yaml --run-all

```



Resolved configuration: `metadata/experiment_config.yaml`; use its paths/limits to reproduce this particular cohort. `--smoke-test` uses a separate output directory, at most 2 scenes and 100 object pairs. Valid offline feature shards are reused only to resume this offline experiment; every distinct image represented in the experiment was originally fully encoded.



## A. Observation Supported by Evidence



In the selected cohort, 2011 valid same-instance pairs at layer_23 have cosine-distance median 0.20816, P95 0.55832, and relative-L2 median 0.66295. The maximum repeated-input object-descriptor cosine distance is 5.9605e-08 (16 controls); 100.0% of same-instance pairs exceed that empirical numerical envelope plus 1e-6. These measurements support non-invariance of the sampled bbox-pooled intermediate representations despite official identity persistence. They do not establish a universal or task-relevant error threshold.



Matched identity margins and overlapping baseline distributions above characterize identity discrimination separately from high absolute cosine similarity. No monotone layer/time/motion effect is presumed.



## B. Claims Not Established



This experiment does not establish that historical visual features cannot be cached; that a similarity decline implies driving-accuracy loss; or that any feature-error cutoff is an optimal reuse threshold. It does not establish that World Alignment resolves variation, that WorldStream State Validity works, that reuse reduces FLOPs by a particular amount, that incremental execution accelerates a GPU, or that a Deadline Scheduler is necessary. No features were replaced, no historical inference computations were reused, and no model was trained or fine-tuned.