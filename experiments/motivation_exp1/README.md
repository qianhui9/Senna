# WorldStream Motivation Experiment 1

Measure temporal image similarity, global annotated instance persistence, and
original Senna visual execution. This directory is independent of the existing
inference code. It adds no alignment, feature cache, pruning, scheduler, training,
or weight changes. All commands below run from the repository root.

## Environment and configuration

```bash
conda activate Senna
python -m pip install -r experiments/motivation_exp1/requirements.txt
```

The local `Senna` environment already has these dependencies. Tests use Python's
`unittest`; SSIM uses SciPy rather than requiring scikit-image. One independent
SSIM reference test uses OpenCV, which is an existing Senna dependency.

Edit `config.yaml`, pass CLI flags, or set `SENNA_MODEL_PATH` and `NUSCENES_ROOT`.
Relative configuration paths resolve against the repository root, regardless of
the configuration file's location. CLI flags override YAML. No data/model
absolute paths are hardcoded in the implementation.

This machine's known paths can be supplied explicitly:

```bash
export SENNA_MODEL_PATH=/data/qh/phdCode/work4/models/Senna
export NUSCENES_ROOT=/data/qh/nuscenes/v1.0-trainval
export MPLCONFIGDIR=/tmp/senna_motivation_matplotlib
```

The actual dataroot contains `samples/`, `sweeps/`, and the metadata subdirectory
`v1.0-trainval/`. The default evaluation JSON is the local corrected validation
set. The original local evaluation helper requires fp16; unsupported precision
is rejected. No LoRA adapters or quantization are loaded.

## Run and reproduce

Check configuration, metadata tables, selected evaluation images, weights, and
CUDA availability without loading weights or producing experiment output:

```bash
python -m experiments.motivation_exp1.run --dry-run
```

Run a genuine minimum workload (2 scenes, 36 pairs, 2 requests, 1 warmup,
1 repeat; also export one detailed trace):

```bash
python -m experiments.motivation_exp1.run --smoke-test --profile-mode both \
  --output-dir outputs/motivation_exp1_smoke --existing overwrite
```

Run the full configured E1 (10 scenes, 500 pairs, 100 original requests,
10 warmups, 3 repeats, 1 detailed trace):

```bash
python -m experiments.motivation_exp1.run --profile-mode both \
  --output-dir outputs/motivation_exp1 --existing overwrite
```

Run only data analysis, aggregation, plots, and a partial report (no GPU needed):

```bash
python -m experiments.motivation_exp1.run --mode data \
  --output-dir outputs/motivation_exp1_data --existing overwrite
```

The `index`, `objects`, `redundancy`, `profile`, `aggregate`, `plot`, and `report`
modes run independently. Keep the same science configuration and output directory
between stages. For example:

```bash
python -m experiments.motivation_exp1.run --mode index
python -m experiments.motivation_exp1.run --mode redundancy --existing resume
python -m experiments.motivation_exp1.run --mode objects
python -m experiments.motivation_exp1.run --mode profile --profile-mode timing --existing resume
python -m experiments.motivation_exp1.run --mode profile --profile-mode detailed --existing resume
python -m experiments.motivation_exp1.run --mode aggregate
python -m experiments.motivation_exp1.run --mode plot
python -m experiments.motivation_exp1.run --mode report
```

Module entry points such as
`python -m experiments.motivation_exp1.build_nuscenes_manifest` also work. Use
`-m` instead of executing an individual `.py` file, so imports resolve correctly.

Important CLI overrides:

```bash
python -m experiments.motivation_exp1.run \
  --checkpoint-path "$SENNA_MODEL_PATH" --dataroot "$NUSCENES_ROOT" \
  --eval-json-path data/senna_nusc_val_plan_corrected.json \
  --version v1.0-trainval --device cuda:0 --device-map single \
  --max-scenes 10 --max-pairs 500 --max-profile-samples 100 \
  --warmup-iterations 10 --profile-repeats 3 --max-new-tokens 64 \
  --temporal-gaps-seconds 0.083 0.25 0.5 --patch-grid 8 8 \
  --analysis-resize 320 192 --seed 42 \
  --profile-mode both --output-dir outputs/motivation_exp1 --existing resume
```

`--existing skip` skips completed stages. `resume` also reuses complete pair
checkpoints and request/repeat rows after interruption; writes are atomic.
`overwrite` recomputes requested stages. Configuration, source, dependency, table,
evaluation JSON, and model-file identity are verified before reuse. If any identity
changes, overwrite clears experiment-owned outputs to prevent stale provenance;
use a new directory for a different workload. Do not run concurrent writers in
one output directory. An interrupted model run still warms up and rechecks
equivalence before continuing. Detailed records never enter timing summaries.

## Local inference path and timing definitions

The repository is a compatible fork (`qianhui9/Senna`), based on the
[official Senna project](https://github.com/hustvl/Senna). Existing local changes
correct camera ordering and navigation and validate planning inputs. The
experiment calls the local `data_tools.senna_qa_utils.eval_multi_img_model_wo_init`
unchanged, using its original questions and prompt template `llava_v1`.

Actual model class: `llava/senna/senna_llava_llama.py::SennaLlavaLlamaForCausalLM`.
The generic `llava/model/llava_arch.py` is not this model's multimodal execution
path. Input order is front, front-left, front-right, back-left, back-right, back;
the explicit evaluation JSON protocol must agree. Actual observed shapes:

- input `[1,6,3,336,336]`, original pad/CLIP preprocessing;
- Vision Tower batched output `[6,576,1024]`;
- per-view MLP Projector output `[1,576,4096]`;
- original `img_adapter` output `[1,128,4096]`, 768 final visual tokens.

These shapes are recorded from real module calls. The collector records no
feature tensors and keeps no state between requests. Temporary wrappers and
hooks return original outputs and restore all attributes even on exceptions.
Equivalence checks compare original and instrumented input hashes, generation
settings, token IDs, decoded outputs, and parsed speed/path predictions.

Low-overhead timing uses synchronized wall time for CPU/loading/transfers and
CUDA events around module and parent intervals. Only transfer boundaries and
the request boundaries synchronize. Current-stream intervals include idle time
and CPU dispatch gaps; detailed traces expose individual CPU operators/GPU
kernels and memory activity. Breakdown definitions are disjoint:

- `vision_tower_ms`, `projector_ms`, `img_adapter_ms`: actual leaf module calls;
- `multimodal_preparation_ms`: parent preparation minus these three leaves;
- `llm_generation_ms`: parent generation minus inclusive preparation;
- `llm_prefill_ms`: the original LLM forward consuming the full multimodal
  prompt with empty incoming KV cache, including the logits for generated token 1;
- `llm_decode_ms`: sum of subsequent original single-token forwards using
  nonempty KV cache (there are `output_tokens - 1` calls);
- `llm_generation_overhead_ms`: `llm_generation_ms - llm_prefill_ms -
  llm_decode_ms`, retaining generation setup, token selection, cache bookkeeping,
  stopping checks and dispatch outside the forward intervals;
- `other_ms`: end-to-end minus loading, preprocessing, transfers and generation.

`preparation_inclusive_ms` and `generate_inclusive_ms` are diagnostic parents;
`llm_generation_ms` is also a parent of the measured prefill, decode and generation
overhead intervals. Never add these parents to their children. M1(d) stacks
separate LLM Prefill and LLM Decode bars. Its Other combines img_adapter, exclusive
preparation, transfers, loading, generation overhead and residual time. All
segments still sum to the synchronized end-to-end request latency.

Phase classification inspects the actual incoming KV cache and forward input
length. Each row records `llm_forward_calls` with input shapes, input/output KV
lengths and CUDA-event durations. Exactly one prefill, single-token decode,
monotonic KV growth and the forward/output-token relationship are verified.
Senna's no-op multimodal preparation checks inside cached forwards remain inside
Decode; only preparation before the LLM is subtracted from the generation parent.
`llm_decode_ms_per_token` divides Decode by the actual number of cached forwards,
excluding the first generated token. Prefill is an LLM forward interval, not
end-to-end time to first token. Timing includes current-stream dispatch/idle
gaps; it is not pure kernel time. Detailed traces contain `senna.llm_prefill` and
`senna.llm_decode` ranges, which are disabled during low-overhead measurements.
Old rows without measured splits cannot produce M1(d); rerun profiling instead
of estimating a split from the old total. End-to-end excludes model load, warmup, equivalence
checks and detailed tracing. CPU OS file caches are not flushed. Output token
counts exclude the artificial BOS prefix in this local inputs_embeds generation
path; returned sequence length and raw IDs are retained.

`--device-map auto` supports the local loader's deployment choices. With CPU
offload or more than one parameter device, synchronized end-to-end wall time and
metadata remain available; stage durations/ratios are unavailable, rather than
reported with misleading precision. Main latency results should use a single
CUDA device. The internal `use_cache=True` is autoregressive KV cache within one
request and does not reuse driving-frame vision features.

The ideal speedup bound is `E2E / (E2E − vision − projector − adapter)`. It assumes
zero added management cost and does not establish achievable reuse or acceleration.

## Data and metric definitions

The official nuScenes devkit follows each camera's `sample_data.next` links from
the scene's first keyframe, with scene/sensor checks. It includes sweeps, uses
real asynchronous camera exposure timestamps, records missing files and interval
anomalies, and never pairs across scenes. The first configured target means
direct adjacent sweep links (it does not impose a 1/12-second interval).
Other gaps choose the nearest strictly later timestamp within ±0.035 seconds.
Sampling balances Scene/Camera/gap strata with seeded hashes unrelated to image
content. Adjacent missing files never turn a non-adjacent pair into an adjacent one.

Analysis resize is separate from model preprocessing. Metrics:

- SSIM: mean across RGB channels of Gaussian local structural similarity,
  11×11 window, sigma=1.5, population moments and valid interior;
- PSNR: `10 log10(1 / MSE)` on RGB scaled to [0,1], +infinity for identical images;
- normalized pixel difference: mean absolute RGB difference on [0,1];
- patch similarity: same metrics on complete regular-grid partitions (including
  non-divisible edges); fractions passing SSIM 0.70, 0.80, 0.90;
- instance persistence: intersection / previous annotated keyframe instance set;
  undefined if previous set is empty; new/disappeared counts and Jaccard retained.

Global 3D instance persistence is not per-camera visibility or sweep-level ground
truth. Scene-equal means and 95% percentile bootstrap confidence intervals resample
scene means, avoiding independent-frame assumptions. CI is undefined with one
scene. Infinite PSNR is retained in raw results and excluded from finite quantiles.

Profiling samples are existing planning records; they are not arbitrary sweeps
with invented questions. If index linkage exists, requests are chosen in successive
pairs within seeded scene groups. Extra matched analysis verifies every one of
the six keyframe image paths against `sample_data` before comparing front-camera
similarity with that request's measured execution. Otherwise M1(e) is explicitly
unavailable. No unrelated average is used to infer a sample-level correlation.

## Outputs

| Artifact | Contents |
|---|---|
| `camera_manifest.csv`, `manifest_summary.json`, `manifest_issues.json` | Camera timeline, missing/abnormal records, selected scene descriptions |
| `temporal_pairs.csv` | Same-scene, same-camera paths/tokens, actual timestamps and gap groups |
| `temporal_similarity.csv`, `patch_similarity.csv` | Traceable raw image and patch metrics |
| `object_persistence.csv`, `object_issues.json` | Global keyframe instance statistics and annotation errors |
| `profiling_samples.csv`, `profiling_summary.json` | Timings, shapes, tokens, predictions, request/repeat/mode, GPU memory |
| `model_metadata.json`, `output_equivalence.json` | Model/processor/config, loading time, original/instrumented equivalence |
| `matched_profile_pairs.csv` | Exact matched inputs, measured similarity and repeated execution |
| `profiler_trace/*.json`, `*_operators.txt` | Chrome/Perfetto trace, operator CPU/GPU/memory summary |
| `aggregate_summary.json`, `scene_statistics.csv` | Distributions, scene bootstrap and stage availability |
| `figures/M1[a-e]*.png`, `*.pdf`, `figure_metadata.json` | PNG 300 DPI, vector plot PDF, honest unavailable panels |
| `experiment_report.md` | Evidence, figures, limitations, supported observations and unestablished claims |
| `run_metadata.json`, `resolved_config.yaml`, `*_status.json` | Provenance, exact configuration, checkpoints and failures |

PDF plots/text are vector. M1(e)'s original photographs remain raster image
content inside the PDF. Missing annotations/profiling/linkage produce clearly
labelled unavailable panels, never simulated curves. Raw JSON serialization uses
null for non-finite summary values; raw CSV retains infinite identical-image PSNR.

## Verification and troubleshooting

```bash
python -m unittest discover -s experiments/motivation_exp1/tests -v
python -m experiments.motivation_exp1.run --smoke-test --profile-mode both \
  --output-dir outputs/motivation_exp1_smoke --existing overwrite
```

Tests use synthetic fixtures only to verify timestamps/boundaries, missing files,
selection, SSIM against an independent Gaussian implementation, metric boundary
cases, scene weighting, non-overlapping timing, hook cleanup and detailed-record
exclusion. Their values are never reported as experimental evidence. The real
smoke test uses installed nuScenes images and existing base model weights.

- **CUDA unavailable:** verify the `Senna` environment and GPU access. In a Codex
  sandbox GPU device access can require an approved execution outside the sandbox.
  `--mode data` still runs without CUDA; full runs record the GPU failure and
  generate a partial report, exiting nonzero.
- **RAM during devkit loading:** full metadata here includes a 1.3 GB sample_data
  JSON plus large ego-pose/annotation tables. Scene caps reduce indexed output,
  not the official devkit's initial table-loading memory. All-mode shares a single
  devkit object for indexing/persistence and releases it before model loading.
- **Missing sweeps:** inspect `manifest_issues.json`; install actual sweep image
  files. Missing pairs are not fabricated from keyframes.
- **Checkpoint/CLIP path missing:** configure the Senna checkpoint and its
  `config.json` vision tower path, then run dry-run. Do not change weights.
- **Legacy evaluation protocol:** use the existing corrected JSON. This experiment
  deliberately uses validated local inputs and rejects mismatched camera order.
- **Output identity differs:** choose a fresh directory, or run all stages with
  overwrite. Changing configuration mid-pipeline cannot silently reuse old data.
- **Failure/interruption:** inspect stage status/CSV; rerun the same command with
  `--existing resume`. No concurrent writes to the same directory are supported.

For reproduction, retain the raw artifacts, resolved config, source hashes,
table/model file identities, environment versions, and original evaluation JSON.
Warmup and profiling hardware conditions affect numerical timings. Treat smoke
results as functional validation; conclusions about a representative driving
workload require the configured multi-scene run and its reported uncertainty.
