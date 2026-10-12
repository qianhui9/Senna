"""Render evidence, missing stages, and limitations from recorded outputs."""
import json

from .utils.data_utils import read_csv
from .utils.stats_utils import finite_values


def fmt(value):
    return "unavailable" if value is None else f"{value:.4g}" if isinstance(value, (float, int)) else str(value)


def interval(stats):
    return "undefined (<2 scenes)" if not stats["ci95"] else f"[{fmt(stats['ci95'][0])}, {fmt(stats['ci95'][1])}]"


def run(cfg, args, out):
    path = out / "aggregate_summary.json"
    if not path.exists():
        raise ValueError("Run --mode aggregate first")
    summary = json.loads(path.read_text())
    metadata = json.loads((out / "run_metadata.json").read_text())
    model_path = out / "model_metadata.json"
    model = json.loads(model_path.read_text()) if model_path.exists() else {}
    figures_path = out / "figure_metadata.json"
    figures = json.loads(figures_path.read_text()) if figures_path.exists() else {}
    equivalence_path = out / "output_equivalence.json"
    equivalence = json.loads(equivalence_path.read_text()) if equivalence_path.exists() else []
    counts = summary["counts"]
    lines = ["# WorldStream Motivation Experiment 1", "", "## Hypothesis and scope", "", "Continuous driving observations may contain persistent visual content while the original request-based VLM repeats visual encoding. This experiment measures existing behavior only; it implements no WorldStream mechanism. Similarity, persistence and cost are measured independently except for explicitly verified matched records.", "", "## Reproduction and provenance", "", f"- Senna commit: `{metadata['git_commit']}`; local source SHA256 hashes and working-tree changes are recorded in `run_metadata.json`.", f"- Checkpoint: `{cfg['senna']['checkpoint_path']}`; original base weights, no LoRA or weight changes.", f"- nuScenes: `{cfg['nuscenes']['version']}` at `{cfg['nuscenes']['dataroot']}`.", f"- Evaluation JSON: `{cfg['senna']['eval_json_path']}`; SHA256 `{metadata.get('eval_json_path_sha256', 'unavailable')}`.", f"- Seed: {cfg['experiment']['seed']}; resolved settings in `resolved_config.yaml`.", f"- Python: `{metadata['python'].splitlines()[0]}`.", f"- Camera order: `{', '.join(metadata['camera_order'])}`.", "- Local fork contains corrected camera ordering/navigation and validated evaluation inputs. The experiment invokes this local `eval_multi_img_model_wo_init` directly. Official reference: https://github.com/hustvl/Senna.", "", "## Workload and indexing", "", "`sample_data.next` is followed separately for all six cameras, including non-keyframe sweeps. Scene boundaries, calibrated sensor channels, cycles, file existence, reciprocal links and timestamp anomalies are checked. The first gap group uses direct adjacent links; remaining targets use nearest actual timestamps within the configured tolerance. Pairs are selected with seeded content-independent hashes, round-robin over Scene/Camera/gap strata.", "", "| Quantity | Recorded count |", "|---|---:|"]
    lines += [f"| {key} | {value} |" for key, value in counts.items()]
    lines += ["", "## Image-space similarity", "", f"Analysis-only resize: `{cfg['experiment']['analysis_resize']}` pixels; patch grid: `{cfg['experiment']['patch_grid']}`. SSIM uses RGB channel averaging, Gaussian local moments (11×11, sigma 1.5, population covariance) and valid interior pixels. Small patches use a documented smaller odd window. PSNR uses peak=1 after division by 255; identical images have +infinity PSNR. Infinite values are excluded from finite PSNR quantiles, with identical-pixel cases retained in raw CSV. Normalized pixel difference is mean absolute RGB difference in [0,1].", "", summary["statistical_unit"], "", "| Camera | Gap | Pairs | Scenes | Actual Δt ms (scene mean) | SSIM scene mean | Scene bootstrap 95% CI |", "|---|---|---:|---:|---:|---:|---|"]
    for group in summary["similarity"]:
        s = group["metrics"]["ssim"]
        lines.append(f"| {group['camera_channel']} | {group['gap_group']} | {group['records']} | {s['scenes']} | {fmt(group['metrics']['actual_delta_t_ms']['scene_equal_weight_mean'])} | {fmt(s['scene_equal_weight_mean'])} | {interval(s)} |")
    lines += ["", "All PSNR, normalized difference and patch threshold statistics, including confidence intervals, are in `scene_statistics.csv` and `aggregate_summary.json`. Thresholds are sensitivity analyses; high patch similarity is not object identity or a FLOPs-saving fraction.", "", "## Annotated physical instance persistence", "", "Scope: global official 3D annotations at adjacent keyframes. Persistence ratio = size(intersection of instance tokens) / previous instance count; an empty previous set is undefined. Jaccard is also recorded. These are neither per-camera visibility estimates nor fabricated sweep annotations."]
    for group in summary["objects"]:
        s = group["metrics"]["persistence_ratio"]
        lines.append(f"\n{group['records']} annotated keyframe pairs, {s['scenes']} scenes; scene-equal persistence mean {fmt(s['scene_equal_weight_mean'])}, scene bootstrap 95% CI {interval(s)}.")
    if not summary["objects"]:
        lines.append("\nReliable annotated persistence results are unavailable.")
    lines += ["", "## Original Senna architecture and runtime", "", "Actual entry: `llava/model/builder.py::load_senna_pretrained_model` → `SennaLlavaLlamaForCausalLM`, with the implementation in `llava/senna/`, rather than generic `llava/model/llava_arch.py`. CLIP ViT-L/14-336 selects layer -2 patch features (576 tokens per image). Each of six views passes through the MLP Projector and the original img_adapter (128 tokens per image, 768 total). These are confirmed at runtime by recorded module shapes rather than assumed in the statistics.", "", "Read-only module hooks surround `get_vision_tower()`, `get_model().mm_projector` and `get_model().img_adapter`. Temporary transparent wrappers time loading/preprocessing, bracket multimodal preparation and `generate()`, and measure the original LLM `forward()` calls without modifying their arguments or outputs. All attributes/hooks are restored even on exceptions. LLM KV cache remains within each generation request; no visual state crosses requests.", "", "Single-device stage durations use CUDA events on the current stream. They include stream idle/CPU-dispatch gaps between events, and are not pure kernel-only costs. Preparation and generation are parent intervals: exclusive preparation = preparation − Vision − Projector − adapter; LLM generation = generation − inclusive preparation before the LLM. The no-op preparation checks inside cached forwards remain in Decode to avoid overlap. End-to-end uses synchronized wall clock and includes loading, preprocessing, transfer, prompt processing and answer decoding. Other is its residual. Prefill is the actual full-prompt forward with empty incoming KV cache; Decode sums subsequent single-token forwards with nonempty KV cache. CUDA events bracket the original forwards. Prefill produces logits for generated token 1, so Decode has output_tokens − 1 calls. Generation setup, token selection, cache bookkeeping, stopping checks and dispatch outside forward are retained as llm_generation_overhead_ms. The measured Prefill + Decode + generation overhead equals the exclusive LLM generation parent. Prefill is not end-to-end time to first token, and these are stream intervals rather than pure kernel times. Multiple-device models get synchronized wall time and metadata, with unreliable stage decomposition left unavailable. Detailed Profiler records are excluded from latency summaries.", "", f"Model load: {fmt(model.get('model_loading_ms'))} ms (excluded from steady-state results). GPU: `{model.get('gpu', 'unavailable')}`. Warmup iterations: {cfg['experiment']['warmup_iterations']}; repeats per request: {cfg['experiment']['profile_repeats']}; max new tokens: {cfg['senna']['max_new_tokens']}; temperature=0, beam=1, template=llava_v1, use_cache=True.", "", f"Output equivalence: {sum(r['passed'] for r in equivalence)}/{len(equivalence)} fixed samples passed exact token IDs, decoded answers, prediction plans, input tensor hashes and generation configuration checks.", "", "| Stage/metric | Finite records | Mean | P50 | P95 | P99 |", "|---|---:|---:|---:|---:|---:|"]
    for metric, s in summary["profiling"]["metrics"].items():
        lines.append(f"| {metric} | {s['n']} | {fmt(s['mean'])} | {fmt(s['p50'])} | {fmt(s['p95'])} | {fmt(s['p99'])} |")
    lines += ["", "`profile_scene_statistics` in `aggregate_summary.json` records scene-level latency confidence intervals. Request output lengths exclude the artificial initial BOS token in the inspected inputs_embeds generation path; returned sequence length and BOS removal are also recorded. Shapes, allocated/reserved GPU peaks, call counts and per-forward CUDA durations/input lengths/incoming and outgoing KV lengths are in `profiling_samples.csv`. Exactly one Prefill, one-token cached Decode, monotonic KV growth and the forward/output-token count relationship are checked per request. `llm_decode_ms_per_token` divides Decode by the cached-forward count, excluding generated token 1. Detailed traces expose `senna.llm_prefill` and `senna.llm_decode` ranges; these ranges are disabled in low-overhead timing. The ideal bound removes measured Vision + Projector + adapter time at zero added management cost; it is not a measured or attainable speedup."]
    timed = [r for r in read_csv(out / "profiling_samples.csv") if r["measurement_mode"] == "timing"]
    for field in ("input_shape", "vision_tokens_per_image", "visual_tokens_total", "output_tokens"):
        values = sorted({r[field] for r in timed})
        lines.append(f"\nRecorded {field}: `{', '.join(values) or 'unavailable'}`.")
    memory = defaultdict_memory(timed)
    for device, value in memory.items():
        lines.append(f"\nPeak allocated memory on {device}: {value / 2**30:.3f} GiB (includes resident weights; per-request peaks in CSV).")
    lines += ["", "## Figures and matched workload", "", "Similarity and original inference use distinct input sources. Sample-level comparisons use only `matched_profile_pairs.csv`, which verifies sample tokens and all six exact image paths against the keyframe manifest. Front-camera similarity is an explanatory covariate of the same six-view request, not a camera-specific latency measurement. No correlation is inferred from unrelated averages."]
    for name, fig in figures.items():
        lines += ["", f"![{name}](figures/{name}.png)", "", f"[Vector PDF](figures/{name}.pdf). Availability: {fig['available']}. {fig['reason']}"]
    lines += ["", "M1(d) replaces the former LLM Generation bar with measured LLM Prefill and LLM Decode. Other includes img_adapter, loading, transfer, exclusive preparation, generation overhead and residual costs. Its stacked segments still sum to synchronized end-to-end latency. The first 12 requests are shown, averaging 3 repeats per request; full-workload statistics use all timing records. Parent timings are never stacked with their children. M1(e) has no fabricated values when exact linkage is absent.", "", "## Missing data, anomalies and limitations", ""]
    for stage, status in summary["statuses"].items():
        lines.append(f"- {stage}: {status.get('status')}; {status.get('reason', '')}")
    for source, records in summary["issues"].items():
        lines.append(f"- {source}: {len(records)} recorded issues; inspect `{source}.json`.")
    lines += ["- CPU OS page cache is not flushed; file-loading cost reflects the warm workload. Instrumentation has small overhead; detailed tracing is kept separate.", "- Scene bootstrap with few scenes is exploratory; a smoke test establishes executability, not a representative workload claim. No hypothesis significance or practical-cost threshold is predeclared.", "- Camera exposure timestamps are asynchronous. Temporal similarity is camera-specific; global annotations use sample timestamps.", "- Scene selection prefers the configured evaluation workload when scene tokens are available. Results describe that sampled population.", "", "## Observation Supported by Evidence", ""]
    if summary["similarity"]:
        means = [g["metrics"]["ssim"]["scene_equal_weight_mean"] for g in summary["similarity"]]
        lines.append(f"Measured group-level scene-mean SSIM spans {fmt(min(means))}–{fmt(max(means))}; the tables and M1(a/b) quantify content similarity and its variation with actual interval and camera. The measurements, rather than a preset redundancy assumption, determine the observation.")
    else:
        lines.append("Image redundancy has not been measured successfully in this run.")
    if summary["objects"]:
        lines.append(f"\nGlobal annotated instances persist at the recorded ratio of {fmt(summary['objects'][0]['metrics']['persistence_ratio']['scene_equal_weight_mean'])}; camera-visible persistence remains unmeasured.")
    calls = finite_values([r["vision_tower_calls"] for r in timed])
    if len(calls):
        lines.append(f"\nAll {len(calls)} recorded low-overhead requests execute {fmt(calls.min())}–{fmt(calls.max())} Vision Tower calls; runtime shapes/call counts establish repeated encoding for these requests. Vision latency fraction mean is {fmt(summary['profiling']['metrics']['vision_latency_ratio']['mean'])}; its practical importance must be judged against the measured end-to-end cost and workload variability.")
        lines.append("\nFor each recorded input shape, `input_shape_groups` reports encoding token counts and latency. Fixed token counts establish a fixed encoding scale in the tested inputs; they do not establish that runtime cost is independent of scene content. The hypothesis receives evidence only for the recorded components and sampled workload.")
    else:
        lines.append("\nRuntime execution and cost remain unmeasured; the full motivation hypothesis cannot be assessed from image statistics alone.")
    lines += ["", "## Claims Not Established", "", "- Image similarity does not establish direct reuse of visual features.", "- Persistent annotated objects do not establish that all visual encoding is redundant.", "- World Alignment benefits have not been demonstrated.", "- Incremental inference GPU speedups have not been demonstrated.", "- Deadline misses have not been demonstrated.", "- No new WorldStream mechanism has been evaluated.", "- Statistical association is not causation, and separate dataset averages do not establish per-request correlation.", ""]
    (out / "experiment_report.md").write_text("\n".join(lines))


def defaultdict_memory(rows):
    result = {}
    for row in rows:
        for device, value in json.loads(row["peak_allocated_bytes"]).items():
            result[device] = max(result.get(device, 0), value)
    return result


if __name__ == "__main__":
    from .run import main
    raise SystemExit(main("report"))
