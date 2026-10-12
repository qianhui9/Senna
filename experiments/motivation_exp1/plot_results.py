"""Export measured motivation panels as 300 DPI PNG and vector PDF."""
from collections import defaultdict
import json
import os

import numpy as np

from .utils.data_utils import CAMERAS, read_csv, write_json
from .utils.stats_utils import finite_values


def run(cfg, args, out):
    os.environ.setdefault("MPLCONFIGDIR", str(out / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    summary_path = out / "aggregate_summary.json"
    if not summary_path.exists():
        raise ValueError("Run --mode aggregate first")
    summary = json.loads(summary_path.read_text())
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    metadata = {}
    def save(fig, name, available, reason=""):
        fig.savefig(fig_dir / (name + ".png"), dpi=300, bbox_inches="tight")
        fig.savefig(fig_dir / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)
        metadata[name] = {"available": available, "reason": reason, "png_dpi": 300, "pdf": "vector text/plots; frame photographs embedded as raster"}
    def unavailable(ax, reason):
        ax.text(.5, .5, "Unavailable\n" + reason, ha="center", va="center", transform=ax.transAxes, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])

    fig, ax = plt.subplots(figsize=(6.5, 3.7), layout="constrained")
    data = summary["similarity"]
    if data:
        for camera in CAMERAS:
            groups = [g for g in data if g["camera_channel"] == camera]
            groups.sort(key=lambda g: g["metrics"]["actual_delta_t_ms"]["scene_equal_weight_mean"])
            xs, ys = [], []
            for g in groups:
                stat = g["metrics"]["ssim"]
                x, y = g["metrics"]["actual_delta_t_ms"]["scene_equal_weight_mean"], stat["scene_equal_weight_mean"]
                xs.append(x)
                ys.append(y)
                if stat["ci95"]:
                    ax.vlines(x, *stat["ci95"], alpha=.35)
            ax.plot(xs, ys, "o-", label=camera.removeprefix("CAM_"), linewidth=1)
        ax.legend(ncol=3, fontsize=7)
        ax.set_ylim(-.05, 1.25)
        ax.set_yticks(np.arange(0, 1.01, .2))
    else:
        unavailable(ax, "No measured image similarity")
    ax.set(xlabel="Actual time interval (ms; scene means)", ylabel="SSIM (scene-equal-weight mean)", title="M1(a) Temporal Visual Similarity")
    save(fig, "M1a_temporal_similarity", bool(data))

    fig, ax = plt.subplots(figsize=(6.5, 3.7), layout="constrained")
    similarity = [r for r in read_csv(out / "temporal_similarity.csv") if r.get("status") == "ok"]
    if similarity:
        from .utils.stats_utils import scene_bootstrap
        for gap in sorted({r["gap_group"] for r in similarity}):
            records = [r for r in similarity if r["gap_group"] == gap]
            stats = [scene_bootstrap(records, f"high_similarity_ratio_{t:.2f}", cfg["experiment"]["seed"], cfg["experiment"]["bootstrap_iterations"]) for t in cfg["experiment"]["ssim_thresholds"]]
            ax.plot(cfg["experiment"]["ssim_thresholds"], [s["scene_equal_weight_mean"] for s in stats], "o-", label=gap)
            for x, s in zip(cfg["experiment"]["ssim_thresholds"], stats):
                if s["ci95"]:
                    ax.vlines(x, *s["ci95"], alpha=.35)
        ax.legend()
        ax.set_ylim(0, 1.05)
    else:
        unavailable(ax, "No measured patches")
    ax.set(xlabel="Patch SSIM threshold", ylabel="High-similarity patch fraction", title="M1(b) High-Similarity Region Ratio")
    save(fig, "M1b_patch_similarity", bool(similarity))

    fig, ax = plt.subplots(figsize=(6.5, 3.7), layout="constrained")
    objects = read_csv(out / "object_persistence.csv")
    object_groups = defaultdict(list)
    for r in objects:
        object_groups[r["scene_name"]].append(r["persistence_ratio"])
    usable = [(s, v) for s, values in sorted(object_groups.items()) if len(v := finite_values(values))]
    if usable:
        ax.boxplot([v for _, v in usable], labels=[s for s, _ in usable], showfliers=False)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, 1.05)
        ax.text(.02, .02, "Global 3D annotations; camera visibility not evaluated", fontsize=7, transform=ax.transAxes)
    else:
        unavailable(ax, "Reliable keyframe annotations unavailable")
    ax.set(xlabel="Scene", ylabel="Persistent / previous annotated instances", title="M1(c) Physical Object Persistence")
    save(fig, "M1c_object_persistence", bool(usable))

    fig, ax = plt.subplots(figsize=(6.5, 3.7), layout="constrained")
    profile = [r for r in read_csv(out / "profiling_samples.csv") if r["measurement_mode"] == "timing" and str(r["timing_decomposition_valid"]).lower() == "true" and str(r.get("llm_split_valid", "")).lower() == "true"]
    if profile:
        stages = {"Preprocessing": ["image_preprocessing_ms"], "Vision Tower": ["vision_tower_ms"], "Projector": ["projector_ms"], "LLM Prefill": ["llm_prefill_ms"], "LLM Decode": ["llm_decode_ms"], "Other": ["image_file_loading_ms", "host_to_device_ms", "img_adapter_ms", "multimodal_preparation_ms", "llm_generation_overhead_ms", "other_ms"]}
        groups = defaultdict(list)
        for r in profile:
            groups[r["sample_token"]].append(r)
        # Limit plotted requests; statistics still include the full workload.
        keys = list(groups)[:12]
        bottom = np.zeros(len(keys))
        for label, fields in stages.items():
            values = [np.mean([sum(float(r[f]) for f in fields) for r in groups[k]]) for k in keys]
            ax.bar(np.arange(len(keys)), values, bottom=bottom, label=label)
            bottom += values
        ax.set_xticks(np.arange(len(keys)), [k[:6] for k in keys], rotation=45)
        ax.set_ylim(0, max(bottom) * 1.35)
        ax.legend(ncol=3, fontsize=7, loc="upper left")
        ax.text(.02, .84, "Other: loading, transfer, adapter, preparation + generation overhead\nPrefill produces token 1; Decode sums subsequent cached forwards", va="top", fontsize=7, transform=ax.transAxes)
    else:
        unavailable(ax, "No valid single-device measured prefill/decode timing; rerun profiling")
    ax.set(xlabel="Original six-view request (sample token prefix)", ylabel="Latency (ms)", title="M1(d) Senna Inference Latency Breakdown")
    save(fig, "M1d_latency_breakdown", bool(profile))
    metadata["M1d_latency_breakdown"].update(llm_split="measured forward intervals; generation overhead in Other", plotted_requests=min(12, len({r["sample_token"] for r in profile})), timing_records=len(profile))

    matched = [r for r in read_csv(out / "matched_profile_pairs.csv") if r.get("camera_channel") == "CAM_FRONT"]
    # Show frames and execution for exactly matched evaluation requests only.
    if matched:
        records = matched[:4]
        fig = plt.figure(figsize=(max(6.5, 2.6 * len(records)), 5.6), layout="constrained")
        grid = fig.add_gridspec(3, len(records), height_ratios=[1, 1, 1.2])
        for i, row in enumerate(records):
            for j, field in enumerate(("previous_image_path", "current_image_path")):
                ax = fig.add_subplot(grid[j, i])
                with Image.open(row[field]) as image:
                    ax.imshow(image.convert("RGB"))
                ax.axis("off")
                ax.set_title(("Previous" if j == 0 else "Current") + (f"\nSSIM={float(row['ssim']):.3f}, Δt={float(row['actual_delta_t_ms']):.1f}ms" if j else f"\n{row['scene_name']}"), fontsize=8)
        ax = fig.add_subplot(grid[2, :])
        values = [float(r["current_vision_tower_calls"]) for r in records]
        ax.bar(np.arange(len(records)), values, width=.5)
        for i, r in enumerate(records):
            ms = r["current_vision_tower_ms"]
            label = f"{float(r['current_visual_tokens_total']):g} visual tokens"
            if ms:
                label += f"\nVision {float(ms):.1f}ms"
            ax.text(i, values[i] + .02, label, ha="center", fontsize=8)
        ax.set_xticks(np.arange(len(records)), [r["current_sample_token"][:6] for r in records])
        ax.set(ylabel="Vision Tower calls per current request", xlabel="Exact matched six-view request; front camera shown", ylim=(0, max(values) * 1.65))
        fig.suptitle("M1(e) Visual Redundancy and Measured Execution", fontsize=11)
    else:
        fig, ax = plt.subplots(figsize=(6.5, 3.7), layout="constrained")
        unavailable(ax, "No exact linkage between similarity and timed requests")
        ax.set_title("M1(e) Visual Redundancy and Measured Execution")
    save(fig, "M1e_repeated_execution", bool(matched), "Exact six-camera sample_data linkage required")
    if matched and any(r.get("current_vision_tower_ms") for r in matched):
        fig, ax = plt.subplots(figsize=(5.4, 3.7), layout="constrained")
        valid = [r for r in matched if r.get("current_vision_tower_ms")]
        ax.scatter([float(r["ssim"]) for r in valid], [float(r["current_vision_tower_ms"]) for r in valid], s=25)
        ax.set(xlabel="Matched front-camera SSIM", ylabel="Current six-view Vision Tower latency (ms)", title="Matched requests only; descriptive, not causal")
        save(fig, "matched_similarity_latency", True)
    write_json(out / "figure_metadata.json", metadata)


if __name__ == "__main__":
    from .run import main
    raise SystemExit(main("plot"))
