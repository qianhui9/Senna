"""M2 panels from actual CSV measurements, including scene-level 95% CIs."""
import json
import shutil
import numpy as np
import pandas as pd

from .analyze_image_displacement import BBOX_FIELDS
from .utils.data_utils import LOG, read_table, standalone, write_json
from .utils.statistics_utils import metric_summary
from .utils.visualization_utils import setup, save_figure, unavailable, qualitative_selection, overlay, render_projection_regressions

NAMES = ["M2a_physical_persistence", "M2b_image_displacement", "M2c_ego_motion_displacement", "M2d_grid_overlap", "M2e_qualitative_examples"]


def mean_ci(ax, x, stats, color, marker="o"):
    y = stats["scene_equal_weight_mean"]
    if y is None:
        return
    ax.plot(x, y, marker, color=color, markersize=4)
    if stats["ci95"]:
        ax.vlines(x, *stats["ci95"], color=color, linewidth=1.4)


def binned_curve(ax, data, x, y, bins, cfg, color, label, marker="o"):
    work = data.copy()
    work["bin"] = pd.cut(work[x], bins, right=False)
    xs, ys, bin_metadata = [], [], []
    for _, group in work.groupby("bin", observed=True):
        # Each physical camera observation's motion contributes once to x.
        xx = float(group[["pair_id", "camera_channel", x]].drop_duplicates()[x].median())
        stats = metric_summary(group, y, cfg)
        if stats["scene_equal_weight_mean"] is None:
            continue
        xs.append(xx)
        ys.append(stats["scene_equal_weight_mean"])
        ax.plot(xx, ys[-1], marker, color=color, markerfacecolor="white" if stats["scenes"] == 1 else color, markersize=4)
        ax.annotate(f"S={stats['scenes']}", (xx,ys[-1]), xytext=(3,4 if marker == "o" else -12), textcoords="offset points", color=color, fontsize=7)
        bin_metadata.append({"x_median_unique_camera_pairs": xx, "n": stats["n"], "scenes": stats["scenes"],
                             "mean": stats["scene_equal_weight_mean"], "ci95": stats["ci95"]})
        if stats["ci95"]:
            ax.vlines(xx, *stats["ci95"], color=color, alpha=.65, linewidth=1)
    ax.plot(xs, ys, "--", color=color, linewidth=1)
    ax.plot([], [], "--", color=color, label=label, marker=marker, markersize=4, linewidth=1)
    return bin_metadata


def run(cfg, out, only=None):
    summary = json.loads((out / "statistics" / "summary.json").read_text())
    plt = setup()
    a = cfg["analysis"]
    gaps = sorted(cfg["pairs"]["keyframe_gaps"])
    ego = read_table(out, "ego_pairs")
    actual = ego.groupby("gap_steps").delta_t_seconds.mean().to_dict()
    labels = [f"{actual[g]:.2f}s\n(gap {g})" for g in gaps]
    motion = read_table(out, "ego_motion_displacement")
    metadata = {}
    bin_counts = {}
    colors = ["#0072b2", "#e69f00", "#009e73", "#d55e00", "#cc79a7", "#56b4e9", "#333333"]
    def save(fig, name, caption, n):
        save_figure(fig, out, name)
        metadata[name] = {"available": bool(n), "n": n, "caption": caption, "png_dpi": 300,
            "pdf": "Vector axes/text; real camera photographs are embedded raster", "bootstrap_unit": "scene",
            "script": f"experiments/motivation_exp2/gen_{name[:3]}.py"}

    if only in [None, "M2a"]:
        fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
        for j, camera in enumerate(["GLOBAL"]+cfg["dataset"]["cameras"]):
            groups = summary["global_persistence"] if camera == "GLOBAL" else [g for g in summary["camera_persistence"] if g["camera_channel"] == camera]
            index = {g["gap_steps"]: g for g in groups}
            xs, ys = [], []
            n = 0
            for i, gap in enumerate(gaps):
                stats = index[gap]["metrics"]["persistent_ratio"]
                n += stats["n"]
                x = i+(j-3)*.018
                if stats["scene_equal_weight_mean"] is not None:
                    xs.append(x)
                    ys.append(stats["scene_equal_weight_mean"])
                    mean_ci(ax, x, stats, colors[j], "s" if camera == "GLOBAL" else "o")
            ax.plot(xs, ys, "--" if camera == "GLOBAL" else "-", color=colors[j], linewidth=1,
                    label=f"{camera.removeprefix('CAM_')} (n={n})")
        ax.set_xticks(range(len(gaps)), labels)
        ax.set_xlabel("Actual sample time interval")
        ax.set_ylabel("Persistent object ratio")
        ax.set_ylim(-.02, 1.03)
        ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=7)
        ax.text(.98,.03, f"{ego.scene_token.nunique()} scenes", ha="right", transform=ax.transAxes, fontsize=8)
        save(fig, NAMES[0], "Global annotated persistence and camera projected-in-view persistence use separate sets and different denominators; the curve gap is not a decomposition. Scene-equal pair means; 95% scene-bootstrap intervals; n counts pairs with a nonempty initial set, summed over gaps in each legend. Actual mean sample gaps are shown; camera exposure gaps are recorded separately.", len(ego.drop_duplicates('pair_id')))
        metadata[NAMES[0]]["counts"] = {"global_valid_pairs_by_gap": {str(g['gap_steps']): g['metrics']['persistent_ratio']['n'] for g in summary['global_persistence']},
                "camera_valid_pairs": [{"gap_steps":g['gap_steps'], "camera":g['camera_channel'], "n":g['metrics']['persistent_ratio']['n']} for g in summary['camera_persistence']]}
    if only in [None, "M2b"]:
        fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
        for i, gap in enumerate(gaps):
            group = motion[motion.gap_steps == gap]
            if len(group):
                box = ax.boxplot(group.normalized_displacement, positions=[i], widths=.45, whis=[5,95], showfliers=False, patch_artist=True,
                                 medianprops={"color":"black"}, boxprops={"facecolor":"#56b4e9", "alpha":.6})
                stats = metric_summary(group, "normalized_displacement", cfg)
                mean_ci(ax, i+.28, stats, "#d55e00", "D")
            ax.text(i, 1.025, f"n={len(group)}; {group.scene_token.nunique()} scenes", ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=8)
        if motion.empty:
            unavailable(ax, "No same-instance valid camera projections")
        ax.set_xticks(range(len(gaps)), labels)
        ax.set_xlabel("Actual sample time interval")
        ax.set_ylabel("Center displacement / image diagonal")
        ax.set_ylim(bottom=0)
        ax.plot([],[], "D", color="#d55e00", label="Scene-equal mean and 95% scene-bootstrap interval")
        ax.legend(loc="upper left", bbox_to_anchor=(0,1.22), frameon=False)
        save(fig, NAMES[1], "Same official instance ID, same camera and scene, valid projections at both times. Boxes: pooled 25–75%, median; whiskers: 5–95%. Orange diamonds: scene-equal means with 95% scene-bootstrap intervals. n is camera-pair instance observations; repeated temporal pairs, repeated instances and cross-camera projections are dependent.", len(motion))
    if only in [None, "M2c"]:
        fig, axes = plt.subplots(2, len(gaps), figsize=(6.8, 5.8), squeeze=False, sharey=True, layout="constrained")
        for col, gap in enumerate(gaps):
            group = motion[motion.gap_steps == gap]
            for row, (x, xlabel, bins) in enumerate([("ego_translation_m", "Ego translation (m)", a["translation_bins_m"]), ("ego_rotation_deg", "Ego rotation (deg)", a["yaw_bins_deg"])]):
                ax = axes[row,col]
                for k, kind in enumerate(["approximately_static", "moving"]):
                    part = group[group.motion_group == kind]
                    sampled = part.sample(n=min(1200,len(part)), random_state=a["seed"])
                    ax.scatter(sampled[x], sampled.normalized_displacement, s=3, alpha=.12, color=colors[k], rasterized=False)
                    bin_counts[f"M2c:{gap}:{x}:{kind}"] = binned_curve(ax, part, x, "normalized_displacement", bins, cfg, colors[k], f"{'Static approx.' if k == 0 else 'Moving'} n={len(part)}", "o" if k == 0 else "s")
                ax.text(.02,.98, f"{actual[gap]:.2f}s; {group.scene_token.nunique()} scenes", va="top", transform=ax.transAxes, fontsize=8)
                ax.set_xlabel(xlabel)
                if col == 0:
                    ax.set_ylabel("Displacement / image diagonal")
                ax.set_ylim(bottom=0)
                if len(motion):
                    ax.set_ylim(0, max(.01, motion.normalized_displacement.max()*1.06))
                ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0,1.25), fontsize=8)
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        save(fig, NAMES[2], f"Time-gap strata; approximately static means global annotation-center speed ≤{a['static_speed_threshold_mps']} m/s over actual sample timestamps. Scatter is seeded and independently capped at 1,200 observations per group, so it does not encode group frequency; binned summaries use all rows. X medians count each unique camera pair once; y is scene-equal displacement mean with 95% scene-bootstrap intervals. S labels give scenes per bin; open markers denote one-scene bins without CIs. Dashed segments connect bin summaries, not a fitted causal response. Associations do not establish ego-only causation.", len(motion))
    if only in [None, "M2d"]:
        gm = read_table(out, "grid_ego_motion")
        gm = gm[(gm.grid_rule == a["primary_grid_rule"]) & (gm.motion_group == "approximately_static")]
        fig, axes = plt.subplots(2, len(gaps), figsize=(6.8, 5.8), squeeze=False, sharey=True, layout="constrained")
        for col, gap in enumerate(gaps):
            group = gm[gm.gap_steps == gap]
            for row, (x,xlabel,bins) in enumerate([("ego_translation_m","Ego translation (m)",a["translation_bins_m"]), ("ego_rotation_deg","Ego rotation (deg)",a["yaw_bins_deg"])]):
                ax = axes[row,col]
                for j, shape in enumerate(a["grid_shapes"]):
                    part = group[(group.grid_rows == shape[0]) & (group.grid_cols == shape[1])]
                    bin_counts[f"M2d:{gap}:{x}:{shape}"] = binned_curve(ax, part, x, "grid_overlap", bins, cfg, colors[j], f"{shape[0]}×{shape[1]}, n={len(part)}", ["o","s","D"][j%3])
                ax.text(.02,.04, f"{actual[gap]:.2f}s; {group.scene_token.nunique()} scenes", transform=ax.transAxes, fontsize=8)
                ax.set_xlabel(xlabel)
                if col == 0:
                    ax.set_ylabel("Fixed-grid Jaccard overlap")
                ax.set_ylim(-.02,1.02)
                ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0,1.25), fontsize=8)
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        rule_definition = {"any":"positive-area box/cell intersection", "coverage":f"intersection/cell area ≥{a['grid_min_coverage_ratio']}", "center":"cell center inside the box"}[a['primary_grid_rule']]
        save(fig, NAMES[3], f"Approximately static annotated objects only: global annotation-center speed ≤{a['static_speed_threshold_mps']} m/s over actual sample timestamps. Main rule {a['primary_grid_rule']}: {rule_definition}. Grids are diagnostic image regions, not CLIP tokens. X medians count unique camera pairs once; y is scene-equal Jaccard mean with 95% scene-bootstrap intervals. S labels count scenes per bin; open markers denote singleton-scene bins without CIs. Grid variants share the same underlying observations and are not independent objects. Dashed segments connect bin summaries.", len(gm.drop_duplicates(['pair_id','camera_channel','instance_token'])))
    if only in [None, "M2e"]:
        chosen = qualitative_selection(motion, cfg["output"]["visualization_samples"] if cfg["output"]["save_visualizations"] else 0, a["seed"])
        frames = read_table(out, "camera_frames").set_index("sample_data_token")
        gallery = chosen.iloc[:min(12,len(chosen))]
        page_count = max(1,int(np.ceil(len(gallery)/3)))
        pages = [plt.subplots(min(3,max(1,len(gallery)-page*3)),2,figsize=(6.8,6.0 if len(gallery)-page*3 >= 3 else 2.1*max(1,len(gallery)-page*3)),squeeze=False,layout="constrained") for page in range(page_count)]
        examples = []
        for i, r in enumerate(chosen.to_dict("records")):
            text = (f"{r['scene_name']} | {r['camera_channel']} | Δt={r['camera_delta_t_seconds']:.3f}s | "
                    f"ego {r['ego_translation_m']:.2f}m, {r['ego_rotation_deg']:.2f}° | "
                    f"displacement {r['center_displacement_px']:.1f}px | J={r['grid_overlap']:.3f} | {r['motion_group']}")
            individual, iaxes = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
            for t in [0,1]:
                frame = frames.loc[r[f"sample_data_token_t{t}"]]
                box = [r[f"{k}_t{t}"] for k in BBOX_FIELDS]
                overlay(iaxes[t], frame.image_path, box, r["instance_token"], a["ego_grid_shape"])
                iaxes[t].text(.01,.97, "t" if t == 0 else "t + Δt", transform=iaxes[t].transAxes, color="white", va="top", fontsize=9)
                if i < len(gallery):
                    fig, axes = pages[i//3]
                    local = i%3
                    overlay(axes[local,t], frame.image_path, box, r["instance_token"], a["ego_grid_shape"])
                    axes[local,t].text(.01,.97, "t" if t == 0 else "t + Δt", transform=axes[local,t].transAxes, color="white", va="top", fontsize=9)
            individual.text(.5, .015, text, ha="center", fontsize=8)
            individual.savefig(out / "visualizations" / f"example_{i:02d}.png", dpi=300, bbox_inches="tight")
            plt.close(individual)
            if i < len(gallery):
                display = f"{r['scene_name']} | {r['camera_channel']} | Δt={r['camera_delta_t_seconds']:.3f}s | {r['motion_group']}\n"+f"ego {r['ego_translation_m']:.2f}m, {r['ego_rotation_deg']:.2f}° | displacement {r['center_displacement_px']:.1f}px | J={r['grid_overlap']:.3f}"
                axes[local,0].text(0,-.08, display, transform=axes[local,0].transAxes, fontsize=8)
            examples.append({"file": f"visualizations/example_{i:02d}.png", "caption": text, "instance_token": r["instance_token"], "pair_id": r["pair_id"], "motion_group": r["motion_group"]})
        if len(chosen) == 0:
            fig, axes = pages[0]
            for ax in axes.flat:
                unavailable(ax, "No valid image pairs or visualizations disabled")
        for page, (fig, axes) in enumerate(pages):
            name = NAMES[4]+(f"_page{page+1}" if page else "")
            save(fig, name, f"The two gold rectangles within each row mark the matched official instance; labels show ten-character token prefixes (full IDs in metadata/CSV). Fixed {a['ego_grid_shape']} grid, {a['primary_grid_rule']} overlap. Seeded scene/camera/gap/motion round-robin strata with 5–95% quantile targets include both small and large displacement; cases are illustrative, not statistical evidence. Page {page+1}/{page_count}; {len(chosen)} individual examples saved.", min(3,max(0,len(gallery)-page*3)))
        write_json(out / "metadata" / "qualitative_examples.json", examples)
        if cfg["output"]["save_visualizations"]:
            render_projection_regressions(out)
    old = out / "metadata" / "figure_metadata.json"
    previous = json.loads(old.read_text()) if only and old.exists() else {}
    write_json(old, {**previous, **metadata})
    bin_file = out / "metadata" / "figure_bin_counts.json"
    previous_bins = json.loads(bin_file.read_text()) if only and bin_file.exists() else {}
    write_json(bin_file, {**previous_bins, **bin_counts})
    snippets = []
    for name, meta in {**previous, **metadata}.items():
        caption = meta["caption"].replace("%", r"\%").replace("_", r"\_").replace("×", r"$\times$")
        snippets.append("\\begin{figure*}[t]\n\\centering\n"+f"\\includegraphics[width=0.95\\textwidth]{{figure_pdf/{name}.pdf}}\n\\caption{{{caption}}}\n\\label{{fig:{name}}}\n\\end{{figure*}}\n")
    (out / "figure_pdf" / "latex_includes.tex").write_text("\n".join(snippets))
    # Keep the exact plotting source beside the outputs as well as in the repository.
    source = __import__("pathlib").Path(__file__).resolve().parent
    scripts = out / "figures" / "scripts"
    scripts.mkdir(exist_ok=True)
    for p in [source / "plot_results.py", source / "utils" / "visualization_utils.py", source / "utils" / "statistics_utils.py", *source.glob("gen_M2*.py")]:
        shutil.copy2(p, scripts / p.name)
    LOG.info("Rendered %d figure panels with real measurements", len(metadata))


if __name__ == "__main__":
    standalone("plot")
