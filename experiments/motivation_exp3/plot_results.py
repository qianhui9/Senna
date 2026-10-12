"""Six evidence-only figures, editable publication PDF and unbiased real examples."""
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/senna_exp3_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

from .utils.common import read, json_read, write_json, standalone
from .analyze_layer_stability import common_layer_cohort
from .utils.statistics import grouped

NAMES = ["M3a_same_vs_different_similarity", "M3b_temporal_feature_variation", "M3c_layer_stability",
         "M3d_displacement_feature_variation", "M3e_static_dynamic_variation", "M3f_qualitative_examples"]
COLORS = ["#286995", "#d17b2c", "#458e67", "#956daf"]


def boxplot(ax, values, labels, ylabel):
    good = [(v, label) for v, label in zip(values, labels) if len(v)]
    if good:
        ax.boxplot([v for v, _ in good], labels=[f"{label}\nn={len(v)}" for v, label in good], showfliers=False)
    else:
        ax.text(.5, .5, "No valid observations", ha="center", transform=ax.transAxes)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.25)


def choose_examples(data, limit):
    """Predeclared quartile conditions; take nearest median, then seeded remainder."""
    if not len(data) or limit <= 0:
        return [], []
    qd = data.cosine_distance.quantile([.25, .75]).to_numpy()
    qx = data.normalized_displacement.quantile([.25, .75]).to_numpy()
    subsets = [("stable (lower distance quartile)", data[data.cosine_distance <= qd[0]]),
               ("variable (upper distance quartile)", data[data.cosine_distance >= qd[1]]),
               ("large displacement / stable", data[(data.normalized_displacement >= qx[1]) & (data.cosine_distance <= qd[0])]),
               ("small displacement / variable", data[(data.normalized_displacement <= qx[0]) & (data.cosine_distance >= qd[1])])]
    selected, missing = [], []
    used = set()
    for label, group in subsets:
        group = group[~group.object_pair_id.isin(used)]
        if not len(group):
            missing.append(label)
            continue
        med = group[["cosine_distance", "normalized_displacement"]].median()
        scale = group[["cosine_distance", "normalized_displacement"]].std().replace(0, 1).fillna(1)
        scores = ((group[["cosine_distance", "normalized_displacement"]]-med)/scale).pow(2).sum(axis=1)
        row = group.loc[scores.idxmin()].to_dict()
        row["example_condition"] = label
        selected.append(row)
        used.add(row["object_pair_id"])
        if len(selected) >= limit:
            return selected, missing
    remaining = data[~data.object_pair_id.isin(used)].sample(frac=1, random_state=42)
    for row in remaining.head(max(0, limit-len(selected))).to_dict("records"):
        row["example_condition"] = "seeded reference sample"
        selected.append(row)
    return selected, missing


def example_axes(fig, axes, row, data):
    for t, ax in enumerate(axes):
        with Image.open(row[f"image_path_t{t}"]) as im:
            original_size = im.size
            im.thumbnail((1000, 600))
            ax.imshow(np.asarray(im), extent=[0, original_size[0], original_size[1], 0])
        x0, y0, x1, y1 = json.loads(row[f"bbox_t{t}"])
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#ffdf38", linewidth=2))
        ax.set_title(f"t{t}, camera timestamp {row[f'timestamp_t{t}']} µs", fontsize=8)
        ax.axis("off")
    related = data[(data.object_pair_id == row["object_pair_id"]) & (data.aggregation == "area") & (data.comparison_group == "B1_same_instance")]
    sims = ", ".join(f"{r.layer_id}: {r.cosine_similarity:.3f}" for r in related.itertuples() if r.numeric_status == "valid")
    return (f"{row['example_condition']} | {row['scene_name']} / {row['camera_channel']} | {row['category_name']} | instance {row['instance_token']}\n"
            f"Δt={row['delta_t_seconds']:.3f} s; displacement={row['normalized_displacement']:.3f} image diagonals; "
            f"ego translation={row['ego_translation_m']:.2f} m; rotation={row['ego_rotation_deg']:.2f}°\nCosine similarity: {sims}")


def run(cfg, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    data = common_layer_cohort(read(out, "feature_similarity"))
    layer = "layer_"+str(json_read(out/"metadata/model_metadata.json")["selected_layer"])
    primary = data[(data.layer_id == layer) & (data.aggregation == "area") & (data.comparison_group == "B1_same_instance")]
    artifacts = []
    def finish(fig, name):
        fig.savefig(out/"figures"/(name+".png"), dpi=180, bbox_inches="tight")
        fig.savefig(out/"figure_pdf"/(name+".pdf"), bbox_inches="tight")
        artifacts.extend(["figures/"+name+".png", "figure_pdf/"+name+".pdf"])
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    groups = ["B1_same_instance", "B2_different_instance_same_category", "B3_different_category"]
    for ax, group, label in zip(axes[:2], groups[1:], ["Same category", "Different category"]):
        controls = data[(data.layer_id == layer) & (data.aggregation == "area") & (data.comparison_group == group)]
        anchors = primary[primary.object_pair_id.isin(controls.object_pair_id)]
        boxplot(ax, [anchors.cosine_similarity.to_numpy(), controls.cosine_similarity.to_numpy()],
                ["Same ID", "Different ID"], "Cosine similarity")
        ax.set_title(label+" / matched anchors")
    fixed = read(out, "fixed_position_similarity")
    fixed = fixed[(fixed.layer_id == layer) & (fixed.numeric_status == "valid")]
    fixed = fixed.groupby(["pair_id", "camera_channel"]).cosine_similarity.mean()
    boxplot(axes[2], [fixed.to_numpy()], ["Fixed coordinate"], "Cosine similarity")
    axes[2].set_title("Frame-pair sampled-patch mean")
    fig.suptitle(f"Identity and coordinate controls — {layer}, area pooling")
    fig.tight_layout()
    finish(fig, NAMES[0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    gaps = sorted(primary.time_gap_group.unique(), key=lambda x: float(x) if x != "other" else np.inf)
    boxplot(ax, [primary[primary.time_gap_group == g].cosine_distance.to_numpy() for g in gaps], [g+" s" for g in gaps], "Cosine distance (1 − similarity)")
    ax.set_title(f"Same instance across actual camera gaps — {layer}")
    ax.set_xlabel("Gap group (actual camera timestamps)")
    finish(fig, NAMES[1])

    stats = read(out, "layer_stability")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for color, group in zip(COLORS, groups):
        x = stats[(stats.aggregation == "area") & (stats.comparison_group == group) & (stats.metric == "cosine_similarity") & stats.layer_id.str.startswith("layer_")].copy()
        x["index"] = x.layer_id.str.replace("layer_", "", regex=False).astype(int)
        x = x.sort_values("index")
        if len(x):
            ax.plot(x["index"], x.scene_equal_weight_mean, "o-", label=f"{group} (n={x.n.min()})", color=color)
            ax.fill_between(x["index"].to_numpy(), x.ci95_low.to_numpy(float), x.ci95_high.to_numpy(float), alpha=.15, color=color)
    ax.set(xlabel="Hidden-state index (0 = embedding after pre-LN)", ylabel="Scene-equal mean cosine similarity", title="Identical valid object/control cohort across CLIP layers")
    if ax.lines:
        ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    finish(fig, NAMES[2])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, motion, color in zip(axes, ["approximately_static", "moving"], COLORS):
        x = primary[primary.motion_group == motion]
        ax.scatter(x.normalized_displacement, x.cosine_distance, s=10, alpha=.22, color=color)
        bins = [0, .03, .1, .25, np.inf]
        xb, yb = [], []
        for left, right in zip(bins, bins[1:]):
            b = x[(x.normalized_displacement >= left) & (x.normalized_displacement < right)]
            if len(b):
                xb.append(b.normalized_displacement.median())
                yb.append(b.cosine_distance.median())
        ax.plot(xb, yb, "o-", color="black", label="Bin medians")
        ax.set(title=f"{motion}, n={len(x)}", xlabel="Center displacement / image diagonal")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Cosine distance")
    fig.suptitle(f"Feature variation vs. image displacement — {layer}")
    fig.tight_layout()
    finish(fig, NAMES[3])

    cats = primary.category_name.value_counts().head(4).index.tolist()
    fig, axes = plt.subplots(1, max(1, len(cats)), figsize=(max(6, 3*len(cats)), 4.5), squeeze=False, sharey=True)
    for ax, category in zip(axes[0], cats):
        x = primary[primary.category_name == category]
        boxplot(ax, [x[x.motion_group == g].cosine_distance.to_numpy() for g in ["approximately_static", "moving", "unknown"]], ["Static", "Moving", "Unknown"], "Cosine distance")
        ax.set_title(category.replace(".", ".\n"), fontsize=9)
    fig.suptitle(f"Within-category comparison; static ≤ {cfg['analysis']['static_speed_threshold_mps']} m/s; gaps {primary.delta_t_seconds.min():.2f}–{primary.delta_t_seconds.max():.2f} s")
    fig.tight_layout()
    finish(fig, NAMES[4])

    samples, missing = choose_examples(primary, max(4, cfg["output"]["visualization_samples"]))
    core = samples[:4]
    fig, axes = plt.subplots(max(1, len(core)), 2, figsize=(14, max(4, 4.2*len(core))), squeeze=False)
    if not core:
        axes[0, 0].text(.5, .5, "No valid real example", ha="center")
    for i, row in enumerate(core):
        label = example_axes(fig, axes[i], row, data)
        axes[i, 0].text(0, -.12, label, transform=axes[i, 0].transAxes, fontsize=8, va="top")
    fig.subplots_adjust(hspace=.55)
    finish(fig, NAMES[5])
    if cfg["output"]["save_visualizations"]:
        for i, row in enumerate(samples[:cfg["output"]["visualization_samples"]]):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            label = example_axes(fig, axes, row, data)
            fig.text(.02, .02, label, fontsize=9)
            fig.subplots_adjust(bottom=.23)
            p = out/"visualizations"/f"example_{i:02d}.png"
            fig.savefig(p, dpi=160, bbox_inches="tight")
            artifacts.append(str(p.relative_to(out)))
            plt.close(fig)
    write_json(out/"metadata/figure_manifest.json", {"artifacts": artifacts, "layer": layer,
        "core_examples": core, "unavailable_example_conditions": missing,
        "selection": "Quartile conditions, nearest within-condition median, then seeded random reference examples. No hand-picked extremes.",
        "caption_notes": "Boxes are official 3D-box projected rectangles; not masks. Box plots omit plotted outliers but statistics retain every valid value. Layer bands bootstrap scene means."})


if __name__ == "__main__":
    standalone("plot")
