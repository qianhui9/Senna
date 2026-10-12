"""Shared publication style, real-image overlays and deterministic examples."""
import json
import numpy as np


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
    return plt


def unavailable(ax, reason):
    ax.text(.5, .5, "Unavailable\n"+reason, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def save_figure(fig, out, name):
    fig.savefig(out / "figures" / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure_pdf" / f"{name}.pdf", bbox_inches="tight")
    __import__("matplotlib.pyplot", fromlist=["close"]).close(fig)


def qualitative_selection(data, count, seed):
    """Round-robin strata, quantile targets, no hand-picked extreme-only examples."""
    if data.empty or count == 0:
        return data.iloc[:0]
    groups = [g.sort_values("normalized_displacement") for _, g in data.groupby(["gap_steps", "camera_channel", "motion_group", "scene_name"])]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    targets = np.linspace(.05, .95, min(count, len(data)))
    selected, used = [], set()
    for i, q in enumerate(targets):
        for shift in range(len(groups)):
            group = groups[order[(i+shift) % len(groups)]]
            positions = sorted(range(len(group)), key=lambda k: abs(k/(max(1, len(group)-1))-q))
            for position in positions:
                index = group.index[position]
                row = group.loc[index]
                key = (row.pair_id, row.camera_channel) # avoid duplicate image pairs in gallery
                if key not in used:
                    selected.append(index)
                    used.add(key)
                    break
            else:
                continue
            break
    return data.loc[selected]


def overlay(ax, path, box, instance, shape):
    from PIL import Image
    from matplotlib.patches import Rectangle
    with Image.open(path) as image:
        width, height = image.size
        ax.imshow(image)
    rows, cols = shape
    for x in np.linspace(0, width, cols+1):
        ax.axvline(x, color="white", alpha=.35, linewidth=.35)
    for y in np.linspace(0, height, rows+1):
        ax.axhline(y, color="white", alpha=.35, linewidth=.35)
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#e69f00", linewidth=1.6))
    ax.text(x0, max(0, y0-8), instance[:10], color="#e69f00", fontsize=7,
            bbox={"facecolor": "black", "alpha": .6, "pad": 1, "edgecolor": "none"})
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()


def render_projection_regressions(out):
    """Real official 3D wireframe and measured 2D rectangle, for human inspection."""
    from PIL import Image
    from matplotlib.patches import Rectangle
    plt = setup()
    records = json.loads((out / "metadata" / "projection_regression.json").read_text())
    # At most twelve views spanning the selected scenes and camera channels.
    chosen = records[::max(1, len(records)//12)][:12]
    for i, record in enumerate(chosen):
        fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
        with Image.open(record["image_path"]) as image:
            ax.imshow(image)
        p = np.array(record["official_pixels"])
        for indices in [[0,1,2,3,0], [4,5,6,7,4], [0,4], [1,5], [2,6], [3,7]]:
            ax.plot(p[0, indices], p[1, indices], color="#56b4e9", linewidth=1)
        x0, y0, x1, y1 = record["box"]
        ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor="#e69f00", linewidth=1.5))
        ax.set_axis_off()
        fig.savefig(out / "visualizations" / f"projection_regression_{i:02d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
