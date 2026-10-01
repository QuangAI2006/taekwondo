# scripts/plot_confusion_heatmap.py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def parse_cm(cm_str: str) -> np.ndarray:
    """
    Parse confusion matrix from string like:
      "26,7,6;5,15,5;8,5,27"
    """
    rows = cm_str.strip().split(";")
    mat = []
    for r in rows:
        if not r.strip():
            continue
        mat.append([float(x) for x in r.split(",")])
    cm = np.array(mat, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"CM must be square. Got shape={cm.shape}")
    return cm


def normalize_cm(cm: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return cm
    eps = 1e-12
    if mode == "true":      # row-normalized
        denom = cm.sum(axis=1, keepdims=True) + eps
        return cm / denom
    if mode == "pred":      # col-normalized
        denom = cm.sum(axis=0, keepdims=True) + eps
        return cm / denom
    if mode == "all":
        denom = cm.sum() + eps
        return cm / denom
    raise ValueError(f"Unknown normalize mode: {mode}")


def plot_heatmap(cm: np.ndarray,
                 labels: list[str],
                 normalize: str,
                 annotate: bool,
                 out_path: str,
                 dpi: int = 300):
    cm_norm = normalize_cm(cm, normalize)

    # Figure size: scale mildly with class count
    n = cm.shape[0]
    figsize = (3.2 + 0.35 * n, 2.8 + 0.35 * n)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Reds", vmin=0.0, vmax=1.0 if normalize != "none" else None)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if normalize == "none":
        cbar.set_label("Count")
    else:
        cbar.set_label("Proportion")

    # Ticks/labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Gridlines to make it publication-friendly
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotation
    if annotate:
        # choose format
        if normalize == "none":
            fmt = "{:d}"
            disp = cm.astype(int)
        else:
            fmt = "{:.2f}"
            disp = cm_norm

        # text color threshold
        thresh = (disp.max() + disp.min()) / 2.0 if disp.size else 0.5
        for i in range(n):
            for j in range(n):
                val = disp[i, j]
                text_color = "white" if val > thresh else "black"
                s = fmt.format(int(val)) if normalize == "none" else fmt.format(float(val))
                ax.text(j, i, s, ha="center", va="center", color=text_color, fontsize=10)

    # IMPORTANT: no title (per your request)
    # ax.set_title(...)  # intentionally omitted

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm", required=True, help='e.g. "26,7,6;5,15,5;8,5,27"')
    ap.add_argument("--labels", nargs="+", required=True, help="class labels, e.g. front roundhouse axe")
    ap.add_argument("--normalize", choices=["none", "true", "pred", "all"], default="true",
                    help="normalization: none/counts, true/row, pred/col, all/global")
    ap.add_argument("--annotate", action="store_true", help="draw numeric values in cells")
    ap.add_argument("--out", required=True, help="output image path, e.g. figures/cm_uni.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    cm = parse_cm(args.cm)
    if len(args.labels) != cm.shape[0]:
        raise ValueError(f"labels length {len(args.labels)} != cm size {cm.shape[0]}")
    plot_heatmap(cm, args.labels, args.normalize, args.annotate, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
