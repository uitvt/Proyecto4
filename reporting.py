import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_rows_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(
            {
                key: f"{value:.3f}".replace(".", ",") if isinstance(value, float) else value
                for key, value in row.items()
            }
            for row in rows
        )


def plot_cdf(errors, title, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, err in errors.items():
        x = np.sort(err)
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, label=name)
    ax.set_xlabel("Error de posición (m)")
    ax.set_ylabel("Probabilidad acumulada")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_rmse_vs_sigma(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(results.keys())
    variants = list(next(iter(results.values())).keys())
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for v in variants:
        values = [float(np.sqrt(np.mean(results[lab][v] ** 2))) for lab in labels]
        ax.plot(labels, values, marker="o", label=v)
    ax.set_xlabel("Escenario")
    ax.set_ylabel("RMSE (m)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_layout(anchors_xy, node_xy, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(anchors_xy[:, 0], anchors_xy[:, 1], marker="^", s=70, label="Anclas")
    ax.scatter(node_xy[:, 0], node_xy[:, 1], s=6, alpha=0.3, label="Posiciones sorteadas")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
