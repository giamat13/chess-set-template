"""Render the store images into store/images/ from the STL files of one version (default: plain).

  banner.png      1920x640  wide header: all six pieces in a row with the title
  cover.png       1600x1200 4:3 cover (MakerWorld / Thingiverse thumbnail ratio)
  piece_<n>.png   1200x1200 one image per piece, with its name

Usage: python src/store_images.py [model]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "plain"
OUT = os.path.join(ROOT, "store", "images")
BG = "#0e1117"
PIECES = [  # file stem, short name, what it is
    ("pawn_neuron", "Pawn", "Neuron"),
    ("rook_gpu", "Rook", "GPU"),
    ("knight_mcp", "Knight", "MCP connector"),
    ("bishop_terminal", "Bishop", "Terminal agent"),
    ("queen_context", "Queen", "Context window"),
    ("king", "King", "The model"),
]
LIGHT = np.array([0.4, -0.7, 0.6]) / np.linalg.norm([0.4, -0.7, 0.6])


def load_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = np.frombuffer(f.read(4), np.uint32)[0]
        rec = np.frombuffer(f.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    return rec["v"], rec["n"]


MESH = {s: load_stl(os.path.join(ROOT, "stl", MODEL, f"{s}_{MODEL}.stl")) for s, _, _ in PIECES}
ZMAX = max(m[0][..., 2].max() for m in MESH.values())


def draw(ax, stem, same_scale=True, zoom=1.0):
    tri, nrm = MESH[stem]
    shade = np.clip(nrm @ LIGHT, 0, 1) * 0.8 + 0.2
    col = np.stack([0.35 * shade + 0.05, 0.80 * shade + 0.05, 1.0 * shade, np.ones_like(shade)], 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(col, 0, 1), edgecolors="none"))
    pts = tri.reshape(-1, 3)
    mn, mx = pts.min(0), pts.max(0)
    c = (mn + mx) / 2
    h = ZMAX if same_scale else mx[2] - mn[2]  # same scale shows the real size order
    r = max(mx[0] - mn[0], mx[1] - mn[1], h / 1.6) * 1.05
    ax.set_xlim(c[0] - r / 2, c[0] + r / 2)
    ax.set_ylim(c[1] - r / 2, c[1] + r / 2)
    ax.set_zlim(0, r * 1.6)
    ax.set_box_aspect((1, 1, 1.6), zoom=zoom)
    ax.view_init(elev=16, azim=-90 if stem.startswith(("knight", "king", "queen", "bishop")) else -58)
    ax.set_axis_off()
    ax.set_facecolor(BG)


def row(fig, bottom, top, labels=True):
    w = 1 / len(PIECES)
    for i, (stem, name, what) in enumerate(PIECES):
        ax = fig.add_axes([i * w, bottom, w, top - bottom], projection="3d")
        draw(ax, stem, zoom=1.3)
        if labels:
            fig.text((i + 0.5) * w, bottom - 0.01, f"{name}\n{what}", color="#9fd3ff", ha="center", va="top",
                     fontsize=13, linespacing=1.3)


def banner():
    fig = plt.figure(figsize=(19.2, 6.4), dpi=100, facecolor=BG)
    fig.text(0.5, 0.93, "AI CHESS SET", color="white", ha="center", va="top", fontsize=40, weight="bold")
    fig.text(0.5, 0.80, "every piece is part of the AI stack  ·  one-piece prints  ·  open source",
             color="#9fd3ff", ha="center", va="top", fontsize=17)
    row(fig, 0.15, 0.78)
    fig.savefig(os.path.join(OUT, "banner.png"), facecolor=BG)
    plt.close(fig)


def cover():
    fig = plt.figure(figsize=(16, 12), dpi=100, facecolor=BG)
    fig.text(0.5, 0.95, "AI CHESS SET", color="white", ha="center", va="top", fontsize=54, weight="bold")
    fig.text(0.5, 0.87, "Neuron · GPU · MCP · Terminal · Context window · Model", color="#9fd3ff",
             ha="center", va="top", fontsize=22)
    for i, (stem, name, what) in enumerate(PIECES):
        x, y = i % 3, 1 - i // 3
        ax = fig.add_axes([x / 3, 0.07 + y * 0.40, 1 / 3, 0.36], projection="3d")
        draw(ax, stem, same_scale=False)
        fig.text((x + 0.5) / 3, 0.04 + y * 0.40, f"{name} = {what}", color="white", ha="center", fontsize=17)
    fig.savefig(os.path.join(OUT, "cover.png"), facecolor=BG)
    plt.close(fig)


def singles():
    for i, (stem, name, what) in enumerate(PIECES, 1):
        fig = plt.figure(figsize=(12, 12), dpi=100, facecolor=BG)
        ax = fig.add_axes([0, 0.08, 1, 0.92], projection="3d")
        draw(ax, stem, same_scale=False)
        fig.text(0.5, 0.05, f"{name} = {what}", color="white", ha="center", fontsize=30, weight="bold")
        fig.savefig(os.path.join(OUT, f"piece_{i}_{name.lower()}.png"), facecolor=BG)
        plt.close(fig)
        print("@@sub done", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    banner()
    print("banner saved", flush=True)
    cover()
    print("cover saved", flush=True)
    singles()
    print("piece images saved", flush=True)
