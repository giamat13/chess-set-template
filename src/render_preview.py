"""Render images/preview_<model>.png of the six pieces from stl/<model>/.  Usage: python src/render_preview.py [version]"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "plain"
SFX = "_" + MODEL
NAMES = [
    ("pawn_neuron" + SFX, "Pawn = NEURON"),
    ("rook_gpu" + SFX, "Rook = GPU"),
    ("knight_mcp" + SFX, "Knight = MCP CONNECTOR"),
    ("bishop_terminal" + SFX, "Bishop = TERMINAL AGENT"),
    ("queen_context" + SFX, "Queen = CONTEXT WINDOW"),
    ("king" + SFX, "King = THE MODEL"),
]


def load_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = np.frombuffer(f.read(4), np.uint32)[0]
        rec = np.frombuffer(f.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    return rec["v"], rec["n"]


fig = plt.figure(figsize=(15, 13), facecolor="#0e1117")
light = np.array([0.4, -0.7, 0.6])
light /= np.linalg.norm(light)
for i, (name, title) in enumerate(NAMES):
    tri, nrm = load_stl(os.path.join(ROOT, "stl", MODEL, name + ".stl"))
    ax = fig.add_subplot(2, 3, i + 1, projection="3d", facecolor="#0e1117")
    shade = np.clip(nrm @ light, 0, 1) * 0.8 + 0.2
    col = np.stack([0.35 * shade + 0.05, 0.80 * shade + 0.05, 1.0 * shade, np.ones_like(shade)], 1)
    pc = Poly3DCollection(tri, facecolors=np.clip(col, 0, 1), edgecolors="none")
    ax.add_collection3d(pc)
    pts = tri.reshape(-1, 3)
    mn, mx = pts.min(0), pts.max(0)
    c = (mn + mx) / 2
    hgt = mx[2] - mn[2]
    r = max(mx[0] - mn[0], mx[1] - mn[1], hgt / 1.6) * 1.05
    ax.set_xlim(c[0] - r / 2, c[0] + r / 2)
    ax.set_ylim(c[1] - r / 2, c[1] + r / 2)
    ax.set_zlim(0, r * 1.6)
    ax.set_box_aspect((1, 1, 1.6))
    ax.view_init(elev=16, azim=-90 if name.startswith(("knight", "king", "queen", "bishop")) else -58)
    ax.set_axis_off()
    ax.set_title(title, color="white", fontsize=13, pad=0)
plt.subplots_adjust(left=0, right=1, top=0.93, bottom=0, wspace=0)
plt.savefig(os.path.join(ROOT, "images", f"preview_{MODEL}.png"), dpi=110, facecolor=fig.get_facecolor())
print("preview saved")
