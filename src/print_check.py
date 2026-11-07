"""Printability and stability check for a Bambu Lab P1S (256 x 256 x 256 mm, 0.4 mm nozzle).

Usage: python src/print_check.py [plain|star|example ...]   (default: all versions found in stl/)

For every STL in stl/<version>/ it reports:
  - size: fits the P1S build volume
  - bed:  first-layer contact area (bed adhesion)
  - tip:  centre of mass vs footprint, i.e. how far the piece can lean before it falls over
  - islands: points that start in mid-air (lowest point of a feature with nothing under it),
             these need supports
  - overhang: area facing down steeper than 45 degrees (prints, but may sag) - info only
  - walls: share of surface where the solid is thinner than 0.8 mm (two 0.4 mm lines)
  - gaps:  share of surface where a slot or gap is narrower than 0.4 mm (it will fuse shut)
  - necks: thin horizontal sections with more material above them, where the piece can snap
"""
import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BED = (256.0, 256.0, 256.0)
NOZZLE = 0.4
MIN_WALL = 2 * NOZZLE
MIN_GAP = NOZZLE
MIN_TIP_DEG = 12.0  # a standard chess piece is roughly 15-25 degrees
SNAP_MM2 = 4.0      # a neck under ~4 mm2 (about 2.2 mm across) that carries material above can snap


def load_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = np.frombuffer(f.read(4), np.uint32)[0]
        rec = np.frombuffer(f.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    return rec["v"].astype(np.float64)


def convex_hull(p):
    p = sorted(set(map(tuple, np.round(p, 4))))
    if len(p) < 3:
        return np.array(p)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and cross(h[-2], h[-1], q) <= 0:
                h.pop()
            h.append(q)
        return h

    lo, up = half(p), half(p[::-1])
    return np.array(lo[:-1] + up[:-1])


def inside_margin(pt, hull):
    """Signed distance from pt to the hull boundary (positive = inside)."""
    d = np.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        e = b - a
        n = np.array([e[1], -e[0]]) / np.linalg.norm(e)  # outward normal for a CCW hull
        d = min(d, -np.dot(pt - a, n))
    return d


def ray_hits(orig, dirs, tri, chunk=256):
    """Distance to the nearest triangle along each ray (Moller-Trumbore), inf if none."""
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    out = np.full(len(orig), np.inf)
    for s in range(0, len(orig), chunk):
        o, d = orig[s:s + chunk, None, :], dirs[s:s + chunk, None, :]
        p = np.cross(d, e2[None])
        det = (e1[None] * p).sum(-1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
        tv = o - v0[None]
        u = (tv * p).sum(-1) * inv
        q = np.cross(tv, e1[None])
        v = (d * q).sum(-1) * inv
        t = (e2[None] * q).sum(-1) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-3)
        out[s:s + chunk] = np.where(hit, t, np.inf).min(1)
    return out


def _manifold(tri):
    from manifold3d import Manifold, Mesh
    v, inv = np.unique(np.round(tri.reshape(-1, 3), 4), axis=0, return_inverse=True)
    return Manifold(Mesh(vert_properties=v.astype(np.float32), tri_verts=inv.reshape(-1, 3).astype(np.uint32))), v


def islands(tri, z_bed, layer=0.2, min_area=0.3):
    """Real mid-air starts, found the way a slicer sees them: a part of one layer with nothing
    under it in the layer below. A short sideways nub on a wall is not an island (it touches
    the layer below through the wall); a dangling tip or a floating bump is."""
    from manifold3d import JoinType
    m, v = _manifold(tri)
    z1 = v[:, 2].max()
    out, prev = [], m.slice(z_bed + layer / 2)
    for z in np.arange(z_bed + 1.5 * layer, z1, layer):
        cur = m.slice(z)
        grown = prev.offset(0.1, JoinType.Round)  # one layer may step out a little (normal overhang)
        for c in cur.decompose():
            if c.area() >= min_area and (c ^ grown).area() < 1e-3:
                b = c.bounds()
                out.append(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, z]))
        prev = cur
    return out


def necks(tri, step=0.4, small=6.0, grow=1.5):
    """Weak links: a horizontal section smaller than `small` mm2 with clearly more material
    right above it (the piece necks down and then widens). Layers bond weakest across z,
    so these are where a piece snaps. Returns [(z, area, x, y)], weakest first."""
    from manifold3d import CrossSection
    m, v = _manifold(tri)
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    out = []
    for z in np.arange(z0 + 1.0, z1 - 2.0, step):
        sec = m.slice(z)
        for c in sec.decompose():
            a = c.area()
            if a >= small or a < 0.25:  # under one extrusion line (0.4 x 0.6 mm): a hair the slicer drops
                continue
            b = c.bounds()
            reg = CrossSection.square((b[2] - b[0] + 6, b[3] - b[1] + 6), center=True).translate(
                ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            above = (m.slice(z + 2.0) ^ reg).area()
            below = (m.slice(z - 2.0) ^ reg).area()
            # a neck has material on both sides; the bottom of a bump only has it above, and the top
            # of a dome has nothing right on top of it (what is "above" is some other feature)
            from manifold3d import JoinType
            goes_on = (m.slice(z + step) ^ c.offset(step, JoinType.Round)).area() > 0.5 * a
            if goes_on and above > grow * a and below > grow * a:
                out.append((z, a, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    # keep the weakest section per location
    out.sort(key=lambda t: t[1])
    keep = []
    for n in out:
        if all(abs(n[2] - k[2]) + abs(n[3] - k[3]) > 2 or abs(n[0] - k[0]) > 3 for k in keep):
            keep.append(n)
    return keep


def check(path, samples=3000, rng=np.random.default_rng(0)):
    tri = load_stl(path)
    pts = tri.reshape(-1, 3)
    mn, mx = pts.min(0), pts.max(0)
    size = mx - mn
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cr, axis=1) / 2
    nrm = cr / np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-12)
    cen = tri.mean(1)
    z0 = mn[2]
    res = {"name": os.path.basename(path)[:-4], "size": size}
    res["fits"] = bool(np.all(size <= BED))
    bed = (nrm[:, 2] < -0.99) & (np.abs(cen[:, 2] - z0) < 0.05)
    res["bed_area"] = area[bed].sum()
    # centre of mass from signed tetrahedra
    vol6 = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    com = ((tri.sum(1) / 4) * vol6[:, None]).sum(0) / vol6.sum()
    res["volume"] = vol6.sum() / 6
    foot = convex_hull(pts[np.abs(pts[:, 2] - z0) < 0.05][:, :2])
    margin = inside_margin(com[:2], foot) if len(foot) >= 3 else -1
    res["tip_deg"] = np.degrees(np.arctan2(margin, com[2] - z0))
    res["com_h"] = com[2] - z0
    over = (nrm[:, 2] < -np.cos(np.radians(45))) & (cen[:, 2] > z0 + 0.2)
    res["overhang_mm2"] = area[over].sum()
    res["islands"] = [] if res["name"].startswith("plate") else islands(tri, z0)
    # wall thickness and gap width by ray casting from sampled surface points (area weighted)
    k = rng.choice(len(tri), size=min(samples, len(tri)), p=area / area.sum())
    o, d = cen[k], nrm[k]
    thick = ray_hits(o - d * 1e-3, -d, tri)
    gap = ray_hits(o + d * 1e-3, d, tri)
    res["thin_pct"] = 100 * np.mean(thick < MIN_WALL)
    res["min_wall"] = np.min(thick)
    res["gap_pct"] = 100 * np.mean(gap < MIN_GAP)
    res["necks"] = [] if res["name"].startswith("plate") else necks(tri)
    return res


def verdict(r):
    bad = []
    if not r["fits"]:
        bad.append("too big for the P1S")
    if r["bed_area"] < 100:
        bad.append("small bed contact")
    if not r["name"].startswith("plate") and r["tip_deg"] < MIN_TIP_DEG:
        bad.append(f"tips over easily ({r['tip_deg']:.0f} deg)")
    if r["islands"]:
        bad.append(f"{len(r['islands'])} mid-air start(s), needs supports")
    weak = [n for n in r["necks"] if n[1] < SNAP_MM2]
    if weak:
        bad.append(f"{len(weak)} thin neck(s) that may snap (weakest {weak[0][1]:.1f} mm2 at z{weak[0][0]:.0f})")
    if r["thin_pct"] > 6.0:
        bad.append(f"{r['thin_pct']:.1f}% of surface thinner than {MIN_WALL} mm")
    return bad


def notes(r):
    """Things worth knowing that do not stop a print."""
    out = []
    if 2.0 < r["thin_pct"] <= 6.0:
        out.append(f"fine detail: {r['thin_pct']:.1f}% of the surface is thinner than {MIN_WALL} mm "
                   "(thin logo lines on a solid face; they print as single lines and cannot break off)")
    return out


def main():
    models = sys.argv[1:] or sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "stl", "*")))
    total_bad = 0
    for m in models:
        print(f"== {m} ==", flush=True)
        for path in sorted(glob.glob(os.path.join(ROOT, "stl", m, "*.stl"))):
            r = check(path, samples=800 if os.path.basename(path).startswith("plate") else 3000)
            bad = verdict(r)
            total_bad += bool(bad)
            s = r["size"]
            isl = "; ".join(f"z{p[2]:.1f} at ({p[0]:.1f},{p[1]:.1f})" for p in r["islands"][:6])
            print(f"{r['name']}: {s[0]:.0f}x{s[1]:.0f}x{s[2]:.0f} mm, bed {r['bed_area']:.0f} mm2, "
                  f"tip {r['tip_deg']:.0f} deg, overhang>45 {r['overhang_mm2']:.0f} mm2, "
                  f"thin<{MIN_WALL} {r['thin_pct']:.1f}% (min {r['min_wall']:.2f}), gaps<{MIN_GAP} {r['gap_pct']:.1f}%"
                  f"  -> {'OK' if not bad else 'FIX: ' + ', '.join(bad)}", flush=True)
            for n in notes(r):
                print(f"    note: {n}", flush=True)
            if isl:
                print(f"    islands: {isl}", flush=True)
            if r["necks"]:
                print("    necks: " + "; ".join(f"{a:.1f} mm2 z{z:.1f} ({x:.1f},{y:.1f})" for z, a, x, y in r["necks"][:8]), flush=True)
            print("@@sub done", flush=True)
    print("ALL OK" if total_bad == 0 else f"{total_bad} file(s) need attention", flush=True)


if __name__ == "__main__":
    main()
