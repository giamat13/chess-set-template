"""
Chess set template - printable STL generator. Every version is the same set with a different emblem.

  King   = THE MODEL : an artificial neural network, its output node is the model emblem
  Queen  = CONTEXT   : a window frame sliding over a long ribbon of tokens, cursor on top
  Bishop = TERMINAL  : the agent - the mitre slit is the terminal prompt  >_
  Knight = MCP       : the connector - plug-prong ears, cable mane, socket on the neck
  Rook   = GPU       : graphics card tower with fans, vents and heat-sink battlements
  Pawn   = NEURON    : a neuron with dendrites, the smallest unit of the network

Every piece stands on a microchip base. All sizes in mm. Run:  python build_set.py
"""
import os
import sys
from functools import reduce

import numpy as np
from manifold3d import CrossSection, FillRule, JoinType, Manifold, OpType

from svg_emblem import normalise, svg_shapes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (this file is in src/)
SEG = 64  # circle resolution


# ---------------------------------------------------------------- helpers
def union(parts):
    parts = [p for p in parts if p is not None]
    if len(parts) == 1:
        return parts[0]
    try:
        return Manifold.batch_boolean(parts, OpType.Add)
    except Exception:
        return reduce(lambda a, b: a + b, parts)


def box(x, y, z, cx=0.0, cy=0.0, z0=0.0):
    """Axis aligned box centred in x/y, bottom at z0."""
    return Manifold.cube((x, y, z), center=True).translate((cx, cy, z0 + z / 2))


def cyl(h, r1, r2=None, z0=0.0, seg=SEG):
    return Manifold.cylinder(h, r1, r1 if r2 is None else r2, seg).translate((0, 0, z0))


def sphere(r, x=0.0, y=0.0, z=0.0, seg=SEG):
    return Manifold.sphere(r, seg).translate((x, y, z))


def rrect(w, d, r):
    r = min(r, w / 2 - 0.01, d / 2 - 0.01)
    return CrossSection.square((w - 2 * r, d - 2 * r), center=True).offset(
        r, JoinType.Round, 2.0, 32
    )


def chamfer_slab(w, d, h, r, ch_top=0.0, ch_bot=0.0, z0=0.0):
    """Rounded-square slab with optional 45 degree chamfers top/bottom."""
    parts = []
    z = z0
    if ch_bot > 0:
        s = (w - 2 * ch_bot) / w
        sec = rrect(w - 2 * ch_bot, d - 2 * ch_bot, max(r - ch_bot, 0.3))
        parts.append(
            Manifold.extrude(sec, ch_bot, 0, 0.0, (w / (w - 2 * ch_bot), d / (d - 2 * ch_bot))).translate((0, 0, z))
        )
        z += ch_bot
    mid = h - ch_top - ch_bot
    parts.append(Manifold.extrude(rrect(w, d, r), mid).translate((0, 0, z)))
    z += mid
    if ch_top > 0:
        parts.append(
            Manifold.extrude(rrect(w, d, r), ch_top, 0, 0.0, ((w - 2 * ch_top) / w, (d - 2 * ch_top) / d)).translate((0, 0, z))
        )
    return union(parts)


def revolve(profile):
    """profile: list of (radius, z) points, closed polygon, radius >= 0."""
    return Manifold.revolve(CrossSection([profile]), SEG)


def ring_groove(r_out, depth, width, z):
    """Torus-like cutter (rectangular section) for a groove on a cylinder of radius r_out."""
    return cyl(width, r_out + 2, z0=z - width / 2) - cyl(width + 0.2, r_out - depth, z0=z - width / 2 - 0.1)


def ycyl(r, y_a, y_b, z, x=0.0, seg=SEG):
    """Cylinder along the y axis between y_a and y_b, centred at (x, z)."""
    lo, hi = min(y_a, y_b), max(y_a, y_b)
    return Manifold.cylinder(hi - lo, r, r, seg).rotate((-90, 0, 0)).translate((x, lo, z))


def fan_rotor(r, y_a, y_b, z, blades=7):
    """Cooling fan: hub + curved swept blades, axis along y."""
    lo, hi = min(y_a, y_b), max(y_a, y_b)
    parts = [ycyl(3.5, lo, hi, z, seg=40)]  # hub, big enough to carry the emblem
    for k in range(blades):
        a0 = 2 * np.pi * k / blades
        pts = []
        for i in range(9):
            t = i / 8
            rad = 1.8 + (r - 1.8) * t
            ang = a0 + 0.9 * t
            pts.append((rad * np.cos(ang), rad * np.sin(ang), 0.5 + 0.5 * t))
        sec = None
        for (x1, y1, w1), (x2, y2, w2) in zip(pts, pts[1:]):
            seg_cs = CrossSection.batch_hull(
                [CrossSection.circle(w1, 16).translate((x1, y1)), CrossSection.circle(w2, 16).translate((x2, y2))]
            )
            sec = seg_cs if sec is None else sec + seg_cs
        bl = Manifold.extrude(sec, hi - lo)
        parts.append(bl.rotate((-90, 0, 0)).translate((0, lo, z)))
    # outer frame ring holds the blade tips
    ring = ycyl(r + 0.5, lo, hi, z) - ycyl(r - 0.3, lo - 1, hi + 1, z)
    parts.append(ring)
    return union(parts)


# ---------------------------------------------------------------- pixel font
FONT = {
    "A": ["010", "101", "111", "101", "101"],
    "B": ["110", "101", "110", "101", "110"],
    "C": ["011", "100", "100", "100", "011"],
    "D": ["110", "101", "101", "101", "110"],
    "E": ["111", "100", "110", "100", "111"],
    "G": ["011", "100", "101", "101", "011"],
    "I": ["111", "010", "010", "010", "111"],
    "K": ["101", "101", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "111"],
    "M": ["10001", "11011", "10101", "10001", "10001"],
    "N": ["1001", "1101", "1011", "1001", "1001"],
    "O": ["010", "101", "101", "101", "010"],
    "P": ["110", "101", "110", "100", "100"],
    "R": ["110", "101", "110", "101", "101"],
    "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"],
    "X": ["101", "101", "010", "101", "101"],
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
}


def text_width_px(text):
    return sum(len(FONT[c][0]) for c in text) + (len(text) - 1)


def pixel_text_cutter(text, px, depth):
    """Pixel text lying in the x-z plane (reading along +x, top = +z),
    occupying y in [0, depth]. Centred on x=0, z=0."""
    total = text_width_px(text) * px
    x = -total / 2
    cubes = []
    for ch in text:
        g = FONT[ch]
        for row, line in enumerate(g):
            for col, bit in enumerate(line):
                if bit == "1":
                    cx = x + (col + 0.5) * px
                    cz = (2 - row) * px
                    cubes.append(
                        Manifold.cube((px * 1.02, depth, px * 1.02), center=True).translate((cx, depth / 2, cz))
                    )
        x += (len(g[0]) + 1) * px
    return union(cubes)


# ---------------------------------------------------------------- chip base
BASE_H = 6.5
DIE_H = 1.2


MODEL = "plain"  # the emblem: plain, star, or the name of an SVG in assets/
if __name__ == "__main__" and len(sys.argv) > 1:
    MODEL = sys.argv[1]  # python build_set.py example


_LOGO_CS = {}  # logos are read from assets/<model>.svg (see logo_file)


def emblem_cs(model):
    """2D model emblem, about 8.8 mm radius. star = starburst, plain = network graph, others from assets/<model>.svg."""
    if model == "star":
        lens = [8.6, 7.4, 8.2, 7.0, 8.8, 7.6, 8.0, 7.2, 8.6, 7.0, 8.4, 7.6]
        cs = CrossSection.circle(2.6, 40)
        for k, L in enumerate(lens):
            a = np.radians(90 + 30 * k + (4 if k % 2 else -3))
            cs = cs + CrossSection.batch_hull([CrossSection.circle(1.25, 24),
                                               CrossSection.circle(0.55, 16).translate((L * np.cos(a), L * np.sin(a)))])
        return cs
    if model == "plain":  # no brand logo: a tiny network graph (3 outer nodes + a centre node, all linked)
        # one node points down: standing upright it rests on the post below it and every other
        # node is carried by a rising link, so nothing starts in mid-air when printed
        pts = [(0.0, 0.0)] + [(6.4 * np.cos(np.radians(-90 + 120 * k)), 6.4 * np.sin(np.radians(-90 + 120 * k)))
                              for k in range(3)]
        cs = None
        for i in range(4):
            for j in range(i + 1, 4):
                e = CrossSection.batch_hull([CrossSection.circle(1.2, 16).translate(pts[i]),
                                             CrossSection.circle(1.2, 16).translate(pts[j])])
                cs = e if cs is None else cs + e
        for i, p in enumerate(pts):
            cs = cs + CrossSection.circle(2.4 if i else 2.0, 40).translate(p)
        return cs
    if model not in _LOGO_CS:  # logo read from assets/<model>.svg
        cs = None
        for polys, rule, op in normalise(svg_shapes(logo_file(model))):
            c = CrossSection(polys, FillRule.EvenOdd if rule == "evenodd" else FillRule.NonZero)
            if op == "sub":
                cs = cs - c if cs is not None else cs
            else:
                cs = c if cs is None else cs + c
        _LOGO_CS[model] = connect_parts(cs)
    return _LOGO_CS[model]


def logo_file(model):
    for name in (f"{model}.svg", f"{model}-icon.svg"):
        p = os.path.join(ROOT, "assets", name)
        if os.path.exists(p):
            return p
    raise SystemExit(f"no logo for '{model}': put {model}.svg in assets/")


def all_models():
    """star and plain are drawn in code; every SVG in assets/ adds one more version."""
    import glob
    svgs = sorted(os.path.basename(p)[:-4].replace("-icon", "") for p in glob.glob(os.path.join(ROOT, "assets", "*.svg")))
    return [m for m in svgs if m not in ("star", "plain")] + ["star", "plain"]


def connect_parts(cs, r=0.7):
    """Join a logo made of separate pieces (e.g. Grok's two strokes) with small bridges at their
    closest points, so a free-standing emblem prints as one part."""
    parts = sorted(cs.decompose(), key=lambda c: -c.area())
    parts = [p for p in parts if p.area() > 0.5]
    if len(parts) < 2:
        return cs
    pts = lambda c: np.vstack([np.array(p) for p in c.to_polygons()])
    out, joined = parts[0], pts(parts[0])
    for p in parts[1:]:
        q = pts(p)
        d = ((joined[:, None, :] - q[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        a, b = joined[i], q[j]
        # push the bridge ends slightly into each part so it overlaps both
        u = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        bridge = CrossSection.batch_hull([CrossSection.circle(r, 16).translate(tuple(a - u * r)),
                                          CrossSection.circle(r, 16).translate(tuple(b + u * r))])
        out = out + p + bridge
        joined = np.vstack([joined, q])
    return out


def emblem_flat(model, radius, h, z0=0.0):
    """Emblem lying flat, extruded upward."""
    f = radius / 8.8
    return Manifold.extrude(emblem_cs(model).scale((f, f)), h).translate((0, 0, z0))


def emblem_xz(model, radius, thick):
    """Emblem standing up in the x-z plane, centred on the origin."""
    f = radius / 8.8
    return Manifold.extrude(emblem_cs(model).scale((f, f)), thick).rotate((90, 0, 0)).translate((0, thick / 2, 0))


def medallion(model, radius, thick):
    """A standing emblem on top of a piece. Brand logos sit on a round coin (thinner than the
    logo, so the logo stands out on both faces): parts of a logo that hang below its centre,
    or thin strokes, would otherwise start in mid-air or snap off. The open network graph of
    the no-logo version is printable as it is."""
    e = emblem_xz(model, radius, thick)
    if model == "plain":
        return e
    # the coin reaches past the logo's farthest point (corners of a square logo, legs of an arch)
    reach = max(np.hypot(*np.array(p).T).max() for p in emblem_cs(model).to_polygons()) * radius / 8.8
    return e + ycyl(reach + 0.5, -thick / 2 + 0.8, thick / 2 - 0.8, 0.0, seg=64)


def on_disc(e, r, y_a, y_b, z, x=0.0):
    """Clip a relief emblem to the flat round face it sits on, so no edge hangs past it."""
    return e ^ ycyl(r, y_a, y_b, z, x=x, seg=48)


def chip_base(w, label):
    """Microchip base. Returns (manifold, z_top_of_die, die_size)."""
    pin_out = 1.6
    body_w = w - 2 * pin_out
    body = chamfer_slab(body_w, w, BASE_H, 2.0, ch_top=1.0, ch_bot=0.4)

    # pins on the +x / -x sides, touching the floor
    pins = []
    n = int((w - 6) // 2.8)
    pitch = 2.8
    start = -(n - 1) * pitch / 2
    for i in range(n):
        y = start + i * pitch
        for sx in (-1, 1):
            pins.append(box(pin_out + 1.0, 1.4, 2.6, cx=sx * (body_w / 2 + pin_out / 2 - 0.5), cy=y))
            # little vertical leg up the side
            pins.append(box(0.9, 1.4, 4.4, cx=sx * (body_w / 2 + 0.2), cy=y))
    base = union([body] + pins)

    # die / heat spreader on top
    die_w = body_w * 0.72
    die = chamfer_slab(die_w, die_w, DIE_H + 0.2, 1.2, ch_top=0.5).translate((0, 0, BASE_H - 0.2))
    base = base + die

    # circuit traces from the die edge to every pin
    cut = []
    top_w = body_w - 2.0  # flat part of the chamfered top
    for i in range(n):
        y = start + i * pitch
        for sx in (-1, 1):
            length = (top_w - die_w) / 2 + 0.6
            cx = sx * (die_w / 2 + length / 2 - 0.3)
            cut.append(box(length, 0.8, 1.0, cx=cx, cy=y, z0=BASE_H - 0.5))
    # pin-1 dot
    cut.append(cyl(1.0, 0.9, z0=BASE_H - 0.6).translate((-top_w / 2 + 1.6, -w / 2 + 2.6, 0)))

    # no text: the shape of each piece speaks for itself (label is only a name)
    base = base - union(cut)
    # pieces start 0.3 mm inside the die so they fuse into one solid
    return base, BASE_H + DIE_H - 0.3, die_w


# ---------------------------------------------------------------- pieces
def pawn():
    """PAWN = a neuron: many small units that together make the network."""
    base, z0, die = chip_base(27, "TOKEN")
    prof = [(0, 0), (10.5, 0), (10.5, 1.6), (9.0, 3.0), (9.0, 4.0), (6.4, 6.5), (4.0, 16.5), (6.8, 17.8),
            (6.8, 18.8), (4.3, 19.8), (3.4, 20.6), (0, 20.6)]
    parts = [base, revolve(prof).translate((0, 0, z0 - 0.01))]
    R = 6.4
    hc = z0 + 20.6 + 5.0
    # the cell body, flattened front and back to carry the emblem
    parts.append(sphere(R, z=hc, seg=96) ^ box(20, 9.2, 20, z0=hc - 10))
    for sy in (-1, 1):
        parts.append(on_disc(emblem_xz(MODEL, 3.9, 1.4).translate((0, sy * 4.6, hc)), 4.2, sy * 3.0, sy * 6.0, hc))
    # dendrites reaching out to other neurons
    c = np.array([0.0, 0.0, hc])
    dirs = []
    for az in (0, 180, 40, 140, 220, 320):
        a, el = np.radians(az), np.radians(22)
        dirs.append(np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el)]))
    dirs.append(np.array([0.0, 0.0, 1.0]))
    for d in dirs:
        p0, p1 = c + d * (R - 0.8), c + d * (R + 2.6)
        parts.append(Manifold.batch_hull([sphere(1.4, *p0, seg=16), sphere(1.0, *p1, seg=16)]))  # >= 2 mm thick
        parts.append(sphere(1.15, *p1, seg=16))
    return union(parts)


def rook():
    base, z0, die = chip_base(30, "GPU")
    s = 19.0
    h = 34.0
    tower = chamfer_slab(s, s, h, 1.2).translate((0, 0, z0 - 0.01))
    cuts = []
    adds = []
    # side faces (x): stacked server blades with vent slots
    for i in range(1, 6):
        zz = z0 + i * h / 6
        for sx in (-1, 1):
            cuts.append(box(1.4, s + 2, 0.8, cx=sx * s / 2, z0=zz - 0.4))
    for i in range(6):
        zc = z0 + (i + 0.5) * h / 6
        for sx in (-1, 1):
            for k in range(-2, 3):
                cuts.append(box(1.6, 1.0, 2.6, cx=sx * s / 2, cy=k * 2.6, z0=zc - 1.3))
    # front and back (y): two big cooling fans, like a graphics card
    fan_r = 7.4
    depth = 1.8
    for sy in (-1, 1):
        for zc in (z0 + h * 0.27, z0 + h * 0.75):
            y_in = sy * (s / 2 - depth)  # bottom of the recess
            cuts.append(ycyl(fan_r, y_in, sy * (s / 2 + 1), zc))
            adds.append(fan_rotor(fan_r - 0.5, y_in - sy * 0.3, sy * (s / 2 - 0.15), zc))
            # the emblem sits on the fan hub (3.5 mm), clipped to it so nothing hangs over the blades
            adds.append(on_disc(emblem_xz(MODEL, 3.1, 1.0).translate((0, sy * (s / 2 - 0.2), zc)),
                                3.4, sy * (s / 2 - 2), sy * (s / 2 + 1), zc))
        # status LED strip
        for k in range(-3, 4):
            cuts.append(box(1.3, 1.6, 0.9, cx=k * 2.2, cy=sy * s / 2, z0=z0 + h * 0.52 - 0.45))
    tower = tower - union(cuts) + union(adds)

    z = z0 + h
    # cap flaring out at 45 degrees (printable), then battlements
    cap_w = 23.0
    flare = (cap_w - s) / 2
    cap = Manifold.extrude(rrect(s, s, 1.2), flare, 0, 0.0, (cap_w / s, cap_w / s)).translate((0, 0, z - 0.3))
    cap = cap + chamfer_slab(cap_w, cap_w, 2.5, 1.4).translate((0, 0, z + flare - 0.6))
    z += flare + 1.9
    wall_h = 6.0
    ring = chamfer_slab(cap_w, cap_w, wall_h, 1.4, ch_top=0.5).translate((0, 0, z - 0.3)) - box(
        cap_w - 5, cap_w - 5, wall_h + 1, z0=z + 0.6
    )
    notch = []
    for a in (0, 90):
        notch.append(box(6.0, cap_w + 2, 3.6, z0=z + wall_h - 3.4).rotate((0, 0, a)))
    ring = ring - union(notch)
    # heat-sink fins inside the battlements
    fins = [box(1.1, cap_w - 5.2, wall_h - 1.8, cx=x, z0=z) for x in (-6.0, -3.0, 0.0, 3.0, 6.0)]
    return union([base, tower, cap, ring] + fins)


# ---------------------------------------------------------------- v3 helpers
def smooth_closed(pts, n=6):
    """Catmull-Rom smoothing of a closed polygon."""
    P = np.asarray(pts, float)
    m = len(P)
    out = []
    for i in range(m):
        p0, p1, p2, p3 = P[i - 1], P[i], P[(i + 1) % m], P[(i + 2) % m]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return [tuple(p) for p in out]


def ccw(poly):
    a = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        a += x1 * y2 - x2 * y1
    return poly if a > 0 else poly[::-1]


def bevel_extrude(cs, thick, bev=1.2, steps=3):
    """Extrude a 2D shape to z in [0, thick] with stepped (printable) bevels on both faces."""
    st = bev / steps
    parts = [Manifold.extrude(cs, thick - 2 * bev + 0.04).translate((0, 0, bev - 0.02))]
    for i in range(steps):
        ins = bev - (i + 0.5) * st
        c = cs.offset(-ins, JoinType.Round, 2.0, 16)
        if c.area() <= 0.01:
            continue
        parts.append(Manifold.extrude(c, st + 0.03).translate((0, 0, i * st - 0.01)))
        parts.append(Manifold.extrude(c, st + 0.03).translate((0, 0, thick - (i + 1) * st - 0.02)))
    return union(parts)


def to_xz(m, thick):
    """Stand a shape drawn in 2D (x, z) and extruded along z up in the x-z plane, centred on y=0."""
    return m.rotate((90, 0, 0)).translate((0, thick / 2, 0))


def rounded_box(sx, sy, sz, r, top_scale=1.0, seg=16):
    """Box with rounded edges (hull of 8 spheres), bottom at z=0; top can be narrower (keycap)."""
    pts = []
    for zz, sc in ((r, 1.0), (sz - r, top_scale)):
        for x in (-1, 1):
            for y in (-1, 1):
                pts.append(sphere(r, x * (sx / 2 - r) * sc, y * (sy / 2 - r) * sc, zz, seg=seg))
    return Manifold.batch_hull(pts)


def tube(points, r, seg=12):
    sph = [sphere(r, *p, seg=seg) for p in points]
    return union([Manifold.batch_hull([a, b]) for a, b in zip(sph, sph[1:])])


def star2d(top, bottom, side, zc, k=0.18, n=14):
    """4-point AI 'sparkle' star with concave curved edges, drawn in (x, z)."""
    T, L, B, R = (np.array(p, float) for p in ((0, top), (-side, zc), (0, bottom), (side, zc)))
    C = np.array([0.0, zc])
    pts = []
    for A, Bp in ((T, L), (L, B), (B, R), (R, T)):
        ctrl = C + (A + Bp - 2 * C) * k
        for i in range(n):
            t = i / n
            pts.append(tuple((1 - t) ** 2 * A + 2 * (1 - t) * t * ctrl + t * t * Bp))
    return CrossSection([ccw(pts)])


def token_tile(w, h, depth=1.7):
    """One token: a small keycap-like tile facing +x, x in [0, depth], z in [0, h]."""
    r = min(0.6, h / 2 - 0.05, w / 2 - 0.05)
    t = Manifold.extrude(rrect(h, w, r), depth, 0, 0.0, (0.8, 0.85))
    return t.rotate((0, 90, 0)).translate((0, 0, h / 2))


# ---------------------------------------------------------------- KNIGHT = MCP connector
KNIGHT_PTS = [
    (11, -3), (10.8, 6), (10.3, 13), (10.8, 20), (11.6, 27), (10.4, 33), (8, 37.8), (5.4, 41),
    (4.6, 43.6), (3.0, 48.5), (1.2, 44.2), (-1.5, 42.8), (-4.5, 41.5), (-8, 39), (-11, 35.2),
    (-13.5, 31), (-15.8, 28.2), (-17.5, 26), (-17.8, 23.5), (-17, 21.2), (-15.5, 20), (-11.5, 17.6),
    (-8, 15.6), (-4.2, 13.5),  # the jaw runs straight down into the throat: no low point in mid-air (-4.4, 10), (-6.5, 6), (-9, 2.5), (-10, -3),
]


def inside(poly, x, z):
    c = False
    n = len(poly)
    for i in range(n):
        (x1, z1), (x2, z2) = poly[i], poly[(i + 1) % n]
        if (z1 > z) != (z2 > z) and x < x1 + (z - z1) * (x2 - x1) / (z2 - z1):
            c = not c
    return c


def knight():
    """KNIGHT = MCP, the connector. The knight jumps where nothing else can go, as MCP
    plugs the model into any tool: the ears are plug prongs, the mane is a cable,
    and each side of the neck has a connector socket."""
    base, z0, die = chip_base(32, "MCP")
    ped_h = 2.5
    parts = [base, chamfer_slab(24, 14, ped_h + 0.3, 1.5, ch_top=0.6).translate((0, 0, z0 - 0.3))]
    zb = z0 + ped_h - 0.3
    th = 12.0
    dx = 2.0
    outline = ccw(smooth_closed(KNIGHT_PTS, 6))
    head = to_xz(bevel_extrude(CrossSection([outline]), th, bev=1.2), th)
    head = (head ^ box(80, 40, 80, z0=0)).translate((dx, 0, zb))
    parts.append(head)

    # the ears are the two prongs of a plug
    for sy in (-1, 1):
        parts.append(rounded_box(2.0, 2.2, 9.0, 0.5, seg=10).translate((3.0 + dx, sy * 2.6, 42.5 + zb)))

    # the mane is a thick cable running down the back, held by clips, into the base
    # the top end is buried in the head, so there is no loose cap
    back = [(4.6, 42.0), (6.2, 40.6), (8.8, 36.8), (11.0, 31.0), (11.8, 25.0), (11.2, 18.0), (11.0, 11.0), (11.4, 5.0), (11.6, 1.0)]
    cable = [(x + dx + 0.6, 0.0, z + zb) for x, z in back]
    parts.append(tube(cable, 1.9, seg=20))
    for k in (2, 4, 6):
        x, _, z = cable[k]
        parts.append(sphere(2.5, x, 0, z, seg=24) ^ box(10, 10, 1.6, cx=x, z0=z - 0.8))

    # the model emblem is the horse eye
    for sy in (-1, 1):
        parts.append(emblem_xz(MODEL, 3.3, 1.6).translate((-5.8 + dx, sy * (th / 2 - 0.2), 33.0 + zb)))
    m = union(parts)

    # connector socket on each side of the neck (slot with a tongue inside), nostril and mouth
    cuts, adds = [], []
    for sy in (-1, 1):
        slot = Manifold.extrude(rrect(6.4, 2.8, 1.35), 3.0).rotate((90, 0, 0))  # y in [-3, 0]
        ty = th / 2 + 1.7 if sy > 0 else -th / 2 + 1.3  # 1.3 mm deep into the neck
        cuts.append(slot.translate((4.6 + dx, ty, 15.0 + zb)))
        adds.append(box(4.4, 1.0, 0.8, cx=4.6 + dx, cy=sy * (th / 2 - 1.0), z0=15.0 + zb - 0.4))

    def face_cut(pts2d, w, depth):
        out = []
        for sy in (-1, 1):
            caps = [ycyl(w / 2, sy * (th / 2 - depth), sy * (th / 2 + 1), zq + zb, x=x + dx, seg=16) for x, zq in pts2d]
            out += caps if len(caps) == 1 else [Manifold.batch_hull([c1, c2]) for c1, c2 in zip(caps, caps[1:])]
        return out

    cuts += face_cut([(-14.0, 24.6)], 1.8, 1.0) + face_cut([(-16.6, 21.0), (-13.0, 19.6)], 0.9, 0.9)
    return (m - union(cuts)) + union(adds)


# ---------------------------------------------------------------- BISHOP = the terminal agent
MITRE = [(-6.2, 0), (-7.6, 4), (-8.3, 8), (-7.6, 12.5), (-5.6, 16.5), (-2.8, 19.5), (0, 21),
         (2.8, 19.5), (5.6, 16.5), (7.6, 12.5), (8.3, 8), (7.6, 4), (6.2, 0)]


def bishop():
    """BISHOP = the agent in the terminal. The bishop slit becomes the
    terminal prompt: a chevron and a cursor, cut deep into both sides of the mitre."""
    base, z0, die = chip_base(31, "AGENT")
    body = revolve(
        [(0, 0), (9.5, 0), (9.5, 1.4), (8.2, 2.8), (6.5, 7), (4.6, 18), (4.2, 23), (6.6, 24.2),
         (6.6, 25.4), (4.6, 26.2), (5.4, 27), (5.4, 28.4), (0, 28.4)]
    ).translate((0, 0, z0 - 0.01))
    zs = z0 + 28.0
    th = 11.0
    mitre = to_xz(bevel_extrude(CrossSection([ccw(smooth_closed(MITRE, 6))]), th, bev=1.6), th)
    mitre = (mitre ^ box(40, 40, 40, z0=0)).translate((0, 0, zs - 0.3))
    m = union([base, body, mitre])

    # each face of the mitre is a terminal window: a sunk screen with the three window
    # buttons on top, and the prompt  >_  standing out of it
    zc_s, sw, sh = 9.3, 9.0, 10.6
    cuts, adds = [], []
    for sy in (-1, 1):
        scr = Manifold.extrude(rrect(sw, sh, 1.4), 2.0).rotate((90, 0, 0))  # y in [-2, 0]
        ty = th / 2 + 1.0 if sy > 0 else -th / 2 + 1.0  # screen sunk 1 mm into the face
        cuts.append(scr.translate((0, ty, zs + zc_s)))
    m = m - union(cuts)
    floor = th / 2 - 1.0

    def stroke(pts2d, w, top):
        out = []
        for sy in (-1, 1):
            caps = [ycyl(w / 2, sy * (floor - 0.3), sy * top, zs + zq, x=x, seg=16) for x, zq in pts2d]
            out += caps if len(caps) == 1 else [Manifold.batch_hull([c1, c2]) for c1, c2 in zip(caps, caps[1:])]
        return out

    adds += stroke([(-3.0, 11.4), (-0.2, 9.2), (-3.0, 7.0)], 1.5, th / 2 + 0.4)  # >
    adds += stroke([(1.0, 7.0), (3.4, 7.0)], 1.4, th / 2 + 0.4)  # _
    for x in (-3.2, -1.9, -0.6):  # window buttons
        adds += stroke([(x, 13.4)], 1.0, th / 2 - 0.2)
    m = m + union(adds)
    # the model emblem is the knob on top of the mitre
    tip = zs - 0.3 + 21.0
    m = m + cyl(5.4, 1.6, z0=tip - 2.2) + medallion(MODEL, 5.0, 3.6).translate((0, 0, tip + 3.2))
    return m


# ---------------------------------------------------------------- QUEEN = the context window
def queen():
    """QUEEN = the context window. A long ribbon of tokens winds up the body; a window
    frame holds only part of it. Tokens inside the window stand out sharp, older ones
    below have dropped out (only their faded prints are left), and the cursor at the
    top of the window is where the next token goes."""
    base, z0, die = chip_base(34, "CONTEXT")
    qz = [0, 2, 4, 14, 30, 40, 48, 54, 58]
    qr = [11.0, 11.0, 9.5, 7.2, 5.6, 5.4, 6.2, 8.2, 9.6]

    def R(zz):
        return float(np.interp(zz, qz, qr))

    prof = [(0, 0)] + [(r, zz) for zz, r in zip(qz, qr)] + [(10.2, 59.4), (10.2, 61.0), (8.0, 61.0), (6.5, 60.2), (0, 60.2)]
    body = revolve(prof).translate((0, 0, z0 - 0.01))
    w1, w2 = 22.0, 40.0  # the window: bottom and top
    mull = [45.0, 135.0, 225.0, 315.0]
    adds, cuts = [], []
    # the token ribbon: a helix of tiles climbing the body
    zz, ang = 6.0, 0.0
    pitch = 5.6
    th_t = 2.6
    last = None
    while True:
        r = R(zz + th_t / 2)
        w = 3.4
        zc = zz + th_t / 2
        if zc > w2 - 2.2:
            break
        near_mull = min(abs(((ang - m_ + 180) % 360) - 180) for m_ in mull) < 11
        if zc < w1 + 0.5:
            # dropped out of the context: only a faded print in the body
            cuts.append(token_tile(w, th_t, depth=2.0).translate((r - 0.7, 0, z0 + zz)).rotate((0, 0, ang)))
        elif not near_mull:
            adds.append(token_tile(w, th_t, depth=2.6).translate((r - 0.9, 0, z0 + zz)).rotate((0, 0, ang)))
            last = (r, zz, ang)
        dang = np.degrees((w + 0.9) / r)
        ang += dang
        zz += pitch * dang / 360
    # cursor right after the newest token
    r, zl, al = last
    adds.append(token_tile(1.0, 4.6, depth=2.8).translate((r - 0.9, 0, z0 + zl - 1.0)).rotate((0, 0, al + np.degrees(3.6 / r))))
    # the window frame: two rails and four bars
    for zr in (w1, w2):
        r = R(zr)
        rail = revolve([(r - 0.4, zr - 2.4), (r + 2.0, zr - 0.2), (r + 2.0, zr + 0.8), (r - 0.4, zr + 0.8)])
        adds.append(rail.translate((0, 0, z0)))
    for m_ in mull:
        pts = [((R(zq) + 1.4) * np.cos(np.radians(m_)), (R(zq) + 1.4) * np.sin(np.radians(m_)), z0 + zq)
               for zq in np.linspace(w1, w2, 5)]
        adds.append(tube(pts, 1.2, seg=12))
    parts = [base, (body - union(cuts)) + union(adds)]
    zt = z0 + 61.0
    tilt = np.radians(16)
    for k in range(9):
        sp = cyl(7.0, 1.6, 1.2, z0=-0.5, seg=20).rotate((0, 16, 0)).translate((8.4, 0, zt))
        tip = rounded_box(2.8, 2.8, 2.8, 0.7, seg=10).translate((8.4 + 7 * np.sin(tilt), 0, zt + 7 * np.cos(tilt) - 1.2))
        parts.append((sp + tip).rotate((0, 0, 360 / 9 * k)))
    # the model emblem is the crown jewel
    parts.append(cyl(6.5, 1.6, z0=zt - 1.3))
    parts.append(medallion(MODEL, 6.0, 3.4).translate((0, 0, zt + 5.2)))
    return union(parts)


# ---------------------------------------------------------------- KING = the model
def king(emblem=None):
    """KING = the model itself: an artificial neural network whose output node is the emblem."""
    emblem = emblem or MODEL
    base, z0, die = chip_base(36, "MODEL")
    prof_z = [0, 1.5, 3.0, 8, 30, 39]
    prof_r = [11, 11, 9.6, 8.0, 6.2, 5.6]
    body = revolve(
        [(0, 0), (11, 0), (11, 1.5), (9.6, 3), (8, 8), (6.2, 30), (5.6, 39), (7.6, 40.6), (7.6, 42),
         (5.8, 43), (5.8, 44.3), (0, 44.3)]
    ).translate((0, 0, z0 - 0.01))
    parts = [base, body]

    def R(zz):
        return float(np.interp(zz, prof_z, prof_r))

    for k in range(8):
        a0 = 2 * np.pi * k / 8
        path = [(a0, 6.0), (a0, 13.0), (a0 + 0.25, 18.0), (a0 + 0.25, 26.0), (a0, 31.0), (a0, 36.5)]
        pts = [((R(zz) + 0.05) * np.cos(a), (R(zz) + 0.05) * np.sin(a), z0 + zz) for a, zz in path]
        parts.append(tube(pts, 0.55))
        for p_ in (pts[0], pts[-1]):
            parts.append(sphere(1.0, *p_, seg=16))

    zn = z0 + 44.0
    parts.append(revolve([(0, 0), (5.8, 0), (10.6, 4.8), (10.6, 8.0), (0, 8.0)]).translate((0, 0, zn - 0.3)))
    for k in range(10):
        parts.append(box(2.0, 1.5, 1.8, cx=10.0, z0=zn + 7.4).rotate((0, 0, 360 / 10 * k + 18)))

    # the network: layers of nodes, every node wired to every node of the next layer
    ct = zn + 7.7
    layers = [(1.2, [-2.5, 2.5]), (8.2, [-5.2, 0.0, 5.2]), (15.2, [-2.5, 2.5])]
    out = (0.0, 0.0, ct + 21.5)
    for rot in (0, 90):
        net = []
        nodes = [[(x, 0.0, ct + zz) for x in xs] for zz, xs in layers]
        for la, lb in zip(nodes, nodes[1:] + [[out]]):
            for a in la:
                for b in lb:
                    net.append(tube([a, b], 1.15, seg=12))
        for la in nodes:
            for a in la:
                net.append(sphere(1.9, *a, seg=24))
        net.append(tube([(0.0, 0.0, ct + layers[1][0]), out], 1.6, seg=16))  # spine: carries the emblem
        parts.append(union(net).rotate((0, 0, rot)))
    # output node = the model emblem, seen from every side
    e = medallion(emblem, 5.6, 3.4)
    parts.append((e + e.rotate((0, 0, 90))).translate(out))
    return union(parts)


PIECES = {
    "king": (king, 1),
    "queen_context": (queen, 1),
    "bishop_terminal": (bishop, 2),
    "knight_mcp": (knight, 2),
    "rook_gpu": (rook, 2),
    "pawn_neuron": (pawn, 8),
}


# ---------------------------------------------------------------- output
def write_stl(m, path):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    t = np.asarray(mesh.tri_verts).astype(np.int64)
    tri = v[t]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1
    n = (n / ln).astype(np.float32)
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    with open(path, "wb") as f:
        f.write(b"AI chess set by GIAMAT13".ljust(80, b" "))
        f.write(np.uint32(len(t)).tobytes())
        f.write(rec.tobytes())
    return v, t


def main():
    OUT = os.path.join(ROOT, "stl", MODEL)  # stl/<version>/
    os.makedirs(OUT, exist_ok=True)
    built = {}
    names = list(PIECES)
    for i, name in enumerate(names):
        fn, count = PIECES[name]
        print(f"@@now building {name} ({i + 1} of {len(names)})", flush=True)
        m = fn()
        # drop zero-volume slivers left by booleans
        m = Manifold.compose([p for p in m.decompose() if p.volume() > 1.0])
        if m.status().name != "NoError":
            print("ERROR", name, m.status(), flush=True)
            sys.exit(1)
        bb = m.bounding_box()
        v, t = write_stl(m, os.path.join(OUT, f"{name}_{MODEL}.stl"))
        built[name] = m
        print(
            f"{name}: {len(t)} triangles, genus {m.genus()}, "
            f"size {bb[3]-bb[0]:.1f} x {bb[4]-bb[1]:.1f} x {bb[5]-bb[2]:.1f} mm, volume {m.volume()/1000:.1f} cm3, x{count}",
            flush=True,
        )
        print("@@sub done", flush=True)

    # one plate per side (16 pieces each), 4 x 4 grid: 
    side_name = MODEL
    for side, kname in ((side_name, "king"),):
        print(f"@@now building {side} plate", flush=True)
        order = ["rook_gpu", "knight_mcp", "bishop_terminal", kname,
                 "queen_context", "bishop_terminal", "knight_mcp", "rook_gpu"] + ["pawn_neuron"] * 8
        pitch = 44.0
        plate = [built[n].translate(((k % 4 - 1.5) * pitch, (k // 4 - 1.5) * pitch, 0)) for k, n in enumerate(order)]
        write_stl(Manifold.compose(plate), os.path.join(OUT, f"plate_{side}_16_pieces.stl"))
        print(f"plate_{side}_16_pieces: done", flush=True)
    print("@@sub done", flush=True)

if __name__ == "__main__":
    main()
