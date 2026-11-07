"""Minimal SVG path reader: turns the path(s) of an SVG file into polygons for manifold3d.

Supports M L H V C S Q T A Z (absolute and relative). Curves and arcs are sampled.
"""
import math
import re

import numpy as np

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARGS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


def _tokens(d):
    i, n, cmd, argi = 0, len(d), None, 0
    while i < n:
        ch = d[i]
        if ch.isalpha():
            cmd, argi = ch, 0
            yield ch
            i += 1
            continue
        if ch in " ,\t\r\n":
            i += 1
            continue
        # arc flags (args 3 and 4) are single characters with no separator needed
        if cmd and cmd.upper() == "A" and argi % 7 in (3, 4):
            yield float(ch)
            i += 1
            argi += 1
            continue
        m = _NUM.match(d, i)
        if not m:
            raise ValueError(f"bad path data at {i}: {d[i:i+20]!r}")
        yield float(m.group())
        i = m.end()
        argi += 1


def _arc(p0, rx, ry, phi, fa, fs, p1, steps=24):
    """SVG endpoint arc -> list of points (excluding p0)."""
    if rx == 0 or ry == 0:
        return [p1]
    phi = math.radians(phi)
    c, s = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1, y1 = c * dx + s * dy, -s * dx + c * dy
    rx, ry = abs(rx), abs(ry)
    lam = x1 ** 2 / rx ** 2 + y1 ** 2 / ry ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1 ** 2 - ry ** 2 * x1 ** 2
    den = rx ** 2 * y1 ** 2 + ry ** 2 * x1 ** 2
    k = math.sqrt(max(0.0, num / den)) * (-1 if fa == fs else 1)
    cx1, cy1 = k * rx * y1 / ry, -k * ry * x1 / rx
    cx = c * cx1 - s * cy1 + (p0[0] + p1[0]) / 2
    cy = s * cx1 + c * cy1 + (p0[1] + p1[1]) / 2

    def ang(ux, uy, vx, vy):
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a

    t1 = ang(1, 0, (x1 - cx1) / rx, (y1 - cy1) / ry)
    dt = ang((x1 - cx1) / rx, (y1 - cy1) / ry, (-x1 - cx1) / rx, (-y1 - cy1) / ry)
    if not fs and dt > 0:
        dt -= 2 * math.pi
    elif fs and dt < 0:
        dt += 2 * math.pi
    out = []
    for j in range(1, steps + 1):
        t = t1 + dt * j / steps
        x, y = rx * math.cos(t), ry * math.sin(t)
        out.append((c * x - s * y + cx, s * x + c * y + cy))
    return out


def _bez(pts, steps=12):
    out = []
    for j in range(1, steps + 1):
        t = j / steps
        if len(pts) == 4:
            p = [(1 - t) ** 3 * a + 3 * (1 - t) ** 2 * t * b + 3 * (1 - t) * t ** 2 * c + t ** 3 * d
                 for a, b, c, d in zip(*pts)]
        else:
            p = [(1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c for a, b, c in zip(*pts)]
        out.append(tuple(p))
    return out


def path_polygons(d):
    toks = list(_tokens(d))
    polys, cur, pos, start = [], [], (0.0, 0.0), (0.0, 0.0)
    last_ctrl, last_cmd = None, None
    i, cmd = 0, None
    while i < len(toks):
        if isinstance(toks[i], str):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                if cur:
                    polys.append(cur)
                cur, pos = [], start
                last_cmd = "Z"
                continue
        U, rel = cmd.upper(), cmd.islower()
        k = _ARGS[U]
        a = toks[i:i + k]
        i += k
        ox, oy = pos if rel else (0.0, 0.0)
        if U == "M":
            if cur:
                polys.append(cur)
            pos = (a[0] + ox, a[1] + oy)
            start, cur = pos, [pos]
            cmd = "l" if rel else "L"  # implicit lineto after moveto
            last_ctrl = None
        elif U == "L":
            pos = (a[0] + ox, a[1] + oy); cur.append(pos); last_ctrl = None
        elif U == "H":
            pos = (a[0] + ox, pos[1]); cur.append(pos); last_ctrl = None
        elif U == "V":
            pos = (pos[0], a[0] + (pos[1] if rel else 0.0)); cur.append(pos); last_ctrl = None
        elif U in "CS":
            if U == "C":
                c1 = (a[0] + ox, a[1] + oy); c2 = (a[2] + ox, a[3] + oy); p = (a[4] + ox, a[5] + oy)
            else:
                c1 = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_cmd in "CS" and last_ctrl else pos
                c2 = (a[0] + ox, a[1] + oy); p = (a[2] + ox, a[3] + oy)
            cur += _bez([pos, c1, c2, p]); last_ctrl, pos = c2, p
        elif U in "QT":
            if U == "Q":
                c1 = (a[0] + ox, a[1] + oy); p = (a[2] + ox, a[3] + oy)
            else:
                c1 = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_cmd in "QT" and last_ctrl else pos
                p = (a[0] + ox, a[1] + oy)
            cur += _bez([pos, c1, p]); last_ctrl, pos = c1, p
        elif U == "A":
            p = (a[5] + ox, a[6] + oy)
            cur += _arc(pos, a[0], a[1], a[2], a[3], a[4], p); pos = p; last_ctrl = None
        last_cmd = U
    if cur:
        polys.append(cur)
    return polys


def _style(el, key):
    v = el.get(key)
    st = el.get("style", "")
    m = re.search(key + r"\s*:\s*([^;]+)", st)
    return m.group(1).strip() if m else v


def _shape_d(el, tag):
    """Path data for basic shapes, so plain <circle>/<rect>/<polygon> icons work too."""
    g = lambda k, dflt=0.0: float(el.get(k, dflt))
    if tag == "path":
        return el.get("d")
    if tag in ("circle", "ellipse"):
        cx, cy = g("cx"), g("cy")
        rx = g("r") if tag == "circle" else g("rx")
        ry = g("r") if tag == "circle" else g("ry")
        return f"M{cx-rx},{cy}A{rx},{ry} 0 1 0 {cx+rx},{cy}A{rx},{ry} 0 1 0 {cx-rx},{cy}Z"
    if tag == "rect":
        x, y, w, h = g("x"), g("y"), g("width"), g("height")
        return f"M{x},{y}H{x+w}V{y+h}H{x}Z"
    if tag in ("polygon", "polyline"):
        return "M" + el.get("points", "").strip() + "Z"
    return None


def svg_shapes(path):
    """Every filled shape in an SVG as (polygons, fill_rule), in SVG coordinates.

    Skips what is not part of the outline: anything inside a filter (blur glows, like
    Gemini's colour blobs), <defs>/<clipPath>/<title>, shapes with fill="none", raster
    images, and exact duplicates (a mask copy of the same outline).
    Transforms are not applied; most icon SVGs do not use them.
    """
    import codecs
    import xml.etree.ElementTree as ET
    raw = open(path, "rb").read()
    enc = "utf-16" if raw[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE) else "utf-8-sig"
    root = ET.fromstring(raw.decode(enc))
    out, seen = [], set()

    def walk(el, rule, fill, skip):
        tag = el.tag.split("}")[-1]
        if tag in ("defs", "clipPath", "title", "desc", "style", "linearGradient", "radialGradient", "filter"):
            return
        skip = skip or el.get("filter") is not None
        rule = _style(el, "fill-rule") or _style(el, "clip-rule") or rule
        fill = _style(el, "fill") or fill
        d = _shape_d(el, tag)
        if d and not skip and fill != "none":
            if d not in seen:
                seen.add(d)
                polys = [p for p in path_polygons(d) if len(p) >= 3]
                if polys:
                    out.append((polys, rule, fill))
        for ch in el:
            walk(ch, rule, fill, skip)

    walk(root, "nonzero", "black", False)
    return _roles(out)


WHITE = ("#fff", "#ffffff", "white", "#fefefe")


def _roles(shapes):
    """Turn painted shapes into (polygons, fill_rule, op) with op "add" or "sub".

    - An app-icon background square (a shape covering ~all of the icon) is dropped.
    - White shapes painted over coloured ones are cut-outs (op "sub"), e.g. Qwen's inner lines.
    """
    if not shapes:
        return []

    def box(polys):
        a = np.vstack([np.array(p) for p in polys])
        return a.min(0), a.max(0)

    lo = np.min([box(p)[0] for p, _, _ in shapes], 0)
    hi = np.max([box(p)[1] for p, _, _ in shapes], 0)
    full = np.prod(hi - lo)
    def fill_area(polys):
        return abs(sum(0.5 * np.sum(np.array(p)[:, 0] * np.roll(np.array(p)[:, 1], -1)
                                    - np.roll(np.array(p)[:, 0], -1) * np.array(p)[:, 1]) for p in polys))

    def is_tile(polys):  # spans the whole icon and is nearly solid, like a rounded square
        b = box(polys)
        return np.prod(b[1] - b[0]) >= 0.9 * full and fill_area(polys) >= 0.8 * np.prod(b[1] - b[0])

    if len(shapes) > 1:
        kept = [s for s in shapes if not is_tile(s[0])]
        if kept:
            shapes = kept
    colour = [s for s in shapes if str(s[2]).lower() not in WHITE]
    out = []
    for polys, rule, fill in shapes:
        white = str(fill).lower() in WHITE
        out.append((polys, rule, "sub" if white and colour else "add"))
    return out


def normalise(shapes, radius=8.8):
    """Centre all shapes on the origin, flip y (SVG y points down) and scale to `radius`."""
    pts = np.vstack([np.array(p) for polys, _, op in shapes if op == "add" for p in polys])
    lo, hi = pts.min(0), pts.max(0)
    ctr = (lo + hi) / 2
    f = radius / (np.max(hi - lo) / 2)
    return [([[((x - ctr[0]) * f, -(y - ctr[1]) * f) for x, y in p] for p in polys], rule, op)
            for polys, rule, op in shapes]
