"""Check that every piece is one solid part.  Usage: python check_parts.py [version]"""
import sys

import build_set as b

if len(sys.argv) > 1:
    b.MODEL = sys.argv[1]
for name, (fn, count) in b.PIECES.items():
    m = fn()
    parts = [p for p in m.decompose() if p.volume() > 1.0]
    print(f"{name} ({b.MODEL}) parts {len(parts)} genus {m.genus()}", flush=True)
    for p in parts[1:]:
        bb = p.bounding_box()
        print(f"   loose vol {p.volume():.2f} at x{(bb[0]+bb[3])/2:.1f} y{(bb[1]+bb[4])/2:.1f} z{(bb[2]+bb[5])/2:.1f}", flush=True)
