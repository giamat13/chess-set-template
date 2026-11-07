"""Build every version: star, plain, and one per SVG in assets/.

Usage: python src/build_all.py            # all versions
       python src/build_all.py grok gemini  # only these

For each version: write stl/<version>/, render images/preview_<version>.png, then run the
P1S print check on its files. To add a new version, drop <name>.svg into assets/ and run this.

Lines starting with @@ drive the progress dashboard (one step per version, one sub-step for
the build, the picture and each checked file); they are harmless elsewhere.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_set import all_models  # noqa: E402

FILES = ["bishop", "king", "knight", "pawn", "plate", "queen", "rook"]  # print_check order (sorted names)


def run(script, m):
    r = subprocess.run([sys.executable, os.path.join(HERE, script), m], capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
        raise SystemExit(f"{script} {m} failed")


models = sys.argv[1:] or all_models()
print(f"versions: {', '.join(models)}", flush=True)
for i, m in enumerate(models):
    print(f"@@step {m}", flush=True)
    for s in ["build pieces~90s", "draw picture~30s"] + [f"check {f}~40s" for f in FILES]:
        print(f"@@sub add {s}", flush=True)
    print(f"@@now {m}: building pieces ({i + 1} of {len(models)})", flush=True)
    run("build_set.py", m)
    print("@@sub done", flush=True)
    print(f"@@now {m}: drawing picture ({i + 1} of {len(models)})", flush=True)
    run("render_preview.py", m)
    print("@@sub done", flush=True)
    print(f"@@now {m}: checking pieces for the P1S ({i + 1} of {len(models)})", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, "print_check.py"), m])  # prints @@sub done per file
    print(f"{m}: built and checked", flush=True)
print("done", flush=True)
