"""Pre-push check for the models in models/.

    python tools/check_models.py

For every model in models/models.json (and any other .onnx in models/) it checks that the file
exists and is real weights (not a Git LFS pointer), loads it with ONNX Runtime, runs one
inference, prints classes / input size / output shape / CPU speed, and warns when a model's
classes differ from the first model's. It also checks that .onnx files are tracked by Git LFS,
because GitHub rejects files over 100 MB.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import SolarPanelDetector  # noqa: E402

MODELS_DIR = ROOT / "models"
GITHUB_LIMIT_MB = 100


def main():
    registry = json.loads((MODELS_DIR / "models.json").read_text(encoding="utf-8"))
    listed = {e["file"] for e in registry}
    entries = registry + [{"file": p.name, "name": p.stem}
                          for p in sorted(MODELS_DIR.glob("*.onnx")) if p.name not in listed]
    ok, reference = True, None

    for e in entries:
        path = MODELS_DIR / e["file"]
        name = e.get("name", path.stem)
        print(f"\n== {name} ({path.relative_to(ROOT).as_posix()})")
        if not path.exists():
            print("  MISSING")
            ok = False
            continue
        if path.read_bytes()[:40].startswith(b"version https://git-lfs"):
            print("  Git LFS pointer, weights not downloaded: run `git lfs pull`")
            ok = False
            continue
        size_mb = path.stat().st_size / 1e6
        try:
            det = SolarPanelDetector(str(path), name=name)
            img = np.full((det.img_size, det.img_size, 3), 114, np.uint8)
            det.detect(img)  # warm-up
            t0 = time.perf_counter()
            for _ in range(3):
                det.detect(img)
            ms = (time.perf_counter() - t0) / 3 * 1000
        except Exception as ex:  # anything ONNX Runtime or the decoder rejects
            print(f"  FAILED to load/run: {ex}")
            ok = False
            continue
        out = det.session.get_outputs()[0]
        print(f"  size {size_mb:.1f} MB | input {det.img_size} | output {out.shape} | "
              f"end2end {det.end2end} | {ms:.0f} ms per image on this CPU")
        print(f"  classes: {det.class_names}")
        if reference is None:
            reference = (name, det.class_names)
        elif det.class_names != reference[1]:
            print(f"  WARNING: classes differ from {reference[0]}; per-class comparison will not line up")

        tracked = subprocess.run(["git", "check-attr", "filter", "--", str(path)], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip().endswith("lfs")
        if not tracked and size_mb > GITHUB_LIMIT_MB:
            print(f"  ERROR: {size_mb:.0f} MB and not tracked by Git LFS; GitHub will reject the push")
            ok = False
        elif tracked:
            print("  Git LFS: tracked")

    print("\nAll models OK." if ok else "\nSome models have problems (see above).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
