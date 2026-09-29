"""Flight/site-based train/val/test split for retraining on labelled drone frames.

Frames from one flight are near-duplicates, so a random frame split leaks test data into
training and overstates accuracy. This script keeps every frame of a flight (or a site) on
the same side of the split.

Input: one or more folders in the layout exported by the app (images/ + labels/, file names
starting with the flight id, e.g. `flight-07_f000120.jpg`) after the labels have been reviewed.

    python tools/split_by_flight.py labelled/ -o dataset/ --test 0.2 --val 0.1
    python tools/split_by_flight.py labelled/ -o dataset/ --test-groups flight-07 flight-12
    python tools/split_by_flight.py labelled/ -o dataset/ --group-map flights.csv   # flight,site

Writes dataset/{train,val,test}/{images,labels} and dataset/data.yaml for Ultralytics.
"""
import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def flight_of(stem):
    m = re.match(r"(.+)_f\d+$", stem)
    return m.group(1) if m else stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", type=Path, help="folders containing images/ and labels/")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--test", type=float, default=0.2, help="fraction of groups for test")
    ap.add_argument("--val", type=float, default=0.1, help="fraction of groups for validation")
    ap.add_argument("--test-groups", nargs="*", default=[], help="put exactly these groups in test")
    ap.add_argument("--group-map", type=Path,
                    help="CSV with columns flight,site to split by site instead of by flight")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    site_of = {}
    if args.group_map:
        with open(args.group_map, newline="") as f:
            site_of = {r["flight"]: r["site"] for r in csv.DictReader(f)}

    groups = defaultdict(list)
    for src in args.sources:
        for img in sorted((src / "images").iterdir()):
            if img.suffix.lower() in IMAGE_EXTS:
                flight = flight_of(img.stem)
                groups[site_of.get(flight, flight)].append((img, src / "labels" / f"{img.stem}.txt"))
    if len(groups) < 2:
        raise SystemExit(f"Need at least 2 flights/sites to split, found {list(groups)}.")

    names = sorted(groups)
    random.Random(args.seed).shuffle(names)
    test = [g for g in names if g in args.test_groups]
    rest = [g for g in names if g not in args.test_groups]
    if not args.test_groups:
        n_test = max(1, round(len(names) * args.test))
        test, rest = rest[:n_test], rest[n_test:]
    n_val = round(len(names) * args.val) if len(rest) > 1 else 0
    split = {"test": test, "val": rest[:n_val], "train": rest[n_val:]}

    for part, gs in split.items():
        (args.out / part / "images").mkdir(parents=True, exist_ok=True)
        (args.out / part / "labels").mkdir(parents=True, exist_ok=True)
        n = 0
        for g in gs:
            for img, lbl in groups[g]:
                shutil.copy2(img, args.out / part / "images" / img.name)
                if lbl.exists():
                    shutil.copy2(lbl, args.out / part / "labels" / lbl.name)
                n += 1
        print(f"{part:5s}: {len(gs)} groups, {n} images  {gs}")

    classes = args.sources[0] / "data.yaml"
    names_block = classes.read_text().split("names:", 1)[1] if classes.exists() else "\n  0: defect\n"
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: train/images\nval: val/images\n"
        f"test: test/images\nnames:{names_block}")
    if not split["val"]:
        print("warning: no validation groups; set val: to a held-out folder before training.")


if __name__ == "__main__":
    main()
