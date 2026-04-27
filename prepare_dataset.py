"""
Normalise a downloaded fruit dataset into the layout train.py expects.

Handles two common Kaggle layouts:

  Layout A (already correct):
      root/
          fresh_apple/   *.jpg
          rotten_apple/  *.jpg
          ...

  Layout B (train/test split — common in Kaggle "Fruits fresh and rotten"):
      root/
          train/
              freshapples/   *.jpg
              rottenapples/  *.jpg
              ...
          test/
              freshapples/   *.jpg
              ...

Outputs a single merged folder at <out_dir> with class names normalised to
'fresh_<fruit>' / 'rotten_<fruit>' (lowercase, singular), which matches the
FRUIT_DB keys used by app/predictor.py.

Usage:
    python prepare_dataset.py <input_dataset_dir> <output_dataset_dir>
"""

import os
import re
import shutil
import sys

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Known fruit stems (singular, lowercase).
KNOWN_FRUITS = {
    "apple", "banana", "orange", "strawberry", "grape", "mango",
    "watermelon", "blueberry", "lemon", "kiwi", "peach", "avocado",
    "cherry", "pineapple", "pomegranate",
}


def _normalise_label(raw: str):
    """Convert 'freshApples', 'rotten_banana', 'Fresh Mango' -> 'fresh_apple' etc."""
    s = re.sub(r"[\s\-]+", "_", raw.strip().lower())
    is_rotten = "rotten" in s
    is_fresh = "fresh" in s
    s = s.replace("rotten", "").replace("fresh", "").strip("_")
    # Strip trailing 's' if it leaves a known fruit.
    if s.endswith("s") and s[:-1] in KNOWN_FRUITS:
        s = s[:-1]
    if s not in KNOWN_FRUITS:
        return None
    if is_rotten:
        return f"rotten_{s}"
    if is_fresh:
        return f"fresh_{s}"
    return None


def _iter_class_dirs(root: str):
    """Yield (label, abs_dir) for every leaf folder that contains images."""
    for dirpath, dirnames, filenames in os.walk(root):
        imgs = [f for f in filenames if os.path.splitext(f)[1].lower() in VALID_EXT]
        if not imgs:
            continue
        label = os.path.basename(dirpath)
        yield label, dirpath, imgs


def prepare(input_dir: str, output_dir: str):
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    counts = {}
    skipped = []
    for raw_label, src_dir, imgs in _iter_class_dirs(input_dir):
        norm = _normalise_label(raw_label)
        if norm is None:
            skipped.append(raw_label)
            continue
        dst_dir = os.path.join(output_dir, norm)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in imgs:
            src = os.path.join(src_dir, fname)
            # Prefix with original folder name to avoid collisions across splits.
            dst = os.path.join(dst_dir, f"{raw_label}__{fname}")
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except OSError as e:
                    print(f"  [warn] copy failed {src}: {e}")
                    continue
        counts[norm] = counts.get(norm, 0) + len(imgs)

    print("\n=== Dataset prepared ===")
    for cls in sorted(counts):
        print(f"  {cls:20s} {counts[cls]} images")
    if skipped:
        print(f"\nSkipped {len(skipped)} folders with unrecognised names, e.g.:")
        for s in skipped[:10]:
            print(f"  - {s}")
    print(f"\nReady. Now run:\n  python train.py \"{output_dir}\"")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_dataset.py <input_dataset_dir> <output_dataset_dir>")
        sys.exit(1)
    prepare(sys.argv[1], sys.argv[2])
