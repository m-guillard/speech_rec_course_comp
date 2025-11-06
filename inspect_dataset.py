#!/usr/bin/env python
# inspect_csv.py
import argparse
from pathlib import Path
import pandas as pd

def find_root(user_root: str | None) -> Path:
    if user_root:
        r = Path(user_root).expanduser().resolve()
        if not r.exists():
            raise FileNotFoundError(f"--root not found: {r}")
        return r
    # Try: script_dir, script_dir/geo, CWD, CWD/.../geo
    candidates = []
    script_dir = Path(__file__).parent.resolve()
    candidates += [script_dir, script_dir / "geo"]
    candidates += [Path.cwd(), Path.cwd() / "geo"]
    for c in candidates:
        if (c / "train.csv").exists() and (c / "dev.csv").exists() and (c / "test.csv").exists():
            return c.resolve()
    raise FileNotFoundError("Could not locate train.csv/dev.csv/test.csv. Pass --root /path/to/geo")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="Path containing train.csv/dev.csv/test.csv and a clips/ dir")
    args = ap.parse_args()
    root = find_root(args.root)
    print(f"Using root: {root}")

    for split in ["train", "dev", "test"]:
        csv_path = root / f"{split}.csv"
        df = pd.read_csv(csv_path)
        print(f"== {csv_path.name} ==")
        print("columns:", list(df.columns))
        print(df.head(5))
        print()

    # Show a glimpse of the audio dir
    clips = root / "clips"
    if clips.exists():
        n = sum(1 for _ in clips.glob("*.wav"))
        print(f"clips/: {n} wav files. Examples:")
        for p in list(clips.glob("*.wav"))[:5]:
            print(" -", p.name)
    else:
        print("WARNING: clips/ not found under root.")

if __name__ == "__main__":
    main()
