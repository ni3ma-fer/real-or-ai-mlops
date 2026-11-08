"""Extrait des images de validation en fichiers, rangées par générateur : data/samples/<générateur>/.

Elles servent à tester l'API, au test de charge et à la démo de drift.
Usage : python scripts/export_samples.py --per-generator 50
"""

import argparse
import glob
import io
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "tiny-genimage" / "data"
OUT = ROOT / "data" / "samples"
GENERATORS = {0: "Real", 1: "ADM", 2: "BigGAN", 3: "GLIDE", 4: "Midjourney",
              5: "SD14", 6: "SD15", 7: "VQDM", 8: "Wukong"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-generator", type=int, default=50)
    args = p.parse_args()

    counts: Counter = Counter()
    for f in sorted(glob.glob(str(RAW / "validation-*.parquet"))):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["image", "generator"]):
            d = batch.to_pydict()
            for img, gen in zip(d["image"], d["generator"]):
                name = GENERATORS[gen]
                if counts[name] >= args.per_generator:
                    continue
                ext = (Image.open(io.BytesIO(img["bytes"])).format or "png").lower()
                dest = OUT / name / f"{name.lower()}_{counts[name]:04d}.{'jpg' if ext == 'jpeg' else ext}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(img["bytes"])
                counts[name] += 1
            if all(counts[n] >= args.per_generator for n in GENERATORS.values() if n != "SD14"):
                break
    print(dict(counts))


if __name__ == "__main__":
    main()
