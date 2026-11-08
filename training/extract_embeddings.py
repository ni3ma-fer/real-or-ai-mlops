"""Transforme chaque image de Tiny-GenImage en vecteur CLIP (embedding de 512 nombres).

C'est l'étape lourde (plusieurs dizaines de minutes sur CPU). On la fait une
seule fois : ensuite, entraîner le classifieur sur les vecteurs prend quelques secondes.

Usage :
    python training/extract_embeddings.py              # tout le dataset
    python training/extract_embeddings.py --limit 200  # test rapide
Sortie : data/embeddings/{train,validation}.npz (X, label, generator)
"""

import argparse
import glob
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import CLIPVisionModelWithProjection

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.preprocessing import preprocess  # noqa: E402

CLIP_NAME = "openai/clip-vit-base-patch32"
RAW = ROOT / "data" / "raw" / "tiny-genimage" / "data"
OUT = ROOT / "data" / "embeddings"
BATCH = 64


def iter_rows(split: str, limit: int | None):
    """Lit les parquet morceau par morceau pour ne pas charger 8 Go en mémoire."""
    seen = 0
    for f in sorted(glob.glob(str(RAW / f"{split}-*.parquet"))):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=BATCH, columns=["image", "label", "generator"]):
            d = batch.to_pydict()
            for img, lab, gen in zip(d["image"], d["label"], d["generator"]):
                yield img["bytes"], lab, gen
                seen += 1
                if limit and seen >= limit:
                    return


def batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


@torch.inference_mode()
def embed_split(model, split: str, limit: int | None, pool: ThreadPoolExecutor) -> None:
    feats, labels, gens = [], [], []
    t0 = time.time()
    for rows in batched(iter_rows(split, limit), BATCH):
        pixels = np.stack(list(pool.map(preprocess, [r[0] for r in rows])))
        emb = model(pixel_values=torch.from_numpy(pixels)).image_embeds
        emb = torch.nn.functional.normalize(emb, dim=-1)  # vecteur de longueur 1
        feats.append(emb.numpy().astype(np.float32))
        labels += [r[1] for r in rows]
        gens += [r[2] for r in rows]
        n = len(labels)
        if n % (BATCH * 10) == 0:
            print(f"  {split}: {n} images, {n / (time.time() - t0):.1f} img/s", flush=True)
    elapsed = time.time() - t0
    X = np.concatenate(feats)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{split}.npz", X=X, label=np.array(labels), generator=np.array(gens))
    print(f"{split}: {len(X)} images en {elapsed / 60:.1f} min ({len(X) / elapsed:.1f} img/s) -> {X.shape}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="nombre max d'images par split (test)")
    args = p.parse_args()

    torch.set_num_threads(8)
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_NAME).eval()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for split in ("train", "validation"):
            embed_split(model, split, args.limit, pool)


if __name__ == "__main__":
    main()
