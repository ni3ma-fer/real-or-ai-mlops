"""Télécharge Tiny-GenImage (Hugging Face) dans data/raw/tiny-genimage.

Usage : python scripts/download_data.py
Relançable : les fichiers déjà téléchargés sont ignorés.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "TheKernel01/Tiny-GenImage"
DEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "tiny-genimage"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {REPO_ID} vers {DEST} (~8,4 Go)...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DEST,
        allow_patterns=["data/*.parquet", "README.md"],
        max_workers=4,
    )
    files = sorted(DEST.glob("data/*.parquet"))
    total_gb = sum(f.stat().st_size for f in files) / 1e9
    print(f"Terminé : {len(files)} fichiers, {total_gb:.2f} Go.")


if __name__ == "__main__":
    main()
