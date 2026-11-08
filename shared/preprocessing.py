"""Préparation d'une image avant CLIP, partagée entre l'entraînement et l'API.

Le même code tourne des deux côtés : si l'API préparait les images autrement
que l'entraînement, le modèle recevrait en production des entrées qu'il n'a
jamais vues (« training/serving skew »).

Dépend seulement de PIL et numpy, pour que l'image Docker de l'API reste légère.
"""

import io

import numpy as np
from PIL import Image

IMAGE_SIZE = 224
JPEG_QUALITY = 90
# Moyenne et écart-type utilisés par OpenAI pour entraîner CLIP.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def load_image(data: bytes) -> Image.Image:
    """Ouvre une image et la réencode en JPEG.

    Dans Tiny-GenImage, les vraies photos sont en JPEG et les images IA en PNG.
    Sans ce réencodage, le modèle pourrait reconnaître le format au lieu du contenu.
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def to_tensor(img: Image.Image) -> np.ndarray:
    """Redimensionne (petit côté à 224), recadre au centre et normalise. Retourne (3, 224, 224)."""
    w, h = img.size
    scale = IMAGE_SIZE / min(w, h)
    img = img.resize((max(IMAGE_SIZE, round(w * scale)), max(IMAGE_SIZE, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return arr.transpose(2, 0, 1)


def preprocess(data: bytes) -> np.ndarray:
    """Octets d'une image -> tableau (3, 224, 224) prêt pour CLIP."""
    return to_tensor(load_image(data))
