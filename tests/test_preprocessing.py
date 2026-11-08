import io

import numpy as np
import pytest
from PIL import Image

from shared.preprocessing import IMAGE_SIZE, load_image, preprocess


def image_bytes(fmt: str, size=(300, 200), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color=128 if mode == "L" else (200, 100, 50)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize("fmt,mode", [("JPEG", "RGB"), ("PNG", "RGB"), ("PNG", "RGBA"), ("JPEG", "L"), ("WEBP", "RGB")])
def test_shape_and_type(fmt, mode):
    x = preprocess(image_bytes(fmt, mode=mode))
    assert x.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert x.dtype == np.float32


@pytest.mark.parametrize("size", [(64, 64), (1024, 1024), (2000, 300), (300, 2000)])
def test_any_size_becomes_224(size):
    assert preprocess(image_bytes("PNG", size=size)).shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_png_is_reencoded_as_jpeg():
    """Le correctif du biais de format : une image PNG ressort avec les pixels d'un JPEG qualité 90."""
    img = Image.effect_noise((128, 128), 50).convert("RGB")
    png, jpg = io.BytesIO(), io.BytesIO()
    img.save(png, format="PNG")
    img.save(jpg, format="JPEG", quality=90)
    out = np.asarray(load_image(png.getvalue()))
    assert np.array_equal(out, np.asarray(Image.open(jpg).convert("RGB")))
    assert not np.array_equal(out, np.asarray(img))  # le PNG d'origine a bien été transformé


def test_same_pixels_same_result_whatever_the_format():
    """Une même image en PNG ou en JPEG qualité max donne presque le même tenseur."""
    img = Image.effect_noise((256, 256), 50).convert("RGB")
    png, jpg = io.BytesIO(), io.BytesIO()
    img.save(png, format="PNG")
    img.save(jpg, format="JPEG", quality=100)
    diff = np.abs(preprocess(png.getvalue()) - preprocess(jpg.getvalue())).mean()
    assert diff < 0.05


def test_not_an_image():
    with pytest.raises(Exception):
        preprocess(b"ceci n'est pas une image")
