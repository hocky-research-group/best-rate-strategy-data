"""Crop the white margin off rendered PNGs in place, leaving a fixed border.

Usage:
    uv run python scripts/crop_png_whitespace.py <png> [<png> ...]
Each file is cropped to the bounding box of its non-white pixels plus the border and
overwritten; sizes before and after are printed.
"""
import sys

from PIL import Image, ImageChops

BORDER = 40  # pixels kept around the content


def crop(path: str) -> None:
    """Crop one PNG in place.

    Args:
        path (str): the PNG file to crop and overwrite.
    """
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box is None:
        raise SystemExit(f"{path}: blank image")
    l, t, r, b = box
    out = im.crop((max(l - BORDER, 0), max(t - BORDER, 0), min(r + BORDER, im.width), min(b + BORDER, im.height)))
    out.save(path)
    print(f"{path}: {im.size} -> {out.size}")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        crop(f)
