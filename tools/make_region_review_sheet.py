#!/usr/bin/env python3
"""Create a visual review sheet for residual/panel regions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Make source/render/diff crops for panel-region review")
    ap.add_argument("source_image")
    ap.add_argument("rendered_image")
    ap.add_argument("regions_json")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--max-regions", type=int, default=24)
    ap.add_argument("--crop-scale", type=int, default=2)
    ap.add_argument("--pad", type=int, default=4)
    args = ap.parse_args()

    source = Image.open(args.source_image).convert("RGB")
    rendered = Image.open(args.rendered_image).convert("RGB")
    regions = _load_regions(args.regions_json)[:args.max_regions]
    sheet = make_sheet(
        source=source,
        rendered=rendered,
        regions=regions,
        crop_scale=max(1, args.crop_scale),
        pad=max(0, args.pad),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(json.dumps({
        "output": str(output),
        "regions": len(regions),
        "size": sheet.size,
    }, indent=2))


def make_sheet(
    *,
    source: Image.Image,
    rendered: Image.Image,
    regions: list[dict],
    crop_scale: int,
    pad: int,
) -> Image.Image:
    font = ImageFont.load_default()
    rows = []
    max_width = 1
    for region in regions:
        x0, y0, x1, y1 = _region_box(region, source.size, pad)
        src_crop = source.crop((x0, y0, x1, y1))
        ren_crop = rendered.crop((x0, y0, x1, y1))
        diff_crop = ImageChops.difference(src_crop, ren_crop)
        if crop_scale != 1:
            size = (src_crop.width * crop_scale, src_crop.height * crop_scale)
            src_crop = src_crop.resize(size, Image.Resampling.NEAREST)
            ren_crop = ren_crop.resize(size, Image.Resampling.NEAREST)
            diff_crop = diff_crop.resize(size, Image.Resampling.NEAREST)
        label = _label(region)
        row = _make_row(label, src_crop, ren_crop, diff_crop, font)
        rows.append(row)
        max_width = max(max_width, row.width)
    if not rows:
        return Image.new("RGB", (640, 120), "white")
    sheet = Image.new(
        "RGB",
        (max_width, sum(row.height for row in rows)),
        "white",
    )
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet


def _load_regions(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return list(data.get("panel_regions") or data.get("regions") or [])


def _region_box(
    region: dict,
    image_size: tuple[int, int],
    pad: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x0, y0, x1, y1 = [float(value) for value in region["bbox"]]
    return (
        max(0, int(round(x0)) - pad),
        max(0, int(round(y0)) - pad),
        min(width, int(round(x1)) + pad),
        min(height, int(round(y1)) + pad),
    )


def _make_row(
    label: str,
    source: Image.Image,
    rendered: Image.Image,
    diff: Image.Image,
    font: ImageFont.ImageFont,
) -> Image.Image:
    gap = 8
    label_w = 190
    label_h = max(source.height, 48)
    row_h = max(label_h, source.height, rendered.height, diff.height) + 8
    row_w = label_w + gap + source.width + gap + rendered.width + gap + diff.width
    row = Image.new("RGB", (row_w, row_h), "white")
    draw = ImageDraw.Draw(row)
    draw.rectangle((0, 0, row_w - 1, row_h - 1), outline=(210, 210, 210))
    draw.text((8, 8), label, fill=(20, 20, 20), font=font)
    x = label_w + gap
    for name, crop in (("source", source), ("render", rendered), ("diff", diff)):
        draw.text((x, 2), name, fill=(80, 80, 80), font=font)
        row.paste(crop, (x, 18))
        x += crop.width + gap
    return row


def _label(region: dict) -> str:
    bbox = ", ".join(str(int(round(float(v)))) for v in region.get("bbox", []))
    density = region.get("density")
    density_text = f"\ndensity={density}" if density is not None else ""
    return f"{region.get('id', 'region')}\n[{bbox}]{density_text}"


if __name__ == "__main__":
    main()
