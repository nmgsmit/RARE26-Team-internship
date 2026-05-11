import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from roi_guidance import crop_image_to_roi


DEFAULT_ROI_JSON = "../gastronet_pretrain_suppro_ROIpretrained_train_rois.json"
DEFAULT_OUTPUT_DIR = "outputs/roi_sampling_preview"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved Grad-CAM ROI guidance and deterministic ROI-sampler crops "
            "for positive training images."
        )
    )
    parser.add_argument(
        "--roi-json",
        type=str,
        default=DEFAULT_ROI_JSON,
        help="Path to the saved ROI JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the PNG grids will be saved.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=5,
        help="Number of rows and columns in the preview grid.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=336,
        help="Square size in pixels for each rendered tile.",
    )
    parser.add_argument(
        "--context-scale",
        type=float,
        default=2.0,
        help="ROI sampler context scale used when generating the crop preview.",
    )
    parser.add_argument(
        "--min-crop-scale",
        type=float,
        default=0.4,
        help="ROI sampler minimum normalized crop scale used for the crop preview.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("score", "path", "smallest_crop", "largest_crop"),
        default="smallest_crop",
        help="How to choose which ROI-positive images are shown.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Seed used for reproducible random crop placement.",
    )
    return parser.parse_args()


def load_roi_payload(roi_json_path):
    with roi_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "roi_records" in payload:
        metadata = dict(payload.get("metadata", {}))
        roi_records = dict(payload.get("roi_records", {}))
    else:
        metadata = {}
        roi_records = dict(payload)
    return metadata, roi_records


def resolve_image_path(image_path_str, roi_json_path):
    raw_path = Path(image_path_str)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                Path.cwd() / raw_path,
                roi_json_path.parent / raw_path,
                Path(__file__).resolve().parent / raw_path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    candidate_text = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not resolve image path '{image_path_str}'. Tried:\n{candidate_text}"
    )


def compute_crop_window(bbox, context_scale, min_crop_scale, jitter_xy=(0.0, 0.0)):
    x0, y0, x1, y1 = [float(value) for value in bbox]
    x0 = min(max(x0, 0.0), 1.0)
    y0 = min(max(y0, 0.0), 1.0)
    x1 = min(max(x1, x0 + 1e-6), 1.0)
    y1 = min(max(y1, y0 + 1e-6), 1.0)

    roi_width = x1 - x0
    roi_height = y1 - y0
    crop_width = min(1.0, max(float(min_crop_scale), roi_width * float(context_scale)))
    crop_height = min(1.0, max(float(min_crop_scale), roi_height * float(context_scale)))

    center_x = 0.5 * (x0 + x1) + float(jitter_xy[0]) * 0.5 * crop_width
    center_y = 0.5 * (y0 + y1) + float(jitter_xy[1]) * 0.5 * crop_height

    left = min(max(center_x - 0.5 * crop_width, 0.0), 1.0 - crop_width)
    top = min(max(center_y - 0.5 * crop_height, 0.0), 1.0 - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return (left, top, right, bottom)


def select_records(roi_records, max_images, selection_mode, context_scale, min_crop_scale):
    items = list(roi_records.items())
    if selection_mode == "score":
        items.sort(
            key=lambda item: (
                -float(item[1].get("score", 0.0)),
                str(item[0]),
            )
        )
    elif selection_mode == "smallest_crop":
        items.sort(
            key=lambda item: (
                (compute_crop_window(item[1]["bbox"], context_scale, min_crop_scale)[2]
                 - compute_crop_window(item[1]["bbox"], context_scale, min_crop_scale)[0])
                * (compute_crop_window(item[1]["bbox"], context_scale, min_crop_scale)[3]
                   - compute_crop_window(item[1]["bbox"], context_scale, min_crop_scale)[1]),
                float(item[1].get("coverage", 0.0)),
                str(item[0]),
            )
        )
    elif selection_mode == "largest_crop":
        items.sort(
            key=lambda item: (
                -(
                    (compute_crop_window(item[1]["bbox"], context_scale, min_crop_scale)[2]
                     - compute_crop_window(item[1]["bbox"], context_scale, min_crop_scale)[0])
                    * (compute_crop_window(item[1]["bbox"], context_scale, min_crop_scale)[3]
                       - compute_crop_window(item[1]["bbox"], context_scale, min_crop_scale)[1])
                ),
                str(item[0]),
            )
        )
    else:
        items.sort(key=lambda item: str(item[0]))
    return items[:max_images]


def get_font():
    return ImageFont.load_default()


def fit_square(image, size):
    return image.resize((size, size), Image.Resampling.BICUBIC)


def draw_text_block(draw, xy, text, fill, font):
    draw.multiline_text(xy, text, fill=fill, font=font, spacing=2)


def build_overlay_tile(image, bbox, crop_bbox, tile_size, caption):
    image = image.convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = [float(value) for value in bbox]
    cx0, cy0, cx1, cy1 = [float(value) for value in crop_bbox]

    left = max(0, min(width - 1, int(round(x0 * width))))
    top = max(0, min(height - 1, int(round(y0 * height))))
    right = max(left + 1, min(width, int(round(x1 * width))))
    bottom = max(top + 1, min(height, int(round(y1 * height))))
    crop_left = max(0, min(width - 1, int(round(cx0 * width))))
    crop_top = max(0, min(height - 1, int(round(cy0 * height))))
    crop_right = max(crop_left + 1, min(width, int(round(cx1 * width))))
    crop_bottom = max(crop_top + 1, min(height, int(round(cy1 * height))))

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [left, top, right, bottom],
        fill=ImageColor.getrgb("#ff5a36") + (80,),
        outline=ImageColor.getrgb("#ffdf5d") + (255,),
        width=max(2, width // 140),
    )
    overlay_draw.rectangle(
        [crop_left, crop_top, crop_right, crop_bottom],
        outline=ImageColor.getrgb("#00d4ff") + (255,),
        width=max(2, width // 120),
    )
    composite = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    tile = fit_square(composite, tile_size)

    caption_height = 34
    canvas = Image.new("RGB", (tile_size, tile_size + caption_height), "white")
    canvas.paste(tile, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw_text_block(draw, (8, tile_size + 6), caption, fill="black", font=get_font())
    return canvas


def build_crop_tile(image, bbox, tile_size, context_scale, min_crop_scale, caption):
    crop = crop_image_to_roi(
        image=image,
        bbox=bbox,
        context_scale=context_scale,
        min_crop_scale=min_crop_scale,
        jitter_xy=(0.0, 0.0),
    ).convert("RGB")
    tile = fit_square(crop, tile_size)

    caption_height = 34
    canvas = Image.new("RGB", (tile_size, tile_size + caption_height), "white")
    canvas.paste(tile, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw_text_block(draw, (8, tile_size + 6), caption, fill="black", font=get_font())
    return canvas


def crop_image_to_window(image, window_bbox):
    width, height = image.size
    left, top, right, bottom = [float(value) for value in window_bbox]

    left_px = max(0, min(width - 1, int(round(left * width))))
    top_px = max(0, min(height - 1, int(round(top * height))))
    right_px = max(left_px + 1, min(width, int(round(right * width))))
    bottom_px = max(top_px + 1, min(height, int(round(bottom * height))))
    return image.crop((left_px, top_px, right_px, bottom_px))


def sample_random_crop_window(crop_bbox, rng):
    left, top, right, bottom = [float(value) for value in crop_bbox]
    crop_width = right - left
    crop_height = bottom - top

    max_left = max(0.0, 1.0 - crop_width)
    max_top = max(0.0, 1.0 - crop_height)
    random_left = rng.uniform(0.0, max_left) if max_left > 0.0 else 0.0
    random_top = rng.uniform(0.0, max_top) if max_top > 0.0 else 0.0
    return (
        random_left,
        random_top,
        random_left + crop_width,
        random_top + crop_height,
    )


def build_random_crop_tile(image, random_crop_bbox, tile_size, caption):
    crop = crop_image_to_window(image=image, window_bbox=random_crop_bbox).convert("RGB")
    tile = fit_square(crop, tile_size)

    caption_height = 34
    canvas = Image.new("RGB", (tile_size, tile_size + caption_height), "white")
    canvas.paste(tile, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw_text_block(draw, (8, tile_size + 6), caption, fill="black", font=get_font())
    return canvas


def build_masked_roi_tile(image, bbox, tile_size, caption):
    image = image.convert("RGB")
    width, height = image.size
    x0, y0, x1, y1 = [float(value) for value in bbox]

    left = max(0, min(width - 1, int(round(x0 * width))))
    top = max(0, min(height - 1, int(round(y0 * height))))
    right = max(left + 1, min(width, int(round(x1 * width))))
    bottom = max(top + 1, min(height, int(round(y1 * height))))

    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    draw.rectangle([left, top, right, bottom], fill="black")
    tile = fit_square(masked, tile_size)

    caption_height = 34
    canvas = Image.new("RGB", (tile_size, tile_size + caption_height), "white")
    canvas.paste(tile, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw_text_block(draw, (8, tile_size + 6), caption, fill="black", font=get_font())
    return canvas


def assemble_grid(tiles, grid_size, title):
    if not tiles:
        raise ValueError("No tiles were provided for the grid.")

    tile_width, tile_height = tiles[0].size
    title_height = 58
    gutter = 10
    grid_width = grid_size * tile_width + (grid_size - 1) * gutter
    grid_height = grid_size * tile_height + (grid_size - 1) * gutter

    canvas = Image.new("RGB", (grid_width, title_height + grid_height), "#f7f3ec")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, grid_width, title_height], fill="#1d2a36")
    draw.text((16, 18), title, fill="white", font=get_font())

    blank_tile = Image.new("RGB", (tile_width, tile_height), "#ddd8cf")
    grid_tiles = list(tiles)
    while len(grid_tiles) < grid_size * grid_size:
        grid_tiles.append(blank_tile)

    for index, tile in enumerate(grid_tiles[: grid_size * grid_size]):
        row = index // grid_size
        col = index % grid_size
        x = col * (tile_width + gutter)
        y = title_height + row * (tile_height + gutter)
        canvas.paste(tile, (x, y))

    return canvas


def build_caption(index, image_path, record):
    image_name = Path(image_path).stem
    short_name = image_name[:14]
    score = float(record.get("score", 0.0))
    coverage = float(record.get("coverage", 0.0))
    return f"{index:02d} {short_name}\ns={score:.3f} cov={coverage:.3f}"


def main():
    args = parse_args()
    roi_json_path = Path(args.roi_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, roi_records = load_roi_payload(roi_json_path)
    max_images = args.grid_size * args.grid_size
    selected_records = select_records(
        roi_records=roi_records,
        max_images=max_images,
        selection_mode=args.selection_mode,
        context_scale=args.context_scale,
        min_crop_scale=args.min_crop_scale,
    )

    if len(selected_records) < max_images:
        print(
            f"Only found {len(selected_records)} ROI records; the remaining grid slots will be blank."
        )

    overlay_tiles = []
    crop_tiles = []
    random_crop_tiles = []
    masked_roi_tiles = []
    rng = random.Random(args.random_seed)

    for index, (image_path_str, record) in enumerate(selected_records, start=1):
        image_path = resolve_image_path(image_path_str, roi_json_path)
        image = Image.open(image_path).convert("RGB")
        caption = build_caption(index, image_path_str, record)
        crop_bbox = compute_crop_window(
            bbox=record["bbox"],
            context_scale=args.context_scale,
            min_crop_scale=args.min_crop_scale,
            jitter_xy=(0.0, 0.0),
        )
        random_crop_bbox = sample_random_crop_window(crop_bbox=crop_bbox, rng=rng)

        overlay_tiles.append(
            build_overlay_tile(
                image=image,
                bbox=record["bbox"],
                crop_bbox=crop_bbox,
                tile_size=args.tile_size,
                caption=caption,
            )
        )
        crop_tiles.append(
            build_crop_tile(
                image=image,
                bbox=record["bbox"],
                tile_size=args.tile_size,
                context_scale=args.context_scale,
                min_crop_scale=args.min_crop_scale,
                caption=caption,
            )
        )
        random_crop_tiles.append(
            build_random_crop_tile(
                image=image,
                random_crop_bbox=random_crop_bbox,
                tile_size=args.tile_size,
                caption=caption,
            )
        )
        masked_roi_tiles.append(
            build_masked_roi_tile(
                image=image,
                bbox=record["bbox"],
                tile_size=args.tile_size,
                caption=caption,
            )
        )

    input_size = metadata.get("input_size", "unknown")
    overlay_title = (
        f"Grad-CAM ROI (yellow) + sampler crop window (cyan) | "
        f"selected={len(selected_records)} | mode={args.selection_mode} | input_size={input_size}"
    )
    crop_title = (
        f"ROI sampler crops | context_scale={args.context_scale} | "
        f"min_crop_scale={args.min_crop_scale} | jitter=0.0"
    )
    random_crop_title = (
        f"Random crops with ROI-matched size | seed={args.random_seed} | "
        f"context_scale={args.context_scale} | min_crop_scale={args.min_crop_scale}"
    )
    masked_roi_title = "Original images with ROI masked out by black box"

    overlay_grid = assemble_grid(overlay_tiles, args.grid_size, overlay_title)
    crop_grid = assemble_grid(crop_tiles, args.grid_size, crop_title)
    random_crop_grid = assemble_grid(random_crop_tiles, args.grid_size, random_crop_title)
    masked_roi_grid = assemble_grid(masked_roi_tiles, args.grid_size, masked_roi_title)

    overlay_output_path = output_dir / "roi_gradcam_overlay_grid.png"
    crop_output_path = output_dir / "roi_sampler_crop_grid.png"
    random_crop_output_path = output_dir / "random_sampler_crop_grid.png"
    masked_roi_output_path = output_dir / "roi_masked_black_box_grid.png"
    overlay_grid.save(overlay_output_path)
    crop_grid.save(crop_output_path)
    random_crop_grid.save(random_crop_output_path)
    masked_roi_grid.save(masked_roi_output_path)

    print(f"Saved Grad-CAM overlay grid to {overlay_output_path}")
    print(f"Saved ROI sampler crop grid to {crop_output_path}")
    print(f"Saved random crop grid to {random_crop_output_path}")
    print(f"Saved ROI-masked black-box grid to {masked_roi_output_path}")


if __name__ == "__main__":
    main()
