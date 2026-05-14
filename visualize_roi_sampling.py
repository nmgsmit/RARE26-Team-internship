import argparse
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

from roi_guidance import (
    DEFAULT_ROI_MAX_ASPECT_RATIO,
    compute_crop_window_from_roi,
    crop_image_to_roi,
    load_roi_records_from_json,
)


DEFAULT_ROI_JSON = "./checkpoints/roi_records/rois.json"
DEFAULT_OUTPUT_DIR = "outputs/roi_sampling_preview_new"


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
        default=12,
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
        "--max-aspect-ratio",
        type=float,
        default=None,
        help=(
            "Maximum ROI aspect ratio used before the shorter side is expanded. "
            "Defaults to the value stored in the ROI JSON metadata, or 1.5 if absent."
        ),
    )
    return parser.parse_args()


def load_roi_payload(roi_json_path):
    roi_records, metadata = load_roi_records_from_json(roi_json_path)
    return metadata, roi_records


def flatten_roi_entries(roi_records):
    flattened_entries = []
    for image_path, record in roi_records.items():
        roi_islands = list(record.get("roi_islands", []))
        if roi_islands:
            for island_record in roi_islands:
                entry_record = dict(island_record)
                entry_record.setdefault("source", record.get("source", "gradcam"))
                entry_record["parent_image_path"] = str(image_path)
                entry_record["parent_island_count"] = int(record.get("island_count", len(roi_islands)))
                flattened_entries.append((str(image_path), entry_record))
        else:
            entry_record = dict(record)
            entry_record["parent_image_path"] = str(image_path)
            entry_record["parent_island_count"] = 1
            flattened_entries.append((str(image_path), entry_record))
    return flattened_entries


def sanitize_path_arg(path_value):
    sanitized = str(path_value)
    for escaped_control_char in ("\r", "\n", "\t"):
        sanitized = sanitized.replace(escaped_control_char, "\\")
    return sanitized.strip().strip('"').strip("'")


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


def select_records(roi_entries, max_images):
    return list(roi_entries)[:max_images]


def get_font():
    return ImageFont.load_default()


def fit_square(image, size):
    return image.resize((size, size), Image.Resampling.BICUBIC)


def draw_text_block(draw, xy, text, fill, font):
    draw.multiline_text(xy, text, fill=fill, font=font, spacing=2)


def build_overlay_tile(image, roi_record, crop_bbox, tile_size, caption):
    image = image.convert("RGB")
    width, height = image.size
    source_bbox = roi_record.get("source_bbox", roi_record["bbox"])
    x0, y0, x1, y1 = [float(value) for value in source_bbox]
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


def build_crop_tile(image, roi_record, tile_size, context_scale, min_crop_scale, max_aspect_ratio, caption):
    crop = crop_image_to_roi(
        image=image,
        roi_record=roi_record,
        context_scale=context_scale,
        min_crop_scale=min_crop_scale,
        jitter_xy=(0.0, 0.0),
        max_aspect_ratio=max_aspect_ratio,
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


def build_source_roi_crop_tile(image, roi_record, tile_size, caption):
    source_bbox = roi_record.get("source_bbox", roi_record["bbox"])
    crop = crop_image_to_window(image=image, window_bbox=source_bbox).convert("RGB")
    tile = fit_square(crop, tile_size)

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
    island_index = int(record.get("island_index", 0)) + 1
    island_total = int(record.get("parent_island_count", record.get("island_count", 1)))
    return (
        f"{index:02d} {short_name}\n"
        f"isl={island_index}/{island_total} s={score:.3f} cov={coverage:.3f}"
    )


def main():
    args = parse_args()
    roi_json_path = Path(sanitize_path_arg(args.roi_json)).resolve()
    output_dir = Path(sanitize_path_arg(args.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata, roi_records = load_roi_payload(roi_json_path)
    roi_entries = flatten_roi_entries(roi_records)
    max_aspect_ratio = float(
        args.max_aspect_ratio
        if args.max_aspect_ratio is not None
        else metadata.get("roi_sampling_max_aspect_ratio", DEFAULT_ROI_MAX_ASPECT_RATIO)
    )
    max_images = args.grid_size * args.grid_size
    selected_records = select_records(roi_entries=roi_entries, max_images=max_images)

    if len(selected_records) < max_images:
        print(
            f"Only found {len(selected_records)} ROI entries; the remaining grid slots will be blank."
        )

    overlay_tiles = []
    crop_tiles = []
    source_roi_crop_tiles = []

    for index, (image_path_str, record) in enumerate(selected_records, start=1):
        image_path = resolve_image_path(image_path_str, roi_json_path)
        image = Image.open(image_path).convert("RGB")
        caption = build_caption(index, image_path_str, record)
        crop_bbox = compute_crop_window_from_roi(
            roi_record=record,
            context_scale=args.context_scale,
            min_crop_scale=args.min_crop_scale,
            jitter_xy=(0.0, 0.0),
            max_aspect_ratio=max_aspect_ratio,
        )

        overlay_tiles.append(
            build_overlay_tile(
                image=image,
                roi_record=record,
                crop_bbox=crop_bbox,
                tile_size=args.tile_size,
                caption=caption,
            )
        )
        crop_tiles.append(
            build_crop_tile(
                image=image,
                roi_record=record,
                tile_size=args.tile_size,
                context_scale=args.context_scale,
                min_crop_scale=args.min_crop_scale,
                max_aspect_ratio=max_aspect_ratio,
                caption=caption,
            )
        )
        source_roi_crop_tiles.append(
            build_source_roi_crop_tile(
                image=image,
                roi_record=record,
                tile_size=args.tile_size,
                caption=caption,
            )
        )

    input_size = metadata.get("input_size", "unknown")
    overlay_title = (
        f"Source ROI (yellow) + sampler crop window (cyan) | "
        f"selected={len(selected_records)} islands | order=json-top-down | input_size={input_size}"
    )
    crop_title = (
        f"ROI sampler crops | context_scale={args.context_scale} | "
        f"min_crop_scale={args.min_crop_scale} | max_aspect_ratio={max_aspect_ratio:.2f} | jitter=0.0"
    )
    source_roi_crop_title = (
        "Direct source-ROI crops | "
        "order=json-top-down"
    )

    overlay_grid = assemble_grid(overlay_tiles, args.grid_size, overlay_title)
    crop_grid = assemble_grid(crop_tiles, args.grid_size, crop_title)
    source_roi_crop_grid = assemble_grid(source_roi_crop_tiles, args.grid_size, source_roi_crop_title)

    overlay_output_path = output_dir / "roi_gradcam_overlay_grid.png"
    crop_output_path = output_dir / "roi_sampler_crop_grid.png"
    source_roi_crop_output_path = output_dir / "source_roi_crop_grid.png"
    overlay_grid.save(overlay_output_path)
    crop_grid.save(crop_output_path)
    source_roi_crop_grid.save(source_roi_crop_output_path)

    print(f"Saved Grad-CAM overlay grid to {overlay_output_path}")
    print(f"Saved ROI sampler crop grid to {crop_output_path}")
    print(f"Saved source ROI crop grid to {source_roi_crop_output_path}")


if __name__ == "__main__":
    main()
