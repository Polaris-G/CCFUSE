"""Pillow rendering for paired, post-NMS detection diagnostics.

The same fused predictions are overlaid on both sensor images for inspection.
The IR row is not a separate IR-only model evaluation.
"""
from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROLES = ("add", "cmfc", "full")
ROLE_NAMES = {"add": "Add", "cmfc": "CMFC-only", "full": "Full v2"}
COLORS = ((72, 190, 255), (255, 155, 65), (100, 225, 145),
          (190, 150, 255), (255, 115, 115))
YELLOW = (255, 232, 65)
GRAY = (165, 170, 180)
ASSOCIATED = (255, 75, 215)
BACKGROUND = (20, 24, 31)
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def _font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow versions before sized default fonts.
        return ImageFont.load_default()


def _xyxy(rows):
    values = np.asarray(rows, dtype=float).reshape(-1, 4)
    return np.concatenate((values[:, :2] - values[:, 2:] / 2,
                           values[:, :2] + values[:, 2:] / 2), axis=1)


def _candidate_scores(gt_box, predictions, names, matching_iou, focus_class, confusing_class):
    """Geometric candidate maxima, independent of the diagnostic assignment."""
    boxes = _xyxy(predictions[:, 1:5])
    target = _xyxy(np.asarray(gt_box).reshape(1, 4))[0]
    inter = np.maximum(np.minimum(boxes[:, 2:], target[2:])
                       - np.maximum(boxes[:, :2], target[:2]), 0).prod(1)
    union = (np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
             + np.maximum(target[2:] - target[:2], 0).prod() - inter)
    ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    output = []
    for name in (focus_class, confusing_class):
        if name not in names:
            continue
        mask = (predictions[:, 0] == names.index(name)) & (ious >= matching_iou)
        output.append(f"{name}={predictions[mask, 5].max():.3f}" if mask.any()
                      else f"{name}=none")
    return ", ".join(output)


def _crop_box(gt_box, size):
    width, height = size
    xc, yc, bw, bh = gt_box * np.array([width, height, width, height])
    crop_width = min(width, max(160, math.ceil(bw * 3)))
    crop_height = min(height, max(160, math.ceil(bh * 3)))
    x0 = max(0, min(width - crop_width, math.floor(xc - crop_width / 2)))
    y0 = max(0, min(height - crop_height, math.floor(yc - crop_height / 2)))
    return int(x0), int(y0), int(x0 + crop_width), int(y0 + crop_height)


def _text_bbox(draw, text, font):
    if hasattr(draw, "textbbox"):
        return draw.textbbox((0, 0), text, font=font)
    width, height = draw.textsize(text, font=font)
    return 0, 0, width, height


def _shorten(draw, text, font, width):
    text = str(text)
    if _text_bbox(draw, text, font)[2] <= width:
        return text
    while text and _text_bbox(draw, text + "...", font)[2] > width:
        text = text[:-1]
    return text + "..."


def _label(draw, xy, text, color, font, bounds):
    left, top, right, bottom = bounds
    text = _shorten(draw, text, font, max(1, right - left - 6))
    bbox = _text_bbox(draw, text, font)
    tw, th = bbox[2] - bbox[0] + 6, bbox[3] - bbox[1] + 5
    x = min(max(left, int(xy[0])), max(left, right - tw))
    y = min(max(top, int(xy[1]) - th), max(top, bottom - th))
    draw.rectangle((x, y, x + tw, y + th), fill=(8, 10, 15))
    draw.text((x + 3, y + 2 - bbox[1]), text, fill=color, font=font)


def _panel(image, gt, gt_index, pred, result, names, crop, confidence,
           matching_iou, sensor, role, confusing_class):
    panel_width, header_height, view_height = 460, 112, 330
    panel = Image.new("RGB", (panel_width, header_height + view_height), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    title_font, font, small_font = _font(21, True), _font(16), _font(14)
    draw.text((12, 8), f"{sensor} | {ROLE_NAMES.get(role, 'Ground truth')}",
              fill="white", font=title_font)
    gt_class = names[int(gt[gt_index, 0])]
    associated_source = -1
    if role is None:
        header = [f"Target GT #{gt_index} (zero-based): {gt_class}",
                  "Yellow = selected GT; gray = neighboring GT",
                  "Same crop and shared annotation coordinates"]
    else:
        status = result["gt_status"][gt_index]
        p = result["gt_pred_indices"][gt_index]
        header = [f"Status: {status}"]
        if p >= 0:
            associated_source = int(result["pred_source_indices"][p])
            associated = result["retained_predictions"][p]
            name = names[int(associated[0])]
            iou = float(result["ious"][gt_index, p])
            header.append(f"Assoc: {name}  score={associated[5]:.3f}  IoU={iou:.3f}")
        else:
            header.append("Assoc: none")
        header.append("Export maxima: " + _candidate_scores(
            gt[gt_index, 1:5], pred, names, matching_iou, gt_class, confusing_class))
    for index, line in enumerate(header):
        draw.text((12, 39 + 23 * index),
                  _shorten(draw, line, font, panel_width - 24),
                  fill=(222, 228, 236), font=font)

    crop_image = image.crop(crop)
    scale = min(panel_width / crop_image.width, view_height / crop_image.height)
    resized = crop_image.resize((max(1, round(crop_image.width * scale)),
                                 max(1, round(crop_image.height * scale))),
                                RESAMPLE)
    offset_x = (panel_width - resized.width) // 2
    offset_y = header_height + (view_height - resized.height) // 2
    panel.paste(resized, (offset_x, offset_y))
    bounds = (offset_x, offset_y, offset_x + resized.width - 1,
              offset_y + resized.height - 1)
    width, height = image.size

    def to_panel(box):
        x1, y1, x2, y2 = box * np.array([width, height, width, height])
        if x2 < crop[0] or x1 >= crop[2] or y2 < crop[1] or y1 >= crop[3]:
            return None
        x1, x2 = np.clip([x1, x2], crop[0], crop[2])
        y1, y2 = np.clip([y1, y2], crop[1], crop[3])
        return (max(bounds[0], min(bounds[2], offset_x + (x1 - crop[0]) * scale)),
                max(bounds[1], min(bounds[3], offset_y + (y1 - crop[1]) * scale)),
                max(bounds[0], min(bounds[2], offset_x + (x2 - crop[0]) * scale)),
                max(bounds[1], min(bounds[3], offset_y + (y2 - crop[1]) * scale)))

    gt_boxes = _xyxy(gt[:, 1:5])
    for index, box in enumerate(gt_boxes):
        location = to_panel(box)
        if index != gt_index and location is not None:
            draw.rectangle(location, outline=GRAY, width=1)

    if pred is not None:
        # Draw low scores first, retaining all visible predictions above the cutoff.
        order = sorted(np.flatnonzero(pred[:, 5] >= confidence),
                       key=lambda index: (index == associated_source, pred[index, 5]))
        pred_boxes = _xyxy(pred[:, 1:5])
        for index in order:
            location = to_panel(pred_boxes[index])
            if location is None:
                continue
            cls = int(pred[index, 0])
            selected = index == associated_source
            color = ASSOCIATED if selected else COLORS[cls % len(COLORS)]
            draw.rectangle(location, outline=color, width=5 if selected else 2)
            label = f"{'Assoc ' if selected else ''}{names[cls]} {pred[index, 5]:.2f}"
            _label(draw, location[:2], label, color, small_font, bounds)

    target = to_panel(gt_boxes[gt_index])
    draw.rectangle(target, outline=YELLOW, width=2)
    # Put the target label below its box to avoid hiding an associated prediction label.
    _label(draw, (target[0], min(bounds[3], target[3] + 23)),
           f"GT #{gt_index}: {gt_class}", YELLOW, small_font, bounds)
    return panel


def render_case(rgb_path, ir_path, gt, gt_index, predictions, results, names,
                output_path, confidence, matching_iou, title, confusing_class='car'):
    """Render an eight-panel PNG without changing inputs or replacing a file.

    ``results`` must contain the same-confidence/IoU ``analyze_image`` results
    for each of add/cmfc/full, computed against the *complete* image GT list.
    Candidate maxima use all saved post-NMS predictions at the chosen matching
    IoU and are not one-to-one matches or raw-network class probabilities.
    """
    rgb_path, ir_path, output_path = map(Path, (rgb_path, ir_path, output_path))
    if output_path.resolve() in {rgb_path.resolve(), ir_path.resolve()}:
        raise ValueError("Output must not replace an input image")
    if output_path.suffix.lower() != ".png":
        raise ValueError("Comparison output must be a PNG")
    if output_path.exists():
        raise FileExistsError(f"Comparison output already exists: {output_path}")
    gt = np.asarray(gt, dtype=float).reshape(-1, 5)
    gt_index = int(gt_index)
    if not 0 <= gt_index < len(gt):
        raise ValueError("GT index outside the complete image GT list")
    if not 0 <= confidence <= 1 or not 0 < matching_iou <= 1:
        raise ValueError("Invalid confidence or matching IoU")
    predictions = {role: np.asarray(predictions[role], dtype=float).reshape(-1, 6)
                   for role in ROLES}
    for role in ROLES:
        if len(results[role]["gt_status"]) != len(gt):
            raise ValueError("Results must be computed against the complete image GT")
    with Image.open(rgb_path) as source:
        rgb = source.convert("RGB")
    with Image.open(ir_path) as source:
        ir = source.convert("RGB")
    if rgb.size != ir.size:
        raise ValueError(f"RGB/IR dimensions differ: {rgb.size} vs {ir.size}")
    crop = _crop_box(gt[gt_index, 1:5], rgb.size)
    gap, top_height, footer_height = 8, 80, 70
    panel_width, panel_height = 460, 442
    canvas = Image.new("RGB", (4 * panel_width + 5 * gap,
                               top_height + 2 * panel_height + gap + footer_height),
                       (11, 14, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), _shorten(draw, title, _font(23, True), canvas.width - 24),
              font=_font(23, True), fill="white")
    draw.text((12, 42),
              f"conf >= {confidence:g} | matching IoU >= {matching_iou:g} | "
              f"GT index is zero-based | crop={crop} | display uses shared GT coordinates",
              font=_font(17), fill=(200, 210, 223))
    for row, (sensor, image) in enumerate((("RGB", rgb), ("IR", ir))):
        for column, role in enumerate((None,) + ROLES):
            panel = _panel(image, gt, gt_index,
                           predictions[role] if role else None,
                           results[role] if role else None, names, crop,
                           confidence, matching_iou, sensor, role, confusing_class)
            canvas.paste(panel, (gap + column * (panel_width + gap),
                                top_height + row * (panel_height + gap)))
    y = top_height + 2 * panel_height + gap + 7
    draw.text((12, y),
              "Yellow: selected GT | Gray: other GT | Magenta: associated prediction "
              "(candidate for non-TP/non-classification statuses)",
              font=_font(16), fill=(220, 227, 238))
    draw.text((12, y + 23),
              "Export maxima: saved boxes with IoU >= matching threshold, independent of "
              "matching/conf filter. Both rows show the SAME fused predictions.",
              font=_font(16), fill=(200, 210, 223))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        canvas.save(stream, format="PNG")
