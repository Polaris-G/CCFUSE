"""CPU-only diagnostics from completed test.py six-column TXT exports.

No model inference, AP recomputation, or changes to input files. Matches are
diagnostic: same-class IoU-greedy pairs first, then cross-class pairs. They do
not reproduce test.py's confidence-ordered AP matching or F1-selected P/R.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


def box_iou(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    inter = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:])
                       - np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2), axis=1)


def read_labels(path, columns, nc, missing_ok=False):
    path = Path(path)
    if columns not in (5, 6):
        raise ValueError('Expected 5 GT columns or 6 prediction columns')
    if not path.is_file():
        if missing_ok and not path.exists():
            return np.empty((0, columns), dtype=float)
        if not path.exists():
            raise FileNotFoundError(f'Missing label file: {path}')
        raise ValueError(f'Missing label file: {path}')
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != columns:
            raise ValueError(f'{path}:{number}: expected {columns} columns, found {len(fields)}')
        try:
            row = [float(v) for v in fields]
        except ValueError as exc:
            raise ValueError(f'{path}:{number}: non-numeric value') from exc
        rows.append(row)
    values = np.asarray(rows, dtype=float).reshape(-1, columns)
    if not len(values):
        return values
    classes, boxes = values[:, 0], values[:, 1:5]
    if not np.isfinite(values).all():
        raise ValueError(f'{path}: non-finite value')
    if np.any(classes != np.floor(classes)) or np.any((classes < 0) | (classes >= nc)):
        raise ValueError(f'{path}: invalid class ID')
    # Clipping padding-only detections to the original image can produce zero-area
    # predictions. Keep those as false positives; zero-area GT is invalid.
    if np.any(boxes < 0) or np.any(boxes > 1) or (columns == 5 and np.any(boxes[:, 2:] <= 0)):
        raise ValueError(f'{path}: invalid normalized xywh')
    corners = xywh_to_xyxy(boxes)
    if np.any(corners < -1e-5) or np.any(corners > 1 + 1e-5):
        raise ValueError(f'{path}: box outside image bounds')
    if columns == 6 and np.any((values[:, 5] < 0) | (values[:, 5] > 1)):
        raise ValueError(f'{path}: confidence outside [0,1]')
    if len(np.unique(values, axis=0)) != len(values):
        raise ValueError(f'{path}: duplicate rows; check repeated --exist-ok exports or duplicate GT')
    return values


def greedy_pairs(ious, eligible):
    gt_ids, pred_ids = np.where(eligible)
    order = np.lexsort((pred_ids, gt_ids, -ious[gt_ids, pred_ids]))
    used_gt, used_pred, pairs = set(), set(), []
    for idx in order:
        g, p = int(gt_ids[idx]), int(pred_ids[idx])
        if g not in used_gt and p not in used_pred:
            pairs.append((g, p))
            used_gt.add(g)
            used_pred.add(p)
    return pairs


def analyze_image(gt, pred, nc, conf, iou_threshold, localization_iou=0.1):
    if not (0 <= conf <= 1 and 0 < localization_iou < iou_threshold <= 1):
        raise ValueError('Require conf in [0,1] and 0 < localization IoU < matching IoU <= 1')
    gt = np.asarray(gt, dtype=float).reshape(-1, 5)
    pred = np.asarray(pred, dtype=float).reshape(-1, 6)
    source_indices = np.flatnonzero(pred[:, 5] >= conf)
    pred = pred[source_indices]
    gcls, pcls = gt[:, 0].astype(int), pred[:, 0].astype(int)
    ious = box_iou(xywh_to_xyxy(gt[:, 1:5]), xywh_to_xyxy(pred[:, 1:5]))
    same = gcls[:, None] == pcls[None, :]
    matched_gt, matched_pred = set(), set()
    gt_status, pred_status = [''] * len(gt), [''] * len(pred)
    gt_pred_indices = [-1] * len(gt)
    matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)

    for g, p in greedy_pairs(ious, same & (ious >= iou_threshold)):
        matched_gt.add(g)
        matched_pred.add(p)
        gt_status[g] = pred_status[p] = 'tp'
        gt_pred_indices[g] = p
        matrix[pcls[p], gcls[g]] += 1

    remaining = ious >= iou_threshold
    if matched_gt:
        remaining[list(matched_gt), :] = False
    if matched_pred:
        remaining[:, list(matched_pred)] = False
    for g, p in greedy_pairs(ious, remaining):
        matched_gt.add(g)
        matched_pred.add(p)
        gt_status[g] = pred_status[p] = 'classification'
        gt_pred_indices[g] = p
        matrix[pcls[p], gcls[g]] += 1

    for g in range(len(gt)):
        if g in matched_gt:
            continue
        candidates = np.flatnonzero(same[g])
        best_same = int(candidates[np.argmax(ious[g, candidates])]) if len(candidates) else -1
        best_any = int(np.argmax(ious[g])) if len(pred) else -1
        same_iou = float(ious[g, best_same]) if best_same >= 0 else 0
        any_iou = float(ious[g, best_any]) if best_any >= 0 else 0
        if same_iou >= iou_threshold:
            kind, associated = 'assignment_conflict', best_same
        elif same_iou >= localization_iou:
            kind, associated = 'localization_candidate', best_same
        elif any_iou >= localization_iou:
            kind, associated = 'class_localization_candidate', best_any
        else:
            kind, associated = 'no_overlap_candidate', best_any if any_iou > 0 else -1
        gt_status[g], gt_pred_indices[g] = kind, associated
        matrix[nc, gcls[g]] += 1

    for p in range(len(pred)):
        if p in matched_pred:
            continue
        same_gt = np.flatnonzero(same[:, p])
        if len(same_gt) and np.max(ious[same_gt, p]) >= iou_threshold:
            pred_status[p] = 'duplicate'
        elif len(gt) and np.max(ious[:, p]) >= localization_iou:
            pred_status[p] = 'other_overlap'
        else:
            pred_status[p] = 'background'
        matrix[pcls[p], nc] += 1
    return dict(confusion=matrix, gt_status=gt_status, gt_pred_indices=gt_pred_indices,
                pred_status=pred_status, retained_predictions=pred,
                pred_source_indices=source_indices, ious=ious)


def write_csv(path, headers, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def associated_gt(result, gt, pred_index):
    """Keep error examples tied to the actual match, not an unrelated higher IoU."""
    kind = result['pred_status'][pred_index]
    if kind in ('tp', 'classification'):
        return next(g for g, p in enumerate(result['gt_pred_indices'])
                    if p == pred_index and result['gt_status'][g] in ('tp', 'classification'))
    if not len(gt):
        return -1
    candidates = np.arange(len(gt))
    if kind == 'duplicate':
        predicted_class = result['retained_predictions'][pred_index, 0]
        candidates = candidates[gt[:, 0] == predicted_class]
    if not len(candidates):
        return -1
    g = int(candidates[np.argmax(result['ious'][candidates, pred_index])])
    return g if result['ious'][g, pred_index] > 0 else -1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/multispectral/DroneVehicle.yaml')
    parser.add_argument('--pred-dir', required=True, help='Completed test.py output labels directory')
    parser.add_argument('--output', required=True, help='New output directory; never overwrite existing reports')
    parser.add_argument('--conf', type=float, nargs='+', default=[0.05, 0.25, 0.5])
    parser.add_argument('--match-iou', type=float, nargs='+', default=[0.5, 0.75])
    parser.add_argument('--localization-iou', type=float, default=0.1)
    parser.add_argument('--export-conf', type=float, default=0.001,
                        help='Declared original export confidence floor; not inferred from TXT')
    parser.add_argument('--focus', nargs='+', default=['bus', 'van'])
    parser.add_argument('--max-examples', type=int, default=20, help='Per class/category/threshold, in filename order')
    parser.add_argument('--expected-images', type=int)
    parser.add_argument('--expected-labels', type=int)
    parser.add_argument('--allow-missing-predictions', action='store_true',
                        help='Absent TXT means zero predictions; only use after confirming complete evaluation')
    args = parser.parse_args()
    if not 0 <= args.export_conf <= 1 or args.max_examples < 0:
        parser.error('Invalid export confidence or max-examples')
    if any(not args.export_conf <= c <= 1 for c in args.conf):
        parser.error('Diagnostic confidence must be >= declared export confidence and <= 1')
    if any(not 0 < args.localization_iou < i <= 1 for i in args.match_iou):
        parser.error('Require 0 < localization IoU < each matching IoU <= 1')
    if len(set(args.conf)) != len(args.conf) or len(set(args.match_iou)) != len(args.match_iou):
        parser.error('Repeated thresholds are not supported')

    data_path = Path(args.data).resolve()
    data = yaml.safe_load(data_path.read_text(encoding='utf-8'))
    names = data['names']
    nc = int(data['nc'])
    if not isinstance(names, list) or len(names) != nc or len(set(names)) != nc:
        raise ValueError('Expected unique data names list matching nc')
    if any(name not in names for name in args.focus):
        raise ValueError('Unknown focus class')
    if not isinstance(data['val_rgb'], str):
        raise ValueError('This tool expects val_rgb to be a single image directory')
    image_root = Path(data['val_rgb']).expanduser().resolve()
    if not image_root.is_dir():
        raise ValueError(f'Image directory missing (run from repository root): {image_root}')
    parts = list(image_root.parts)
    if 'images' not in parts:
        raise ValueError('Cannot infer GT path: image directory needs an images path component')
    parts[parts.index('images')] = 'labels'
    gt_root = Path(*parts)
    pred_root, output = Path(args.pred_dir).resolve(), Path(args.output).resolve()
    if not gt_root.is_dir() or not pred_root.is_dir():
        raise ValueError(f'Missing GT or prediction directory: {gt_root}, {pred_root}')
    if output.exists():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    if any(output == root or root in output.parents for root in (image_root, gt_root, pred_root)):
        raise ValueError('Output must be outside all input directories')
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    images = sorted(p for p in image_root.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
    if not images:
        raise ValueError('No validation images found')
    stems = [p.stem for p in images]
    if len(set(stems)) != len(stems):
        raise ValueError('Repeated image stems: test.py flat TXT export is ambiguous')
    pred_files = list(pred_root.glob('*.txt'))
    stem_set = set(stems)
    extras = sorted(p.name for p in pred_files if p.stem not in stem_set)
    if extras:
        raise ValueError(f'Predictions outside this split (first 10): {extras[:10]}')
    missing_pred = [p.name for p in images if not (pred_root / (p.stem + '.txt')).is_file()]
    if missing_pred and not args.allow_missing_predictions:
        raise ValueError(f'{len(missing_pred)} missing prediction TXT files. Only after confirming complete export, '
                         'use --allow-missing-predictions to count them as zero predictions.')
    if args.expected_images is not None and len(images) != args.expected_images:
        raise ValueError(f'Image count mismatch: {len(images)} != {args.expected_images}')

    cases = []
    for image in images:
        gt = read_labels(gt_root / image.relative_to(image_root).with_suffix('.txt'), 5, nc)
        pred = read_labels(pred_root / (image.stem + '.txt'), 6, nc, args.allow_missing_predictions)
        if len(pred) > 300:
            raise ValueError(f'{image.name}: >300 predictions; check repeated appends or export max_det settings')
        cases.append((image, gt, pred))
    total_gt = sum(len(gt) for _, gt, _ in cases)
    if args.expected_labels is not None and total_gt != args.expected_labels:
        raise ValueError(f'GT count mismatch: {total_gt} != {args.expected_labels}')

    report = {
        'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'command': [sys.executable, *sys.argv],
        'data': str(data_path), 'images': str(image_root), 'gt_labels': str(gt_root),
        'predictions': str(pred_root), 'image_count': len(images), 'gt_count': total_gt,
        'prediction_file_count': len(pred_files), 'missing_prediction_files': missing_pred,
        'zero_gt_images': sum(not len(gt) for _, gt, _ in cases),
        'zero_prediction_images_at_export': sum(not len(pred) for _, _, pred in cases),
        'declared_export_confidence': args.export_conf, 'names': names,
        'method': 'same-class IoU-descending greedy 1:1, then remaining cross-class IoU-descending greedy 1:1',
        'limits': ['Fixed-threshold diagnostics, not AP or test.py F1-selected P/R.',
                   'Localization labels are candidate explanations, not proven causes.',
                   'Only saved post-NMS predictions are visible; NMS-suppressed/low-score candidates cannot be recovered.',
                   'Absent prediction TXT is treated as zero only with explicit flag; completed-export provenance is user-confirmed.'],
        'scenarios': [],
    }
    examples, example_counts, summary_rows = [], Counter(), []
    matrices = []
    for conf in args.conf:
        for match_iou in args.match_iou:
            matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)
            statuses = [Counter() for _ in names]
            pred_statuses = [Counter() for _ in names]
            for image, gt, pred in cases:
                result = analyze_image(gt, pred, nc, conf, match_iou, args.localization_iou)
                matrix += result['confusion']
                kept = result['retained_predictions']
                for g, kind in enumerate(result['gt_status']):
                    name = names[int(gt[g, 0])]
                    statuses[int(gt[g, 0])][kind] += 1
                    key = (conf, match_iou, 'gt', name, kind)
                    if kind == 'tp' or name not in args.focus or example_counts[key] >= args.max_examples:
                        continue
                    p = result['gt_pred_indices'][g]
                    examples.append([conf, match_iou, str(image), 'gt', g, name, kind,
                                     int(result['pred_source_indices'][p]) if p >= 0 else '',
                                     names[int(kept[p, 0])] if p >= 0 else '',
                                     float(kept[p, 5]) if p >= 0 else '',
                                     float(result['ious'][g, p]) if p >= 0 else 0,
                                     json.dumps(gt[g, 1:5].tolist()),
                                     json.dumps(kept[p, 1:5].tolist()) if p >= 0 else ''])
                    example_counts[key] += 1
                for p, kind in enumerate(result['pred_status']):
                    name = names[int(kept[p, 0])]
                    pred_statuses[int(kept[p, 0])][kind] += 1
                    key = (conf, match_iou, 'prediction', name, kind)
                    if kind == 'tp' or name not in args.focus or example_counts[key] >= args.max_examples:
                        continue
                    g = associated_gt(result, gt, p)
                    examples.append([conf, match_iou, str(image), 'prediction', g if g >= 0 else '',
                                     names[int(gt[g, 0])] if g >= 0 else '', kind,
                                     int(result['pred_source_indices'][p]), name, float(kept[p, 5]),
                                     float(result['ious'][g, p]) if g >= 0 else 0,
                                     json.dumps(gt[g, 1:5].tolist()) if g >= 0 else '',
                                     json.dumps(kept[p, 1:5].tolist())])
                    example_counts[key] += 1
            by_class = []
            for cls, name in enumerate(names):
                tp = int(matrix[cls, cls])
                gt_count, pred_count = int(matrix[:, cls].sum()), int(matrix[cls, :].sum())
                fp, fn = pred_count - tp, gt_count - tp
                row = dict(name=name, gt=gt_count, predictions=pred_count, tp=tp, fp=fp, fn=fn,
                           precision=tp / pred_count if pred_count else None,
                           recall=tp / gt_count if gt_count else None,
                           gt_categories=dict(statuses[cls]), prediction_categories=dict(pred_statuses[cls]))
                by_class.append(row)
                summary_rows.append([conf, match_iou, name, gt_count, pred_count, tp, fp, fn,
                                     row['precision'], row['recall'],
                                     statuses[cls]['classification'], statuses[cls]['localization_candidate'],
                                     statuses[cls]['class_localization_candidate'], statuses[cls]['assignment_conflict'],
                                     statuses[cls]['no_overlap_candidate'], pred_statuses[cls]['duplicate'],
                                     pred_statuses[cls]['background']])
            assert int(matrix[:, :nc].sum()) == total_gt
            report['scenarios'].append(dict(confidence=conf, matching_iou=match_iou,
                                            localization_iou=args.localization_iou, classes=by_class,
                                            confusion_predicted_rows_true_columns=matrix.tolist()))
            matrices.append((f'confusion_conf{conf!r}_iou{match_iou!r}.csv', matrix))

    lines = [f'Images={len(images)} GT={total_gt} prediction_TXT={len(pred_files)} missing_TXT={len(missing_pred)}',
             'DIAGNOSTIC ONLY: fixed confidence, TP-first IoU matching; not AP or printed test.py P/R.',
             'GT error categories: classification / localization_candidate / class_localization_candidate / assignment_conflict / no_overlap_candidate.']
    for scenario in report['scenarios']:
        lines.append(f"conf={scenario['confidence']:g}, matching_IoU={scenario['matching_iou']:g}")
        for row in scenario['classes']:
            if row['name'] in args.focus:
                p = '-' if row['precision'] is None else f"{row['precision']:.3f}"
                r = '-' if row['recall'] is None else f"{row['recall']:.3f}"
                errors = {k: v for k, v in row['gt_categories'].items() if k != 'tp'}
                lines.append(f"  {row['name']}: TP={row['tp']} FP={row['fp']} FN={row['fn']} P={p} R={r} errors={errors}")
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_csv(output / 'threshold_summary.csv',
              ['confidence', 'matching_iou', 'class', 'gt', 'predictions', 'tp', 'fp', 'fn', 'precision', 'recall',
               'gt_classification', 'gt_localization_candidate', 'gt_class_localization_candidate', 'gt_assignment_conflict',
               'gt_no_overlap_candidate', 'duplicate_predictions', 'background_predictions'], summary_rows)
    write_csv(output / 'examples.csv',
              ['confidence_threshold', 'matching_iou', 'image', 'record_side', 'gt_index_zero_based', 'gt_class',
               'category', 'pred_index_in_export_zero_based', 'pred_class', 'pred_confidence', 'associated_iou',
               'gt_normalized_xywh', 'pred_normalized_xywh'], examples)
    for filename, matrix in matrices:
        write_csv(output / filename, ['predicted / true'] + names + ['unmatched_prediction'],
                  [[name] + row.tolist() for name, row in zip(names + ['unmatched_gt'], matrix)])
    print('\n'.join(lines))
    print(f'Reports saved to {output}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f'ERROR: {exc}')
