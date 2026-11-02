"""Pair three completed detection exports by (image, GT row), without inference.

The reports must be produced by diagnose_detections.py on the same validation
split. Recomputed matrices and category counts must agree with those reports.
Fixed-threshold TP-first matching is diagnostic, not AP or a causal attribution.
"""
import argparse
import csv
import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from diagnose_detections import analyze_image, box_iou, read_labels, xywh_to_xyxy


ROLES = ('add', 'cmfc', 'full')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
GROUPS = ('all_three_to_confusing_class', 'add_tp_cmfc_miss', 'cmfc_tp_add_miss',
          'full_tp_cmfc_miss', 'cmfc_tp_full_miss', 'full_tp_add_miss',
          'add_tp_full_miss', 'full_only_tp', 'all_three_tp', 'all_three_miss')
VISUAL_GROUPS = GROUPS[:8]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_csv(path, rows, fields):
    with path.open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def state_for(result, g, names):
    status = result['gt_status'][g]
    if status == 'classification':
        p = result['gt_pred_indices'][g]
        return 'as:' + names[int(result['retained_predictions'][p, 0])]
    return status


def membership(states, confusing):
    a, c, f = [s == 'tp' for s in states]
    return dict(zip(GROUPS, (
        all(s == 'as:' + confusing for s in states),
        a and not c, c and not a, f and not c, c and not f,
        f and not a, a and not f, f and not a and not c,
        a and c and f, not a and not c and not f,
    )))


def candidate_info(pred, ious, cls, minimum_iou):
    """Highest-score saved candidate; independent of threshold and assignment."""
    eligible = np.flatnonzero((pred[:, 0] == cls) & (ious >= minimum_iou))
    if not len(eligible):
        return {'score': None, 'iou': None, 'pred_index': None}
    p = int(eligible[np.argmax(pred[eligible, 5])])
    return {'score': float(pred[p, 5]), 'iou': float(ious[p]), 'pred_index': p}


def model_row(result, g, pred, raw_ious, names, focus_id, confusing_id, iou):
    p = result['gt_pred_indices'][g]
    kept = result['retained_predictions']
    row = {'status': result['gt_status'][g], 'state': state_for(result, g, names),
           'associated_class': names[int(kept[p, 0])] if p >= 0 else '',
           'associated_score': float(kept[p, 5]) if p >= 0 else None,
           'associated_iou': float(result['ious'][g, p]) if p >= 0 else None,
           'associated_pred_index': int(result['pred_source_indices'][p]) if p >= 0 else None}
    for label, cls in [('focus', focus_id), ('confusing', confusing_id)]:
        for key, value in candidate_info(pred, raw_ious, cls, iou).items():
            row[f'best_{label}_{key}'] = value
    return row


def class_stats(matrix, gt_categories, pred_categories, cls, names):
    tp = int(matrix[cls, cls])
    gt_count = int(matrix[:, cls].sum())
    predictions = int(matrix[cls, :].sum())
    return dict(name=names[cls], gt=gt_count, predictions=predictions, tp=tp,
                fp=predictions - tp, fn=gt_count - tp,
                precision=tp / predictions if predictions else None,
                recall=tp / gt_count if gt_count else None,
                gt_categories=dict(gt_categories[cls]),
                prediction_categories=dict(pred_categories[cls]))


def verify_scenario(report, conf, iou, matrix, gt_categories, pred_categories, names, role):
    found = [s for s in report['scenarios']
             if s['confidence'] == conf and s['matching_iou'] == iou]
    if len(found) != 1:
        raise ValueError(f'{role}: expected exactly one report scenario {conf}/{iou}')
    saved = found[0]
    if saved['localization_iou'] != 0.1:
        raise ValueError(f'{role}: localization IoU must be 0.1')
    if saved['confusion_predicted_rows_true_columns'] != matrix.tolist():
        raise ValueError(f'{role}: report mismatch at conf={conf}, IoU={iou}; input exports/GT changed')
    if len(saved['classes']) != len(names):
        raise ValueError(f'{role}: incomplete per-class report')
    computed = []
    for cls, name in enumerate(names):
        row = class_stats(matrix, gt_categories, pred_categories, cls, names)
        matches = [c for c in saved['classes'] if c['name'] == name]
        if len(matches) != 1:
            raise ValueError(f'{role}: invalid class report {name}')
        for key in ('gt', 'predictions', 'tp', 'fp', 'fn', 'gt_categories', 'prediction_categories'):
            if row[key] != matches[0][key]:
                raise ValueError(f'{role}: report mismatch for {name}/{key} at {conf}/{iou}')
        computed.append(row)
    return computed


def load_inputs(report_paths, output, no_images):
    reports = [json.loads(p.read_text(encoding='utf-8-sig')) for p in report_paths]
    first = reports[0]
    names = first['names']
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError('Invalid class names')
    nc = len(names)
    rgb_root, gt_root = Path(first['images']).resolve(), Path(first['gt_labels']).resolve()
    data_path = Path(first['data']).resolve()
    data = yaml.safe_load(data_path.read_text(encoding='utf-8-sig'))
    if data['names'] != names or data['nc'] != nc or Path(data['val_rgb']).resolve() != rgb_root:
        raise ValueError('Data YAML no longer agrees with the diagnostic reports; run from repository root')
    ir_root = Path(data['val_ir']).resolve()
    if not rgb_root.is_dir() or not gt_root.is_dir() or (not no_images and not ir_root.is_dir()):
        raise ValueError('Missing RGB, GT or IR input directory')
    images = sorted(p for p in rgb_root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    stems = [p.stem for p in images]
    stem_set = set(stems)
    if not images or len(set(stems)) != len(stems):
        raise ValueError('No images or repeated image stems')
    if len({p.name for p in images}) != len(images):
        raise ValueError('Repeated image names make missing-file provenance ambiguous')
    protected = [rgb_root, ir_root, gt_root] + [p.parent for p in report_paths]
    source_meta = {}
    for role, report, report_path in zip(ROLES, reports, report_paths):
        if (report['names'] != names or Path(report['images']).resolve() != rgb_root
                or Path(report['gt_labels']).resolve() != gt_root
                or Path(report['data']).resolve() != data_path):
            raise ValueError(f'{role}: reports do not describe the same split/GT/names/data YAML')
        if report['image_count'] != len(images):
            raise ValueError(f'{role}: image count changed')
        pred_root = Path(report['predictions']).resolve()
        if not pred_root.is_dir():
            raise ValueError(f'{role}: prediction directory missing: {pred_root}')
        protected.append(pred_root)
        files = list(pred_root.glob('*.txt'))
        if len(files) != report['prediction_file_count'] or any(p.stem not in stem_set for p in files):
            raise ValueError(f'{role}: prediction file inventory changed or includes another split')
        missing = sorted(p.name for p in images if not (pred_root / (p.stem + '.txt')).is_file())
        if missing != sorted(report['missing_prediction_files']):
            raise ValueError(f'{role}: missing prediction inventory changed; no blanket missing-file allowance')
        source_meta[role] = dict(report=str(report_path), report_sha256=sha256(report_path),
                                 predictions=str(pred_root), missing_prediction_files=missing,
                                 declared_export_confidence=report['declared_export_confidence'],
                                 original_diagnostic_tool_sha256=report.get('tool_sha256'),
                                 original_command=report.get('command'))
    if len({m['predictions'] for m in source_meta.values()}) != len(ROLES):
        raise ValueError('Each model must have a distinct prediction directory')
    if output.exists():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    if any(output == p or p in output.parents for p in protected):
        raise ValueError('Output must be outside all input directories, including source reports')
    cases, missing_details = [], {role: [] for role in ROLES}
    digests = {key: hashlib.sha256() for key in ('gt', *ROLES)}
    for image in images:
        relative = image.relative_to(rgb_root)
        label_path = gt_root / relative.with_suffix('.txt')
        gt = read_labels(label_path, 5, nc)
        digests['gt'].update(relative.as_posix().encode() + b'\0' + label_path.read_bytes() + b'\0')
        predictions = {}
        for role, report in zip(ROLES, reports):
            pp = Path(report['predictions']) / (image.stem + '.txt')
            allowed = image.name in report['missing_prediction_files']
            predictions[role] = read_labels(pp, 6, nc, missing_ok=allowed)
            if len(predictions[role]) > 300:
                raise ValueError(f'{role}/{image.name}: >300 predictions; check repeated export appends')
            digests[role].update(relative.as_posix().encode() + b'\0'
                                 + (pp.read_bytes() if pp.is_file() else b'<MISSING>') + b'\0')
            if allowed:
                missing_details[role].append(dict(image=relative.as_posix(), gt_count=len(gt),
                                                  gt_classes=dict(Counter(names[int(c)] for c in gt[:, 0]))))
        ir = ir_root / relative
        if not no_images and not ir.is_file():
            raise ValueError(f'Missing paired IR image: {ir}')
        cases.append(dict(image=relative.as_posix(), rgb=image, ir=ir, gt=gt, predictions=predictions))
    gt_count = sum(len(case['gt']) for case in cases)
    for role, report in zip(ROLES, reports):
        if gt_count != report['gt_count']:
            raise ValueError(f'{role}: GT count changed')
        if sum(not len(case['gt']) for case in cases) != report['zero_gt_images']:
            raise ValueError(f'{role}: zero-GT image count changed')
        zero_pred = sum(not len(case['predictions'][role]) for case in cases)
        if zero_pred != report['zero_prediction_images_at_export']:
            raise ValueError(f'{role}: zero-prediction image count changed')
    return (reports, names, cases, source_meta, missing_details,
            {key: h.hexdigest() for key, h in digests.items()})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs=3, required=True, metavar=('ADD', 'CMFC', 'FULL'),
                        help='Existing summary.json files in Add / CMFC-only / full-v2 order')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--focus', default='van')
    parser.add_argument('--confusing-class', default='car')
    parser.add_argument('--conf', type=float, nargs='+', default=[0.05, 0.25, 0.5])
    parser.add_argument('--match-iou', type=float, nargs='+', default=[0.5, 0.75])
    parser.add_argument('--expected-focus', type=int)
    parser.add_argument('--samples-per-group', type=int, default=10)
    parser.add_argument('--sample-seed', type=int, default=42, help='Only controls visual sampling, not training')
    parser.add_argument('--no-images', action='store_true', help='Write counts/CSV only')
    args = parser.parse_args()
    if args.samples_per_group < 0 or (args.expected_focus is not None and args.expected_focus < 0):
        parser.error('Sample and expected-focus counts must be nonnegative')
    if (len(set(args.conf)) != len(args.conf) or len(set(args.match_iou)) != len(args.match_iou)
            or any(not 0 <= c <= 1 for c in args.conf) or any(not 0.1 < i <= 1 for i in args.match_iou)):
        parser.error('Invalid or repeated confidence/IoU thresholds')
    if 0.25 not in args.conf or 0.5 not in args.match_iou:
        parser.error('Include conf=0.25 and matching-IoU=0.5 for the fixed visual comparison')
    return args


def main():
    args = parse_args()
    output = args.output.resolve()
    report_paths = [p.resolve() for p in args.reports]
    reports, names, cases, source_meta, missing_details, input_hashes = load_inputs(report_paths, output, args.no_images)
    if args.focus not in names or args.confusing_class not in names or args.focus == args.confusing_class:
        raise ValueError('Focus and confusing class must be distinct known classes')
    if any(c < report['declared_export_confidence'] for c in args.conf for report in reports):
        raise ValueError('Diagnostic confidence must not be below any declared export confidence')
    nc, focus_id, confusing_id = len(names), names.index(args.focus), names.index(args.confusing_class)
    focus_count = sum(int((case['gt'][:, 0] == focus_id).sum()) for case in cases)
    if args.expected_focus is not None and focus_count != args.expected_focus:
        raise ValueError(f'Focus GT count mismatch: {focus_count} != {args.expected_focus}')
    paired_rows, transitions, scenarios = [], [], []
    anchor_rows, anchor_results = [], {}
    print(f'Validating {len(cases)} images, {focus_count} {args.focus} targets across three exports...', flush=True)
    for conf, iou in itertools.product(args.conf, args.match_iou):
        matrices = {role: np.zeros((nc + 1, nc + 1), dtype=np.int64) for role in ROLES}
        gt_categories = {role: [Counter() for _ in names] for role in ROLES}
        pred_categories = {role: [Counter() for _ in names] for role in ROLES}
        group_counts, pair_counts = Counter({key: 0 for key in GROUPS}), {}
        transition_counts = Counter()
        for left, right in itertools.combinations(ROLES, 2):
            pair_counts[f'{left}_to_{right}'] = dict(both_tp=0, gained_tp=0, lost_tp=0, both_miss=0,
                                                    both_to_confusing_class=0)
        for case in cases:
            gt, predictions = case['gt'], case['predictions']
            results = {role: analyze_image(gt, predictions[role], nc, conf, iou) for role in ROLES}
            for role, result in results.items():
                matrices[role] += result['confusion']
                for g, kind in enumerate(result['gt_status']):
                    gt_categories[role][int(gt[g, 0])][kind] += 1
                for p, kind in enumerate(result['pred_status']):
                    pred_categories[role][int(result['retained_predictions'][p, 0])][kind] += 1
            focus_indices = np.flatnonzero(gt[:, 0] == focus_id)
            raw_ious = {role: box_iou(xywh_to_xyxy(gt[focus_indices, 1:5]), xywh_to_xyxy(predictions[role][:, 1:5]))
                        for role in ROLES} if len(focus_indices) else {}
            for local_g, g in enumerate(focus_indices):
                g = int(g)
                row = dict(image=case['image'], gt_index_zero_based=g, gt_class=args.focus,
                           gt_normalized_xywh=json.dumps(gt[g, 1:5].tolist()), confidence=conf, matching_iou=iou)
                states = tuple(state_for(results[role], g, names) for role in ROLES)
                for role in ROLES:
                    details = model_row(results[role], g, predictions[role], raw_ious[role][local_g],
                                        names, focus_id, confusing_id, iou)
                    row.update({f'{role}_{key}': value for key, value in details.items()})
                paired_rows.append(row)
                flags = membership(states, args.confusing_class)
                group_counts.update({key: int(value) for key, value in flags.items()})
                for (li, left), (ri, right) in itertools.combinations(enumerate(ROLES), 2):
                    ls, rs = states[li], states[ri]
                    key = 'both_tp' if ls == rs == 'tp' else ('lost_tp' if ls == 'tp' else
                          ('gained_tp' if rs == 'tp' else 'both_miss'))
                    pair_counts[f'{left}_to_{right}'][key] += 1
                    if ls == rs == 'as:' + args.confusing_class:
                        pair_counts[f'{left}_to_{right}']['both_to_confusing_class'] += 1
                    transition_counts[(left, right, ls, rs)] += 1
                if conf == 0.25 and iou == 0.5:
                    anchor_rows.append(row)
                    anchor_results[case['image']] = results
        models = {}
        for role, report in zip(ROLES, reports):
            computed = verify_scenario(report, conf, iou, matrices[role], gt_categories[role], pred_categories[role], names, role)
            models[role] = computed[focus_id]
        scenarios.append(dict(confidence=conf, matching_iou=iou, focus_gt=focus_count, models=models,
                              groups=dict(group_counts), pairwise=pair_counts))
        for (left, right, ls, rs), count in sorted(transition_counts.items()):
            transitions.append(dict(confidence=conf, matching_iou=iou, from_model=left, to_model=right,
                                    from_state=ls, to_state=rs, count=count))
        print(f'  verified conf={conf:g}, matching_IoU={iou:g}', flush=True)

    # All source checks pass before creating any output directory.
    output.mkdir(parents=True, exist_ok=False)
    model_fields = ['status', 'state', 'associated_class', 'associated_score', 'associated_iou',
                    'associated_pred_index', 'best_focus_score', 'best_focus_iou', 'best_focus_pred_index',
                    'best_confusing_score', 'best_confusing_iou', 'best_confusing_pred_index']
    fields = ['image', 'gt_index_zero_based', 'gt_class', 'gt_normalized_xywh', 'confidence', 'matching_iou']
    fields += [f'{role}_{key}' for role in ROLES for key in model_fields]
    dump_csv(output / 'paired_targets.csv', paired_rows, fields)
    dump_csv(output / 'transitions.csv', transitions,
             ['confidence', 'matching_iou', 'from_model', 'to_model', 'from_state', 'to_state', 'count'])
    rng, selected, rendered = random.Random(args.sample_seed), [], {}
    case_lookup = {case['image']: case for case in cases}
    render_errors = []
    if not args.no_images:
        from paired_detection_render import render_case
        (output / 'images').mkdir()
    for group in VISUAL_GROUPS:
        eligible = [row for row in anchor_rows if membership(tuple(row[f'{role}_state'] for role in ROLES),
                                                            args.confusing_class)[group]]
        sample = rng.sample(eligible, min(len(eligible), args.samples_per_group))
        for row in sample:
            identity = (row['image'], row['gt_index_zero_based'])
            token = hashlib.sha256((identity[0] + '\0' + str(identity[1])).encode()).hexdigest()[:12]
            filename = f'images/{Path(identity[0]).stem}_gt{identity[1]}_{token}.png'
            selected.append(dict(group=group, image=identity[0], gt_index_zero_based=identity[1],
                                 **{f'{role}_state': row[f'{role}_state'] for role in ROLES},
                                 panel_file='' if args.no_images else filename))
            if not args.no_images and identity not in rendered:
                case = case_lookup[identity[0]]
                try:
                    render_case(case['rgb'], case['ir'], case['gt'], identity[1], case['predictions'],
                                anchor_results[identity[0]], names, output / filename, 0.25, 0.5,
                                f"{identity[0]} | GT row {identity[1]} | {args.focus}",
                                confusing_class=args.confusing_class)
                    rendered[identity] = filename
                except (OSError, ValueError) as exc:
                    rendered[identity] = None
                    render_errors.append(dict(image=identity[0], gt_index_zero_based=identity[1], error=str(exc)))
    dump_csv(output / 'selection.csv', selected,
             ['group', 'image', 'gt_index_zero_based', *[f'{r}_state' for r in ROLES], 'panel_file'])
    metadata = dict(command=[sys.executable, *sys.argv], names=names, focus=args.focus, confusing_class=args.confusing_class,
                    image_count=len(cases), gt_count=sum(len(c['gt']) for c in cases), focus_gt=focus_count,
                    source_reports=source_meta, input_sha256=input_hashes, missing_predictions=missing_details,
                    tool_sha256=sha256(__file__), matcher_sha256=sha256(Path(__file__).with_name('diagnose_detections.py')),
                    scenarios=scenarios, visual_confidence=0.25, visual_matching_iou=0.5,
                    sample_seed=args.sample_seed, samples_per_group=args.samples_per_group,
                    rendered_panels=sum(v is not None for v in rendered.values()), render_errors=render_errors,
                    limits=['Diagnostic TP-first matching, not AP or proof of model/data causality.',
                            'Model roles follow the CLI order; original checkpoint/NMS settings are not recoverable from TXT.',
                            'Candidate scores use all saved post-NMS predictions with IoU >= matching_iou, independently of matching/confidence; absent candidates are null, not zero.',
                            'Associated boxes for non-TP/non-classification states are explanatory candidates, not assigned detections.',
                            'Groups overlap; all counts cover every focus GT. Panels are fixed-seed samples, not estimates of prevalence.',
                            'Existing diagnostic reports do not hash original input contents; reconciliation detects aggregate drift, not every possible equal-count change.',
                            'Missing TXT files are accepted only when already listed in source reports; provenance still relies on complete original evaluations.'])
    lines = [f'Images={len(cases)} GT={metadata["gt_count"]} {args.focus}_GT={focus_count}',
             'Roles: add / cmfc / full in --reports order. Diagnostic only, not AP.',
             'All six requested report matrices and class categories reconciled.' if len(scenarios) == 6 else 'All requested report scenarios reconciled.',
             'GT identity: relative image path + zero-based nonblank GT row.',
             'Candidate scores: saved post-NMS boxes, IoU >= matching threshold, independent of assignment/confidence.']
    for role in ROLES:
        lines.append(f'Missing TXT {role}: {json.dumps(missing_details[role], ensure_ascii=False)}')
    for s in scenarios:
        lines.append(f'\nconf={s["confidence"]:g}, matching_IoU={s["matching_iou"]:g}')
        for role, row in s['models'].items():
            lines.append(f'  {role}: TP={row["tp"]} FP={row["fp"]} FN={row["fn"]}')
        lines.append(f'  all_three_to_{args.confusing_class}={s["groups"]["all_three_to_confusing_class"]}')
        for pair, values in s['pairwise'].items():
            lines.append(f'  {pair}: {values}')
    lines.extend(['', f'Panels={metadata["rendered_panels"]}; render_errors={len(render_errors)}',
                  'See paired_targets.csv, transitions.csv and selection.csv. Groups may overlap.',
                  f'Reports saved to {output}'])
    (output / 'summary.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / 'README.txt').write_text(
        'Read summary.txt first. All statistics use every focus GT, not the selected images.\n'
        'paired_targets.csv: one row per focus GT per confidence/IoU scenario. GT indices are zero-based nonblank label rows.\n'
        'transitions.csv: full state-to-state counts, e.g. tp -> as:car; each model pair/scenario sums to focus_gt.\n'
        'In A_to_B: gained_tp means A misses and B is TP; lost_tp means A is TP and B misses. Both may be nonzero when net change is zero.\n'
        'selection.csv: group membership and relative PNG filenames. A target may appear in several groups; identical panels are saved once.\n'
        'all_three_to_confusing_class means all models actually received the cross-class assignment at this working point.\n'
        'Groups named X_tp_Y_miss mean X is TP and Y has any non-TP state; full_only_tp means both Add and CMFC miss.\n'
        'Images: RGB/IR rows, GT/Add/CMFC/Full columns. BOTH sensor rows overlay the SAME fused predictions.\n'
        'Yellow = selected GT; gray = other GT; magenta = associated prediction. Association for localization/no-overlap states is only a candidate.\n'
        'best_focus_* and best_confusing_* use all saved post-NMS predictions with IoU >= matching_iou, regardless of confidence threshold or assignment.\n'
        'Empty candidate score fields mean no eligible saved candidate, NOT a probability of zero. Scores combine objectness and class confidence in this exporter.\n'
        'Panels use conf=0.25, matching IoU=0.5 and a fixed sampling seed; they are not unbiased estimates of error prevalence.\n'
        'Report roles and original checkpoint/NMS provenance require the correct CLI inputs; filenames alone do not establish model identity.\n'
        'Input digests record the current contents. The old reports contain no original input digests, so matching counts do not prove byte-for-byte historical identity.\n',
        encoding='utf-8')
    print('\n'.join(lines))
    if render_errors:
        raise ValueError('Some panels failed; counts/CSV are available, inspect summary.json render_errors')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(f'ERROR: {exc}')
