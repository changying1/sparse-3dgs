#!/usr/bin/env python3
"""RD-D1: audit frozen RD-v1 interventions and export compact mechanism evidence.

No GPU, checkpoint loading, training, rendering, or source mutations.
Run from tools/ with `python -m tools.audit_rd_d1_offline`.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

SCENES = ('horns','fern','flower','fortress','orchids','room','leaves','trex')
BATCH = 'RDv1_20261016_153009_622830'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require_ref(ref, label):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise RuntimeError(f'{label}: invalid path/hash binding: {ref!r}')
    p = Path(ref['path'])
    if not p.is_file():
        raise RuntimeError(f'{label}: missing file {p}')
    got = sha(p)
    if got.lower() != ref['sha256'].lower():
        raise RuntimeError(f'{label}: SHA mismatch {p}')
    return p


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def stats(values):
    v = sorted(float(x) for x in values if isinstance(x, (int, float))
               and not isinstance(x, bool) and math.isfinite(x))
    if not v:
        return {'n': 0}
    def q(frac):
        k = (len(v) - 1) * frac
        a, b = math.floor(k), math.ceil(k)
        return v[a] * (b - k) + v[b] * (k - a) if a != b else v[a]
    return {'n': len(v), 'min': v[0], 'q25': q(.25), 'median': q(.5),
            'q75': q(.75), 'max': v[-1], 'mean': statistics.fmean(v)}


def read_metric(file, kind):
    d = load(file)
    check(d.get('iteration') == 5000 and d.get('checkpoint_audit', {}).get('pass') is True,
          f'{file}: formal iteration/checkpoint audit not PASS')
    block = d[kind]
    check(len(block['per_view']) == block['view_count'], f'{file}: view count differs')
    return block


def run(root: Path, batch: str, scenes: tuple[str, ...]):
    audit = root/'reports'/'i2_audit'/batch
    manifest_path = audit/'batch_manifest.json'
    check(manifest_path.is_file(), f'Batch manifest missing: {manifest_path}')
    manifest = load(manifest_path)
    check(manifest.get('batch_id') == batch and manifest.get('status') == 'PASS',
          'Original batch not frozen at PASS')
    now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = audit/f'RD_D1_Offline_{now}'
    check(not output.exists(), 'Output must be a new directory')
    # Validate all existing source paths before writing a report directory.
    entries = []
    for scene in scenes:
        row = manifest['scenes'][scene]
        check(all(row.get(x) == 'PASS' for x in ('preflight','prepare','adaptive_smoke',
              'adaptive_formal','adaptive_control_formal')), f'{scene}: incomplete batch')
        decp = Path(row['decision'])
        check(decp.is_file() and sha(decp) == row['decision_sha256'],
              f'{scene}: decision hash mismatch')
        decision = load(decp)
        check(decision['policy_version'] == 'RD-v1' and decision['scene'] == scene,
              f'{scene}: incorrect decision')
        k = decision['selected_k']; r = decision['receiver_capacity']; d = decision['donor_capacity']
        check(type(k) is int and k == min(r,d) and k == row['K'],
              f'{scene}: violated K=min(R,D)')
        rp = require_ref(decision['evidence']['receiver'], f'{scene} receiver audit')
        dp = require_ref(decision['evidence']['donor'], f'{scene} donor audit')
        pp = require_ref(decision['adaptive_plan'], f'{scene} RD Plan') if k else None
        receiver = load(rp); donor = load(dp); plan = load(pp) if pp else {}
        rd_output = Path(row['adaptive_formal_output'])
        control_output = Path(row['adaptive_control_formal_output'])
        eventp = rd_output/'active_event.json'
        check(eventp.is_file(), f'{scene}: missing formal active_event.json')
        event = load(eventp)
        check(event.get('format') == 'i2_active_event_rd_v1', f'{scene}: wrong event format')
        check(event.get('plan_sha256') == sha(pp), f'{scene}: actual event plan hash differs')
        check(event.get('budget_conserved') is True, f'{scene}: runtime budget not conserved')
        check(len(event.get('released_uids', [])) == k
              and len(event.get('born_uids', [])) == k
              and len(set(event['released_uids'])) == k
              and len(set(event['born_uids'])) == k,
              f'{scene}: bad event release/birth records')
        check(event['gaussian_before'] == event['gaussian_after'],
              f'{scene}: event did not conserve Gaussian total')
        check(event['gaussian_after_release'] == event['gaussian_before'] - k,
              f'{scene}: unexpected event release count')
        plan_donors = plan.get('donors', [])
        plan_receivers = plan.get('receivers', [])
        check(len(plan_donors) == k and len(plan_receivers) == k,
              f'{scene}: Plan selected counts differ')
        check([int(x['uid']) for x in plan_donors] == [int(x) for x in event['released_uids']],
              f'{scene}: actual release UIDs differ from Plan')
        check([str(x['candidate_id']) for x in plan_receivers] == event['receiver_candidate_ids'],
              f'{scene}: actual Receiver IDs differ from Plan')
        check([int(x['anchor_uid']) for x in plan_receivers] == event['anchor_uids'],
              f'{scene}: actual Receiver anchors differ from Plan')
        check(not set(event['released_uids']) & set(event['anchor_uids']),
              f'{scene}: a donor is a live receiver anchor')
        check(receiver.get('coverage', {}).get('complete') is True
              and donor.get('coverage_complete') is True,
              f'{scene}: incomplete enumeration')
        check(receiver['receiver_budget']['receiver_capacity'] == r
              and donor['donor_capacity'] == d,
              f'{scene}: capacity evidence does not match Decision')
        accepted = receiver['receiver_budget']['positive_evidence_ids']
        check([x['candidate_id'] for x in plan_receivers] == accepted[:k],
              f'{scene}: Receiver selected order differs from audited prefix')
        check([x['uid'] for x in plan_donors] == [x['uid'] for x in donor['ranked_eligible'][:k]],
              f'{scene}: Donor selected order differs from audited prefix')
        rd_meta = load(rd_output/'launch_meta.json')
        control_meta = load(control_output/'launch_meta.json')
        check(rd_meta['status'] == 'PASS' and control_meta['status'] == 'PASS',
              f'{scene}: formal metadata missing PASS')
        check(rd_meta['source_checkpoint_sha256'] == control_meta['source_checkpoint_sha256']
              and rd_meta['view_manifest_sha256'] == control_meta['view_manifest_sha256']
              and rd_meta.get('random_seed') == control_meta.get('random_seed'),
              f'{scene}: source/split/seed not paired')
        check(rd_meta.get('adaptive_decision_sha256') == row['decision_sha256'],
              f'{scene}: RD meta wrong decision')
        rd_test = read_metric(rd_output/'formal_metrics_5000.json', 'test')
        ctrl_test = read_metric(control_output/'formal_metrics_5000.json', 'test')
        rd_train = read_metric(rd_output/'formal_metrics_5000.json', 'train')
        ctrl_train = read_metric(control_output/'formal_metrics_5000.json', 'train')
        per_view = []
        check(len(rd_test['per_view']) == len(ctrl_test['per_view']),
              f'{scene}: test view count mismatch')
        for aa,bb in zip(rd_test['per_view'], ctrl_test['per_view']):
            check(aa['image_name'] == bb['image_name'], f'{scene}: test view name mismatch')
            per_view.append({'image_name': aa['image_name'],
                             **{f'delta_{m}': aa[m]-bb[m] for m in ('PSNR','SSIM','LPIPS','L1')}})
        selected_per_view = Counter(x.get('training_camera_id', '<missing>') for x in plan_receivers)
        donor_birth = Counter(str(x.get('birth_type', '<missing>')) for x in plan_donors)
        geometry_reasons = Counter(x.get('geometry_failure_reason') or '<valid>'
                                   for x in receiver['rows'])
        nms = Counter(x.get('nms_status', '<missing>') for x in receiver['rows'])
        alignment = stats(x.get('photometric_alignment') for x in plan_receivers)
        all_geometry = [x['geometry_evidence'] for x in receiver['rows']
                        if x.get('geometry_status') == 'PASS'
                        and isinstance(x.get('geometry_evidence'), dict)]
        geometry_positive = [x for x in all_geometry if x.get('photometric_alignment', 0) > 0]
        details = {
            'scene': scene, 'batch_id': batch, 'K': k, 'R': r, 'D': d,
            'G': decision['receiver_audit']['geometry_nms_capacity_g'],
            'r_over_g': r/decision['receiver_audit']['geometry_nms_capacity_g'] if decision['receiver_audit']['geometry_nms_capacity_g'] else None,
            'source_checkpoint_sha256': rd_meta['source_checkpoint_sha256'],
            'decision_sha256': sha(decp), 'plan_sha256': sha(pp),
            'receiver_audit_sha256': sha(rp), 'donor_audit_sha256': sha(dp),
            'test': {'RD': rd_test['aggregate'], 'Control': ctrl_test['aggregate'],
                     'delta': {m:rd_test['aggregate'][m]-ctrl_test['aggregate'][m]
                               for m in ('PSNR','SSIM','LPIPS','L1')}},
            'train': {'RD': rd_train['aggregate'], 'Control': ctrl_train['aggregate'],
                      'delta': {m:rd_train['aggregate'][m]-ctrl_train['aggregate'][m]
                                for m in ('PSNR','SSIM','LPIPS','L1')}},
            'positive_test_views': sum(v['delta_PSNR'] > 0 for v in per_view),
            'total_test_views': len(per_view), 'per_view': per_view,
            'receiver': {
                'by_training_camera': dict(selected_per_view),
                'geometry_status': dict(geometry_reasons),
                'nms_status': dict(nms),
                'selected_alignment': alignment,
                'all_geometry_positive_count': len(geometry_positive),
                'selected_demand_score': stats(x.get('demand_score') for x in plan_receivers),
                'selected_displacement_norm': stats(x.get('displacement_norm') for x in plan_receivers),
                'selected_opacity': stats(x.get('activated_opacity') for x in plan_receivers),
                'selected_projection_condition': stats(x.get('projection_condition') for x in plan_receivers),
                'selected_pixel_error_reduction': stats(
                    x['source_pixel_offset_before']-x['source_pixel_offset_after']
                    for x in plan_receivers if isinstance(x.get('source_pixel_offset_before'), (int,float))
                    and isinstance(x.get('source_pixel_offset_after'), (int,float))),
            },
            'donor': {
                'by_birth_type': dict(donor_birth),
                'selected_row_field_names': sorted(set().union(*(set(x) for x in plan_donors))),
                'numeric_distributions': {
                    field: stats(x.get(field) for x in plan_donors)
                    for field in sorted(set().union(*(set(x) for x in plan_donors)))
                    if field.lower() not in ('uid','source_uid','rank','rd_rank','generation','birth_iter')
                    and any(type(x.get(field)) in (int,float) for x in plan_donors)
                },
            },
            'active_event': {key:value for key,value in event.items()
                             if key not in ('released_uids','born_uids','receiver_candidate_ids','anchor_uids')},
        }
        entries.append((scene, details, plan_donors, plan_receivers, event))
    output.mkdir(parents=True, exist_ok=False)
    files = []
    def save(name, value):
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
        files.append(path)
    for scene, detail, donors, receivers, event in entries:
        save(f'{scene}/summary.json', detail)
        save(f'{scene}/selected_donors.json', donors)
        save(f'{scene}/selected_receivers.json', receivers)
        save(f'{scene}/active_event.json', event)
    save('overall_summary.json', {
        'protocol': 'RD-D1', 'source_batch': batch,
        'source_manifest_sha256': sha(manifest_path),
        'scene_count': len(entries),
        'scenes': {s: {
            'G':v['G'], 'R':v['R'], 'D':v['D'], 'K':v['K'],
            'receiver_by_training_camera': v['receiver']['by_training_camera'],
            'donor_by_birth_type': v['donor']['by_birth_type'],
            'delta_test': v['test']['delta'], 'delta_train': v['train']['delta'],
        } for s,v,_,_,_ in entries},
    })
    # Portable, minimal archive for follow-up analysis (not giant source checkpoints).
    bundle = output/'RD_D1_Offline_Evidence.zip'
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(output).as_posix())
    print('\nRD-D1 READ-ONLY AUDIT: PASS')
    print('Scene | K | Receiver training-view counts | Donor birth types | ΔTestPSNR')
    for s,v,_,_,_ in entries:
        print(f"{s:9s} | {v['K']:4d} | {v['receiver']['by_training_camera']} | "
              f"{v['donor']['by_birth_type']} | {v['test']['delta']['PSNR']:+.4f}")
    print('OUTPUT:', output)
    print('BUNDLE:', bundle)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--batch',default=BATCH)
    p.add_argument('--scenes',nargs='+',choices=SCENES,default=list(SCENES))
    args=p.parse_args()
    run(args.root.resolve(),args.batch,tuple(args.scenes))

if __name__=='__main__':
    main()
