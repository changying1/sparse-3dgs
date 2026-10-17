#!/usr/bin/env python3
"""RD-D4 read-only two-scene trajectory audit. No GPU or training.

Reads the original RD-v1 and RD-D3 batch manifests and each arm's existing
formal train.log + final JSON. Never changes experiment files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'RDv1_20261016_153009_622830'
DEFAULT_D3 = 'RD_D3_20261017_104858_958268'
SCENES = ('fortress', 'trex')
ARMS = ('control', 'release_only', 'birth_only', 'rd_v1')
STEPS = (1500, 2000, 3000, 4000, 5000)
METRICS = ('L1','PSNR','SSIM','LPIPS')
NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
RX = re.compile(r'\[ITER\s+(\d+)\]\s*Evaluating\s+(test|train):\s*L1\s+(' + NUM + r')\s+PSNR\s+(' + NUM + r')\s+SSIM\s+(' + NUM + r')\s+LPIPS\s+(' + NUM + r')')


def read(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f'Missing input: {path}')
    d = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(d, dict):
        raise RuntimeError(f'JSON object required: {path}')
    return d


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
    return h.hexdigest()


def metric_json(path):
    d=read(path)
    if d.get('iteration')!=5000 or d.get('checkpoint_audit',{}).get('pass') is not True:
        raise RuntimeError(f'Invalid formal checkpoint/metrics: {path}')
    for split in ('test','train'):
        s=d[split]
        rows=s['per_view']
        if len(rows)!=s['view_count'] or len({r['image_name'] for r in rows})!=len(rows):
            raise RuntimeError(f'Invalid view set {path}: {split}')
        for met in METRICS:
            actual=sum(float(r[met]) for r in rows)/len(rows)
            if not math.isclose(actual,float(s['aggregate'][met]),rel_tol=1e-6,abs_tol=1e-8):
                raise RuntimeError(f'Invalid aggregate {path}: {split}.{met}')
    return d


def log_curve(path):
    path=Path(path)
    if not path.is_file():
        raise RuntimeError(f'Missing full train.log: {path}')
    text=path.read_text(encoding='utf-8',errors='replace')
    rows={}
    for x in RX.finditer(text):
        step, split=int(x[1]), x[2]
        values={met:float(v) for met,v in zip(METRICS,x.groups()[2:])}
        key=(step,split)
        if key in rows:
            raise RuntimeError(f'Duplicate train report {key}: {path}')
        rows[key]=values
    for step in STEPS:
        if (step,'test') not in rows:
            raise RuntimeError(f'Missing test training report iter {step}: {path}')
    return rows


def csv_out(path,rows,columns):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--d3-batch',default=DEFAULT_D3)
    args=ap.parse_args()
    source_dir=ROOT/'reports'/'i2_audit'
    base=read(source_dir/BASE/'batch_manifest.json')
    d3=read(source_dir/args.d3_batch/'rd_d3_batch_manifest.json')
    if base.get('batch_id')!=BASE or d3.get('source_batch')!=BASE or d3.get('batch_id')!=args.d3_batch:
        raise RuntimeError('Batch identity mismatch')
    if base.get('status')!='PASS' or d3.get('status')!='PASS':
        raise RuntimeError('Batch has not PASSed')
    output=source_dir/args.d3_batch/('RD_D4_Trajectory_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if output.exists():
        raise RuntimeError('Refusing output overwrite')

    metric_rows=[];combination_rows=[];view_rows=[];references=[]
    for scene in SCENES:
        old=base['scenes'][scene]
        item=d3['scenes'][scene]
        if old['K']!=item['K'] or old['decision_sha256']!=item['decision_sha256']:
            raise RuntimeError(f'{scene}: frozen K/Decision changed')
        paths={
          'control': Path(old['adaptive_control_formal_output']),
          'rd_v1': Path(old['adaptive_formal_output']),
          'release_only': Path(item['arms']['release']['formal']['output']),
          'birth_only': Path(item['arms']['birth_only']['formal']['output']),
        }
        arms={}; curves={}
        for arm in ARMS:
            dest=paths[arm]
            formal=dest/'formal_metrics_5000.json'
            meta_path=dest/'launch_meta.json'
            logfile=dest/'train.log'
            m=metric_json(formal)
            meta=read(meta_path)
            if meta.get('status')!='PASS' or meta.get('scene')!=scene or meta.get('stage')!='formal':
                raise RuntimeError(f'{scene}/{arm}: formal launch metadata not PASS')
            if meta.get('source_checkpoint_sha256')!=item['source_checkpoint_sha256']:
                raise RuntimeError(f'{scene}/{arm}: source checkpoint mismatch')
            if arm in ('release_only','birth_only'):
                if meta.get('plan_sha256')!=item['plan_sha256'] or meta.get('adaptive_decision_sha256')!=item['decision_sha256']:
                    raise RuntimeError(f'{scene}/{arm}: frozen Plan/Decision mismatch')
            arms[arm]=m
            curves[arm]=log_curve(logfile)
            references += [{'scene':scene,'arm':arm,'path':str(p),'sha256':sha(p)} for p in (formal,meta_path,logfile)]
            if not math.isclose(curves[arm][(5000,'test')]['PSNR'],m['test']['aggregate']['PSNR'],rel_tol=1e-5,abs_tol=1e-4):
                raise RuntimeError(f'{scene}/{arm}: train.log iter5000 disagrees with formal metrics')
        for arm in ARMS:
            for split in ('test','train'):
                ref=[r['image_name'] for r in arms['control'][split]['per_view']]
                other=[r['image_name'] for r in arms[arm][split]['per_view']]
                if ref!=other:
                    raise RuntimeError(f'{scene}/{arm}: evaluation view order mismatch')
            if arms[arm]['protocol']!=arms['control']['protocol']:
                raise RuntimeError(f'{scene}/{arm}: evaluation protocol mismatch')
        for step in STEPS:
            for split in ('test','train'):
                for arm in ARMS:
                    if (step,split) not in curves[arm]:
                        continue
                    for metric,value in curves[arm][(step,split)].items():
                        metric_rows.append({'scene':scene,'iteration':step,'split':split,'arm':arm,'metric':metric,'value':value})
                if all((step,split) in curves[arm] for arm in ARMS):
                    for metric in METRICS:
                        v={arm:curves[arm][(step,split)][metric] for arm in ARMS}
                        combination_rows.append({'scene':scene,'iteration':step,'split':split,'metric':metric,
                            'control':v['control'],'release_only':v['release_only'],'birth_only':v['birth_only'],'rd_v1':v['rd_v1'],
                            'delta_release':v['release_only']-v['control'],
                            'delta_birth':v['birth_only']-v['control'],
                            'delta_rd':v['rd_v1']-v['control'],
                            'nonadditivity_contrast':v['rd_v1']-v['release_only']-v['birth_only']+v['control']})
        for split in ('test','train'):
            control={r['image_name']:r for r in arms['control'][split]['per_view']}
            for name,row in control.items():
                other={a:next(x for x in arms[a][split]['per_view'] if x['image_name']==name) for a in ARMS}
                for metric in METRICS:
                    values={a:float(other[a][metric]) for a in ARMS}
                    view_rows.append({'scene':scene,'split':split,'image_name':name,'metric':metric,**values,
                        'delta_release':values['release_only']-values['control'],
                        'delta_birth':values['birth_only']-values['control'],
                        'delta_rd':values['rd_v1']-values['control'],
                        'nonadditivity_contrast':values['rd_v1']-values['release_only']-values['birth_only']+values['control']})
        print(f'[{scene}] 4 arms x 5 formal checkpoints: PASS',flush=True)
    output.mkdir(parents=True,exist_ok=False)
    csv_out(output/'formal_trajectory_long.csv',metric_rows,['scene','iteration','split','arm','metric','value'])
    fields=['scene','iteration','split','metric','control','release_only','birth_only','rd_v1',
        'delta_release','delta_birth','delta_rd','nonadditivity_contrast']
    csv_out(output/'formal_trajectory_comparison.csv',combination_rows,fields)
    csv_out(output/'final_per_view_interactions.csv',view_rows,['scene','split','image_name','metric','control',
        'release_only','birth_only','rd_v1','delta_release','delta_birth','delta_rd','nonadditivity_contrast'])
    provenance={
       'format':'rd_d4_offline_trajectory_v1','source_batch':BASE,'d3_batch':args.d3_batch,
       'dataset_scenes':list(SCENES),'formal_steps':list(STEPS),'reference_files':references,
       'rd_v1_train_py_sha256':base['fingerprints']['train.py'],
       'rd_d3_train_py_sha256':d3['fingerprints']['train.py'],
       'training_code_fingerprint_equal':base['fingerprints']['train.py']==d3['fingerprints']['train.py'],
       'interpretation_guard':'Single seed, unequal point counts, different code fingerprints; metric contrasts are descriptive, not an isolated causal effect. Smoke iteration1252 is a different training run from formal5000 and is intentionally excluded from trajectories.',
       'status':'PASS'
    }
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    zip_path=output/'RD_D4_Trajectory_Evidence.zip'
    with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.iterdir()):
            if p.is_file() and p!=zip_path:
                archive.write(p,p.name)
    print('RD-D4 READ-ONLY TRAJECTORY AUDIT: PASS')
    print('TRAINING CODE FINGERPRINT EQUAL:',provenance['training_code_fingerprint_equal'])
    print('OUTPUT:',output)
    print('BUNDLE:',zip_path)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
