#!/usr/bin/env python3
"""Official COCO/BBOB external validation, independently executed, no surrogate functions.
Runs the *SHA-verified* frozen TFTMO-ADG v1.0.1 on all 24 noiseless BBOB
functions, D=10/20, instances 1..5 at 50D,150D,300D budgets.
The fixed target-budget CEC campaign is NOT replicated by these pilot budgets.
"""
import csv, hashlib, io, json, os, platform, sys, urllib.request, zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import cocoex
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'validation_output'
OUT.mkdir(parents=True,exist_ok=True)
SHA={
 'TFTMO_ADG.py':'8a337b2aef64576f7c6867729a496f6efe54e899e6205d25f7c6eb79517dcad6',
 'TFTMO_DF.py':'04dd9645a74e37bdb123554d7eaff7ebd266fadc05934424c33c99931d013cdb',
 'TFTMO_PG2.py':'7305ddbfc54a2c851fb1d7c532cc8787fa4c4c0fecd456324468069a5f4996fe',
}
SRC=OUT/'frozen_tftmo'
SRC.mkdir(exist_ok=True)
def readurl(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'BBOB-audit/1.0'}),timeout=120) as resp:
        return resp.read()
def acquire():
    payload=json.loads(readurl('https://zenodo.org/api/records/23554461'))
    records=payload.get('files',[])
    files=[]; source={}
    for item in records:
        key=item.get('key',''); links=item.get('links',{})
        if not key.lower().endswith('.zip'): continue
        url=links.get('self') or ('https://zenodo.org/records/23554461/files/'+key+'?download=1')
        raw=readurl(url)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            files+=z.namelist()
            for itemname in z.namelist():
                if itemname.split('/')[-1] not in SHA: continue
                b=z.read(itemname)
                h=hashlib.sha256(b).hexdigest()
                if h==SHA[itemname.split('/')[-1]]: source[itemname.split('/')[-1]]=b
    if set(source)!=set(SHA):
        raise RuntimeError('The linked Zenodo record does not contain the three frozen v1.0.1 source files: '+str(sorted(source)))
    for name,b in source.items(): (SRC/name).write_bytes(b)
    return {'zenodo_doi':'10.5281/zenodo.23554461','zenodo_title':payload.get('metadata',{}).get('title',''),
            'sources_sha256':SHA,'source_archive_names':files[:10]}
def main():
    try:
        info=acquire()
        sys.path.insert(0,str(SRC))
        from TFTMO_ADG import tftmo_adg
        from TFTMO_DF import _lshade_seeded
        suite=cocoex.Suite('bbob','','dimensions:10,20 instance_indices:1-5')
        rows=[]; budgets_multipliers=[50,150,300]; variants=['full','no_handoff','de_stage_only']
        manifest={'status':'started','date_utc':datetime.now(timezone.utc).isoformat(),'cocoex_version':getattr(cocoex,'__version__','not exposed'),
                  'numpy_version':np.__version__,'python_version':platform.python_version(),'algorithm_source':info,
                  'function_indices':list(range(1,25)),'dimensions':[10,20],'instance_indices':[1,2,3,4,5],
                  'budget_multipliers':budgets_multipliers,'variants':variants,
                  'repetitions_per_instance':1,'design':'exploratory independent suite after CEC discovery; no CEC target-budget claims'}
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
        csvpath=OUT/'raw_coco_bbob.csv'
        with csvpath.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['problem_id','function','dimension','instance','budget_multiplier','budget','seed','variant','best_f','optimum_f','error','evaluations'])
            for pi,p in enumerate(suite):
                d=int(p.dimension); fid=int(p.id.split('_f')[1].split('_')[0]);inst=int(p.id.split('_i')[1].split('_')[0])
                lb=np.asarray(p.lower_bounds,dtype=float);ub=np.asarray(p.upper_bounds,dtype=float)
                for factor in budgets_multipliers:
                    budget=factor*d;seed=627000+1000*inst+fid
                    for v in variants:
                        ctr=[0]
                        def fun(x):
                            ctr[0]+=1
                            return float(p(np.asarray(x,dtype=float)))
                        if v=='de_stage_only':
                            x=np.random.default_rng(seed).uniform(lb,ub)
                            val=fun(x)
                            result=_lshade_seeded(fun,d,lb,ub,additional_fes=budget-1,seed=seed+104729,seed_x=x,seed_f=val)
                        elif v=='no_handoff':
                            result=tftmo_adg(fun,d,lb,ub,max_fes=budget,seed=seed,min_frac=1.,max_frac=1.,patience_frac=.08)
                        else:
                            result=tftmo_adg(fun,d,lb,ub,max_fes=budget,seed=seed,min_frac=.60,max_frac=.95,patience_frac=.08)
                        best=float(result[1]); optimum=float(p.optimal_value); error=max(0.,best-optimum)
                        if ctr[0]!=budget or not np.isfinite(error): raise RuntimeError('FE/finite failure: '+str((p.id,factor,v,ctr[0],error)))
                        w.writerow([p.id,fid,d,inst,factor,budget,seed,v,repr(best),repr(optimum),repr(error),ctr[0]])
                        f.flush()
                print(f'{pi+1}/{len(suite)} completed: {p.id}',flush=True)
        manifest['status']='completed'
        manifest['rows']=sum(1 for _ in csvpath.open())-1
        manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
        manifest['raw_sha256']=hashlib.sha256(csvpath.read_bytes()).hexdigest()
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print('COMPLETE',manifest['rows'],manifest['raw_sha256'])
    except BaseException as exc:
        (OUT/'failure.txt').write_text(repr(exc))
        raise
if __name__=='__main__': main()
