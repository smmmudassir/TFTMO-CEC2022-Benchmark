#!/usr/bin/env python3
"""Finalize frozen CTC-DE publication evidence from downloaded GitHub Actions artifacts.

Expected directory layout after 'gh run download':
  inputs/ctc/.../results.csv
  inputs/ablation/.../results.csv
  inputs/lshade/.../results.csv
  inputs/jso/.../results.csv
  inputs/rdex/.../results.csv
  inputs/lsrtde/.../results.csv
"""
from pathlib import Path
import argparse, glob, json, hashlib
import numpy as np
import pandas as pd

FROZEN_SHA="70bc55e8cb21d1ad65ab0f3690c6786ee364591d05c6c37351fb66c5a55427a5"
ALGS={"CTC-DE","L-SHADE-1.0.1","jSO","RDEx-SOP","L-SRTDE"}
VARIANTS={"full","no_decoy","no_escape","no_reallocation","base"}

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",nargs="?",default="inputs")
    ap.add_argument("--out",default="final")
    args=ap.parse_args()
    root=Path(args.root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    frames=[]; sources=[]
    for f in root.rglob("results.csv"):
        d=pd.read_csv(f)
        need={"algorithm","dimension","function","run","seed","error"}
        if not need.issubset(d.columns): continue
        d=d.copy()
        if "variant" not in d.columns: d["variant"]="full"
        d["variant"]=d["variant"].fillna("full").astype(str)
        d["_source"]=str(f)
        frames.append(d); sources.append({"path":str(f),"sha256":sha256(f),"rows":len(d)})
    if not frames: raise SystemExit("No result CSVs found")
    x=pd.concat(frames,ignore_index=True,sort=False)

    # canonical types
    for c in ["dimension","function","run","seed"]: x[c]=pd.to_numeric(x[c],errors="raise").astype(int)
    x["error"]=pd.to_numeric(x["error"],errors="raise")
    if not np.isfinite(x["error"]).all(): raise SystemExit("Non-finite objective errors")

    # protocol checks
    expected_seed=2026220000+x.dimension*10000+x.function*100+(x.run-1)
    if not (x.seed.to_numpy()==expected_seed.to_numpy()).all():
        bad=x.loc[x.seed.to_numpy()!=expected_seed.to_numpy(),["algorithm","variant","dimension","function","run","seed"]]
        bad.to_csv(out/"bad_seeds.csv",index=False); raise SystemExit("Matched seed violation")

    x["label"]=x.algorithm.astype(str)+np.where(x.variant.eq("full"),"","-"+x.variant)
    key=["label","dimension","function","run","seed"]
    dup=x[x.duplicated(key,keep=False)]
    if len(dup):
        conflicts=dup.groupby(key).error.nunique()
        if (conflicts>1).any(): raise SystemExit("Conflicting duplicate rows")
        x=x.drop_duplicates(key)

    # Full campaign completeness: 5 algorithms * 24 blocks * 30.
    full=x[x.variant.eq("full") & x.algorithm.isin(ALGS)]
    fb=full.groupby(["algorithm","dimension","function"]).agg(n=("error","size"),runs=("run","nunique"),seeds=("seed","nunique")).reset_index()
    expected_full=5*2*12*30
    full_ok=(len(full)==expected_full and len(fb)==5*24 and (fb[["n","runs","seeds"]]==30).all().all())

    # Ablation completeness: 4 non-full variants * 24 * 30.
    abl=x[x.algorithm.eq("CTC-DE") & x.variant.isin(VARIANTS-{"full"})]
    ab=abl.groupby(["variant","dimension","function"]).agg(n=("error","size"),runs=("run","nunique"),seeds=("seed","nunique")).reset_index()
    expected_abl=4*2*12*30
    ablation_ok=(len(abl)==expected_abl and len(ab)==4*24 and (ab[["n","runs","seeds"]]==30).all().all())

    fb.to_csv(out/"full_block_validation.csv",index=False)
    ab.to_csv(out/"ablation_block_validation.csv",index=False)
    pd.DataFrame(sources).to_csv(out/"source_manifest.csv",index=False)
    x.to_csv(out/"all_results_canonical.csv",index=False)

    status={
      "frozen_optimizer_sha256":FROZEN_SHA,
      "full_rows":int(len(full)),"expected_full_rows":expected_full,"full_complete":bool(full_ok),
      "ablation_rows":int(len(abl)),"expected_ablation_rows":expected_abl,"ablation_complete":bool(ablation_ok),
      "all_rows":int(len(x))
    }
    (out/"validation_status.json").write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))
    if not full_ok: raise SystemExit("Full comparison campaign incomplete/invalid")
    if not ablation_ok: raise SystemExit("Ablation campaign incomplete/invalid")

if __name__=="__main__": main()
