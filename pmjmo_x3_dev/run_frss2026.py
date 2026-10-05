#!/usr/bin/env python3
import argparse, csv, math, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"pmjmo_x3_dev"))

from pmjmo_x3_dev.pmjmo_x3_alpha import PMJMOX3Alpha
from pmjmo_x3_dev.frss2026 import make_function
from PMJMO_X2F_FROZEN import PMJMOX2F

VARIANTS = {
    "X3-full": lambda seed: PMJMOX3Alpha(seed=seed,use_microgeom=True,use_escape=True),
    "X3-no-geometry": lambda seed: PMJMOX3Alpha(seed=seed,use_microgeom=False,use_escape=True),
    "X3-backbone": lambda seed: PMJMOX3Alpha(seed=seed,use_microgeom=False,use_escape=False),
    "X2f-frozen": lambda seed: PMJMOX2F(seed=seed),
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dim",type=int,required=True,choices=[20,50,100])
    ap.add_argument("--func",type=int,required=True,choices=range(1,11))
    ap.add_argument("--runs",type=int,default=5)
    ap.add_argument("--budget",type=int,required=True)
    ap.add_argument("--base-seed",type=int,required=True)
    ap.add_argument("--instance-seed",type=int,required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    f,name=make_function(a.func,a.dim,a.instance_seed)
    lb=np.full(a.dim,-5.0); ub=np.full(a.dim,5.0)
    rows=[]

    for r in range(1,a.runs+1):
        seed=a.base_seed+(r-1)
        for variant,ctor in VARIANTS.items():
            opt=ctor(seed)
            t0=time.perf_counter()
            res=opt.optimize(f,lb,ub,a.budget)
            elapsed=time.perf_counter()-t0
            err=max(0.0,float(res["fbest"]))
            nfe=int(res["fes"])
            if nfe!=a.budget:
                raise RuntimeError(f"NFE mismatch {variant} D{a.dim} F{a.func} run{r}: {nfe} != {a.budget}")
            if not math.isfinite(err):
                raise RuntimeError("non-finite error")
            rows.append([variant,a.dim,a.func,name,r,seed,a.instance_seed,err,nfe,elapsed])
            print(f"RESULT,{variant},{a.dim},{a.func},{name},{r},{seed},{a.instance_seed},{err:.17e},{nfe},{elapsed:.6f}",flush=True)

    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as fh:
        w=csv.writer(fh)
        w.writerow(["variant","dimension","function","function_name","run","seed","instance_seed","error","nfe","elapsed_s"])
        w.writerows(rows)

if __name__=="__main__":
    main()
