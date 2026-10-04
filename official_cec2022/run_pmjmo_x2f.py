#!/usr/bin/env python3
import argparse, ctypes, csv, hashlib, math, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PMJMO_X2F_FROZEN import PMJMOX2F

OPT={1:300.0,2:400.0,3:600.0,4:800.0,5:900.0,6:1800.0,7:2000.0,8:2200.0,9:2300.0,10:2400.0,11:2600.0,12:2700.0}
EXPECTED_SHA="89d013851d408fb51930d670b95c93e7aeb5f3d186e100fd5a68bc0118ca6976"

class CEC22:
    def __init__(self, libpath, dim, func):
        self.dim=int(dim); self.func=int(func); self.count=0
        self.lib=ctypes.CDLL(str(libpath))
        self.lib.cec22_eval.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.c_int]
        self.lib.cec22_eval.restype=ctypes.c_double
    def __call__(self,x):
        a=np.ascontiguousarray(x,dtype=np.float64)
        if a.size!=self.dim: raise ValueError("dimension mismatch")
        self.count+=1
        return float(self.lib.cec22_eval(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),self.dim,self.func))

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lib",required=True)
    ap.add_argument("--dim",type=int,required=True,choices=[10,20])
    ap.add_argument("--func",type=int,required=True,choices=range(1,13))
    ap.add_argument("--runs",type=int,default=30)
    ap.add_argument("--base-seed",type=int,required=True)
    ap.add_argument("--budget",type=int,required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()

    src=ROOT/"PMJMO_X2F_FROZEN.py"
    got=sha256(src)
    if got!=EXPECTED_SHA:
        raise RuntimeError(f"FROZEN SOURCE HASH MISMATCH: {got} != {EXPECTED_SHA}")

    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in range(1,a.runs+1):
        seed=a.base_seed+(r-1)
        obj=CEC22(a.lib,a.dim,a.func)
        opt=PMJMOX2F(seed=seed)
        lb=np.full(a.dim,-100.0); ub=np.full(a.dim,100.0)
        t0=time.perf_counter()
        res=opt.optimize(obj,lb,ub,a.budget,return_trace=False)
        elapsed=time.perf_counter()-t0
        if obj.count!=a.budget or int(res["fes"])!=a.budget:
            raise RuntimeError(f"FE mismatch run {r}: evaluator={obj.count}, optimizer={res['fes']}, expected={a.budget}")
        bf=float(res["fbest"])
        err=max(0.0,bf-OPT[a.func])
        if err<=1e-8: err=0.0
        if not math.isfinite(err): raise RuntimeError("non-finite result")
        rows.append(["PMJMO-X2f-frozen",a.dim,a.func,r,seed,bf,err,obj.count,elapsed,got])
        print(f"RESULT,{a.dim},{a.func},{r},{seed},{bf:.17e},{err:.17e},{obj.count},{elapsed:.6f}",flush=True)

    with out.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["algorithm","dimension","function","run","seed","best_f","error","nfe","elapsed_s","source_sha256"])
        w.writerows(rows)

if __name__=="__main__":
    main()
