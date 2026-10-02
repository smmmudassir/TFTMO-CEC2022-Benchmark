#!/usr/bin/env python3
import argparse, ctypes, csv, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithms"))
from CTC_DE import ctc_de

OPT = {1:300.0,2:400.0,3:600.0,4:800.0,5:900.0,6:1800.0,
       7:2000.0,8:2200.0,9:2300.0,10:2400.0,11:2600.0,12:2700.0}

class CEC22:
    def __init__(self, libpath, dim, func):
        self.dim = int(dim)
        self.func = int(func)
        self.count = 0
        self.lib = ctypes.CDLL(str(libpath))
        self.lib.cec22_eval.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
        self.lib.cec22_eval.restype = ctypes.c_double
    def __call__(self, x):
        a = np.ascontiguousarray(x, dtype=np.float64)
        if a.size != self.dim:
            raise ValueError("dimension mismatch")
        self.count += 1
        return float(self.lib.cec22_eval(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.dim, self.func))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--func", type=int, required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--base-seed", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for run in range(1, args.runs + 1):
        seed = args.base_seed + run - 1
        obj = CEC22(args.lib, args.dim, args.func)
        t0 = time.perf_counter()
        bx, bf = ctc_de(obj, args.dim, -100.0, 100.0, max_fes=args.budget,
                        seed=seed, variant=args.variant)
        elapsed = time.perf_counter() - t0

        if obj.count != args.budget:
            raise RuntimeError(f"FE mismatch run {run}: {obj.count} != {args.budget}")
        err = max(0.0, float(bf) - OPT[args.func])
        if err <= 1e-8:
            err = 0.0
        if not math.isfinite(err):
            raise RuntimeError("non-finite result")

        rows.append(["CTC-DE", args.variant, args.dim, args.func, run, seed,
                     err, obj.count, elapsed])
        print(f"RESULT,{args.variant},{args.dim},{args.func},{run},{seed},"
              f"{err:.17e},{obj.count},{elapsed:.6f}", flush=True)

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm","variant","dimension","function","run","seed",
                    "error","nfe","elapsed_s"])
        w.writerows(rows)

if __name__ == "__main__":
    main()
