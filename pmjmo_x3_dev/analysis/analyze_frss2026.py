#!/usr/bin/env python3
"""Reproduce the PMJMO-X3 FRSS-2026 screen analysis from extracted results.csv files."""
from pathlib import Path
from itertools import combinations
import argparse
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

VARIANTS=["X3-full","X3-no-geometry","X3-backbone","X2f-frozen"]

def holm(pvals):
    p=np.asarray(pvals,float); m=len(p); order=np.argsort(p); out=np.empty(m); running=0.0
    for k,i in enumerate(order):
        running=max(running,(m-k)*p[i]); out[i]=min(1.0,running)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",type=Path)
    ap.add_argument("--out",type=Path,default=Path("analysis_out"))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    files=sorted(a.root.glob("*/results.csv"))
    if len(files)!=30:
        raise SystemExit(f"expected 30 results.csv files, found {len(files)}")
    df=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    if len(df)!=600: raise SystemExit(f"expected 600 rows, found {len(df)}")
    if df.duplicated(["variant","dimension","function","run"]).any(): raise SystemExit("duplicate keys")
    if (~np.isfinite(df.error)).any() or (df.error<0).any(): raise SystemExit("invalid errors")
    if (df.nfe != 1000*df.dimension).any(): raise SystemExit("NFE mismatch")
    block=df.groupby(["dimension","function","function_name","variant"],as_index=False).agg(median_error=("error","median"))
    block["rank"]=block.groupby(["dimension","function"])["median_error"].rank(method="average",ascending=True)
    piv=block.pivot_table(index=["dimension","function","function_name"],columns="variant",values="median_error")
    rkpiv=block.pivot_table(index=["dimension","function","function_name"],columns="variant",values="rank")
    overall=block.groupby("variant")["rank"].mean().sort_values().rename("mean_rank")
    dim=block.groupby(["variant","dimension"])["rank"].mean().unstack()
    fr=friedmanchisquare(*[piv[v].values for v in VARIANTS])
    rows=[]
    for x,y in combinations(VARIANTS,2):
        z=wilcoxon((rkpiv[x]-rkpiv[y]).values,zero_method="pratt",alternative="two-sided",method="auto")
        rows.append([x,y,float(z.statistic),float(z.pvalue)])
    pair=pd.DataFrame(rows,columns=["A","B","W","p"]); pair["p_holm"]=holm(pair.p.values)
    df.to_csv(a.out/"merged.csv",index=False)
    block.to_csv(a.out/"block_summary.csv",index=False)
    overall.to_csv(a.out/"overall_ranks.csv")
    dim.to_csv(a.out/"dimension_ranks.csv")
    pair.to_csv(a.out/"wilcoxon_holm.csv",index=False)
    print(overall.to_string()); print(dim.to_string())
    print(f"Friedman chi2={fr.statistic:.6f} p={fr.pvalue:.6g}")

if __name__=="__main__":
    main()
