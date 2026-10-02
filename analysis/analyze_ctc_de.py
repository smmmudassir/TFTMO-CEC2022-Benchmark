#!/usr/bin/env python3
"""Publication-grade summaries for CTC-DE campaign CSV files."""
from pathlib import Path
import argparse, glob, math
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata

def holm(rows, alpha=0.05):
    rows=sorted(rows,key=lambda x:x["p_raw"])
    m=len(rows)
    for j,r in enumerate(rows):
        r["holm_threshold"]=alpha/(m-j)
        r["reject_holm"]=r["p_raw"]<=r["holm_threshold"]
        r["p_holm"]=min(1.0,max((m-k)*rows[k]["p_raw"] for k in range(j+1)))
    return rows

def rank_biserial(x,y):
    d=np.asarray(x)-np.asarray(y)
    d=d[d!=0]
    if len(d)==0:return 0.0
    ranks=rankdata(np.abs(d))
    pos=ranks[d>0].sum(); neg=ranks[d<0].sum()
    return float((pos-neg)/(pos+neg))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("inputs",nargs="+")
    ap.add_argument("--outdir",default="analysis/ctc_de_results")
    args=ap.parse_args()
    files=[]
    for pat in args.inputs: files+=glob.glob(pat,recursive=True)
    frames=[]
    for f in files:
        try:
            d=pd.read_csv(f)
            need={"algorithm","dimension","function","run","seed","error"}
            if need.issubset(d.columns): frames.append(d)
        except Exception: pass
    if not frames: raise SystemExit("No compatible result CSVs found")
    d=pd.concat(frames,ignore_index=True)
    if "variant" not in d: d["variant"]="full"
    d["label"]=d["algorithm"].astype(str)+np.where(d["variant"].astype(str).eq("full"),"","-"+d["variant"].astype(str))

    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    summary=(d.groupby(["label","dimension","function"])["error"]
             .agg(["count","mean","median","std","min","max"]).reset_index())
    summary.to_csv(out/"descriptive.csv",index=False)

    # Median error per problem is the unit for cross-problem Friedman ranks.
    med=d.groupby(["label","dimension","function"])["error"].median().reset_index()
    pivot=med.pivot(index=["dimension","function"],columns="label",values="error").dropna()
    labels=list(pivot.columns)
    rank_rows=[]
    if len(labels)>=2 and len(pivot)>=2:
        ranks=np.vstack([rankdata(row,method="average") for row in pivot.to_numpy()])
        avg=ranks.mean(axis=0)
        rank_rows=pd.DataFrame({"label":labels,"friedman_avg_rank":avg}).sort_values("friedman_avg_rank")
        rank_rows.to_csv(out/"average_ranks.csv",index=False)
        if len(labels)>=3:
            stat,p=friedmanchisquare(*[pivot[c].values for c in labels])
            pd.DataFrame([{"friedman_chi2":stat,"p_value":p,"n_problems":len(pivot)}]).to_csv(out/"friedman.csv",index=False)

    # Paired run-level Wilcoxon for every problem; Holm correction within each comparator pair.
    full=d[d["variant"].astype(str).eq("full")]
    base_labels=sorted(full["algorithm"].unique())
    rows=[]
    if "CTC-DE" in base_labels:
        for other in [x for x in base_labels if x!="CTC-DE"]:
            pair=[]
            for (dim,fun),g in full[full["algorithm"].isin(["CTC-DE",other])].groupby(["dimension","function"]):
                a=g[g["algorithm"].eq("CTC-DE")][["seed","error"]].rename(columns={"error":"a"})
                b=g[g["algorithm"].eq(other)][["seed","error"]].rename(columns={"error":"b"})
                z=a.merge(b,on="seed")
                if len(z)<5: continue
                diff=z["a"].to_numpy()-z["b"].to_numpy()
                if np.allclose(diff,0):
                    p=1.0
                else:
                    p=float(wilcoxon(z["a"],z["b"],zero_method="pratt",alternative="two-sided").pvalue)
                pair.append({"comparator":other,"dimension":dim,"function":fun,"n":len(z),
                             "p_raw":p,"rank_biserial_ctc_minus_other":rank_biserial(z["a"],z["b"]),
                             "median_ctc":float(np.median(z["a"])),"median_other":float(np.median(z["b"]))})
            rows.extend(holm(pair))
    pd.DataFrame(rows).to_csv(out/"wilcoxon_holm.csv",index=False)

    # Win/tie/loss by median error.
    wtl=[]
    for other in [x for x in base_labels if x!="CTC-DE"]:
        j=(med[med["label"].eq("CTC-DE")].merge(
           med[med["label"].eq(other)],on=["dimension","function"],suffixes=("_ctc","_other")))
        if len(j):
            tol=1e-12
            w=int((j["error_ctc"]<j["error_other"]-tol).sum())
            l=int((j["error_ctc"]>j["error_other"]+tol).sum())
            t=len(j)-w-l
            wtl.append({"comparator":other,"wins":w,"ties":t,"losses":l,"problems":len(j)})
    pd.DataFrame(wtl).to_csv(out/"win_tie_loss.csv",index=False)
    print("WROTE",out)

if __name__=="__main__": main()
