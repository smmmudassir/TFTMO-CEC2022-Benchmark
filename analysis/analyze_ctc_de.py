#!/usr/bin/env python3
"""Publication-grade summaries for frozen CTC-DE and matched CEC-2022 references."""
from pathlib import Path
import argparse, glob
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata

REQUIRED={"algorithm","dimension","function","run","seed","error"}

def holm(rows, alpha=0.05):
    rows=sorted(rows,key=lambda x:x["p_raw"])
    m=len(rows)
    running=0.0
    for j,r in enumerate(rows):
        r["holm_threshold"]=alpha/(m-j)
        r["reject_holm"]=bool(r["p_raw"]<=r["holm_threshold"])
        running=max(running,(m-j)*r["p_raw"])
        r["p_holm"]=min(1.0,running)
    return rows

def rank_biserial(x,y):
    d=np.asarray(x,float)-np.asarray(y,float)
    d=d[np.isfinite(d) & (d!=0)]
    if len(d)==0: return 0.0
    ranks=rankdata(np.abs(d))
    pos=ranks[d>0].sum(); neg=ranks[d<0].sum()
    return float((pos-neg)/(pos+neg))

def ranks_table(med, labels, outpath):
    p=med[med["label"].isin(labels)].pivot(index=["dimension","function"],columns="label",values="error").dropna()
    if len(p)<2 or len(p.columns)<2: return
    arr=np.vstack([rankdata(row,method="average") for row in p.to_numpy()])
    pd.DataFrame({"label":list(p.columns),"average_rank":arr.mean(axis=0)})\
      .sort_values("average_rank").to_csv(outpath,index=False)
    if len(p.columns)>=3:
        stat,pv=friedmanchisquare(*[p[c].values for c in p.columns])
        pd.DataFrame([{"friedman_chi2":stat,"p_value":pv,"n_problems":len(p),"n_algorithms":len(p.columns)}])\
          .to_csv(outpath.with_name(outpath.stem+"_friedman.csv"),index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("inputs",nargs="+")
    ap.add_argument("--outdir",default="analysis/ctc_de_results")
    ap.add_argument("--strict-30",action="store_true",help="Require 30 matched runs for every algorithm/problem block")
    args=ap.parse_args()

    files=[]
    for pat in args.inputs: files += glob.glob(pat,recursive=True)
    frames=[]
    for f in sorted(set(files)):
        try:
            x=pd.read_csv(f)
        except Exception:
            continue
        if REQUIRED.issubset(x.columns):
            x=x.copy()
            x["_source"]=f
            frames.append(x)
    if not frames: raise SystemExit("No compatible result CSVs found")

    d=pd.concat(frames,ignore_index=True,sort=False)
    if "variant" not in d.columns: d["variant"]="full"
    d["variant"]=d["variant"].fillna("full").astype(str)
    d["algorithm"]=d["algorithm"].astype(str)
    d["label"]=d["algorithm"]+np.where(d["variant"].eq("full"),"","-"+d["variant"])
    d["dimension"]=d["dimension"].astype(int); d["function"]=d["function"].astype(int)
    d["run"]=d["run"].astype(int); d["seed"]=d["seed"].astype(int)
    d["error"]=pd.to_numeric(d["error"],errors="coerce")
    if not np.isfinite(d["error"]).all(): raise SystemExit("Non-finite error value detected")

    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)

    # Deduplicate exact repeated rows (e.g. same artifact discovered twice), but reject conflicting duplicates.
    key=["label","dimension","function","run","seed"]
    dup=d[d.duplicated(key,keep=False)].sort_values(key)
    if len(dup):
        conflicts=dup.groupby(key)["error"].nunique().reset_index()
        conflicts=conflicts[conflicts["error"]>1]
        if len(conflicts): raise SystemExit("Conflicting duplicate result rows detected")
        d=d.drop_duplicates(key,keep="first")

    blocks=(d.groupby(["label","dimension","function"])
              .agg(n=("error","size"),unique_runs=("run","nunique"),unique_seeds=("seed","nunique"),
                   mean=("error","mean"),median=("error","median"),std=("error","std"),
                   min=("error","min"),max=("error","max"))
              .reset_index())
    blocks.to_csv(out/"descriptive.csv",index=False)

    if args.strict_30:
        bad=blocks[(blocks["n"]!=30)|(blocks["unique_runs"]!=30)|(blocks["unique_seeds"]!=30)]
        if len(bad):
            bad.to_csv(out/"invalid_blocks.csv",index=False)
            raise SystemExit("Campaign validation failed: not every block contains 30 unique runs/seeds")

    # Check matched deterministic seed protocol where applicable.
    expected=2026220000+d["dimension"]*10000+d["function"]*100+(d["run"]-1)
    seed_ok=(d["seed"].to_numpy()==expected.to_numpy())
    pd.DataFrame([{"rows":len(d),"matched_seed_rows":int(seed_ok.sum()),"all_matched":bool(seed_ok.all())}])\
      .to_csv(out/"seed_validation.csv",index=False)
    if args.strict_30 and not seed_ok.all(): raise SystemExit("Matched-seed protocol violation")

    med=d.groupby(["label","dimension","function"])["error"].median().reset_index()
    full_labels=sorted(d.loc[d["variant"].eq("full"),"label"].unique())
    ablation_labels=sorted(d.loc[d["algorithm"].eq("CTC-DE"),"label"].unique())
    ranks_table(med,full_labels,out/"full_average_ranks.csv")
    ranks_table(med,ablation_labels,out/"ablation_average_ranks.csv")

    full=d[d["variant"].eq("full")]
    base_algorithms=sorted(full["algorithm"].unique())
    tests=[]
    if "CTC-DE" in base_algorithms:
        for other in [a for a in base_algorithms if a!="CTC-DE"]:
            pair=[]
            for (dim,fun),g in full[full["algorithm"].isin(["CTC-DE",other])].groupby(["dimension","function"]):
                a=g[g["algorithm"].eq("CTC-DE")][["seed","error"]].rename(columns={"error":"ctc"})
                b=g[g["algorithm"].eq(other)][["seed","error"]].rename(columns={"error":"other"})
                z=a.merge(b,on="seed")
                if len(z)<5: continue
                delta=z["ctc"].to_numpy()-z["other"].to_numpy()
                p=1.0 if np.allclose(delta,0) else float(wilcoxon(
                    z["ctc"],z["other"],zero_method="pratt",alternative="two-sided").pvalue)
                pair.append({"comparator":other,"dimension":dim,"function":fun,"n":len(z),
                    "p_raw":p,"rank_biserial_ctc_minus_other":rank_biserial(z["ctc"],z["other"]),
                    "median_ctc":float(np.median(z["ctc"])),"median_other":float(np.median(z["other"]))})
            tests.extend(holm(pair))
    pd.DataFrame(tests).to_csv(out/"wilcoxon_holm.csv",index=False)

    wtl=[]
    ctc=med[med["label"].eq("CTC-DE")]
    for other in [a for a in full_labels if a!="CTC-DE"]:
        j=ctc.merge(med[med["label"].eq(other)],on=["dimension","function"],suffixes=("_ctc","_other"))
        if len(j):
            scale=np.maximum(1.0,np.maximum(np.abs(j["error_ctc"]),np.abs(j["error_other"])))
            tol=1e-12*scale
            w=int((j["error_ctc"]<j["error_other"]-tol).sum())
            l=int((j["error_ctc"]>j["error_other"]+tol).sum())
            wtl.append({"comparator":other,"wins":w,"ties":int(len(j)-w-l),"losses":l,"problems":len(j)})
    pd.DataFrame(wtl).to_csv(out/"win_tie_loss.csv",index=False)

    # Compact machine-readable campaign status.
    status={"rows":len(d),"labels":len(d["label"].unique()),"problems":len(d[["dimension","function"]].drop_duplicates()),
            "strict_30":args.strict_30,"matched_seeds":bool(seed_ok.all())}
    pd.DataFrame([status]).to_csv(out/"campaign_status.csv",index=False)
    print("WROTE",out)
    print(status)

if __name__=="__main__": main()
