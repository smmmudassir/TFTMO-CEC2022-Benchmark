#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata

def holm(pvals):
    items=sorted(pvals.items(), key=lambda kv: kv[1])
    m=len(items); out={}; prev=0.0
    for i,(k,p) in enumerate(items):
        adj=min(1.0,(m-i)*p)
        adj=max(prev,adj); prev=adj; out[k]=adj
    return out

def cliffs_delta(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    gt=sum(a>b for a in x for b in y)
    lt=sum(a<b for a in x for b in y)
    return (gt-lt)/(len(x)*len(y))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--outdir",required=True)
    ap.add_argument("--candidate",default="TFTMO-ADG")
    a=ap.parse_args()
    df=pd.read_csv(a.input)
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)

    req={"algorithm","dimension","function","run","error"}
    if not req.issubset(df.columns): raise ValueError(req-set(df.columns))
    df=df.copy()
    df["error"]=pd.to_numeric(df["error"])
    if not np.isfinite(df["error"]).all(): raise ValueError("non-finite errors")

    # Per-function summaries
    summary=df.groupby(["algorithm","dimension","function"])["error"].agg(
        count="count",mean="mean",std="std",median="median",min="min",max="max"
    ).reset_index()
    succ=(df.assign(success=df.error<=1e-8)
             .groupby(["algorithm","dimension","function"])["success"].sum().reset_index(name="success_le_1e-8"))
    summary=summary.merge(succ,on=["algorithm","dimension","function"])
    summary.to_csv(out/"summary.csv",index=False)

    # Median-based block ranks, tie-aware, lower error is better.
    med=df.groupby(["algorithm","dimension","function"])["error"].median().unstack("algorithm")
    rank_rows=[]
    for idx,row in med.iterrows():
        ranks=rankdata(row.values,method="average")
        for alg,r in zip(row.index,ranks):
            rank_rows.append({"dimension":idx[0],"function":idx[1],"algorithm":alg,"rank":r,"median_error":row[alg]})
    ranks=pd.DataFrame(rank_rows)
    ranks.to_csv(out/"block_ranks.csv",index=False)
    mean_ranks=ranks.groupby(["dimension","algorithm"])["rank"].mean().reset_index(name="mean_rank")
    mean_ranks.to_csv(out/"mean_ranks.csv",index=False)

    # Friedman by dimension across function medians.
    fried=[]
    for dim in sorted(df.dimension.unique()):
        p=med.loc[dim]
        algs=list(p.columns)
        stat,pv=friedmanchisquare(*[p[c].values for c in algs])
        fried.append({"dimension":dim,"statistic":stat,"p_value":pv,"algorithms":len(algs),"functions":len(p)})
    pd.DataFrame(fried).to_csv(out/"friedman.csv",index=False)

    # Candidate vs every comparator: paired run-level Wilcoxon within each function,
    # then aggregate function medians for a dimension-level paired test.
    cand=a.candidate
    if cand not in df.algorithm.unique(): raise ValueError(f"candidate {cand} absent")
    others=[x for x in sorted(df.algorithm.unique()) if x!=cand]
    pair_rows=[]
    dim_rows=[]
    for dim in sorted(df.dimension.unique()):
        dim_pvals={}
        temp={}
        for alg in others:
            function_diffs=[]
            wins=ties=losses=0
            for f in sorted(df.function.unique()):
                c=df[(df.algorithm==cand)&(df.dimension==dim)&(df.function==f)].sort_values("run")
                o=df[(df.algorithm==alg)&(df.dimension==dim)&(df.function==f)].sort_values("run")
                merged=c.merge(o,on=["dimension","function","run"],suffixes=("_cand","_other"))
                if len(merged)==0: continue
                xc=merged.error_cand.values; xo=merged.error_other.values
                medc=float(np.median(xc)); medo=float(np.median(xo))
                if medc < medo: wins+=1
                elif medc > medo: losses+=1
                else: ties+=1
                try:
                    wstat,pv=wilcoxon(xc,xo,zero_method="pratt",alternative="two-sided")
                except ValueError:
                    wstat,pv=0.0,1.0
                pair_rows.append({
                    "dimension":dim,"function":f,"candidate":cand,"comparator":alg,
                    "candidate_median":medc,"comparator_median":medo,
                    "wilcoxon_stat":wstat,"p_value":pv,
                    "cliffs_delta_candidate_vs_comparator":cliffs_delta(xc,xo)
                })
                function_diffs.append((medc,medo))
            aa=np.array([x[0] for x in function_diffs]); bb=np.array([x[1] for x in function_diffs])
            try:
                ws,pv=wilcoxon(aa,bb,zero_method="pratt",alternative="two-sided")
            except ValueError:
                ws,pv=0.0,1.0
            key=alg; dim_pvals[key]=pv
            temp[alg]={"dimension":dim,"candidate":cand,"comparator":alg,"wins":wins,"ties":ties,"losses":losses,
                       "wilcoxon_over_function_medians_stat":ws,"p_value":pv}
        adj=holm(dim_pvals)
        for alg,row in temp.items():
            row["holm_adjusted_p"]=adj[alg]
            row["significant_0_05"]=adj[alg] < 0.05
            dim_rows.append(row)
    pd.DataFrame(pair_rows).to_csv(out/"candidate_pairwise_by_function.csv",index=False)
    pd.DataFrame(dim_rows).to_csv(out/"candidate_dimension_tests.csv",index=False)

    # Overall pooled descriptive ranks across the 24 dimension-function blocks.
    overall=ranks.groupby("algorithm")["rank"].mean().sort_values().reset_index(name="mean_rank_24_blocks")
    overall.to_csv(out/"overall_mean_ranks.csv",index=False)

    report={
        "candidate":cand,
        "algorithms":sorted(df.algorithm.unique().tolist()),
        "rows":int(len(df)),
        "blocks":int(df.groupby(["dimension","function"]).ngroups),
        "runs_per_algorithm_expected":int(df.groupby("algorithm").size().min()),
        "overall_mean_ranks":overall.to_dict(orient="records"),
        "friedman":fried,
        "candidate_dimension_tests":dim_rows
    }
    (out/"statistical_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
