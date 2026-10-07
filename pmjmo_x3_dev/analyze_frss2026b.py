#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

EXPECTED_VARIANTS=["X3-beta","X3-alpha-no-geometry","X3-alpha-full","X2f-frozen"]
EXPECTED_DIMS=[20,50,100]
EXPECTED_FUNCS=list(range(1,11))
EXPECTED_RUNS=30

def holm_adjust(pvals):
    pairs=sorted(enumerate(pvals),key=lambda x:x[1]); out=[None]*len(pvals)
    running=0.0; m=len(pvals)
    for k,(idx,p) in enumerate(pairs):
        adj=min(1.0,(m-k)*p); running=max(running,adj); out[idx]=running
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-root",required=True)
    ap.add_argument("--out-dir",required=True)
    a=ap.parse_args()
    root=Path(a.input_root); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    csvs=sorted(root.rglob("results.csv"))
    if len(csvs)!=30: raise RuntimeError(f"Expected 30 results.csv files, found {len(csvs)}")
    frames=[]
    for p in csvs:
        d=pd.read_csv(p); d["source_file"]=str(p); frames.append(d)
    df=pd.concat(frames,ignore_index=True)
    expected_rows=30*EXPECTED_RUNS*len(EXPECTED_VARIANTS)
    if len(df)!=expected_rows: raise RuntimeError(f"Expected {expected_rows} rows, found {len(df)}")
    if set(df["variant"])!=set(EXPECTED_VARIANTS): raise RuntimeError("Variant coverage mismatch")
    if sorted(df["dimension"].unique().tolist())!=EXPECTED_DIMS: raise RuntimeError("Dimension coverage mismatch")
    if sorted(df["function"].unique().tolist())!=EXPECTED_FUNCS: raise RuntimeError("Function coverage mismatch")
    if not df["error"].map(math.isfinite).all(): raise RuntimeError("Non-finite errors present")
    if df.duplicated(["dimension","function","variant","run"]).any(): raise RuntimeError("Duplicate run keys")
    nrun=df.groupby(["dimension","function","variant"])["run"].nunique()
    if not nrun.eq(EXPECTED_RUNS).all(): raise RuntimeError("Run count mismatch")
    seed_match=df.groupby(["dimension","function","run"])["seed"].nunique()
    if not seed_match.eq(1).all(): raise RuntimeError("Algorithm seeds not matched across variants")
    inst=df.groupby(["dimension","function"])["instance_seed"].nunique()
    if not inst.eq(1).all(): raise RuntimeError("Instance seed mismatch within block")
    expected_nfe=df["dimension"]*1000
    if not (df["nfe"].astype(int).to_numpy()==expected_nfe.astype(int).to_numpy()).all():
        raise RuntimeError("NFE budget mismatch")
    df.to_csv(out/"pmjmo_x3beta_frss2026b_merged.csv",index=False)

    block=(df.groupby(["dimension","function","variant"],as_index=False)
             .agg(median_error=("error","median"),mean_error=("error","mean"),
                  std_error=("error","std"),min_error=("error","min"),
                  max_error=("error","max"),median_elapsed_s=("elapsed_s","median")))
    ranks=[]
    for (D,F),g in block.groupby(["dimension","function"],sort=True):
        gg=g.set_index("variant").loc[EXPECTED_VARIANTS]
        rr=rankdata(gg["median_error"].to_numpy(),method="average")
        for v,r in zip(EXPECTED_VARIANTS,rr): ranks.append([D,F,v,float(r)])
    rank_df=pd.DataFrame(ranks,columns=["dimension","function","variant","rank"])
    block=block.merge(rank_df,on=["dimension","function","variant"],how="left")
    block.to_csv(out/"pmjmo_x3beta_frss2026b_block_summary.csv",index=False)

    overall=(block.groupby("variant",as_index=False)
                  .agg(mean_rank=("rank","mean"),median_error=("median_error","median"))
                  .sort_values(["mean_rank","median_error"]))
    overall.to_csv(out/"pmjmo_x3beta_frss2026b_overall_ranks.csv",index=False)
    by_dim=(block.groupby(["dimension","variant"],as_index=False)
                 .agg(mean_rank=("rank","mean"),median_error=("median_error","median"))
                 .sort_values(["dimension","mean_rank"]))
    by_dim.to_csv(out/"pmjmo_x3beta_frss2026b_dimension_ranks.csv",index=False)

    pivot=block.pivot(index=["dimension","function"],columns="variant",values="median_error")
    pivot=pivot.loc[:,EXPECTED_VARIANTS]
    fried=friedmanchisquare(*[pivot[v].to_numpy() for v in EXPECTED_VARIANTS])
    beta=pivot["X3-beta"].to_numpy(); comps=[]; raw=[]
    for v in EXPECTED_VARIANTS[1:]:
        other=pivot[v].to_numpy(); diff=beta-other
        wins=int(np.sum(diff<0)); ties=int(np.sum(diff==0)); losses=int(np.sum(diff>0))
        try:
            stat,p=wilcoxon(beta,other,alternative="two-sided",zero_method="pratt",method="auto")
            stat=float(stat); p=float(p)
        except ValueError:
            stat,p=0.0,1.0
        comps.append({"comparison":f"X3-beta vs {v}","opponent":v,
                      "wins":wins,"ties":ties,"losses":losses,
                      "wilcoxon_stat":stat,"p_raw":p,
                      "median_signed_difference":float(np.median(diff))})
        raw.append(p)
    for row,padj in zip(comps,holm_adjust(raw)):
        row["p_holm"]=float(padj); row["holm_significant_0.05"]=bool(padj<0.05)
    comp_df=pd.DataFrame(comps)
    comp_df.to_csv(out/"pmjmo_x3beta_frss2026b_wilcoxon_holm.csv",index=False)

    best=str(overall.iloc[0]["variant"])
    parent=comp_df.loc[comp_df["opponent"]=="X3-alpha-no-geometry"].iloc[0]
    freeze=bool(best=="X3-beta" and int(parent["wins"])>int(parent["losses"])
                and float(parent["median_signed_difference"])<0.0
                and bool(parent["holm_significant_0.05"]))
    rule=("Freeze only if X3-beta is #1 by overall mean block rank, has more block wins "
          "than losses versus X3-alpha-no-geometry, has a negative median signed block-error "
          "difference versus that parent, and the paired Wilcoxon comparison versus that parent "
          "remains significant at alpha=0.05 after Holm correction across the three pre-specified "
          "X3-beta comparisons.")
    decision={"suite":"FRSS-2026B","rows":int(len(df)),"blocks":30,
              "runs_per_block_variant":EXPECTED_RUNS,
              "friedman_chi2":float(fried.statistic),"friedman_p":float(fried.pvalue),
              "best_variant_by_mean_rank":best,
              "x3_beta_mean_rank":float(overall.loc[overall["variant"]=="X3-beta","mean_rank"].iloc[0]),
              "parent_comparison":parent.to_dict(),"freeze_x3_beta":freeze,"freeze_rule":rule}
    (out/"pmjmo_x3beta_frss2026b_decision.json").write_text(json.dumps(decision,indent=2,default=str)+"\n")
    lines=["# PMJMO-X3beta FRSS-2026B confirmation","",
           f"- Records: {len(df)}",f"- Friedman chi2: {fried.statistic:.6g}",
           f"- Friedman p: {fried.pvalue:.6g}",f"- Best mean-rank variant: **{best}**",
           f"- Freeze X3-beta: **{'YES' if freeze else 'NO'}**","","## Overall ranks","",
           overall.to_markdown(index=False),"","## X3-beta pairwise confirmation tests","",
           comp_df.to_markdown(index=False),"","## Pre-specified freeze rule","",rule,""]
    (out/"PMJMO_X3BETA_FRSS2026B_CONFIRMATION.md").write_text("\n".join(lines))
    print(json.dumps(decision,indent=2,default=str))

if __name__=="__main__":
    main()
