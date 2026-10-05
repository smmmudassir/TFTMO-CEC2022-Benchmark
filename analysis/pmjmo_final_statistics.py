#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon, f as fdist

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"analysis"
pm=pd.read_csv(A/"pmjmo_x2f_cec2022_720runs.csv")
ref=pd.read_csv(A/"reference_4alg_block_summary.csv")

assert len(pm)==720
assert pm.groupby(["dimension","function"]).size().eq(30).all()
assert pm["seed"].nunique()==720
assert np.isfinite(pm["error"]).all()
assert (pm["nfe"]==pm["dimension"].map({10:200000,20:1000000})).all()

pms=(pm.groupby(["algorithm","dimension","function"])["error"]
       .agg(["count","mean","std","median","min","max"]).reset_index())
pms["success_le_1e-8"]=(pm.assign(success=pm.error<=1e-8)
    .groupby(["algorithm","dimension","function"])["success"].sum().to_numpy())
blocks=pd.concat([ref,pms],ignore_index=True,sort=False)

algs=["PMJMO-X2f-frozen","RDEx-SOP","jSO","L-SRTDE","L-SHADE-1.0.1"]
blocks=blocks[blocks.algorithm.isin(algs)].copy()
assert len(blocks)==120
assert blocks.groupby(["algorithm","dimension"]).size().eq(12).all()

# rank each dimension/function block on median error
blocks["rank"]=blocks.groupby(["dimension","function"])["median"].rank(method="average",ascending=True)
per=blocks.pivot_table(index=["dimension","function"],columns="algorithm",values=["median","rank"],aggfunc="first")
per=per.sort_index()

rank_table=[]
for alg in algs:
    d10=float(blocks[(blocks.algorithm==alg)&(blocks.dimension==10)]["rank"].mean())
    d20=float(blocks[(blocks.algorithm==alg)&(blocks.dimension==20)]["rank"].mean())
    overall=float(blocks[blocks.algorithm==alg]["rank"].mean())
    rank_table.append([alg,d10,d20,overall])
rank_df=pd.DataFrame(rank_table,columns=["algorithm","D10_mean_rank","D20_mean_rank","overall_mean_rank"])
rank_df["overall_rank_position"]=rank_df["overall_mean_rank"].rank(method="min").astype(int)
rank_df=rank_df.sort_values("overall_mean_rank")

def omnibus(sub):
    piv=sub.pivot(index=["dimension","function"],columns="algorithm",values="median")[algs]
    chi,p=friedmanchisquare(*[piv[a].to_numpy() for a in algs])
    n=len(piv); k=len(algs)
    F=((n-1)*chi)/(n*(k-1)-chi) if (n*(k-1)-chi)>0 else np.inf
    fp=fdist.sf(F,k-1,(k-1)*(n-1)) if np.isfinite(F) else 0.0
    return n,chi,p,F,fp
om=[]
for scope,sub in [("D10",blocks[blocks.dimension==10]),("D20",blocks[blocks.dimension==20]),("overall_24_blocks",blocks)]:
    om.append([scope,*omnibus(sub)])
om_df=pd.DataFrame(om,columns=["scope","n_blocks","friedman_chi2","friedman_p","iman_davenport_F","iman_davenport_p"])

# W/T/L by block medians and Wilcoxon on per-block ranks
comparators=["RDEx-SOP","jSO","L-SRTDE","L-SHADE-1.0.1"]
pair=[]
for scope,sub in [("D10",blocks[blocks.dimension==10]),("D20",blocks[blocks.dimension==20]),("overall_24_blocks",blocks)]:
    med=sub.pivot(index=["dimension","function"],columns="algorithm",values="median")
    rr=sub.pivot(index=["dimension","function"],columns="algorithm",values="rank")
    for c in comparators:
        a=med["PMJMO-X2f-frozen"].to_numpy(); b=med[c].to_numpy()
        eq=np.isclose(a,b,rtol=1e-12,atol=1e-12)
        W=int(np.sum((a<b)&~eq)); L=int(np.sum((a>b)&~eq)); T=int(np.sum(eq))
        d=(rr[c]-rr["PMJMO-X2f-frozen"]).to_numpy() # positive => PMJMO rank advantage
        nz=d[np.abs(d)>1e-15]
        if len(nz)==0:
            stat=0.0; pv=1.0; rbc=0.0
        else:
            stat,pv=wilcoxon(d,zero_method="wilcox",alternative="two-sided",method="auto")
            ar=rankdata(np.abs(nz),method="average")
            wp=float(ar[nz>0].sum()); wm=float(ar[nz<0].sum())
            rbc=(wp-wm)/(wp+wm) if (wp+wm)>0 else 0.0
        pair.append([scope,c,W,T,L,float(stat),float(pv),float(rbc)])
pair_df=pd.DataFrame(pair,columns=["scope","comparator","W","T","L","wilcoxon_stat_on_block_ranks","p_raw","rank_biserial_pmjmo_advantage"])

# Holm within each scope
pair_df["p_holm"]=np.nan
for scope,g in pair_df.groupby("scope"):
    ix=g.index.to_list()
    order=sorted(ix,key=lambda i:pair_df.loc[i,"p_raw"])
    m=len(order); adj=[]; prev=0.0
    for j,i in enumerate(order):
        val=min(1.0,(m-j)*pair_df.loc[i,"p_raw"])
        val=max(val,prev); prev=val; adj.append((i,val))
    for i,val in adj: pair_df.loc[i,"p_holm"]=val

# per-block printable table
pb=blocks.pivot(index=["dimension","function"],columns="algorithm",values="median")[algs].reset_index()
rpb=blocks.pivot(index=["dimension","function"],columns="algorithm",values="rank")[algs].reset_index()
for alg in algs: pb[alg+"_rank"]=rpb[alg]

rank_df.to_csv(A/"pmjmo_final_rank_table.csv",index=False)
om_df.to_csv(A/"pmjmo_final_omnibus.csv",index=False)
pair_df.to_csv(A/"pmjmo_final_pairwise.csv",index=False)
pb.to_csv(A/"pmjmo_final_per_function.csv",index=False)
pms.to_csv(A/"pmjmo_block_summary.csv",index=False)

report=[]
report.append("# PMJMO-X2f Final CEC-2022 Comparison\n")
report.append("## Integrity\n")
report.append(f"- PMJMO runs: {len(pm)}\n- PMJMO blocks: {pm.groupby(['dimension','function']).ngroups}\n- All blocks have 30 runs: {pm.groupby(['dimension','function']).size().eq(30).all()}\n- Exact FE budgets verified: {(pm['nfe']==pm['dimension'].map({10:200000,20:1000000})).all()}\n- Finite errors: {np.isfinite(pm['error']).all()}\n")
report.append("## Mean ranks\n"+rank_df.to_markdown(index=False)+"\n")
report.append("## Omnibus tests\n"+om_df.to_markdown(index=False)+"\n")
report.append("## Pairwise PMJMO comparisons\n"+pair_df.to_markdown(index=False)+"\n")
report.append("## Per-function median errors and ranks\n"+pb.to_markdown(index=False)+"\n")
(A/"PMJMO_FINAL_CEC2022_REPORT.md").write_text("\n".join(report))

print("===INTEGRITY===")
print(json.dumps({"pmjmo_runs":len(pm),"blocks":pm.groupby(["dimension","function"]).ngroups,
"all_30":bool(pm.groupby(["dimension","function"]).size().eq(30).all()),
"nfe_ok":bool((pm["nfe"]==pm["dimension"].map({10:200000,20:1000000})).all()),
"finite":bool(np.isfinite(pm["error"]).all())}))
print("===RANKS===")
print(rank_df.to_csv(index=False))
print("===OMNIBUS===")
print(om_df.to_csv(index=False))
print("===PAIRWISE===")
print(pair_df.to_csv(index=False))
print("===PER_FUNCTION===")
print(pb.to_csv(index=False))
