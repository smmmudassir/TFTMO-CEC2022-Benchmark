import numpy as np
from TFTMO_PG2 import tftmo_pg2

def _sample_cauchy(rng,loc,scale=0.1):
    for _ in range(50):
        v=loc+scale*np.tan(np.pi*(rng.random()-0.5))
        if v>0: return min(1.0,v)
    return min(1.0,max(1e-6,loc))

def _lshade_seeded(fun,dim,lb,ub,additional_fes,seed,seed_x,seed_f,np_init=None,np_min=4,H=6):
    rng=np.random.default_rng(seed)
    lbv=np.full(dim,lb,float) if np.isscalar(lb) else np.asarray(lb,float)
    ubv=np.full(dim,ub,float) if np.isscalar(ub) else np.asarray(ub,float)
    if np_init is None: np_init=18*dim
    NP=max(np_min,int(np_init)); NP0=NP
    X=rng.uniform(lbv,ubv,size=(NP,dim)); X[0]=np.asarray(seed_x,float)
    fit=np.empty(NP,float); fit[0]=float(seed_f)
    for j in range(1,NP): fit[j]=float(fun(X[j]))
    fes=NP-1
    MF=np.full(H,0.5); MCR=np.full(H,0.5); k=0; archive=[]
    gi=int(np.argmin(fit)); g=X[gi].copy(); gf=float(fit[gi])
    while fes<additional_fes and NP>=4:
        order=np.argsort(fit); succF=[]; succCR=[]; improvements=[]; new_archive=[]
        pmin=2.0/NP
        for i in rng.permutation(NP):
            if fes>=additional_fes: break
            r=int(rng.integers(H)); F=_sample_cauchy(rng,MF[r],0.1)
            CR=0.0 if MCR[r]<0 else float(np.clip(rng.normal(MCR[r],0.1),0,1))
            p=0.2 if pmin>=0.2 else rng.uniform(pmin,0.2); pnum=max(2,min(NP,int(np.ceil(p*NP))))
            pbest=int(rng.choice(order[:pnum]))
            ids=np.arange(NP); ids=ids[ids!=i]; r1=int(rng.choice(ids))
            total=NP+len(archive)
            while True:
                rr=int(rng.integers(total))
                if rr<NP:
                    if rr!=i and rr!=r1: xr2=X[rr]; break
                else: xr2=archive[rr-NP]; break
            v=X[i]+F*(X[pbest]-X[i])+F*(X[r1]-xr2)
            low=v<lbv; high=v>ubv
            v[low]=(lbv[low]+X[i][low])/2.0; v[high]=(ubv[high]+X[i][high])/2.0
            mask=rng.random(dim)<CR; mask[int(rng.integers(dim))]=True
            u=np.where(mask,v,X[i]); fu=float(fun(u)); fes+=1
            if fu<=fit[i]:
                oldf=float(fit[i]); new_archive.append(X[i].copy()); X[i]=u; fit[i]=fu
                if fu<oldf:
                    succF.append(F); succCR.append(CR); improvements.append(oldf-fu)
                if fu<gf: gf=fu; g=u.copy()
        archive.extend(new_archive)
        if len(archive)>NP:
            sel=rng.choice(len(archive),size=NP,replace=False); archive=[archive[int(j)] for j in sel]
        if succF:
            w=np.asarray(improvements,float); w/=w.sum()+1e-30; Fs=np.asarray(succF); CRs=np.asarray(succCR)
            MF[k]=np.sum(w*Fs*Fs)/(np.sum(w*Fs)+1e-30)
            MCR[k]=np.sum(w*CRs*CRs)/(np.sum(w*CRs)+1e-30) if np.sum(w*CRs)>0 else -1.0
            k=(k+1)%H
        target=int(round(np_min+(NP0-np_min)*(1-fes/max(1,additional_fes))))
        target=max(np_min,min(NP,target))
        if target<NP:
            keep=np.argsort(fit)[:target]; X=X[keep]; fit=fit[keep]; NP=target
            if len(archive)>NP:
                sel=rng.choice(len(archive),size=NP,replace=False); archive=[archive[int(j)] for j in sel]
    return g,gf
