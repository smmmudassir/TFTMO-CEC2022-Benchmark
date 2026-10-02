import numpy as np
from TFTMO_DF import _lshade_seeded

def _cauchy_pos(rng,loc=.5,scale=.1):
    for _ in range(24):
        x=loc+scale*np.tan(np.pi*(rng.random()-.5))
        if x>0:return min(1.,x)
    return min(1.,max(1e-6,loc))

def _pg2_until_plateau(fun,dim,lb,ub,max_fes,seed,pop_size=36,min_frac=.35,max_frac=.85,patience_frac=.10):
    rng=np.random.default_rng(seed)
    lbv=np.full(dim,lb,float) if np.isscalar(lb) else np.asarray(lb,float);ubv=np.full(dim,ub,float) if np.isscalar(ub) else np.asarray(ub,float)
    span=ubv-lbv;X=rng.uniform(lbv,ubv,(pop_size,dim));fit=np.array([float(fun(x)) for x in X]);fes=pop_size
    gi=int(np.argmin(fit));g=X[gi].copy();gf=float(fit[gi]);diag=np.ones((pop_size,dim));stagn=np.zeros(pop_size,int);archive=[];muF=.5;muCR=.85
    scale0=max(float(np.std(fit)),1e-12); checkpoint=gf;last_meaningful=fes
    patience=max(pop_size,int(patience_frac*max_fes));minfes=int(min_frac*max_fes);maxfes=int(max_frac*max_fes)
    while fes<max_fes:
        progress=fes/max_fes;idxsort=np.argsort(fit);pcount=max(2,int(np.ceil(.2*pop_size)));succF=[];succCR=[]
        pop_sigma=max(np.mean(np.std(X,axis=0)),float(np.mean(span))*1e-8)
        for i in rng.permutation(pop_size):
            if fes>=max_fes:break
            F=_cauchy_pos(rng,muF,.1);CR=float(np.clip(rng.normal(muCR,.1),0,1));pbest=X[rng.choice(idxsort[:pcount])]
            ids=np.arange(pop_size);ids=ids[ids!=i];r1,r2,r3=rng.choice(ids,3,replace=False);xr2=archive[rng.integers(len(archive))] if archive and rng.random()<.5 else X[r2]
            rays=[X[i]+F*(pbest-X[i])+F*(X[r1]-xr2)]
            if progress<.72 or rng.random()<.45:rays.append(X[r1]+F*(X[r2]-X[r3]))
            center=X[i]+(.15+.45*rng.random())*(g-X[i]);sig=(.12*(1-progress)**1.7+.0025)*pop_sigma
            rays.append(center+sig*rng.standard_t(3,size=dim)*diag[i])
            Y=[];vals=[]
            for raw in rays:
                if fes>=max_fes:break
                m=rng.random(dim)<CR;m[rng.integers(dim)]=True;y=np.where(m,raw,X[i]);y=np.clip(y,lbv,ubv);fy=float(fun(y));fes+=1;Y.append(y);vals.append(fy)
            if not vals:break
            j=int(np.argmin(vals));anchor=Y[j];af=vals[j];bx=anchor;bf=af
            if fes<max_fes and (af<fit[i] or rng.random()<(.35*(1-progress)+.08)):
                ls=max(np.linalg.norm(anchor-X[i])/np.sqrt(dim),float(np.mean(span))*(.02*(1-progress)**2+1e-7))
                zraw=anchor+(.22+.28*(1-progress))*ls*rng.normal(size=dim)*diag[i];fcr=max(.15,.65*(1-progress)+.15)
                zm=rng.random(dim)<fcr;zm[rng.integers(dim)]=True;z=np.where(zm,zraw,anchor);z=np.clip(z,lbv,ubv);zf=float(fun(z));fes+=1
                if zf<bf:bx,bf=z,zf
            if bf<fit[i]:
                old=X[i].copy();oldf=float(fit[i])
                if len(archive)>=pop_size:archive.pop(int(rng.integers(len(archive))))
                archive.append(old);X[i]=bx.copy();fit[i]=bf;stagn[i]=0;succF.append((F,oldf-bf));succCR.append((CR,oldf-bf))
                step=np.abs(X[i]-old);shape=step/(step.mean()+1e-12);diag[i]=.95*diag[i]+.05*np.clip(shape,.35,2.5)
                if bf<gf:g=X[i].copy();gf=bf
            else:stagn[i]+=1
            if stagn[i]>=12 and fes<max_fes:
                top=X[rng.choice(idxsort[:max(3,pop_size//2)])];rr=(.05*(1-progress)+.003)*span;esc=np.clip(top+rr*rng.standard_t(2,size=dim),lbv,ubv);ef=float(fun(esc));fes+=1
                if ef<fit[i]:
                    if len(archive)>=pop_size:archive.pop(int(rng.integers(len(archive))))
                    archive.append(X[i].copy());X[i]=esc;fit[i]=ef
                    if ef<gf:g=esc.copy();gf=ef
                stagn[i]=0
        if succF:
            Fs=np.array([a for a,_ in succF]);w=np.array([b for _,b in succF]);w/=w.sum();leh=np.sum(w*Fs*Fs)/(np.sum(w*Fs)+1e-12);muF=.9*muF+.1*leh
            Cs=np.array([a for a,_ in succCR]);wc=np.array([b for _,b in succCR]);wc/=wc.sum();muCR=.9*muCR+.1*np.sum(wc*Cs)
        tol=max(1e-12,1e-6*scale0)
        if checkpoint-gf>tol:checkpoint=gf;last_meaningful=fes
        if fes>=maxfes or (fes>=minfes and fes-last_meaningful>=patience):break
    return g,gf,fes

def tftmo_adg(fun,dim,lb,ub,max_fes=10000,seed=1,pop_size=36,min_frac=.60,max_frac=.95,patience_frac=.08):
    x1,f1,used=_pg2_until_plateau(fun,dim,lb,ub,max_fes,seed,pop_size,min_frac,max_frac,patience_frac)
    rem=max_fes-used
    if rem<=0:return x1,f1
    x2,f2=_lshade_seeded(fun,dim,lb,ub,additional_fes=rem,seed=seed+104729,seed_x=x1,seed_f=f1)
    return (x2,f2) if f2<f1 else (x1,f1)
