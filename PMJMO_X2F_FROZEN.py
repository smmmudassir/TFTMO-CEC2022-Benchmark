from __future__ import annotations
import numpy as np


def _reflect(x, lb, ub):
    span = ub - lb
    y = (x - lb) % (2.0 * span)
    return lb + np.where(y <= span, y, 2.0 * span - y)


class PMJMOX2F:
    """Frozen-candidate PMJMO-X2f.

    Two-core role-adaptive architecture:
      J0 differential thrust jet with low-amplitude torque correction;
      J1 confidence-gated learned-geometry jet with orthogonal-complement recovery.

    Shared mechanisms: success-history F/CR, external archive, diversity/stagnation
    sensing, synchronous reverse escape, nonlinear population reduction, and cached
    successful-direction geometry to avoid high-dimensional SVD overhead.
    """
    VERSION = "PMJMO-X2f-candidate-2026-10-04"

    def __init__(self, pop_size=None, seed=None, memory_size=6, geometry_interval=5):
        self.pop_size = pop_size
        self.seed = seed
        self.memory_size = memory_size
        self.geometry_interval = geometry_interval

    def optimize(self, func, lb, ub, max_fes, return_trace=False):
        rng = np.random.default_rng(self.seed)
        lb = np.asarray(lb, dtype=float); ub = np.asarray(ub, dtype=float)
        D = lb.size
        NP0 = self.pop_size or max(40, min(80, 10 * int(np.sqrt(D))))
        fes = 0
        checkpoints = np.unique(np.maximum(NP0, np.geomspace(max(NP0,1), max_fes, 32).astype(int)))
        cpidx=0; trace=[]

        def eval1(x):
            nonlocal fes
            if fes >= max_fes: raise StopIteration
            fes += 1
            return float(func(np.asarray(x,float)))

        X = rng.uniform(lb,ub,(NP0,D)); fit=np.array([eval1(x) for x in X])
        ib=int(np.argmin(fit)); best=X[ib].copy(); fbest=float(fit[ib])
        span_norm=np.linalg.norm(ub-lb)+1e-12
        d0=np.mean(np.linalg.norm(X-X.mean(0),axis=1))/span_norm

        H=self.memory_size; MF=np.full(H,.5); MCR=np.full(H,.5); mem=0
        archive=[]; success_dirs=[]
        qop=np.zeros(2); stagn=0; generation=0
        B_cache=None; conf_cache=0.0; last_geom_success_count=-999
        geom_interval = 1 if D <= 30 else max(2, min(self.geometry_interval, 4))

        while fes < max_fes:
            generation += 1
            NP=len(X); prog=fes/max_fes
            div=np.mean(np.linalg.norm(X-X.mean(0),axis=1))/span_norm
            q=min(1.0,div/(d0+1e-12))

            # Two-core online operator race with a weak prior favouring the robust DE jet.
            temp=.14+.55*(1-prog)
            prior=np.array([.72,.28])
            z=np.exp((qop-qop.max())/temp)*prior
            probs=z/z.sum()

            order=np.argsort(fit); ranks=np.empty(NP,int); ranks[order]=np.arange(NP)

            # Recompute learned geometry only periodically and after new directional evidence.
            if (generation % geom_interval == 0 and len(success_dirs) >= 8 and
                (D <= 30 or len(success_dirs) - last_geom_success_count >= 3)):
                R=np.asarray(success_dirs[-min(50,len(success_dirs)):])
                R=R/(np.linalg.norm(R,axis=1,keepdims=True)+1e-12)
                try:
                    _,s,Vt=np.linalg.svd(R,full_matrices=False)
                    conf=float(s[0]**2/(np.sum(s*s)+1e-12))
                    if conf>.22:
                        energy=np.cumsum(s*s)/(np.sum(s*s)+1e-12)
                        rr=max(1,min(5,int(np.searchsorted(energy,.75)+1)))
                        B_cache=Vt[:rr].T; conf_cache=conf
                    else:
                        B_cache=None; conf_cache=conf
                    last_geom_success_count=len(success_dirs)
                except np.linalg.LinAlgError:
                    B_cache=None; conf_cache=0.0
            B=B_cache; confidence=conf_cache

            Xn=X.copy(); fn=fit.copy(); rewards=[[],[]]
            SF=[]; SCR=[]; gains=[]; improved_global=False

            for i in range(NP):
                if fes>=max_fes: break
                role=int(rng.choice(2,p=probs)); mi=int(rng.integers(H))
                Fi=-1.0
                while Fi<=0: Fi=MF[mi]+.1*rng.standard_cauchy()
                Fi=min(Fi,1.0); CRi=float(np.clip(rng.normal(MCR[mi],.1),0,1))
                pfrac=.08+.12*prog; p=max(2,int(np.ceil(pfrac*NP))); pb=int(rng.choice(order[:p]))
                pool=[j for j in range(NP) if j!=i and j!=pb]
                if len(pool)<2: pool=[j for j in range(NP) if j!=i]
                r1,r2=rng.choice(pool,2,replace=False)
                g=best-X[i]; gh=g/(np.linalg.norm(g)+1e-12)
                d=X[r1]-X[r2]; dpar=np.dot(d,gh)*gh; dperp=d-dpar

                if role==0:
                    xr2=archive[rng.integers(len(archive))] if archive and rng.random()<.5 else X[r2]
                    y=X[i]+Fi*(X[pb]-X[i])+Fi*(X[r1]-xr2)
                    # Subordinate thrust-vectoring correction: preserves torque novelty without
                    # allowing the damaging standalone torque role found in ablation.
                    if q>.12 and prog<.8:
                        y=y+.05*(.5+.5*q)*Fi*dperp
                    mask=rng.random(D)<CRi
                else:
                    if B is not None and confidence>.22:
                        spread=np.median(np.linalg.norm(X-X.mean(0),axis=1))
                        if q<.18:
                            zvec=rng.normal(size=D); zvec=zvec-B@(B.T@zvec)
                            zvec/=np.linalg.norm(zvec)+1e-12
                            y=X[i]+spread*(.35+.4*(1-prog))*zvec
                        else:
                            coeff=rng.normal(size=B.shape[1])
                            y=X[pb]+B@coeff*spread*(.25*(1-prog)+.04)
                        mask=np.ones(D,bool)
                    else:
                        y=X[i]+Fi*(X[pb]-X[i])+.5*Fi*dperp
                        mask=rng.random(D)<max(CRi,.4)
                mask[int(rng.integers(D))]=True; y=np.where(mask,y,X[i])

                if stagn>=4 and q<.16 and ranks[i]>.7*(NP-1) and rng.random()<.35:
                    rho=.4+min(1.0,.08*stagn)
                    y=best-rho*(X[i]-best)+.5*Fi*d

                y=_reflect(y,lb,ub); fy=eval1(y)
                if fy<fit[i]:
                    gain=float(fit[i]-fy); rel=gain/(abs(fit[i])+1e-12)
                    rewards[role].append(np.tanh(8*rel))
                    archive.append(X[i].copy())
                    if len(archive)>NP0: archive.pop(int(rng.integers(len(archive))))
                    success_dirs.append(y-X[i])
                    if len(success_dirs)>100: success_dirs.pop(0)
                    Xn[i]=y; fn[i]=fy; SF.append(Fi); SCR.append(CRi); gains.append(max(gain,1e-300))
                    if fy<fbest: best=y.copy(); fbest=float(fy); improved_global=True
                if return_trace:
                    while cpidx<len(checkpoints) and fes>=checkpoints[cpidx]:
                        trace.append((int(fes),float(fbest))); cpidx+=1

            X,fit=Xn,fn
            for k in range(2):
                reward=np.mean(rewards[k]) if rewards[k] else 0.0
                qop[k]=.8*qop[k]+.2*reward
            if SF:
                w=np.asarray(gains)/np.sum(gains); sf=np.asarray(SF); scr=np.asarray(SCR)
                MF[mem]=np.sum(w*sf*sf)/(np.sum(w*sf)+1e-12); MCR[mem]=np.sum(w*scr); mem=(mem+1)%H
            stagn=0 if improved_global else stagn+1
            target=max(10,int(round(NP0-(NP0-10)*(fes/max_fes)**.7)))
            if len(X)>target:
                keep=np.argsort(fit)[:target]; X=X[keep]; fit=fit[keep]

        out={"xbest":best,"fbest":fbest,"fes":fes}
        if return_trace: out["trace"]=trace
        return out
