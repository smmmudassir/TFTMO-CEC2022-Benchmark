import numpy as np

def _cauchy_pos(rng, loc=0.5, scale=0.1):
    for _ in range(20):
        x = loc + scale * np.tan(np.pi * (rng.random() - 0.5))
        if x > 0:
            return min(x, 1.0)
    return loc

def tftmo_pg2(fun, dim, lb, ub, max_fes=10000, seed=1, pop_size=36):
    rng = np.random.default_rng(seed)
    lbv = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ubv = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)
    span = ubv - lbv
    X = rng.uniform(lbv, ubv, size=(pop_size, dim))
    fit = np.array([fun(x) for x in X], dtype=float)
    fes = pop_size
    gi = int(np.argmin(fit)); g = X[gi].copy(); gf = float(fit[gi])
    diag = np.ones((pop_size, dim), dtype=float)
    stagn = np.zeros(pop_size, dtype=int)
    archive = []
    muF = 0.5; muCR = 0.85
    while fes < max_fes:
        progress = fes / max_fes
        idxsort = np.argsort(fit)
        pcount = max(2, int(np.ceil(0.20 * pop_size)))
        succ_F, succ_CR = [], []
        pop_sigma = max(np.mean(np.std(X, axis=0)), float(np.mean(span)) * 1e-8)
        for i in rng.permutation(pop_size):
            if fes >= max_fes: break
            F = _cauchy_pos(rng, muF, 0.1)
            CR = float(np.clip(rng.normal(muCR, 0.1), 0, 1))
            pbest = X[rng.choice(idxsort[:pcount])]
            ids = np.arange(pop_size); ids = ids[ids != i]
            r1, r2, r3 = rng.choice(ids, 3, replace=False)
            xr2 = archive[rng.integers(len(archive))] if archive and rng.random() < 0.5 else X[r2]
            rays = [X[i] + F*(pbest-X[i]) + F*(X[r1]-xr2)]
            if progress < 0.72 or rng.random() < 0.45:
                rays.append(X[r1] + F*(X[r2]-X[r3]))
            center = X[i] + (0.15 + 0.45*rng.random())*(g-X[i])
            sig = (0.12*(1-progress)**1.7 + 0.0025)*pop_sigma
            rays.append(center + sig*rng.standard_t(df=3,size=dim)*diag[i])
            Y, vals = [], []
            for raw in rays:
                if fes >= max_fes: break
                mask = rng.random(dim) < CR; mask[rng.integers(dim)] = True
                y = np.where(mask, raw, X[i]); y = np.clip(y, lbv, ubv)
                fy = float(fun(y)); fes += 1
                Y.append(y); vals.append(fy)
            if not vals: break
            j = int(np.argmin(vals)); anchor = Y[j]; anchor_f = vals[j]
            best_x, best_f = anchor, anchor_f
            if fes < max_fes and (anchor_f < fit[i] or rng.random() < (0.35*(1-progress)+0.08)):
                local_scale = max(np.linalg.norm(anchor-X[i])/np.sqrt(dim),
                                  float(np.mean(span))*(0.02*(1-progress)**2+1e-7))
                zraw = anchor + (0.22+0.28*(1-progress))*local_scale*rng.normal(size=dim)*diag[i]
                fcr = max(0.15,0.65*(1-progress)+0.15)
                zmask = rng.random(dim) < fcr; zmask[rng.integers(dim)] = True
                z = np.where(zmask,zraw,anchor); z = np.clip(z,lbv,ubv)
                zf=float(fun(z)); fes+=1
                if zf < best_f: best_x,best_f=z,zf
            if best_f < fit[i]:
                old=X[i].copy(); oldf=float(fit[i])
                if len(archive)>=pop_size: archive.pop(int(rng.integers(len(archive))))
                archive.append(old); X[i]=best_x.copy(); fit[i]=best_f; stagn[i]=0
                succ_F.append((F,oldf-best_f)); succ_CR.append((CR,oldf-best_f))
                step=np.abs(X[i]-old); shape=step/(step.mean()+1e-12)
                diag[i]=0.95*diag[i]+0.05*np.clip(shape,0.35,2.5)
                if best_f < gf: g=X[i].copy(); gf=best_f
            else:
                stagn[i]+=1
            if stagn[i]>=12 and fes<max_fes:
                top=X[rng.choice(idxsort[:max(3,pop_size//2)])]
                rr=(0.05*(1-progress)+0.003)*span
                esc=np.clip(top+rr*rng.standard_t(2,size=dim),lbv,ubv)
                ef=float(fun(esc)); fes+=1
                if ef < fit[i]:
                    if len(archive)>=pop_size: archive.pop(int(rng.integers(len(archive))))
                    archive.append(X[i].copy()); X[i]=esc; fit[i]=ef
                    if ef < gf: g=esc.copy(); gf=ef
                stagn[i]=0
        if succ_F:
            Fs=np.array([a for a,_ in succ_F]); w=np.array([b for _,b in succ_F],dtype=float)
            w/=w.sum(); lehmer=np.sum(w*Fs*Fs)/(np.sum(w*Fs)+1e-12); muF=0.9*muF+0.1*lehmer
            CRs=np.array([a for a,_ in succ_CR]); wc=np.array([b for _,b in succ_CR],dtype=float)
            wc/=wc.sum(); muCR=0.9*muCR+0.1*np.sum(wc*CRs)
    return g,gf
