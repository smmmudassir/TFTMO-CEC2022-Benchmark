"""PE-TFTMO-CRF Stage-C v0.3 DEVELOPMENT implementation. NOT frozen for publication.

The 14-variable Stage-B DAB surrogate evaluator is immutable. All objective calls
made during optimization count against `budget`. Archive stores prior evaluated
candidates; rereading metrics or archive vectors does not consume hidden FEs.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .eval import evaluate
from scipy.optimize import minimize

DIM=14
LO=-100.
HI=100.
GROUPS=(np.arange(0,6),np.arange(6,12),np.arange(12,14))

@dataclass
class Entry:
    x: np.ndarray
    score: float
    m: dict

class CountedEvaluator:
    def __init__(self, budget: int):
        self.budget=int(budget)
        self.fes=0
        self.first_feasible=None
        self.best=None
    def call(self, x):
        if self.fes >= self.budget:
            return None
        a=np.clip(np.asarray(x,float),LO,HI)
        m=evaluate(a,metrics=True)
        self.fes+=1
        e=Entry(a.copy(),float(m['score']),m)
        if m['feasible'] and self.first_feasible is None: self.first_feasible=self.fes
        if self.best is None or e.score < self.best.score: self.best=e
        return e

class DiversityArchive:
    def __init__(self, cap=12, minimum_distance=0.03):
        self.cap=cap
        self.threshold=minimum_distance
        self.items=[]
    def add(self, e):
        if not np.isfinite(e.score):return
        distances=[np.linalg.norm((e.x-a.x)/200)/np.sqrt(DIM) for a in self.items]
        if distances and min(distances)<self.threshold:
            j=int(np.argmin(distances))
            if e.score<self.items[j].score:self.items[j]=e
        else: self.items.append(e)
        self.items.sort(key=lambda a:a.score)
        del self.items[self.cap:]
    def sample(self,rng):
        if not self.items: return None
        return self.items[int(rng.integers(len(self.items)))]

# Surrogate uses sum of positive normalized violations (not the squared-CV alternative).
def boundary_recovery(x,m,rng,scale=1.0):
    z=np.clip(np.asarray(x,float).copy(),LO,HI)
    c=max(m['max_ipk']/150.-1,m['max_irms']/120.-1)
    tracking=m['max_track']/.08-1
    zvs=(.70-m['min_zvs_high'])/.70
    if c>0:
        z[1]+=min(15,3+13*c)*scale
        z[2]+=min(5,3*c)*scale
    elif tracking>0:
        z[1]-=min(14,3+12*tracking)*scale
        z[2]-=min(6,4*tracking)*scale
    if zvs>0:
        z[1]-=min(10,2+13*zvs)*scale
        z[3]+=min(5,2+5*zvs)*scale
    if m['max_ripple']>.02: z[4]+=min(20,7+20*(m['max_ripple']/.02-1))*scale
    if m['max_tj']>150:z[5]+=min(25,8+15*(m['max_tj']/150-1))*scale
    if m['bpk']>.30:z[2]+=min(15,6+10*(m['bpk']/.30-1))*scale
    # Discrete grid in min_zvs_high makes gradient-only repair unreliable.
    z+=rng.normal(0,0.9*scale,DIM)
    return np.clip(z,LO,HI)

def reactance_repair(x,rng):
    """Restore grossly invalid transfer reactance without using the calibration anchor."""
    z=np.clip(np.asarray(x,float).copy(),LO,HI)
    u=(z+100.)/200
    n=.8+1.4*u[0]; L=(5+75*u[1])*1e-6; fs=(50+150*u[2])*1e3
    X=2*np.pi*fs*L
    if X<11 or X>23:
        target=np.clip(15.5+2*(n-1.7)+rng.normal(0,1.8),11.,23.)
        Ln=target/(2*np.pi*fs)
        z[1]=200*np.clip((Ln*1e6-5)/75,0,1)-100
    return np.clip(z,LO,HI)

def solve(seed:int,budget:int=10000, npop:int=36, *, ablate_recovery=False,
          ablate_archive=False,ablate_switch=False):
    """Return best design and auditable diagnostics; no anchor injection."""
    if budget<npop: raise ValueError('budget must exceed initial population')
    rng=np.random.default_rng(int(seed));ev=CountedEvaluator(budget)
    X=rng.uniform(LO,HI,(npop,DIM));pop=[]
    for i,x in enumerate(X):
        if i<npop//3 and not ablate_recovery: x=reactance_repair(x,rng)
        pop.append(ev.call(x))
    Afeas=DiversityArchive(12); Anear=DiversityArchive(12)
    def archives(e):
        if e is None or ablate_archive:return
        if e.m['feasible']:Afeas.add(e)
        elif e.m['cv']<1.0:Anear.add(e)
    for e in pop:archives(e)
    step_diag=np.ones((npop,DIM));stagn=np.zeros(npop,int)
    mu_f=.5;mu_cr=.85;phase_counts={'recovery':0,'capture':0,'optimize':0,'refine':0,'sqp':0};
    last_best=ev.best.score;last_improvement=ev.fes
    until=int(.72*budget)
    while ev.fes < until:
        order=np.argsort([e.score for e in pop]); feasp=float(np.mean([e.m['feasible'] for e in pop]))
        phase=('recovery' if len(Afeas.items)==0 and ev.first_feasible is None else
               'capture' if feasp<.20 else 'optimize')
        if ablate_switch:phase='optimize'
        succF=[];succCR=[];deltas=[]
        for i in rng.permutation(npop):
            if ev.fes>=until: break
            before=pop[i]
            fi=float(np.clip(mu_f+.1*rng.standard_cauchy(),.08,1.))
            cr=float(np.clip(rng.normal(mu_cr,.12),.05,1))
            ids=np.array([k for k in range(npop) if k!=i]); r1,r2,r3=rng.choice(ids,3,replace=False)
            pb=pop[int(rng.choice(order[:max(2,npop//5)]))]
            archive_parent=Afeas.sample(rng) if phase!='recovery' else Anear.sample(rng)
            if archive_parent is None:archive_parent=pb
            xp=before.x; best_new=None
            rays=[xp+fi*(pb.x-xp)+fi*(pop[r1].x-pop[r2].x),
                  pop[r1].x+fi*(pop[r2].x-pop[r3].x)]
            g=GROUPS[int(rng.integers(len(GROUPS)))];ray=xp.copy()
            ray[g]=archive_parent.x[g]+rng.normal(0,12 if phase=='recovery' else 3,len(g))*step_diag[i,g]
            rays.append(ray)
            for raw in rays:
                if ev.fes>=until:break
                mask=rng.random(DIM)<cr;mask[int(rng.integers(DIM))]=True
                y=np.where(mask,raw,xp)
                if phase=='recovery' and not ablate_recovery:y=reactance_repair(y,rng)
                trial=ev.call(y);phase_counts[phase]+=1
                archives(trial)
                if trial is not None and (best_new is None or trial.score<best_new.score):best_new=trial
                if trial is not None and not trial.m['feasible'] and not ablate_recovery and phase!='optimize' and ev.fes<until:
                    q=ev.call(boundary_recovery(trial.x,trial.m,rng));phase_counts[phase]+=1;archives(q)
                    if q is not None and q.score<best_new.score:best_new=q
            if best_new is None: break
            # Local, groupwise tactile foveation around valid region or improving candidate
            if ev.fes<until and (best_new.score<before.score or phase=='capture'):
                z=best_new.x.copy();gg=GROUPS[int(rng.integers(len(GROUPS)))];
                z[gg]+=rng.normal(0,2.5 if best_new.m['feasible'] else 7,len(gg))*step_diag[i,gg]
                alt=ev.call(z);phase_counts[phase]+=1;archives(alt)
                if alt is not None and alt.score<best_new.score:best_new=alt
            if best_new.score<before.score:
                pop[i]=best_new;stagn[i]=0
                shape=np.abs(best_new.x-xp);shape=shape/(np.mean(shape)+1e-12)
                step_diag[i]=.95*step_diag[i]+.05*np.clip(shape,.35,2.5)
                succF.append(fi);succCR.append(cr);deltas.append(before.score-best_new.score)
            else:stagn[i]+=1
            if stagn[i]>=9 and ev.fes<until:
                base=Afeas.sample(rng) or Anear.sample(rng) or pb
                y=base.x.copy();g=GROUPS[int(rng.integers(len(GROUPS)))];y[g]+=rng.standard_t(3,len(g))*4
                alt=ev.call(y);phase_counts[phase]+=1;archives(alt)
                if alt is not None and alt.score<pop[i].score:pop[i]=alt
                stagn[i]=0
        if succF:
            w=np.asarray(deltas)/sum(deltas);fs=np.array(succF);cs=np.array(succCR)
            mu_f=.9*mu_f+.1*np.sum(w*fs*fs)/(np.sum(w*fs)+1e-12)
            mu_cr=.9*mu_cr+.1*np.sum(w*cs)
        if ev.best.score<last_best-1e-11: last_improvement=ev.fes;last_best=ev.best.score
        # Keep a nonzero budget for seeded-DE refinement.
        if ev.fes>int(.55*budget) and ev.fes-last_improvement>int(.12*budget): break
    # Stage-C near-feasible coordinate recovery: counted exploratory search, never a free local solver.
    # This stage receives at most 12% total budget, even when no feasible solution has appeared.
    if not ablate_recovery and ev.best is not None and not ev.best.m['feasible']:
        limit=min(budget,ev.fes+int(.12*budget))
        best=ev.best
        while ev.fes<limit and not best.m['feasible']:
            step=(7.0 if best.score>1.3 else 2.5) * (0.65 if ev.fes>limit-.04*budget else 1.)
            improved=False
            perm=rng.permutation(DIM)
            # First recover variables most directly coupled to current, ZVS and tracking.
            for j in [1,2,0,6,8,10,3,4]+[int(v) for v in perm if v not in (1,2,0,6,8,10,3,4)]:
                if ev.fes>=limit or best.m['feasible']: break
                for direction in (-1.,1.):
                    if ev.fes>=limit:break
                    cand=best.x.copy();cand[j]+=direction*step
                    ne=ev.call(cand);phase_counts['recovery']+=1;archives(ne)
                    if ne is not None and ne.score < best.score-1e-12:
                        best=ne;improved=True
                        if best.m['feasible']:break
            if not improved:
                cand=best.x+rng.normal(0,step,DIM)
                ne=ev.call(cand);phase_counts['recovery']+=1;archives(ne)
                if ne is not None and ne.score<best.score:best=ne
        wi=int(np.argmax([e.score for e in pop]))
        if best.score<pop[wi].score:pop[wi]=best
    # Stage 2: seeded, success-history current-to-pbest with all FEs accounted.
    H=6; MF=np.full(H,.5); MCR=np.full(H,.8);pos=0;old=[]
    refinement_ceiling=max(ev.fes,int(.80*budget))
    while ev.fes<refinement_ceiling:
        order=np.argsort([e.score for e in pop]);ss=[];sc=[];dw=[]
        for i in range(npop):
            if ev.fes>=refinement_ceiling:break
            h=int(rng.integers(H));f=-1
            while f<=0:f=MF[h]+.1*rng.standard_cauchy()
            f=min(f,1.0);cr=float(np.clip(rng.normal(MCR[h],.1),0,1));
            pb=pop[int(rng.choice(order[:max(2,int(.2*npop))]))]
            r1=int(rng.choice([k for k in range(npop) if k!=i]));pool=pop+old
            r2=pool[int(rng.integers(len(pool)))].x
            y=pop[i].x+f*(pb.x-pop[i].x)+f*(pop[r1].x-r2)
            mask=rng.random(DIM)<cr;mask[int(rng.integers(DIM))]=True
            y=np.where(mask,y,pop[i].x)
            if (not pop[i].m['feasible']) and not ablate_recovery and rng.random()<.4:y=reactance_repair(y,rng)
            trial=ev.call(y);phase_counts['refine']+=1;archives(trial)
            if trial is not None and trial.score<pop[i].score:
                gain=pop[i].score-trial.score;old.append(pop[i]);pop[i]=trial
                ss.append(f);sc.append(cr);dw.append(gain)
        if ss:
            w=np.asarray(dw)/sum(dw);sf=np.asarray(ss);scc=np.asarray(sc)
            MF[pos]=np.sum(w*sf*sf)/(np.sum(w*sf)+1e-12);MCR[pos]=np.sum(w*scc);pos=(pos+1)%H
        old=old[-2*npop:]
    # Stage 3: explicit constrained SQP local recovery/polishing (finite-difference
    # gradients computed by SciPy). This is a hybrid local solver, not a claim that
    # the solution arises from pure evolutionary search. All unique physical
    # evaluator calls—including constraint-only calls—consume the FE budget.
    class EvaluationLimit(Exception): pass
    target_end=int(.97*budget)
    tries=0
    while ev.fes<target_end and tries<4:
        if target_end-ev.fes<max(100,4*DIM):break
        if tries==0:
            source=ev.best
        else:
            source=(Afeas.sample(rng) if ev.best.m['feasible'] else Anear.sample(rng)) or ev.best
        if source is None:source=ev.best
        y0=source.x.copy()
        if tries>0:
            group=GROUPS[int(rng.integers(len(GROUPS)))]
            y0[group]+=rng.normal(0.,min(2.,tries*.65),len(group))
        # Use the same physical model for the objective and all constraints.
        # Exact duplicate evaluations requested by the nonlinear optimizer are
        # cached; each unique candidate actually evaluated is counted once.
        last=[None,None]
        def metrics(q):
            t=np.clip(np.asarray(q,dtype=float),LO,HI)
            if last[0] is not None and np.array_equal(t,last[0]):return last[1]
            if ev.fes>=target_end:raise EvaluationLimit
            e=ev.call(t)
            if e is None:raise EvaluationLimit
            archives(e);phase_counts['sqp']+=1
            last[0]=t.copy();last[1]=e.m
            return e.m
        def obj(q):return float(metrics(q)['j_loss'])
        def feas(q):
            m=metrics(q)
            return np.array([(150.-m['max_ipk'])/150.,
                (120.-m['max_irms'])/120.,(150.-m['max_tj'])/150.,
                (.02-m['max_ripple'])/.02,(.08-m['max_track'])/.08,
                (m['min_zvs_high']-.70)/.70,(.30-m['bpk'])/.30],float)
        try:
            minimize(obj,np.clip(y0,LO,HI),method='SLSQP',
                     bounds=[(LO,HI)]*DIM,
                     constraints=[{'type':'ineq','fun':feas}],
                     options={'maxiter':min(100,max(8,int((target_end-ev.fes)/17))),
                              'ftol':1e-9,'eps':1e-3})
        except EvaluationLimit:pass
        tries+=1
    # Fill the remaining fixed FE quota with feasibility-archive-aware foveation.
    while ev.fes<budget:
        best=ev.best;y=best.x.copy();q=ev.fes/budget
        if not best.m['feasible']:
            dims=GROUPS[int(rng.integers(len(GROUPS)))]
            y[dims]+=rng.normal(0.,.9 if q<.99 else .25,len(dims))
            if not ablate_recovery and rng.random()<.2:y=boundary_recovery(y,best.m,rng,.1)
        else:
            dims=GROUPS[int(rng.integers(len(GROUPS)))]
            y[dims]+=rng.normal(0.,.65 if q<.99 else .12,len(dims))
        e2=ev.call(y);phase_counts['refine']+=1;archives(e2)
    e=ev.best
    assert ev.fes==budget,(ev.fes,budget)
    return {'seed':int(seed),'budget':budget,'fes':ev.fes,
            'first_feasible_fe':ev.first_feasible,'score':e.score,
            'feasible':bool(e.m['feasible']),'x':e.x.tolist(),
            'metrics':{k:float(v) for k,v in e.m.items() if k!='feasible'},
            'phase_evals':phase_counts,
            'feasible_archive':len(Afeas.items),'near_archive':len(Anear.items),
            'ablation':{'recovery':ablate_recovery,'archive':ablate_archive,'switch':ablate_switch},'implementation':'Stage-C-v0.3-explicit-SLSQP-polish'}
