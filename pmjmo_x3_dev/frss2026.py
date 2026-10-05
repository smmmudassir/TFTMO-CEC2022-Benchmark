from __future__ import annotations
import numpy as np

class BlockRotation:
    def __init__(self,D,seed,block=8):
        rng=np.random.default_rng(seed); self.perm=rng.permutation(D); self.sign=rng.choice([-1.,1.],D); self.blocks=[]
        for a in range(0,D,block):
            b=min(D,a+block); A=rng.normal(size=(b-a,b-a)); Q,_=np.linalg.qr(A); self.blocks.append((a,b,Q))
    def __call__(self,x):
        z=np.asarray(x)[self.perm]*self.sign; y=z.copy()
        for a,b,Q in self.blocks: y[a:b]=Q@z[a:b]
        return y

def make_function(fid,D,instance_seed):
    rng=np.random.default_rng(instance_seed); shift=rng.uniform(-3,3,D); rot=BlockRotation(D,instance_seed+1009,block=min(8,D))
    def t(x): return rot(np.asarray(x)-shift)
    if fid==1:
        w=10**np.linspace(0,6,D); return lambda x: float(np.sum(w*t(x)**2)),'ellipsoid'
    if fid==2: return lambda x: float(t(x)[0]**2+1e6*np.sum(t(x)[1:]**2)),'bent_cigar'
    if fid==3: return lambda x: float(1e6*t(x)[0]**2+np.sum(t(x)[1:]**2)),'discus'
    if fid==4:
        def f(x):
            z=t(x)+1; return float(np.sum(100*(z[1:]-z[:-1]**2)**2+(z[:-1]-1)**2))
        return f,'rosenbrock'
    if fid==5: return lambda x: float(10*D+np.sum(t(x)**2-10*np.cos(2*np.pi*t(x)))),'rastrigin'
    if fid==6:
        def f(x):
            z=t(x); return float(-20*np.exp(-.2*np.sqrt(np.mean(z*z)))-np.exp(np.mean(np.cos(2*np.pi*z)))+20+np.e)
        return f,'ackley'
    if fid==7:
        def f(x):
            z=t(x); return float(np.sum(z*z)/4000-np.prod(np.cos(z/np.sqrt(np.arange(1,D+1))))+1)
        return f,'griewank'
    if fid==8:
        def f(x):
            z=t(x); s=np.sum(.5*np.arange(1,D+1)*z); return float(np.sum(z*z)+s*s+s**4)
        return f,'zakharov'
    if fid==9:
        def f(x): return float(np.sum(np.cumsum(t(x))**2))
        return f,'schwefel_1_2'
    if fid==10:
        def f(x):
            z=t(x); a=max(1,D//3); b=max(a+1,2*D//3); z1=z[:a]; z2=z[a:b]+1; z3=z[b:]
            f1=10*len(z1)+np.sum(z1*z1-10*np.cos(2*np.pi*z1))
            f2=np.sum(100*(z2[1:]-z2[:-1]**2)**2+(z2[:-1]-1)**2) if len(z2)>1 else np.sum((z2-1)**2)
            ww=10**np.linspace(0,4,len(z3)) if len(z3) else np.array([])
            return float(f1+f2+(np.sum(ww*z3*z3) if len(z3) else 0.))
        return f,'hybrid_rastrigin_rosenbrock_ellipsoid'
    raise ValueError(fid)
