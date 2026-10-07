from __future__ import annotations
import numpy as np

def _reflect(x, lb, ub):
    span = ub - lb
    y = (x - lb) % (2.0 * span)
    return lb + np.where(y <= span, y, 2.0 * span - y)

class PMJMOX3Beta:
    """
    PMJMO-X3 beta candidate for independent FRSS confirmation.

    Primary candidate policy:
      * micro-geometry OFF by default (inherits the X3-no-geometry winner);
      * dimension-aware escape pressure: conservative at D~20, progressively
        stronger toward D~100;
      * optional geometry is hard-suppressed above D=30 and requires stronger
        success/failure alignment evidence than X3-alpha.

    NOT FROZEN and NOT for external superiority claims.
    """
    VERSION = "PMJMO-X3-beta-confirmation-candidate-2026-10-06"

    def __init__(self, seed=None, pop_size=None, memory_size=8,
                 use_microgeom=False, use_escape=True):
        self.seed = seed
        self.pop_size = pop_size
        self.memory_size = memory_size
        self.use_microgeom = use_microgeom
        self.use_escape = use_escape

    @staticmethod
    def dimension_policy(D):
        lo, hi = np.sqrt(20.0), np.sqrt(100.0)
        ds = float(np.clip((np.sqrt(float(D)) - lo) / (hi - lo), 0.0, 1.0))
        return {
            "dscale": ds,
            "escape_soft_stagn": int(round(8.0 - 4.0 * ds)),
            "escape_hard_stagn": int(round(10.0 - 4.0 * ds)),
            "escape_probability": 0.05 + 0.17 * ds,
            "escape_tail_fraction": 0.20 + 0.15 * ds,
            "escape_scale_low": 0.16 + 0.14 * ds,
            "escape_scale_high": 0.28 + 0.30 * ds,
            "escape_noise": 0.010 + 0.012 * ds,
            "success_rate_gate": 0.05 + 0.03 * ds,
        }

    def optimize(self, func, lb, ub, max_fes):
        rng = np.random.default_rng(self.seed)
        lb = np.asarray(lb, float)
        ub = np.asarray(ub, float)
        D = lb.size
        policy = self.dimension_policy(D)

        NP0 = self.pop_size or max(48, min(96, int(round(12 * np.sqrt(D)))))
        Nmin = max(6, min(12, int(round(1.5 * np.sqrt(D)))))
        fes = 0

        def ev(x):
            nonlocal fes
            if fes >= max_fes:
                raise StopIteration
            fes += 1
            return float(func(np.asarray(x, float)))

        X = rng.uniform(lb, ub, (NP0, D))
        fit = np.array([ev(x) for x in X])
        ib = int(np.argmin(fit))
        best = X[ib].copy()
        fbest = float(fit[ib])

        H = self.memory_size
        MF = np.full(H, .5)
        MCR = np.full(H, .7)
        mem = 0
        archive = []
        succ_dirs = []
        B = None
        spectral = 0.0
        last_basis = -999
        succ_align = .5
        fail_align = .5
        prev_sr = .25
        stagn = 0
        generation = 0
        range_rms = np.sqrt(np.mean((ub - lb) ** 2)) + 1e-15

        while fes < max_fes:
            generation += 1
            NP = len(X)
            prog = fes / max_fes
            order = np.argsort(fit)
            center = np.median(X, axis=0)
            div = np.median(np.sqrt(np.mean((X - center) ** 2, axis=1))) / range_rms

            if (self.use_microgeom and D <= 30 and len(succ_dirs) >= 16
                    and generation - last_basis >= 8):
                R = np.asarray(succ_dirs[-min(48, len(succ_dirs)):])
                R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-15)
                try:
                    _, s, Vt = np.linalg.svd(R, full_matrices=False)
                    spectral = float(s[0] ** 2 / (np.sum(s * s) + 1e-15))
                    energy = np.cumsum(s * s) / (np.sum(s * s) + 1e-15)
                    r = max(1, min(3, int(np.searchsorted(energy, .80) + 1)))
                    B = Vt[:r].T
                except np.linalg.LinAlgError:
                    B = None
                    spectral = 0.0
                last_basis = generation

            utility_gap = succ_align - fail_align
            geom_gate = (
                self.use_microgeom and D <= 30 and B is not None
                and spectral > .30 and utility_gap > .12
                and prev_sr > .12 and stagn < 4 and div < .45
            )

            soft_stag = policy["escape_soft_stagn"]
            hard_stag = policy["escape_hard_stagn"]
            rugged = (prev_sr < policy["success_rate_gate"] or stagn >= hard_stag)
            tail_start = int(np.floor((1.0 - policy["escape_tail_fraction"]) * NP))
            tail_mask = np.zeros(NP, dtype=bool)
            tail_mask[order[max(0, min(NP - 1, tail_start)):]] = True

            Xn = X.copy()
            fn = fit.copy()
            SF, SCR, gains = [], [], []
            gen_succ = 0
            improved = False
            align_s, align_f = [], []

            for i in range(NP):
                if fes >= max_fes:
                    break

                mi = int(rng.integers(H))
                Fi = -1.0
                while Fi <= 0:
                    Fi = MF[mi] + .1 * rng.standard_cauchy()
                Fi = float(min(Fi, 1.0))
                CR = float(np.clip(rng.normal(MCR[mi], .08), 0, 1))

                pfrac = .20 - .11 * prog
                p = max(2, int(np.ceil(pfrac * NP)))
                pb = int(rng.choice(order[:p]))

                rw = np.exp(-3 * np.arange(NP) / NP)
                rw /= rw.sum()
                r1 = int(rng.choice(order, p=rw))
                guard = 0
                while r1 in (i, pb) and guard < 20:
                    r1 = int(rng.choice(order, p=rw))
                    guard += 1

                pool = [j for j in range(NP) if j not in (i, pb, r1)]
                r2 = int(rng.choice(pool))
                xr2 = (archive[int(rng.integers(len(archive)))]
                       if archive and rng.random() < min(.75, .25 + .5 * prog)
                       else X[r2])

                if rng.random() < .35 and p >= 3:
                    pb2 = int(rng.choice(order[:p]))
                    target = .5 * (X[pb] + X[pb2])
                else:
                    target = X[pb]

                step = Fi * (target - X[i]) + Fi * (X[r1] - xr2)

                if geom_gate and rng.random() < .12:
                    proj = B @ (B.T @ (target - X[i]))
                    step = step + (.010 + .015 * (1 - prog)) * proj

                y = X[i] + step
                mask = rng.random(D) < CR
                mask[int(rng.integers(D))] = True
                y = np.where(mask, y, X[i])

                if (self.use_escape and rugged and stagn >= soft_stag
                        and tail_mask[i]
                        and rng.random() < policy["escape_probability"]):
                    lo = policy["escape_scale_low"]
                    hi = policy["escape_scale_high"]
                    esc_scale = lo + (hi - lo) * rng.random()
                    noise = rng.normal(
                        0.0,
                        policy["escape_noise"] * (ub - lb) / np.sqrt(D),
                        D,
                    )
                    y = best + esc_scale * (X[r1] - xr2) + noise

                y = _reflect(y, lb, ub)

                al = None
                if B is not None:
                    u = y - X[i]
                    u = u / (np.linalg.norm(u) + 1e-15)
                    al = float(np.sum((B.T @ u) ** 2))

                fy = ev(y)
                success = fy <= fit[i]
                if al is not None:
                    (align_s if success else align_f).append(al)

                if success:
                    gain = max(0.0, float(fit[i] - fy))
                    archive.append(X[i].copy())
                    if len(archive) > NP0:
                        del archive[int(rng.integers(len(archive)))]
                    succ_dirs.append(y - X[i])
                    if len(succ_dirs) > 80:
                        succ_dirs.pop(0)

                    Xn[i] = y
                    fn[i] = fy
                    gen_succ += 1
                    SF.append(Fi)
                    SCR.append(CR)
                    gains.append(max(gain, 1e-300))
                    if fy < fbest:
                        best = y.copy()
                        fbest = float(fy)
                        improved = True

            X, fit = Xn, fn
            prev_sr = gen_succ / max(1, NP)

            if align_s:
                succ_align = .8 * succ_align + .2 * float(np.mean(align_s))
            if align_f:
                fail_align = .8 * fail_align + .2 * float(np.mean(align_f))

            if SF:
                w = np.asarray(gains)
                w /= w.sum()
                sf = np.asarray(SF)
                cr = np.asarray(SCR)
                MF[mem] = np.sum(w * sf * sf) / (np.sum(w * sf) + 1e-15)
                denom = np.sum(w * cr)
                MCR[mem] = np.sum(w * cr * cr) / (denom + 1e-15)
                mem = (mem + 1) % H

            stagn = 0 if improved else stagn + 1
            target = max(
                Nmin,
                int(round(Nmin + (NP0 - Nmin) * (1 - prog) ** 1.35))
            )
            if NP > target:
                keep = np.argsort(fit)[:target]
                X = X[keep]
                fit = fit[keep]

        return {
            "xbest": best,
            "fbest": fbest,
            "fes": fes,
            "version": self.VERSION,
            "dimension_policy": policy,
        }
