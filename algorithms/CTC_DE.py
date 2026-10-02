import numpy as np

def _sample_cauchy(rng, loc, scale=0.1):
    for _ in range(50):
        v = loc + scale * np.tan(np.pi * (rng.random() - 0.5))
        if v > 0:
            return min(1.0, float(v))
    return max(1e-8, min(1.0, float(loc)))

def _state(X, fit, stagn, lbv, ubv):
    n, dim = X.shape
    order = np.argsort(fit)
    ranks = np.empty(n, float)
    ranks[order] = np.arange(n, dtype=float) / max(1, n - 1)
    z = (X - lbv) / (ubv - lbv + 1e-300)
    dist = np.sqrt(np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2) / dim)
    np.fill_diagonal(dist, np.inf)
    nn = np.min(dist, axis=1)
    med = np.median(nn[np.isfinite(nn)]) + 1e-12
    crowd = np.clip(1.0 - nn / (1.8 * med), 0.0, 1.0)
    st = np.clip(stagn / max(6.0, 0.7 * np.sqrt(dim) + 3.0), 0.0, 1.0)
    threat = np.clip(0.45 * st + 0.35 * crowd + 0.20 * ranks, 0.0, 1.0)
    return order, ranks, dist, crowd, st, threat

def ctc_de(fun, dim, lb, ub, max_fes=10000, seed=1, pop_mult=8,
           max_pop=180, np_min=4, H=6, variant="full", return_trace=False):
    """
    Caterpillar Threat-Conditioned Differential Evolution (CTC-DE).

    Development algorithm. Biological metaphor is limited to the controller:
    low threat -> concealment/refinement, medium threat -> counterfactual decoy,
    high threat -> startle escape. Stable elites may skip an evaluation and the
    saved evaluation is reallocated to high-threat agents.

    variant:
      full             : all mechanisms
      no_decoy         : replace decoy mutation with standard pbest mutation
      no_escape        : replace escape mutation with standard pbest mutation
      no_reallocation  : evaluate every individual once per generation
      base             : success-history current-to-pbest/1 only
    """
    rng = np.random.default_rng(int(seed))
    lbv = np.full(dim, lb, float) if np.isscalar(lb) else np.asarray(lb, float)
    ubv = np.full(dim, ub, float) if np.isscalar(ub) else np.asarray(ub, float)
    span = ubv - lbv

    np_init = max(20, min(int(max_pop), int(pop_mult) * int(dim)))
    X = rng.uniform(lbv, ubv, size=(np_init, dim))
    fit = np.array([float(fun(x)) for x in X], dtype=float)
    fes = np_init
    NP = np_init
    stagn = np.zeros(NP, dtype=int)

    MF = np.full(H, 0.5, dtype=float)
    MCR = np.full(H, 0.8, dtype=float)
    mempos = 0
    archive = []

    bi = int(np.argmin(fit))
    best = X[bi].copy()
    bestf = float(fit[bi])

    op_success = np.ones(3, dtype=float)
    op_trials = np.full(3, 3.0, dtype=float)
    generation = 0

    trace = []
    next_frac = 0.01
    if return_trace:
        trace.append((fes, bestf))

    while fes < max_fes and NP >= np_min:
        generation += 1
        progress = fes / float(max_fes)
        order, ranks, dist, crowd, st, threat = _state(X, fit, stagn, lbv, ubv)

        if variant in ("no_reallocation", "base"):
            freeze = np.zeros(NP, dtype=bool)
        else:
            elite_count = max(1, int(np.floor(0.10 * NP)))
            freeze = np.zeros(NP, dtype=bool)
            for i in order[:elite_count]:
                freeze[i] = (st[i] < 0.25 and crowd[i] < 0.55)

        active = [i for i in range(NP) if not freeze[i]]
        saved = NP - len(active)
        bonus = []
        if saved > 0 and active and variant not in ("no_reallocation", "base"):
            a = np.asarray(active, dtype=int)
            bonus = list(a[np.argsort(-threat[a])[:saved]])
        targets = active + bonus

        candidates = {}
        candidate_fit = {}
        candidate_meta = {}
        succF, succCR, improvements = [], [], []

        for i in targets:
            if fes >= max_fes:
                break

            T = float(threat[i])
            if variant == "base":
                op = 0
            else:
                op = 0 if T < 0.30 else (1 if T < 0.67 else 2)
                if variant == "no_decoy" and op == 1:
                    op = 0
                if variant == "no_escape" and op == 2:
                    op = 0

                rates = op_success / op_trials
                if rng.random() < 0.15 and rates[op] < np.max(rates):
                    alt = int(np.argmax(rates))
                    if (variant != "no_decoy" or alt != 1) and (variant != "no_escape" or alt != 2):
                        op = alt

            k = int(rng.integers(H))
            F = _sample_cauchy(rng, MF[k], 0.1)
            CR = 0.0 if MCR[k] < 0 else float(np.clip(rng.normal(MCR[k], 0.1), 0.0, 1.0))

            pmin = 2.0 / NP
            pmax = 0.10 + 0.10 * (1.0 - progress)
            p = pmax if pmin >= pmax else rng.uniform(pmin, pmax)
            pnum = max(2, min(NP, int(np.ceil(p * NP))))
            pbest_idx = int(rng.choice(order[:pnum]))

            ids = np.arange(NP)
            ids = ids[(ids != i) & (ids != pbest_idx)]
            r1 = int(rng.choice(ids))

            while True:
                rr = int(rng.integers(NP + len(archive)))
                if rr < NP:
                    if rr != i and rr != r1:
                        xr2 = X[rr]
                        break
                else:
                    xr2 = archive[rr - NP]
                    break

            standard = X[i] + F * (X[pbest_idx] - X[i]) + F * (X[r1] - xr2)

            if op == 0:
                mutant = standard
            elif op == 1:
                neigh = np.argsort(dist[pbest_idx])[:max(3, min(7, NP - 1))]
                centroid = np.mean(X[neigh], axis=0)
                kappa = (0.25 + 0.75 * (1.0 - progress)) * (0.5 + rng.random())
                decoy = X[pbest_idx] + kappa * (X[pbest_idx] - centroid)
                decoy = np.clip(decoy, lbv, ubv)
                mutant = X[i] + F * (decoy - X[i]) + F * (X[r1] - xr2)
            else:
                neigh = np.argsort(dist[i])[:max(3, min(7, NP - 1))]
                centroid = np.mean(X[neigh], axis=0)
                repel = X[i] - centroid
                nr = np.linalg.norm(repel)
                if nr < 1e-12:
                    repel = rng.normal(size=dim)
                    nr = np.linalg.norm(repel)
                repel /= nr + 1e-300
                heavy = float(np.clip(rng.standard_t(2.5), -5.0, 5.0))
                radius = (0.03 * (1.0 - progress) + 0.002) * float(np.mean(span))
                mutant = standard + heavy * radius * repel

            low = mutant < lbv
            high = mutant > ubv
            mutant[low] = (lbv[low] + X[i][low]) / 2.0
            mutant[high] = (ubv[high] + X[i][high]) / 2.0

            mask = rng.random(dim) < CR
            mask[int(rng.integers(dim))] = True
            trial = np.where(mask, mutant, X[i])
            ftrial = float(fun(trial))
            fes += 1
            op_trials[op] += 1.0

            if i not in candidate_fit or ftrial < candidate_fit[i]:
                candidates[i] = trial.copy()
                candidate_fit[i] = ftrial
                candidate_meta[i] = (F, CR, op)

            if return_trace and fes / float(max_fes) >= next_frac:
                trace.append((fes, bestf))
                next_frac += 0.01

        for i, ftrial in candidate_fit.items():
            F, CR, op = candidate_meta[i]
            if ftrial <= fit[i]:
                oldf = float(fit[i])
                oldx = X[i].copy()
                X[i] = candidates[i]
                fit[i] = ftrial
                stagn[i] = 0

                if len(archive) < np_init:
                    archive.append(oldx)
                else:
                    archive[int(rng.integers(len(archive)))] = oldx

                if ftrial < oldf:
                    succF.append(F)
                    succCR.append(CR)
                    improvements.append(oldf - ftrial)
                    op_success[op] += 1.0

                if ftrial < bestf:
                    bestf = ftrial
                    best = X[i].copy()
            else:
                stagn[i] += 1

        if improvements:
            w = np.asarray(improvements, float)
            w /= np.sum(w) + 1e-300
            Fs = np.asarray(succF, float)
            CRs = np.asarray(succCR, float)
            MF[mempos] = np.sum(w * Fs * Fs) / (np.sum(w * Fs) + 1e-300)
            if np.sum(w * CRs) > 0:
                MCR[mempos] = np.sum(w * CRs * CRs) / (np.sum(w * CRs) + 1e-300)
            else:
                MCR[mempos] = -1.0
            mempos = (mempos + 1) % H

        if generation % 20 == 0:
            op_success = 1.0 + 0.8 * (op_success - 1.0)
            op_trials = 3.0 + 0.8 * (op_trials - 3.0)

        target_np = int(round(np_min + (np_init - np_min) * (1.0 - fes / float(max_fes))))
        target_np = max(np_min, min(NP, target_np))
        if target_np < NP:
            keep = np.argsort(fit)[:target_np]
            X = X[keep]
            fit = fit[keep]
            stagn = stagn[keep]
            NP = target_np
            if len(archive) > NP:
                sel = rng.choice(len(archive), size=NP, replace=False)
                archive = [archive[int(j)] for j in sel]

    if return_trace:
        trace.append((fes, bestf))
        return best, bestf, trace
    return best, bestf
