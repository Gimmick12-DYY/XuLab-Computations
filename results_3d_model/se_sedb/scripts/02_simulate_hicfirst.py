"""Hi-C-first step 2: as the hybrid 02_simulate_hyb.py, but restraints come from restraints.tsv (01_prep_hicfirst.py) with
per-bond k and r0 and a flat-bottom form on*k*step(r-r0)*(r-r0)^2 (pixels r0 1.5, loops/red edges r0 1); each restraint is
ON for a whole trajectory with probability p_on (red edges p_on = 1); the global multiplier is ramped 0 -> 1 over --ramp
steps. --eps 0 gives the Hi-C-only model (TF particles inert, WCA only).

Hybrid notes: the v2 TF-switching, multi-species bridging polymer (unchanged) + Peakachu WT 0.7 loops as stochastic
soft restraints (integrative-restraint style, cf. Nuebler 2018 PNAS; Di Pierro 2016 PNAS).

Loop restraints: harmonic bond kl*(r - r0)^2 between the two anchor beads, r0 = 1 sigma, kl = --k-loop (kT/sigma^2), only
when the loop is ON. Each loop is ON for a whole trajectory with probability p_on (0.2-0.8 from its Hi-C O/E rank;
loop_restraints.tsv), re-drawn independently per trajectory. kl is ramped linearly from 0 over the first --ramp steps of
equilibration so distant anchors are pulled together gently.

v2 notes:

Same chain, confinement and units as v1 (910 x 2 kb beads, harmonic bonds k = 50, bending kb = 2, WCA, sphere at
total volume fraction 0.1, Langevin dt = 0.01 tau, kT = 1). Changes:
  * K TF species, n_tf/K TFs each. A TF of species k in the ON state attracts bead b with a truncated-shifted LJ
    (cut 1.8 sigma) of depth eps * w_bk (w from 01_prep_v2: only the sparse binding beads have w > 0).
    OFF TFs and all other pairs: WCA only.
  * Switching: every --sw-every steps each TF flips ON<->OFF with probability k_sw * dt * sw_every (symmetric rates,
    so half the TFs are ON on average). Implemented by updating per-particle parameters in the context.
No Hi-C, loops, Cicero edges or loop extrusion enter the model.
Saves per kept conformation: positions, TF ON state; plus TF species.
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
import openmm as mm
from openmm import unit as u

ap = argparse.ArgumentParser()
ap.add_argument("--beads", required=True)
ap.add_argument("--eps", type=float, required=True)
ap.add_argument("--n-tf", type=int, default=200)
ap.add_argument("--phi", type=float, default=0.1)
ap.add_argument("--k-sw", type=float, default=0.001, help="switching rate per tau (each direction)")
ap.add_argument("--sw-every", type=int, default=100)
ap.add_argument("--traj", default="0")
ap.add_argument("--n-eq", type=int, default=50000)
ap.add_argument("--n-keep", type=int, default=50)
ap.add_argument("--n-every", type=int, default=1000)
ap.add_argument("--dt", type=float, default=0.01)
ap.add_argument("--loops", required=True, help="loop_restraints.tsv")
ap.add_argument("--ramp", type=int, default=20000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

B = pd.read_csv(a.beads, sep="\t")
W = B[[c for c in B.columns if c.startswith("w_")]].values * a.eps
N, K = W.shape; NT = a.n_tf; NP = N + NT
species = np.arange(NT) % K
R = (NP * np.pi / 6 / a.phi * 3 / (4 * np.pi)) ** (1 / 3)
T = 1.0 / (u.MOLAR_GAS_CONSTANT_R.value_in_unit(u.kilojoule_per_mole / u.kelvin))
LC = 4 * (1.8 ** -12 - 1.8 ** -6)
p_flip = a.k_sw * a.dt * a.sw_every

def tf_params(j, on):
    return [1.0] + [float(on and species[j] == k) for k in range(K)] + [0.0] * K

def build():
    s = mm.System()
    for _ in range(NP): s.addParticle(1.0)
    bf = mm.HarmonicBondForce()
    for i in range(N - 1): bf.addBond(i, i + 1, 1.0, 50.0)
    s.addForce(bf)
    af = mm.CustomAngleForce("kb*(1+cos(theta))"); af.addGlobalParameter("kb", 2.0)
    for i in range(N - 2): af.addAngle(i, i + 1, i + 2, [])
    s.addForce(af)
    aa = " + ".join(f"s{k}1*w{k}2 + s{k}2*w{k}1" for k in range(K))
    nb = mm.CustomNonbondedForce(
        "select(aa, eatt, erep);"
        f"eatt = aa*(4*(ir6*ir6-ir6) - ({LC}))/(1+({LC}));"
        "erep = step(1.122462048309373 - r)*(4*(ir6*ir6-ir6) + 1);"
        f"aa = {aa};"
        "ir6 = 1/r^6")
    nb.addPerParticleParameter("istf")
    for k in range(K): nb.addPerParticleParameter(f"s{k}")
    for k in range(K): nb.addPerParticleParameter(f"w{k}")
    for i in range(N): nb.addParticle([0.0] + [0.0] * K + [float(x) for x in W[i]])
    for j in range(NT): nb.addParticle(tf_params(j, True))
    nb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic); nb.setCutoffDistance(1.8)
    s.addForce(nb)
    cf = mm.CustomExternalForce("10*step(rr-Rs)*(rr-Rs)^2; rr=sqrt(x^2+y^2+z^2)")
    cf.addGlobalParameter("Rs", R)
    for i in range(NP): cf.addParticle(i, [])
    s.addForce(cf)
    lf = mm.CustomBondForce("on*kl*kb*step(r-r0)*(r-r0)^2"); lf.addGlobalParameter("kl", 0.0)
    for nm in ("on", "kb", "r0"): lf.addPerBondParameter(nm)
    for b1, b2, k_, r0_ in zip(LR.b1, LR.b2, LR.k, LR.r0): lf.addBond(int(b1), int(b2), [0.0, float(k_), float(r0_)])
    s.addForce(lf)
    return s, nb, lf

def init_conf(rng):
    Rin = R - 1.0
    X = np.zeros((NP, 3)); X[0] = rng.normal(size=3) * 0.5
    i, tries = 1, 0
    while i < N:
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        p = X[i - 1] + v
        if np.linalg.norm(p) < Rin and (i < 2 or np.min(np.sum((X[:i - 1] - p) ** 2, 1)) > 0.81):
            X[i] = p; i += 1; tries = 0
        else:
            tries += 1
            if tries > 200: i = max(1, i - 10); tries = 0
    for j in range(N, NP):
        while True:
            p = rng.uniform(-Rin, Rin, 3)
            if np.linalg.norm(p) < Rin and np.min(np.sum((X[:j] - p) ** 2, 1)) > 0.81:
                X[j] = p; break
    return X

LR = pd.read_csv(a.loops, sep="\t")
system, nbf, lff = build()
plat = mm.Platform.getPlatformByName("CPU")
ids = a.traj.split("-"); ids = range(int(ids[0]), int(ids[-1]) + 1)
print(f"restraints {LR.kind.value_counts().to_dict()} ramp={a.ramp}; N={N} K={K} NT={NT} R={R:.2f} eps={a.eps} binding beads={(W.sum(1) > 0).sum()} k_sw={a.k_sw}/tau p_flip={p_flip:.3g} per {a.sw_every} steps", flush=True)
for t in ids:
    f = out / f"traj{t:03d}.npz"
    if f.exists(): print("skip", f); continue
    t0 = time.time(); seed = int(1000 * a.eps) * 10000 + t
    rng = np.random.default_rng(seed)
    integ = mm.LangevinMiddleIntegrator(T, 1.0, a.dt); integ.setRandomNumberSeed(seed % 2**31)
    ctx = mm.Context(system, integ, plat, {"Threads": "1"})
    on = rng.random(NT) < 0.5
    for j in range(NT): nbf.setParticleParameters(N + j, tf_params(j, on[j]))
    nbf.updateParametersInContext(ctx)
    lon = rng.random(len(LR)) < LR.p_on.values
    for q in range(len(LR)): lff.setBondParameters(q, int(LR.b1[q]), int(LR.b2[q]), [float(lon[q]), float(LR.k[q]), float(LR.r0[q])])
    lff.updateParametersInContext(ctx)
    ctx.setPositions(init_conf(rng))
    mm.LocalEnergyMinimizer.minimize(ctx, 1.0, 2000)
    ctx.setVelocitiesToTemperature(T, seed % 2**31)
    def run(nsteps):
        done = 0
        while done < nsteps:
            st = min(a.sw_every, nsteps - done); integ.step(st); done += st
            flip = rng.random(NT) < p_flip
            if flip.any():
                for j in np.where(flip)[0]:
                    on[j] = ~on[j]; nbf.setParticleParameters(N + j, tf_params(j, on[j]))
                nbf.updateParametersInContext(ctx)
    nr = (a.ramp // a.sw_every) * a.sw_every
    for k in range(0, nr, a.sw_every):
        ctx.setParameter("kl", (k + a.sw_every) / max(1, nr)); run(a.sw_every)
    ctx.setParameter("kl", 1.0)
    run(a.n_eq - nr)
    keep = np.zeros((a.n_keep, NP, 3), np.float32); ON = np.zeros((a.n_keep, NT), bool); ener = np.zeros(a.n_keep)
    for k in range(a.n_keep):
        run(a.n_every)
        st = ctx.getState(getPositions=True, getEnergy=True)
        keep[k] = st.getPositions(asNumpy=True).value_in_unit(u.nanometer); ON[k] = on
        ener[k] = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    if not np.isfinite(keep).all(): print(f"traj {t}: NaN, not saved", flush=True); continue
    np.savez_compressed(f, X=keep, ON=ON, loop_on=lon, species=species, E=ener, eps=a.eps, seed=seed, R=R)
    print(f"traj {t}: {time.time()-t0:.0f}s E/particle {ener[0]/NP:.2f}->{ener[-1]/NP:.2f} on {ON.mean():.2f} loops on {lon.sum()}/{len(lon)}", flush=True)
    del ctx, integ
