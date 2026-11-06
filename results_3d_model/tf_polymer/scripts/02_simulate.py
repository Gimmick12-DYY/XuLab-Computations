"""Step 2: Langevin dynamics of the TF-binding (bridging-induced attraction) polymer, Brackley et al. 2016 style. OpenMM, CPU.

Reduced units: sigma = 1 (one 2 kb bead), kT = 1, tau = 1 (OpenMM nm / kJ mol^-1 / ps with T set so kB*T = 1).
  chain    910 beads, harmonic bonds (k = 50 kT/sigma^2, r0 = 1), bending kb*(1 + cos theta) with kb = 2 kT
  TFs      n_tf free spheres (diameter sigma), always "on" (no switching)
  pairs    WCA (eps 1, cut 2^(1/6)) for every pair, except TF-binding-bead pairs: truncated-shifted LJ, cut 1.8,
           normalised so the well depth equals the attraction of the bead's class (eps_top * {0.6, 0.8, 1.0} for
           1, 2-4, >=5 TFs with a summit in the bead)
  sphere   harmonic wall at radius R set by total volume fraction phi (chain + TFs)
No Hi-C, loops, Cicero edges or loop extrusion enter the model.

Each trajectory: self-avoiding random growth inside the sphere -> energy minimisation -> n_eq steps -> n_keep conformations
every n_every steps. Saves positions (float32) of every kept conformation.
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
import openmm as mm
from openmm import unit as u

ap = argparse.ArgumentParser()
ap.add_argument("--beads", required=True)
ap.add_argument("--eps", type=float, required=True, help="attraction (kT) of the >=5-TF class")
ap.add_argument("--mult", default="0.6,0.8,1.0")
ap.add_argument("--n-tf", type=int, default=200)
ap.add_argument("--phi", type=float, default=0.1)
ap.add_argument("--traj", default="0", help="trajectory ids, e.g. 0-9")
ap.add_argument("--n-eq", type=int, default=200000)
ap.add_argument("--n-keep", type=int, default=50)
ap.add_argument("--n-every", type=int, default=4000)
ap.add_argument("--dt", type=float, default=0.01)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

B = pd.read_csv(a.beads, sep="\t")
N = len(B); NT = a.n_tf; NP = N + NT
mult = np.r_[0.0, [float(x) for x in a.mult.split(",")]]
w = a.eps * mult[B.cls.values]
R = (NP * np.pi / 6 / a.phi * 3 / (4 * np.pi)) ** (1 / 3)
kT_kJ = 1.0
T = kT_kJ / (u.MOLAR_GAS_CONSTANT_R.value_in_unit(u.kilojoule_per_mole / u.kelvin))
LC = 4 * (1.8 ** -12 - 1.8 ** -6)

def build():
    s = mm.System()
    for _ in range(NP): s.addParticle(1.0)
    bf = mm.HarmonicBondForce()
    for i in range(N - 1): bf.addBond(i, i + 1, 1.0, 50.0)
    s.addForce(bf)
    af = mm.CustomAngleForce("kb*(1+cos(theta))"); af.addGlobalParameter("kb", 2.0)
    for i in range(N - 2): af.addAngle(i, i + 1, i + 2, [])
    s.addForce(af)
    nb = mm.CustomNonbondedForce(
        "select(aa, eatt, erep);"
        f"eatt = aa*(4*(ir6*ir6-ir6) - ({LC}))/(1+({LC}));"
        "erep = step(1.122462048309373 - r)*(4*(ir6*ir6-ir6) + 1);"
        "aa = tf1*w2 + tf2*w1;"
        "ir6 = 1/r^6")
    nb.addPerParticleParameter("tf"); nb.addPerParticleParameter("w")
    for i in range(N): nb.addParticle([0.0, float(w[i])])
    for _ in range(NT): nb.addParticle([1.0, 0.0])
    nb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic); nb.setCutoffDistance(1.8)
    s.addForce(nb)
    cf = mm.CustomExternalForce("10*step(rr-Rs)*(rr-Rs)^2; rr=sqrt(x^2+y^2+z^2)")
    cf.addGlobalParameter("Rs", R)
    for i in range(NP): cf.addParticle(i, [])
    s.addForce(cf)
    return s

def init_conf(rng):
    """self-avoiding growth (min distance 0.9) inside radius R-1, with backtracking; then TFs at free random sites"""
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
            if tries > 200:
                i = max(1, i - 10); tries = 0
    for j in range(N, NP):
        while True:
            p = rng.uniform(-Rin, Rin, 3)
            if np.linalg.norm(p) < Rin and np.min(np.sum((X[:j] - p) ** 2, 1)) > 0.81:
                X[j] = p; break
    return X

system = build()
plat = mm.Platform.getPlatformByName("CPU")
ids = a.traj.split("-"); ids = range(int(ids[0]), int(ids[-1]) + 1)
print(f"N={N} NT={NT} R={R:.2f} eps_top={a.eps} class eps={np.round(a.eps*mult[1:],2).tolist()} T={T:.2f}K dt={a.dt}", flush=True)
for t in ids:
    f = out / f"traj{t:03d}.npz"
    if f.exists(): print("skip", f); continue
    t0 = time.time()
    seed = int(1000 * a.eps) * 10000 + t
    rng = np.random.default_rng(seed)
    integ = mm.LangevinMiddleIntegrator(T, 1.0, a.dt); integ.setRandomNumberSeed(seed % 2**31)
    ctx = mm.Context(system, integ, plat, {"Threads": "1"})
    ctx.setPositions(init_conf(rng))
    mm.LocalEnergyMinimizer.minimize(ctx, 1.0, 2000)
    ctx.setVelocitiesToTemperature(T, seed % 2**31)
    integ.step(a.n_eq)
    keep = np.zeros((a.n_keep, NP, 3), np.float32); ener = np.zeros(a.n_keep)
    for k in range(a.n_keep):
        integ.step(a.n_every)
        st = ctx.getState(getPositions=True, getEnergy=True)
        keep[k] = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        ener[k] = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    if not np.isfinite(keep).all(): print(f"traj {t}: NaN, not saved", flush=True); continue
    np.savez_compressed(f, X=keep, E=ener, eps=a.eps, seed=seed, R=R)
    rmax = np.linalg.norm(keep, axis=2).max()
    print(f"traj {t}: {time.time()-t0:.0f}s E/particle {ener[0]/NP:.2f}->{ener[-1]/NP:.2f} rmax {rmax:.1f}", flush=True)
    del ctx, integ
