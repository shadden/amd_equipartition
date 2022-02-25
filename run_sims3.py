import numpy as np
import pickle
from amd_equipartition_utils import run_secular_system_simulation, generate_simulations
from celmech.secular import SecularSystemSimulation
import sys

Nsims = 10
Tfin = 2e9*np.pi # 1Gyr
Nout = 8192
Npl = 5
Idelta = int(sys.argv[1])
Ndelta = int(sys.argv[2])

deltas = np.linspace(0.3,1,Ndelta)
delta = deltas[Idelta]
masses = 3e-5 * np.ones(Npl)
semimajor_axes = (1+delta)**np.arange(Npl)
sims = generate_simulations(masses,semimajor_axes,Nsims,fcrit=1.)
times = np.linspace(0,Tfin,Nout)
for i,sim in enumerate(sims):
    sec_sim = SecularSystemSimulation.from_Simulation(
        sim,
        method='RK',
        dtFraction=0.05,
        rk_kwargs={'rk_method':'GL6'}
    )
    sec_sim._integrator.atol = 1e-10 * np.sqrt(sec_sim.calculate_AMD())
    results = run_secular_system_simulation(sec_sim,times)
    finame = "./results/runs3/si3_{}_delta_{}_of_{}.pkl".format(i,Idelta+1,Ndelta)
    with open(finame,"wb") as fi:
        pickle.dump(results,fi)
