import numpy as np
import pickle
from amd_equipartition_utils import run_secular_system_simulation, generate_simulations
from celmech.secular import SecularSystemSimulation
import sys

Nsims = 50
Idelta = int(sys.argv[1])
Ndelta = int(sys.argv[2])

deltas = np.linspace(0.3,1,Ndelta)

delta = deltas[Idelta]

masses = 3e-5 * np.ones(4)
semimajor_axes = (1+delta)**np.arange(4)
sims = generate_simulations(masses,semimajor_axes,Nsims)
times = np.linspace(0,2e6*np.pi,10)#np.linspace(0,2e8*np.pi,4096)
for i,sim in enumerate(sims):
    sec_sim = SecularSystemSimulation.from_Simulation(
        sim,
        method='RK',
        dtFraction=0.05,
        rk_kwargs={'rk_method':'GL6'}
    )
    sec_sim._integrator.atol = 1e-10 * np.sqrt(sec_sim.calculate_AMD())
    results = run_secular_system_simulation(sec_sim,times)
    finame = "secular_integration_{}_delta_{}_of_{}.pkl".format(i,Idelta+1,Ndelta)
    with open(finame,"wb") as fi:
        pickle.dump(results,fi)
