import numpy as np
import rebound as rb
from celmech import Poincare
from celmech.nbody_simulation_utilities import align_simulation
from celmech.secular import LaplaceLagrangeSystem
from celmech.miscellaneous import critical_relative_AMD
from sympy import S
def get_samples(n):
    """
    Generate a sample from joint distribution
    of n uniform random variables between
    0 and 1 subject to the constraint that
    their sum is equl to 1.
    """
    x = np.zeros(n)
    u = 1
    for i in range(n-1):
        x[i] = np.random.uniform(low=0,high=u)
        u-= x[i]
    x[n-1]=u
    return np.random.permutation(x)

def generate_simulations(masses,semimajor_axes,N,fcrit=1):
    """
    Generate N rebound simulations with specified planet
    masses and semi-major axes and AMD values set to 
    ``fcrit`` times the critical value of AMD for stability.
    """
    # set up initial simulation
    sim0 = rb.Simulation()
    sim0.add(m=1)
    for m,a in zip(masses,semimajor_axes):
        sim0.add(m=m,a=a)
    pvars = Poincare.from_Simulation(sim0)

    # Get LL solution
    llsys = LaplaceLagrangeSystem.from_Poincare(pvars)
    Te,De = llsys.diagonalize_eccentricity()
    Ti,Di = llsys.diagonalize_inclination()

    # Determine critical AMD
    ps = pvars.particles
    alphas = [ps[i].a/ps[i+1].a for i in range(1,sim0.N-1)]
    gammas = [ps[i].m/ps[i+1].m for i in range(1,sim0.N-1)]
    amdCrit = np.min([p.Lambda * critical_relative_AMD(alpha,gamma) for p,alpha,gamma in zip(ps[2:],alphas,gammas)])

    # Initialize sims with amd randomly 
    # distributed uniformly among the LL modes.
    Npl = pvars.N - 1
    sims = []
    for _ in range(N):
        actions = fcrit * get_samples(2*Npl - 1) * amdCrit
        e_actions = actions[:Npl]
        i_actions = np.concatenate(([0],actions[Npl:]))
        e_angles = np.random.uniform(-np.pi,np.pi,size = Npl) 
        i_angles = np.random.uniform(-np.pi,np.pi,size = Npl)
        u = np.sqrt(e_actions) * np.exp(1j * e_angles)
        v = np.sqrt(i_actions) * np.exp(1j * i_angles)
        x = Te @ u
        y = Ti @ v
        for i in range(pvars.N-1):
            pvars.qp[S('kappa{}'.format(i+1))] = np.sqrt(2) * np.real(x[i])
            pvars.qp[S('sigma{}'.format(i+1))] = np.sqrt(2) * np.real(y[i])
            pvars.qp[S('eta{}'.format(i+1))] = np.sqrt(2) * np.imag(x[i])
            pvars.qp[S('rho{}'.format(i+1))] = np.sqrt(2) * np.imag(y[i])
            pvars.qp[S('lambda{}'.format(i+1))] = np.random.uniform(-np.pi,np.pi)
        sim = pvars.to_Simulation()
        align_simulation(sim)
        sims.append(sim)
    return sims

def run_secular_system_simulation(sec_sim,times):
    r"""
    Integrate the input secular simulation and get
    output at the specified times. Results returned
    as a dictionary with the trajectory as stored as
    ``qp`` along with the times, energy, and AMD.
    """
    Nout = len(times)
    qp0 = sec_sim.state_to_qp_vec()
    qp_solution = np.zeros((Nout, qp0.shape[0]))
    energy = np.zeros(Nout)
    amd = np.zeros(Nout)
    times_done = np.zeros(Nout)
    for i,t in enumerate(times):
        sec_sim.integrate(t)
        times_done[i] = sec_sim.t
        qp_solution[i] = sec_sim.state_to_qp_vec()
        energy[i] = sec_sim.calculate_energy()
        amd[i] = sec_sim.calculate_AMD()

    soln = dict()
    soln["times"] = times_done
    soln["energy"] = energy
    soln["AMD"] = amd
    soln["qp"] = qp_solution
    return soln
