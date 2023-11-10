import numpy as np
import rebound as rb
from celmech import Poincare
from celmech.nbody_simulation_utilities import align_simulation
from celmech.secular import LaplaceLagrangeSystem, SecularSystemSimulation
from celmech.miscellaneous import critical_relative_AMD, AMD_stability_coefficients
from sympy import S,Poly
# The old, bad way of doing this
def old_get_samples(n):
    x = np.zeros(n)
    u = 1
    for i in range(n-1):
        x[i] = np.random.uniform(low=0,high=u)
        u-= x[i]
    x[n-1]=u
    return np.random.permutation(x)

# another way of doing it wrong dammit!
def old2_get_samples(n):
    x = np.random.uniform(0,1,size=n)
    x /= np.sum(x)
    return x

def get_samples(n):
    """
    Generate a sample from joint distribution
    of n uniform random variables between
    0 and 1 subject to the constraint that
    their sum is equal to 1.
    """
    # put down n-1 random "fence posts"
    x = np.random.uniform(0,1,size=n-1)
    xs = np.sort(x)
    y = np.zeros(n)
    y[0] = xs[0]
    y[1:n-1] = xs[1:] - xs[:-1]
    y[-1] = 1 - xs[-1]
    return y

def generate_simulations(masses,semimajor_axes,N,fcrit=1):
    """
    Generate multiple rebound N-body simulations with fixed masses, semi-major
    axes and total AMD. The total AMD is set to a fixed fraction of the critical
    AMD and distributed randomly among the secular modes of the system.

    Parameters
    ----------
    masses : array-like
        List of planet masses.
    semimajor_axes : array-like
        List of planet semi-major axes.
    N : int
        Number of randomized systems to generate
    fcrit : int, optional
        Set the total AMD of the system as a fraction of the total AMD.
        `fcrit`=1 by default.

    Returns
    -------
    list
        A list of randomly generated rebound simulations.
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

def _remove_indices_from_tuple(original_tuple, indices_to_remove):
    return tuple(item for idx, item in enumerate(original_tuple) if idx not in indices_to_remove)

def _are_parallel(vector1, vector2):
    # Ensure vectors are non-empty and of the same length
    if len(vector1) != len(vector2):
        return False

    # Initialize a variable to store the ratio
    ratio = None

    for a, b in zip(vector1, vector2):
        # If both components are zero, continue to the next component
        if a == 0 and b == 0:
            continue

        # If one component is zero but the other is not, vectors are not parallel
        if a == 0 or b == 0:
            return False

        # Calculate the current ratio
        current_ratio = a / b

        # If the ratio has not been set yet, set it
        if ratio is None:
            ratio = current_ratio

        # If the current ratio does not match the previous, vectors are not parallel
        elif ratio != current_ratio:
            return False

    return True
def _normalize_positive(arr):
    # Find the first non-zero element
    nonzero_elements = arr[arr != 0]

    if nonzero_elements.size == 0:
        # If all elements are zero, return the original array
        return arr

    # Get the sign of the first non-zero element
    sign = np.sign(nonzero_elements[0])

    # Multiply the array by this sign
    return arr * sign

from celmech.poisson_series_manipulate import PSTerm, PoissonSeries
from collections import defaultdict
def keyval_to_psterm(key,val,Npl):
    Ndof = 2*Npl-1
    kkbar = np.array(key,dtype=int)
    term = PSTerm(val,kkbar[:Ndof],kkbar[Ndof:],[],[])
    return term

def simulation_to_secular_poisson_series(sim):
    """
    Given an input `rebound.Simulation` object, construct `PoissonSeries`
    objects that capture the secular Hamiltonian of the system.

    Parameters
    ----------
    sim : rebound.Simulation
        Target simulation for which to compute secular Hamiltonian.

    Returns
    -------
    series_sec: PoissonSeries
        Poisson Series representing the integrable part of the secular
        Hamiltonian
    series_res: dict
        Dictionary containing the resonant terms appearing in the Hamiltonian
        for different wave vectors.
        
    """
    sec_sim = SecularSystemSimulation.from_Simulation(sim,dtFraction=0.1)
    Hpoly = sec_sim.Hamiltonian_as_polynomial(transformed=True)
    T_ecc, T_inc, D_ecc, D_inc = sec_sim.diagonalizing_tranformations()
    # identify inclinatinon mode with freq. zero
    zero_mode_index = np.argmin(np.diagonal(D_inc))
    Ndof = 2*sec_sim.Npl
    Npl = sec_sim.Npl
    gens = Hpoly.gens
    # substitute 0 amplitude for modes w/ frequency
    zero_mode_gens = gens[Npl+zero_mode_index],gens[Ndof+Npl+zero_mode_index]

    # new polynomial with 0-freq inclination mode removed
    new_poly = Hpoly.subs(dict(zip(zero_mode_gens,np.zeros(2))))
    new_gens = _remove_indices_from_tuple(gens,[Npl+zero_mode_index,Ndof+Npl+zero_mode_index])
    new_poly = Poly(new_poly,new_gens)
    Hdict = new_poly.as_dict()

    terms_sec = []
    terms_res = defaultdict(list)
    kvecs = []
    for key,val in Hdict.items():        
        kkbar = np.array(key,dtype=int)
        k,kbar = kkbar[:Ndof-1],kkbar[Ndof-1:]
        kres = k-kbar
        gcd = np.gcd.reduce(kres)
        term = keyval_to_psterm(key,val,Npl)
        if gcd==0:
            terms_sec.append(term)
        else:
            kres = _normalize_positive(kres // gcd)
            for kres0 in kvecs:
                if _are_parallel(kres,kres0):
                    terms_res[tuple(kres)].append(term)
                    break
            else:
                terms_res[tuple(kres)] = [term]
                kvecs.append(kres)
    series_res = dict()
    for key,val in terms_res.items():
        series_res[key] = PoissonSeries.from_PSTerms(val,Ndof-1,0,cvar_symbols=new_gens)
    series_sec = PoissonSeries.from_PSTerms(terms_sec,cvar_symbols=new_gens)
    return series_sec, series_res

def H0_series_to_omega_Domega(H0):
    r"""
    Given a Poisson series for the integrable part of a Hamiltonian written as

    .. math::
        H_0 \approx \omega \cdot J + \frac{1}{2} J^\mathrm{T}\cdot \Delta\omega \cdot J
    
    return the vector :math:`\omega` and matrix :math:`\Delta\omega`.

    Parameters
    ----------
    H0 : PoissonSeries
        Poisson series representing integrable part of the Hamiltonian

    Returns
    -------
    omega, Domega
        ndarrays representing frequency vector and curvature matrix
    """
    omega = np.zeros(H0.N)
    Domega = np.zeros((H0.N,H0.N))
    for term in H0.terms:
        imax = np.argmax(term.k)
        order = np.sum(term.k)
        if order==1:
            omega[imax] = term.C
        elif order==2:
            if term.k[imax]==2:
                Domega[imax,imax] = term.C
            else:
                j = 1 + imax + np.argmax(term.k[imax+1:])
                Domega[imax,j]= 0.5 * term.C
                Domega[j,imax]= 0.5 * term.C
                continue
    return omega, Domega