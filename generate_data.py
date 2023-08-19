import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
from chem_ode import fex
from chem_commons import nspecies, idx_C, idx_O, idx_H2, names
import sys

from scipy.stats import loguniform

import pandas as pd

from joblib import Parallel, delayed

import h5py

def run_simulation(idx, simulation_parameters):
    # seconds per year
    spy = 3600.0 * 24 * 365.0

    # Initialize all species at the numerical minimum of 10^-20
    y0 = np.zeros(nspecies + 1) + 1e-20 * simulation_parameters["ntot"]
    # Gas temperature
    y0[-1] = simulation_parameters["t_gas_init"]
    # The initial molecular hydrogen abundance
    y0[idx_H2] = simulation_parameters["ntot"]
    # The intial carbon and oxygen abundances
    y0[idx_O] = simulation_parameters["ntot"] * simulation_parameters["O_fraction"]
    y0[idx_C] = simulation_parameters["ntot"] * simulation_parameters["C_fraction"]

    # Cosmic ray ionisation rate and radiation field

    # Integrate the system for 1 Myr
    tend = 1e6 * spy
    # Solve the system using the BDF method
    sol = solve_ivp(fex,(0, tend),y0,"BDF",atol=1e-40,rtol=1e-12,args=(simulation_parameters["cr_rate"], simulation_parameters["gnot"]),)
    if sol.success:
        arr=np.vstack((sol.t.reshape(1, -1), sol.y))
        # np.save(f"data/{idx}.npy",arr)
        with h5py.File(f"data/{idx}.h5", "w") as file:
            file.create_dataset("data", data=arr)


    return sol.success

if __name__=="__main__":

    n_samples = int(1e4)

    grid_parameters = pd.DataFrame({
        # Hydrogen number density
        "ntot": loguniform.rvs(1e2, 1e6, size=n_samples, random_state=12),
        # Fractional abunadnce of oxygen
        "O_fraction": loguniform.rvs(1e-5, 1e-3, size=n_samples, random_state=13),  # [1e-5, 1e-3]
        # Fractional abundance of carbon
        "C_fraction": loguniform.rvs(1e-5, 1e-3, size=n_samples, random_state=14),  # [1e-5, 1e-3]
        # Cosmic ray ionisation rate
        "cr_rate": loguniform.rvs(1e-18, 1e-14, size=n_samples, random_state=15),  # enchance up to 1e-14
        # Radiation field
        "gnot": loguniform.rvs(1e-1, 1e4, size=n_samples, random_state=16),  # enchance up to 1e5
        # t_gas_init
        "t_gas_init": loguniform.rvs(1e1, 5e5, size=n_samples, random_state=17),  # [1e1, 1e6]
    })


    grid_parameters.to_csv("data/.grid_parameters.csv")

    workers = 16

    results = Parallel(n_jobs=workers, verbose=1)(
        delayed(run_simulation)(idx, row) for idx, row in grid_parameters.iterrows()
    )
    grid_parameters["run_result"] = results

    grid_parameters.to_csv("data/grid_parameters.csv")