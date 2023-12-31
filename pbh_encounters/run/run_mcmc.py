"""
This script provides functionality to run Monte Carlo Markov Chain
(MCMC) simulations of primordial black hole (PBH) encounters with the solar
system. It allows for simulating different scenarios including Mars only,
inner planets only, and the full solar system.

"""

import os
import argparse
import emcee

from ..ephemeris import BodyGroup
from ..data import SOLAR_SYSTEM_SIMPLIFIED as SSS
from ..mcmc import SolarSystemMonteCarlo


def run(n_steps, n_walkers, output, case):
    """
    Run the MCMC simulation for a specified solar system case.

    Args:
        n_steps (int): The number of steps to use in the MCMC simulation.
        n_walkers (int): The number of walkers to use in the MCMC simulation.
        output (str): File path where the MCMC chains will be saved.
        case (str): Specifies the simulation case. Options are 'mars_only',
            'inner_only', or 'full'.
    """
    dist_bodies = BodyGroup(
        SSS.n['Mars'],
        SSS.n['Venus']
    )
    kwargs = dict(
        n_steps=n_steps,
        n_walkers=n_walkers,
        sim_kwargs=dict(
            sim_attrs=dict(
                integrator='ias15'
            )
        )
    )
    mc_cases = dict(
        mars_only=SolarSystemMonteCarlo(
            bodies=BodyGroup(
                SSS.n['Sun'],
                SSS.n['Earth'],
                SSS.n['Mars']
            ),
            dist_bodies=BodyGroup(
                SSS.n['Mars'],
            ), **kwargs
        ),
        inner_only=SolarSystemMonteCarlo(
            bodies=BodyGroup(
                SSS.n['Sun'],
                SSS.n['Mercury'],
                SSS.n['Venus'],
                SSS.n['Earth'],
                SSS.n['Mars']
            ),
            dist_bodies=dist_bodies, **kwargs
        ),
        full=SolarSystemMonteCarlo(
            bodies=SSS,
            dist_bodies=dist_bodies, **kwargs
        )
    )
    mc = mc_cases[case]

    resume = os.path.exists(output)
    backend = emcee.backends.HDFBackend(output)
    mc.run(
        skip_initial_state_check=True, mpi=True,
        init_kwargs=dict(
            backend=backend
        ),
        resume=resume
    )


def main():
    """
    Parse command-line arguments and run the MCMC simulation.
    """
    # Create the parser
    parser = argparse.ArgumentParser(description='Run MCMC simulations.')

    # Add arguments
    parser.add_argument(
        '--n-steps', 
        type=int, 
        required=True, 
        help='Number of steps to use in the MCMC simulation.'
    )
    parser.add_argument(
        '--n-walkers', 
        type=int, 
        required=True, 
        help='Number of walkers to use in the MCMC simulation.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help='File path to which to save the MCMC chains.'
    )
    parser.add_argument(
        '--case', 
        type=str, 
        required=True, 
        choices=['mars_only', 'inner_only', 'full'],
        help="Simulation case: 'mars_only', 'inner_only', or 'full'."
    )

    # Parse arguments
    args = parser.parse_args()

    # Run main function with provided arguments
    run(args.n_steps, args.n_walkers, args.output, args.case)
