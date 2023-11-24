"""
This script provides functionality to run Monte Carlo Markov Chain
(MCMC) simulations of primordial black hole (PBH) encounters with the solar
system. It allows for simulating different scenarios including Mars only,
inner planets only, and the full solar system.

"""


import argparse
import emcee

from pbh_encounters.ephemeris import BodyGroup
from pbh_encounters.data import SOLAR_SYSTEM_SIMPLIFIED
from pbh_encounters.mcmc import SolarSystemMonteCarlo


def main(n_steps, output, case):
    """
    Run the MCMC simulation for a specified solar system case.

    Args:
        n_steps (int): The number of steps to be used in the MCMC simulation.
        output (str): File path where the MCMC chains will be saved.
        case (str): Specifies the simulation case. Options are 'mars_only',
            'inner_only', or 'full'.
    """
    dist_bodies = BodyGroup(
        SOLAR_SYSTEM_SIMPLIFIED.n['Mars'],
        SOLAR_SYSTEM_SIMPLIFIED.n['Venus']
    )
    kwargs = dict(
        n_steps=n_steps,
        sim_kwargs=dict(
            sim_attrs=dict(
                integrator='ias15'
            )
        )
    )
    mc_cases = dict(
        mars_only=SolarSystemMonteCarlo(
            bodies=BodyGroup(
                SOLAR_SYSTEM_SIMPLIFIED.n['Sun'],
                SOLAR_SYSTEM_SIMPLIFIED.n['Earth'],
                SOLAR_SYSTEM_SIMPLIFIED.n['Mars']
            ),
            dist_bodies=BodyGroup(
                SOLAR_SYSTEM_SIMPLIFIED.n['Mars'],
            ), **kwargs
        ),
        inner_only=SolarSystemMonteCarlo(
            bodies=BodyGroup(
                SOLAR_SYSTEM_SIMPLIFIED.n['Sun'],
                SOLAR_SYSTEM_SIMPLIFIED.n['Mercury'],
                SOLAR_SYSTEM_SIMPLIFIED.n['Venus'],
                SOLAR_SYSTEM_SIMPLIFIED.n['Earth'],
                SOLAR_SYSTEM_SIMPLIFIED.n['Mars']
            ),
            dist_bodies=dist_bodies, **kwargs
        ),
        full=SolarSystemMonteCarlo(
            bodies=SOLAR_SYSTEM_SIMPLIFIED,
            dist_bodies=dist_bodies, **kwargs
        )
    )
    mc = mc_cases[case]
    backend = emcee.backends.HDFBackend(output)
    backend.reset(mc.n_walkers, mc.n_dim)
    mc.run(
        skip_initial_state_check=True, mpi=True,
        init_kwargs=dict(
            backend=backend
        )
    )


if __name__ == '__main__':
    """
    Parse command-line arguments and run the MCMC simulation.
    """
    # Create the parser
    parser = argparse.ArgumentParser(description='Run MCMC simulations.')

    # Add arguments
    parser.add_argument(
        '--n_steps', 
        type=int, 
        required=True, 
        help='Number of steps to use in the MCMC simulation.'
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
    main(args.n_steps, args.output, args.case)
