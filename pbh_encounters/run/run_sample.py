import argparse
from schwimmbad import choose_pool

from ..ephemeris import BodyGroup
from ..data import SOLAR_SYSTEM_SIMPLIFIED as SSS
from ..sample import Sampler


def run(n_samples, output, case):
    kwargs = dict(
        n_samples=n_samples,
        output=output,
        dist_bodies=BodyGroup(
            SSS.n['Mars'],
            SSS.n['Venus']
        )
    )
    if case == 'mars_only':
        kwargs.update(
            bodies=BodyGroup(
                SSS.n['Sun'],
                SSS.n['Earth'],
                SSS.n['Mars']
            ),
            dist_bodies=BodyGroup(SSS.n['Mars'])
        )
    elif case == 'inner_only':
        kwargs.update(
            bodies=BodyGroup(
                SSS.n['Sun'],
                SSS.n['Mercury'],
                SSS.n['Venus'],
                SSS.n['Earth'],
                SSS.n['Mars']
            )
        )
    elif case == 'full':
        kwargs.update(bodies=SSS)
    else:
        raise ValueError("Unrecognized case")

    sampler = Sampler(**kwargs)
    return sampler
    with choose_pool(mpi=True, processes=None) as pool:
        sampler.sample(pool)


def main():
    """
    Parse command-line arguments and run the MCMC simulation.
    """
    # Create the parser
    parser = argparse.ArgumentParser(description='Sample distance deltas.')

    # Add arguments
    parser.add_argument(
        '-n', '--n-samples',
        type=int,
        required=True,
        help="Number of samples to generate."
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
    run(args.n_samples, args.output, args.case)
