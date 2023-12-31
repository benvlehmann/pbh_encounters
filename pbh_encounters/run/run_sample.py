import argparse
from schwimmbad import choose_pool

from ..ephemeris import BodyGroup
from ..data import SOLAR_SYSTEM_SIMPLIFIED as SSS
from ..sample import MultiDistanceSampler as Sampler


def run(n_samples, output, case, batch_size, no_mpi):
    kwargs = dict(
        n_samples=n_samples,
        batch_size=batch_size,
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
        kwargs.update(bodies=SSS, dist_bodies=SSS)
    else:
        raise ValueError("Unrecognized case")

    sampler = Sampler(**kwargs)
    if no_mpi:
        pool_kwargs = dict(processes=4)
    else:
        pool_kwargs = dict(mpi=True, processes=None)
    with choose_pool(**pool_kwargs) as pool:
        sampler.sample(pool)
    return sampler


def main():
    """
    Parse command-line arguments and run the sampler
    """
    # Create the parser
    parser = argparse.ArgumentParser(description='Sample distance deltas.')

    # Add arguments
    parser.add_argument(
        '-n', '--n-samples',
        type=int,
        required=True,
        help="Log base 2 of the number of samples to generate."
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help='File path to which to save the samples.'
    )
    parser.add_argument(
        '--case', 
        type=str, 
        required=True, 
        choices=['mars_only', 'inner_only', 'full'],
        help="Simulation case: 'mars_only', 'inner_only', or 'full'."
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        required=False,
        default=int(1e4),
        help="Batch size for sample computation."
    )
    parser.add_argument(
        '--no-mpi', 
        action='store_true',
        help="Flag to use multiprocessing instead of MPI."
    )

    # Parse arguments
    args = parser.parse_args()

    # Run main function with provided arguments
    run(args.n_samples, args.output, args.case, args.batch_size, args.no_mpi)
