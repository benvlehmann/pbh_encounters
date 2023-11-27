"""
This module defines a class `Sampler` for performing a parametric analysis of
a celestial mechanics system by sampling parameter space using a Sobol
sequence and evaluating the system's response to each set of parameters. The
primary focus is on perturbations to planetary orbits due to flybys of small
objects.

This module uses the Sobol sequence method for generating samples in a
high-dimensional parameter space, evaluates the system's response to these
samples using the `delta` function, and saves the results in an HDF5 file. It
is designed to run either in serial mode or in parallel using MPI through the
`schwimmbad` package.

"""

import numpy as np
from scipy.stats.qmc import Sobol
import h5py

from .common import AU, KILOMETER, SECOND, GRAM
from .simulation import DistanceGenerator
from .ephemeris import PbhBody, BodyGroup
from .data import SOLAR_SYSTEM_SIMPLIFIED as SSS


# Define the batch size of samples between disk writes
SAVE_INTERVAL = int(1e8)
PBH_MASS = 1e27*GRAM
PBH_SPEED = 220*KILOMETER/SECOND


class Sampler(object):
    def __init__(self, *args, **kwargs):
        """Initialize core attributes of the simulation scenario."""
        self.bodies = kwargs.get('bodies', SSS)
        self.n_samples = kwargs.get('n_samples')
        self.all_bodies = BodyGroup(*([PbhBody()] + list(self.bodies)))
        self.output = kwargs.get('output')

        # Names of parameters specific to the PBH
        self.param_names = ('r', 'cos_theta', 'phi', 'cos_alpha', 'beta')
        self.param_index = {
            param: i for i, param in enumerate(self.param_names)
        }
        self.bounds = kwargs.get(
            'bounds', np.array([
                [300*AU,  700*AU],
                [    -1,       1],
                [     0, 2*np.pi],
                [    -1,       1],
                [     0, 2*np.pi]
            ])
        )

        # Total number of dimensions in the parameter space
        self.n_dim = len(self.param_names)

        # The group of bodies used for computing distances in the simulation
        self.dist_bodies = kwargs.get('dist_bodies')

        # Create a DistanceGenerator for evaluating the deltas
        kwargs['bodies'] = self.bodies
        self.simulator = DistanceGenerator(**kwargs)

        # Array of observational uncertainties for each body
        self.dist_uncertainty = np.array([
            body.obs_noise for body in self.dist_bodies
        ])

    def func(self, params):
        r, cos_theta, phi, cos_alpha, beta = params
        theta = np.arccos(cos_theta)
        alpha = np.arccos(cos_alpha)
        deltas = self.simulator.delta_distances(
            PBH_MASS, r, theta, phi, PBH_SPEED, alpha, beta)
        fom = np.sqrt(
            np.sum((deltas / self.dist_uncertainty[:, np.newaxis])**2))
        dof = deltas.compressed().size
        return np.array([fom, dof])

    def save(self, results):
        """
        Save the simulation results to an HDF5 file.

        Parameters:
        results (list): A list of results to be saved.
        filename (str): Name of the file to save the results in.

        We need to save both the Sobol samples and the values.
        """
        data = np.vstack((self.points.T, np.array(results).T)).T
        with h5py.File(self.output, 'w') as f:
            dataset = f.create_dataset("results", data=data, dtype='float64')
            dataset.attrs['PBH_MASS'] = PBH_MASS
            dataset.attrs['PBH_SPEED'] = PBH_SPEED

    def sample(self, pool):
        """
        Main function to generate samples, evaluate them, and save the results.

        Parameters:
        pool (schwimmbad.JoblibPool or similar): A pool for parallel or
            serial processing.
        num_samples (int): Number of samples to generate.
        dim (int): The number of dimensions in the parameter space.
        bounds (list of tuples): The boundaries for each dimension in the
            parameter space.

        Returns:
        array: A list of results for each sample.

        """
        if pool.is_master():
            # Generate parameter points from a Sobol sequence
            sobol_sampler = Sobol(d=self.n_dim, scramble=False)
            self.points = sobol_sampler.random_base2(
                m=int(np.log2(self.n_samples)))

            # Rescale Sobol samples to the given parameter bounds
            lower_bounds, upper_bounds = np.array(self.bounds).T
            diffs = upper_bounds - lower_bounds
            self.points = lower_bounds + self.points * diffs

            # Evaluate the function on the sample points
            tasks = pool.map(self.func, self.points)
            results = np.array([result for task in tasks for result in task])

            # Save results if requested
            if self.output is not None:
                self.save(results)
            return results
        else:
            pool.wait_workers()
