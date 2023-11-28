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
import h5py
from scipy.stats.qmc import Sobol
from tqdm import tqdm

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
        self.batch_size = kwargs.get('batch_size', int(1e4))
        self.log2_n_samples = kwargs.get('n_samples')
        self.n_samples = 2**self.log2_n_samples
        self.all_bodies = BodyGroup(*([PbhBody()] + list(self.bodies)))
        self.output = kwargs.get('output')
        self.output_offset = 0

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
        raise NotImplementedError

    def save(self, points, results):
        """
        Save the simulation results to an HDF5 file.

        Parameters:
        results (list): A list of results to be saved.
        filename (str): Name of the file to save the results in.

        We need to save both the Sobol samples and the values.
        """
        dataset_name = "results"
        data = np.vstack((points.T, results.T)).T
        with h5py.File(self.output, 'a') as f:
            if dataset_name not in f:
                # Create the dataset if it doesn't exist
                maxshape = (None,) + data.shape[1:]
                dataset = f.create_dataset(
                    dataset_name, data=data, maxshape=maxshape,
                    dtype='float64', chunks=True)
                dataset.attrs['PBH_MASS'] = PBH_MASS
                dataset.attrs['PBH_SPEED'] = PBH_SPEED
            else:
                # Resize and append
                dataset = f[dataset_name]
                index = dataset.shape[0]
                dataset.resize((index + data.shape[0]), axis=0)
                dataset[index:index + data.shape[0]] = data

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
        try:
            master = pool.is_master()
        except AttributeError:
            master = True
            mpi = False
        else:
            mpi = True

        if mpi and not master:
            pool.wait_workers()
            return

        # Generate parameter points from a Sobol sequence
        sobol_sampler = Sobol(d=self.n_dim, scramble=False)
        self.points = sobol_sampler.random_base2(m=self.log2_n_samples)
        np.random.shuffle(self.points)

        # Rescale Sobol samples to the given parameter bounds
        lower_bounds, upper_bounds = np.array(self.bounds).T
        diffs = upper_bounds - lower_bounds
        self.points = lower_bounds + self.points * diffs
        batches = []
        # Evaluate the function on the sample points
        for start_idx in np.arange(
            0, self.n_samples, self.batch_size, dtype=int
        ):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batches.append(self.points[start_idx:end_idx])

        # This line prevents the per-batch speed from increasing with the
        # number of samples. I have NO IDEA why.
        del self.points

        # Evaluate the function on the sample points
        for batch in tqdm(batches):
            batch_results = list(pool.map(self.func, batch))
            batch_results = np.array(batch_results)
            self.save(batch, batch_results)


class DistanceSampler(Sampler):
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


def fourier_transform(times, signal):
    # Calculate time step (assuming uniform sampling)
    dt = times[1] - times[0]
    
    # Perform the FFT
    fft_result = np.fft.fft(signal)

    # Get frequencies corresponding to the FFT components
    frequencies = np.fft.fftfreq(len(signal), d=dt)

    # Get the amplitudes (absolute values of FFT result)
    fft_amplitudes = np.abs(fft_result)

    phases = np.angle(fft_result)
    frequencies = np.abs(frequencies)
    order = np.argsort(frequencies)

    return frequencies[order], fft_amplitudes[order], phases[order]


class SpectralRatioSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Determine the inverse period of each distance body
        self.inverse_periods = np.zeros(len(self.dist_bodies))
        for i, distances in enumerate(self.simulator.base_dists):
            times = self.simulator.times[i]
            deviations = distances - np.mean(distances)
            frequencies, amplitudes, _ = fourier_transform(times, deviations)
            self.inverse_periods[i] = \
                np.abs(frequencies[np.argmax(amplitudes)])

    def statistic(self, times, signal, inverse_period):
        frequencies, amplitudes, _ = fourier_transform(times, signal)
        mask = (0.5*inverse_period < frequencies) & \
               (frequencies < 1.5*inverse_period)
        return np.sum(amplitudes[mask]) / np.sum(amplitudes)

    def func(self, params):
        r, cos_theta, phi, cos_alpha, beta = params
        theta = np.arccos(cos_theta)
        alpha = np.arccos(cos_alpha)
        deltas = self.simulator.delta_distances(
            PBH_MASS, r, theta, phi, PBH_SPEED, alpha, beta)

        # Noise the deltas
        noise = np.random.multivariate_normal(
            mean=np.zeros_like(self.dist_uncertainty),
            cov=np.diag(self.dist_uncertainty**2),
            size=deltas.shape[1]
        ).T
        deltas += noise

        # Compute the test statistics
        signal_stats = np.zeros(len(self.dist_bodies))
        noise_stats = np.zeros_like(signal_stats)
        for i, delta in enumerate(deltas):
            times = self.simulator.times[i].compressed()
            signal_stats[i] = self.statistic(
                times,
                delta.compressed(),
                self.inverse_periods[i]
            )
            noise_stats[i] = self.statistic(
                times,
                noise[i],
                self.inverse_periods[i]
            )
        return np.hstack((signal_stats, noise_stats))
