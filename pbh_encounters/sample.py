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
                dataset.attrs['DIST_BODIES'] = \
                    [b.name for b in self.dist_bodies]
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

        self.masses = np.geomspace(1e18, 1e27, 10) * GRAM

        # Determine the inverse period of each distance body
        self.inverse_periods = np.zeros(len(self.dist_bodies))
        self.f_bounds = np.zeros((len(self.dist_bodies), 2))
        for i, distances in enumerate(self.simulator.base_dists):
            times = self.simulator.times[i]
            deviations = distances - np.mean(distances)
            frequencies, amplitudes, _ = fourier_transform(times, deviations)
            i_peak = np.argmax(amplitudes)
            f_peak = frequencies[i_peak]
            a_peak = amplitudes[i_peak]
            self.inverse_periods[i] = np.abs(f_peak)
            # Find half-max to the left
            a_i = a_peak
            i_left = i_peak
            while a_i > a_peak / 2:
                i_left -= 1
                a_i = amplitudes[i_left]
            f_left = frequencies[i_left]
            # Find half-max to the right
            a_i = a_peak
            i_right = i_peak
            while a_i > a_peak / 2:
                i_right += 1
                a_i = amplitudes[i_right]
            f_right = frequencies[i_right]
            self.f_bounds[i] = f_left, f_right

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        dataset_name = "results"
        with h5py.File(self.output, 'a') as f:
            dataset = f[dataset_name]
            dataset.attrs['SIGNAL_MASSES'] = self.masses

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
        noises = np.random.multivariate_normal(
            mean=np.zeros_like(self.dist_uncertainty),
            cov=np.diag(self.dist_uncertainty**2),
            size=deltas.shape[1]
        ).T

        # Compute the test statistics: each body, then repeated for each mass
        signal_stats = np.zeros((self.masses.size, len(self.dist_bodies)))
        noise_stats = np.zeros(len(self.dist_bodies))
        for i, mass in enumerate(self.masses):
            for j, delta in enumerate(deltas):
                times = self.simulator.times[j].compressed()
                signal_stats[i, j] = self.statistic(
                    times,
                    (delta*mass/PBH_MASS + noises[j]).compressed(),
                    self.inverse_periods[j]
                )
        for i, noise in enumerate(noises):
            noise_stats[i] = self.statistic(
                times,
                noise,
                self.inverse_periods[i]
            )
        return np.hstack((signal_stats.reshape(-1), noise_stats))


class CombinedSpectralRatioSampler(SpectralRatioSampler):
    def _divided_statistic(self, times, signal, f_bounds, t0):
        mask_1 = times < t0
        mask_2 = ~mask_1
        if np.count_nonzero(mask_1) < 2 or np.count_nonzero(mask_2) < 2:
            raise ValueError
        frequencies_1, amplitudes_1, _ = fourier_transform(
            times[mask_1].compressed(), signal[mask_1].compressed())
        frequencies_2, amplitudes_2, _ = fourier_transform(
            times[mask_2].compressed(), signal[mask_2].compressed())
        f_mask_1 = \
            (f_bounds[0] <= frequencies_1) & (frequencies_1 <= f_bounds[1])
        f_mask_2 = \
            (f_bounds[0] <= frequencies_2) & (frequencies_2 <= f_bounds[1])
        if not np.any(f_mask_1) or not np.any(f_mask_2):
            raise ValueError
        bin_power_1 = np.sum(amplitudes_1[f_mask_1])
        bin_power_2 = np.sum(amplitudes_2[f_mask_2])
        return (bin_power_2 / np.sum(amplitudes_2)) \
            * (bin_power_2 / bin_power_1)

    def _combined_divided_statistic(self, signals, t0):
        stats = []
        for time, signal, f_bound in zip(
            self.simulator.times, signals, self.f_bounds
        ):
            stats.append(self._divided_statistic(time, signal, f_bound, t0))
        return np.prod(stats)

    def _slide_statistic(self, signals, n_shifts=100):
        stats = []
        max_time = np.amax(self.simulator.times)
        for t0 in np.linspace(1/np.amin(self.f_bounds), max_time, n_shifts):
            try:
                stats.append(self._combined_divided_statistic(signals, t0))
            except ValueError:
                continue
        return stats

    def statistic(self, signal):
        return np.amax(self._slide_statistic(signal))

    def func(self, params):
        r, cos_theta, phi, cos_alpha, beta = params
        theta = np.arccos(cos_theta)
        alpha = np.arccos(cos_alpha)
        deltas = self.simulator.delta_distances(
            PBH_MASS, r, theta, phi, PBH_SPEED, alpha, beta)

        # Noise the deltas
        noises = np.random.multivariate_normal(
            mean=np.zeros_like(self.dist_uncertainty),
            cov=np.diag(self.dist_uncertainty**2),
            size=deltas.shape[1]
        ).T

        # Compute the test statistics
        signal_stats = np.zeros((self.masses.size))
        noise_stat = np.zeros(1)
        for i, mass in enumerate(self.masses):
            signal_stats[i] = self.statistic(
                deltas*mass/PBH_MASS + noises
            )
        noise_stat[0] = self.statistic(noises)
        return np.hstack((signal_stats, noise_stat))
