"""
Module: mcmc.py

This module provides the SolarSystemMonteCarlo class, which is used for
conducting Monte Carlo simulations on solar system models. This class
leverages Markov Chain Monte Carlo (MCMC) techniques, specifically using the
emcee package, to explore parameter space in solar system simulations that
may include a Primordial Black Hole (PBH). It allows for parameter inference
and uncertainty quantification in the context of celestial mechanics.

Classes:
- SolarSystemMonteCarlo: A class that sets up and runs Monte Carlo simulations
  on a solar system model. It handles parameter transformations, defines
  log-prior, log-likelihood, and log-probability functions for the MCMC, and
  provides a method to run the simulation.

The class tracks various parameters, including the mass of the PBH in log
space, PBH's position and velocity in angular coordinates, and deviations in
mass (dm) for other celestial objects. It also contains methods to transform
parameters for MCMC, compute distances based on parameters, and run the MCMC
simulation.

Example Usage:
>>> mc = SolarSystemMonteCarlo(
        bodies=my_bodies, dist_bodies=my_dist_bodies, n_steps=1000)
>>> sampler = mc.run()

Dependencies:
- numpy: Used for numerical operations and handling arrays.
- emcee: A Python module for performing MCMC.
- multiprocessing: For parallel processing capabilities.

"""

import os
import numpy as np
import emcee
import multiprocessing
from multiprocessing import Pool
from schwimmbad import MPIPool

from .common import (
    BadMassError, AU, KILOMETER, SECOND, GRAM)
from .ephemeris import PbhBody, BodyGroup
from .simulation import DistanceGenerator
from .data import SOLAR_SYSTEM_SIMPLIFIED


class SolarSystemMonteCarlo(object):
    """
    A class for running Monte Carlo simulations on a solar system model,
    potentially including a Primordial Black Hole (PBH).

    This class uses Markov Chain Monte Carlo (MCMC) techniques to explore the
    parameter space of a solar system model. It tracks parameters related to
    the PBH and other celestial bodies and calculates the likelihood of these
    parameters based on observational data.

    For the PBH, we need all 7 parameters. For other objects, we only need
    the masses: these are the only things that are measured from the
    simulation.

    We'll track the PBH mass in log space. Note that this is natural log, for
    simplicity of inversion. The PBH position and velocity are coordinatized
    using the angular coordinate systems of `DistanceGenerator`. Since we have
    a tight prior on the speed `s`, we will vary `ds`, measured from 220 km/s.

    So, we need:

        (log_m_pbh, r, theta, phi, ds, alpha, beta, dm1, dm2, ..., dmn).

    A crucial point: the actual parameters will be deviations away from the
    true parameters in the absence of a PBH. (This is why I'm writing "dm".)
    We use `dist_bodies` for computing the loglikelihood, and we rescale
    certain parameters for conditioning reasons.

    Attributes:
    bodies (BodyGroup): List of celestial bodies in the simulation.
    all_bodies (BodyGroup): Group of all bodies, including the PBH and other
        celestial bodies.
    n_steps (int): Number of steps for the MCMC simulation.
    dist_bodies (BodyGroup): Group of bodies used for computing the
        log-likelihood.
    simulator (DistanceGenerator): A DistanceGenerator instance for
        calculating distances in the simulation.
    dist_uncertainty (numpy.ndarray): Array of observational uncertainties
        for each body.
    pbh_mass_min (float): minimum allowed mass for the PBH.
    pbh_mass_max (float): maximum allowed mass for the PBH.
    pbh_param_names (tuple): Names of the PBH parameters.
    pbh_param_scales (numpy.ndarray): Scaling factors for the PBH parameters.
    base_s (float): Base speed for the PBH.
    dm_scale (float): Scaling factor for mass differences.
    pbh_param_offset (dict): Offset indices for PBH parameters.
    body_index (dict): Mapping of body hashes to their indices.
    all_param_names (list): List of all parameter names.
    n_dim (int): Number of dimensions in the parameter space.
    n_walkers (int): Number of walkers for the MCMC simulation.
    origins (numpy.ndarray): Initial positions for the MCMC walkers.
    add_noise (bool): Flag to add noise to mock observed distances.

    Methods:
    transform(params): Transforms physical parameters to MCMC parameters.
    transform_inverse(params): Inverse transformation from MCMC to physical
        parameters.
    param_index(body, param): Gets the index of a parameter for a given body.
    log_prior(scaled_params): Computes the log-prior for a set of parameters.
    distances(params): Computes the distances based on the current parameters.
    log_likelihood(scaled_params): Computes the log-likelihood for a set of
        parameters.
    log_probability(params): Computes the total log-probability
        (log-prior + log-likelihood).
    run(*args, init_kwargs, parallel, **kwargs): Runs the MCMC simulation.

    The class is designed to be flexible and can be adapted for different
    solar system models and observational datasets.

    Example Usage:
    >>> mc = SolarSystemMonteCarlo(
            bodies=my_bodies, dist_bodies=my_dist_bodies, n_steps=1000)
    >>> sampler = mc.run()

    """
    def __init__(self, *args, **kwargs):
        """Initialize core attributes of the Monte Carlo simulation."""
        self.bodies = kwargs.get('bodies', SOLAR_SYSTEM_SIMPLIFIED)
        self.all_bodies = BodyGroup(*([PbhBody()] + list(self.bodies)))

        # The number of steps for each walker in the MCMC simulation
        self.n_steps = kwargs.get('n_steps', 100)

        # The group of bodies used for computing distances in the simulation
        self.dist_bodies = kwargs.get('dist_bodies')

        # Create a DistanceGenerator for evaluating the likelihood
        kwargs['bodies'] = self.bodies
        self.simulator = DistanceGenerator(**kwargs)

        # Array of observational uncertainties for each body
        self.dist_uncertainty = np.array([
            body.obs_noise for body in self.dist_bodies
        ])

        # Mass bounds for PBH
        self.pbh_mass_min = kwargs.get('pbh_mass_min', 1e17*GRAM)
        self.pbh_mass_max = kwargs.get('pbh_mass_max', 1e25*GRAM)

        # Names of parameters specific to the PBH
        self.pbh_param_names = (
            'log_mass', 'r', 'theta', 'phi', 'ds', 'alpha', 'beta')

        # Scaling factors for the PBH parameters
        self.pbh_param_scales = np.array([
            10, 100*AU, 1, 1, 1*KILOMETER/SECOND, 1e-2, 1
        ])

        # Base speed for the PBH
        self.base_s = 220*KILOMETER/SECOND

        # Scaling factor for the mass differences of celestial bodies
        self.dm_scale = 1e-8

        # Mapping of PBH parameter names to their index offsets
        self.pbh_param_offset = {
            param: i for i, param in enumerate(self.pbh_param_names)
        }

        # Mapping of body hashes to their indices in all_bodies
        self.body_index = {b.hash: i for i, b in enumerate(self.all_bodies)}

        # List of all parameter names (mass differences of bodies)
        self.all_param_names = list(self.pbh_param_names) + [
            f'[{body.name}].dm' for body in self.bodies
        ]

        # Total number of dimensions in the parameter space
        self.n_dim = len(self.all_param_names)

        # Number of walkers to use in the MCMC simulation
        self.n_walkers = kwargs.get('n_walkers', 2*self.n_dim)

        # Set the noise flag
        self.add_noise = kwargs.get('add_noise', True)

        # Generate starting points
        self.origins = np.zeros((self.n_walkers, self.n_dim))
        for i in range(self.n_walkers):
            # Take PBH mass log-uniform between min and max
            self.origins[i, 0] = np.random.uniform(
                np.log(self.pbh_mass_min), np.log(self.pbh_mass_max)
            )
            # Sample the PBH distance uniformly but far
            r = np.random.uniform(300, 700) * AU
            # Sample the PBH angles uniformly on the sphere
            cos_theta = np.random.uniform(-1, 1)
            theta = np.arccos(cos_theta)
            phi = np.random.uniform(0, 2*np.pi)
            self.origins[i, 1:4] = r, theta, phi
            # Sample the PBH vxyz in a cone towards the origin. The max
            # opening angle is chosen to correspond to an impact parameter of
            # 5 AU with the sun.
            max_alpha = np.arctan(5*AU / r)
            ds = np.random.normal(scale=5) * KILOMETER/SECOND
            alpha = np.random.uniform(low=0, high=max_alpha)
            beta = np.random.uniform(low=0, high=2*np.pi)
            self.origins[i, 4:7] = ds, alpha, beta
            # Sample the mass differences for the other bodies
            self.origins[i, 7:] = \
                self.simulator.base_masses * np.random.normal(
                    loc=0., scale=1e-8, size=self.simulator.base_masses.size
                )
            self.origins[i] = self.transform(self.origins[i])

    def transform(self, params):
        """
        Transforms physical parameters to a scale suitable for MCMC
        exploration.

        This method is used to rescale parameters from their physical
        dimensions to a normalized scale that is more suitable for MCMC
        exploration. This includes scaling down large numerical values and
        applying periodic boundary conditions to angular parameters.

        Parameters:
        params (numpy.ndarray): An array of physical parameters to transform.

        Returns:
        numpy.ndarray: Transformed parameters suitable for MCMC.

        This transformation is crucial for the effective exploration of
        parameter space by the MCMC algorithm, particularly when dealing with
        parameters that span a wide range of magnitudes or are cyclic in
        nature.

        """
        # Copy the parameters to avoid modifying the original array
        transformed = np.copy(params)

        # Rescale the PBH parameters by their respective scales
        # The first few parameters in 'params' correspond to the PBH
        transformed[..., :len(self.pbh_param_scales)] /= self.pbh_param_scales

        # Rescale the mass differences (dm) for other celestial bodies
        # These parameters follow the PBH parameters in the 'params' array
        transformed[..., len(self.pbh_param_scales):] /= self.dm_scale

        # Apply periodic boundary conditions to angular parameters
        # This ensures that angles are properly normalized for MCMC exploration
        # Indices of angular parameters (theta, phi, alpha, beta) in `params`
        index = [2, 3, 5, 6]
        # Modulo operation to put the angular parameters into [0, 2pi)
        transformed[..., index] = np.mod(transformed[..., index], 2*np.pi)

        return transformed

    def transform_inverse(self, params):
        """
        Inversely transforms the MCMC parameters back to their physical scale.

        This method reverses the transformations applied by the `transform`
        method, converting the parameters from the MCMC scale back to their
        original physical dimensions. This is crucial for interpreting the
        results of the MCMC simulation in a physically meaningful way.

        Parameters:
        params (numpy.ndarray): An array of MCMC parameters to be transformed.

        Returns:
        numpy.ndarray: Inversely transformed parameters in their original
            physical scale.

        The inverse transformation is essential for converting the MCMC
        exploration results back into a physical context, especially for
        parameters like PBH mass, position, and velocity.

        """
        transformed = np.copy(params)
        transformed[..., :len(self.pbh_param_scales)] *= self.pbh_param_scales
        transformed[..., len(self.pbh_param_scales):] *= self.dm_scale
        index = [2, 3, 5, 6]
        transformed[..., index] = np.mod(transformed[..., index], 2*np.pi)
        return transformed

    def param_index(self, body, param):
        """
        Retrieves the index of a specific parameter for a given body within
        the parameter array.

        This method is used to identify the location (index) of a particular
        parameter in the flattened parameter array used by the MCMC
        simulation. The index is determined based on the body (either a PBH
        or another celestial body) and the parameter of interest.

        Parameters:
        body (Body or PbhBody): The celestial body for which the parameter
            index is needed.
        param (str): The name of the parameter whose index is to be retrieved.

        Returns:
        int: The index of the specified parameter in the MCMC parameter array.

        This method facilitates the manipulation and retrieval of specific
        parameters from the  multi-dimensional parameter space used in the
        MCMC simulation.

        """
        # Determine the base index depending on whether the body is a PBH or
        # another celestial body
        if body.hash == PbhBody.hash:
            # For a PBH, the index is based on the order of PBH parameters
            index_base = 0
            index_offset = self.pbh_param_offset[param]
        else:
            # For other celestial bodies, the index is based on the position
            # in the all_bodies list
            index_base = self.body_index[body.hash]
            # Offset is determined by the number of PBH-specific parameters
            index_offset = len(self.pbh_param_names)

        # Calculate the total index by adding the base index and the offset
        return index_base + index_offset

    def log_prior(self, scaled_params):
        """
        Computes the log-prior of a set of parameters.

        This method evaluates the log-prior for a given set of parameters in
        the MCMC simulation. The log-prior reflects the initial assumptions
        about the probability distribution of the parameters before
        considering the observational data. It includes conditions and
        constraints on various parameters, such as limits on the PBH mass and
        position.

        Parameters:
        scaled_params (numpy.ndarray): An array of scaled parameters used in
            the MCMC simulation.

        Returns:
        float: The logarithm of the prior probability of the given parameters.
            Returns -np.inf if the parameters are outside the acceptable
            range, indicating a zero prior probability.

        The log-prior function is crucial in Bayesian inference as it encodes
        prior beliefs about the parameters before observing the data. It is
        used in conjunction with the log-likelihood to compute the
        log-posterior.

        """
        # Inversely transform the parameters to their original physical scale
        params = self.transform_inverse(scaled_params)

        # Unpack the PBH-specific parameters from the transformed parameters
        log_mass, r, theta, phi, ds, alpha, beta = params[:7]
        # The remaining parameters represent the mass differences (dm) for
        # other celestial bodies
        dm = np.asarray(params[7:])

        def _log_normal(mean, sigma, value):
            # Helper function to compute the contribution to the log-prior
            # from a normal distribution
            return -1/(2*sigma**2) * (value - mean)**2

        log_value = 0

        # log_mass: Penalize for going out of bounds
        if log_mass > np.log(self.pbh_mass_max):
            return -np.inf
        if log_mass < np.log(self.pbh_mass_min):
            return -np.inf

        # r: Penalize departure from a specified window
        if not 300*AU <= r <= 700*AU:
            return -np.inf

        # theta, phi: Flat priors, no penalization
        log_value += 0

        # ds: Penalize deviation from the Milky Way halo's typical velocity
        log_value += _log_normal(0, 1*KILOMETER/SECOND, ds)

        # alpha: Restrict to angles within a cone of impact parameter 5 AU.
        alpha_scale = np.arctan(5*AU / r)
        log_value += _log_normal(0, alpha_scale, alpha)

        # beta: Flat prior, no penalization
        log_value += 0

        # dm: Gaussian prior with 1e-3 precision
        log_value += np.sum(
            _log_normal(0, 1e-3*self.simulator.base_masses, dm))

        return log_value

    def delta_distances(self, params):
        """
        Computes the distances of celestial bodies from Earth in the
        simulation based on the given parameters, and returns differences from
        the base case.

        This method calculates the distances of the celestial bodies,
        including the PBH, from Earth at various time points in the
        simulation. It utilizes the current set of parameters to determine
        the positions and velocities of the bodies, and then calculates the
        distances based on these positions.

        Parameters:
        params (numpy.ndarray): An array of parameters, including those for
            the PBH and mass adjustments for other celestial bodies.

        Returns:
        numpy.ndarray: An array of delta distances from simulation.

        The distance calculations are an essential part of evaluating the
        likelihood of the parameter set in the context of the observed data.

        """
        # Unpack the PBH-specific parameters from the given parameters
        log_mass, r, theta, phi, ds, alpha, beta = params[:7]
        # The remaining parameters represent the mass differences (dm) for
        # other celestial bodies
        dm = np.asarray(params[7:])

        # Calculate the actual speed of the PBH by adding the base speed to
        # the speed deviation (ds)
        s = self.base_s + ds

        # Convert the log_mass to actual mass
        pbh_mass = np.exp(log_mass)

        # Use the simulator to calculate distances
        deltas = self.simulator.delta_pivot(
            1e27*GRAM, pbh_mass, r, theta, phi, s, alpha, beta, dm=dm)

        return deltas

    def log_likelihood(self, scaled_params):
        """
        Computes the log-likelihood of a set of parameters given the
        observational data.

        This method evaluates the log-likelihood for the given set of
        parameters based on how well the simulated distances (derived from
        these parameters) match the observed distances. The likelihood is a
        key component in Bayesian inference, determining how probable the
        observed data is under the assumed model and parameter set.

        Note that here, "observed" data is actually simulated data in the
        absence of a PBH.

        Parameters:
        scaled_params (numpy.ndarray): An array of scaled parameters used in
            the MCMC simulation.

        Returns:
        float: The log-likelihood of the given parameters based on the
            observational data.

        The log-likelihood is combined with the log-prior to determine the
        log-posterior probability in MCMC. A higher log-likelihood indicates
        a better fit of the model to the observational data.

        """
        # Inversely transform the parameters to their original physical scale
        params = self.transform_inverse(scaled_params)

        # Compute the distances of celestial bodies with these parameters
        try:
            deltas = self.delta_distances(params)
        except BadMassError:
            return -np.inf

        if self.add_noise:
            # Add noise to the deltas sampled based on the uncertainties.
            # The shape of `deltas` is (n_bodies, n_times).
            noise = np.random.multivariate_normal(
                mean=np.zeros_like(self.dist_uncertainty),
                cov=np.diag(self.dist_uncertainty**2),
                size=deltas.shape[1]
            ).T
            deltas += noise

        # Compute the total log-likelihood.
        # This is based on the difference between the simulated distances and
        # the distances in the base (no-PBH) simulation, normalized by the
        # observational uncertainties.
        total = -1/2 * np.sum(
            (deltas / self.dist_uncertainty[:, np.newaxis])**2,
        )

        # Check for numeric stability
        if not np.isfinite(total):
            raise RuntimeError

        return total

    def log_probability(self, params):
        """
        Computes the total logarithm of the posterior probability of a set of
        parameters.

        This method combines the log-prior and log-likelihood to compute the
        log-posterior, which represents the probability of the parameters
        given both the prior beliefs and the observed data. This is the key
        metric used in MCMC to explore and sample the parameter space.

        Parameters:
        params (numpy.ndarray): An array of parameters for which the
            log-probability is to be computed.

        Returns:
        float: The total logarithm of the posterior probability of the given
            parameters.

        In Bayesian inference, the log-posterior is central to updating
        beliefs about the parameters based on observed data. A higher
        log-posterior indicates a higher probability of the parameters being
        correct given the model and data.

        """
        # First, compute the log-prior for the given parameters
        # This reflects our initial beliefs about the parameter values before
        # considering the data
        log_prior = self.log_prior(params)

        # If the log-prior is -inf, the parameters are outside the acceptable
        # range, so return -inf immediately
        if not np.isfinite(log_prior):
            return -np.inf

        # Compute the log-likelihood, which measures how well the parameters
        # fit the observed data. The total log-probability is the sum of the
        # log-prior and log-likelihood.
        return log_prior + self.log_likelihood(params)

    def run(self, *args, init_kwargs=dict(), parallel=True,
            mpi=False, **kwargs):
        """
        Runs the MCMC simulation using the emcee package.

        This method sets up and executes the MCMC simulation, sampling from
        the parameter space to infer the most probable values and their
        uncertainties. It can run in parallel to utilize multiple CPU cores
        for faster computation.

        Parameters:
        *args: Variable length argument list, not currently used.
        init_kwargs (dict, optional): Initial keyword arguments for the
            emcee.EnsembleSampler.
        parallel (bool, optional): Flag to indicate if the simulation should
            run in parallel. Defaults to True.
        mpi (bool, optional): Flag to indicate if parallelization should use
            MPI instead of multiprocessing. If True, the `parallel` keyword
            is ignored. Defaults to False.
        **kwargs: Additional keyword arguments for the emcee's run_mcmc method.

        Returns:
        emcee.EnsembleSampler: The MCMC sampler after running the simulation.

        This method is the primary entry point for executing the Monte Carlo
        simulation. It manages the setup, execution, and parallelization of
        the MCMC process.

        """

        # Set up parallel processing if enabled
        if mpi:
            pool_function = MPIPool
            pool_args = ()
        elif parallel:
            pool_function = Pool
            pool_args = multiprocessing.cpu_count() - 1
        else:
            def pool_function():
                return None
            pool_args = ()
        with pool_function(*pool_args) as pool:
            # Initialize the MCMC sampler from emcee
            sampler = emcee.EnsembleSampler(
                self.n_walkers, self.n_dim, self.log_probability,
                **init_kwargs, pool=pool
            )

            # Set the starting point, resuming existing chains if supplied
            start = self.origins
            backend = init_kwargs.get('backend')
            if backend is not None:
                if os.path.exists(backend.filename):
                    start = None

            # Run the MCMC simulation
            # The origins are the starting points for each walker
            # The number of steps each walker takes is defined by self.n_steps
            sampler.run_mcmc(start, self.n_steps, progress=True, **kwargs)

        # Return the sampler for further analysis or diagnostics
        return sampler
