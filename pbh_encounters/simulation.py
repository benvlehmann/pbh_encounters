"""
Module: simulation.py

This module provides classes for simulating the solar system with the ability
to include additional celestial objects like primordial black holes(PBHs). It
allows for creating, customizing, and running simulations to study the
dynamics and positions of celestial bodies in the solar system.

Classes:
- SolarSystemSimulation: Extends rebound.Simulation to create a simulation
  environment specifically for the solar system. It includes functionality
  for setting up a simulation with specific parameters and celestial bodies,
  including Earth.

- SolarSystemGenerator: A class for generating and configuring solar system
  simulations. It supports various customization options, including the
  selection of integrator, adding specific celestial bodies, and setting
  simulation attributes.

- DistanceGenerator: A class for generating distance measurements in a solar
  system simulation. It can simulate the impact of a PBH on the distances of
  specified bodies from Earth and provides functionality to analyze these
  changes over time.

These classes interact to create detailed simulations of the solar system,
allowing users to specify parameters like the types of celestial bodies
included, the time span of the simulation, and observing offsets. Users can
also introduce PBHs into the simulation to study their potential impact on
the solar system's dynamics.

Example Usage:
>>> from simulation import SolarSystemGenerator, DistanceGenerator
>>> generator = SolarSystemGenerator(
        bodies=my_bodies, sim_attrs={'integrator': 'ias15'})
>>> distance_generator = DistanceGenerator(
        bodies=my_bodies, dist_bodies=my_dist_bodies)
>>> simulation = generator.make_simulation()
>>> distances = distance_generator.distances(pbh_params, dm=mass_changes)

Dependencies:
- numpy: Used for mathematical operations and handling arrays.
- rebound: An astrophysical Python package for N-body simulations.

"""


import numpy as np

import rebound

from .common import (
    BadMassError,
    GRAVITATIONAL_CONSTANT, YEAR
)
from .ephemeris import BodyGroup, PbhBody, Ephemeris
from .data import SOLAR_SYSTEM_SIMPLIFIED, BODY_EARTH


class SolarSystemSimulation(rebound.Simulation):
    """
    A simulation class for modeling the solar system, extending the
    rebound.Simulation class.

    This class provides a specialized simulation environment for the solar
    system. It inherits from rebound.Simulation, which is a part of the
    REBOUND N-body simulation package used for astrophysical simulations. The
    SolarSystemSimulation class allows for additional customization specific
    to solar system simulations, such as setting a default value for Earth or
    other planetary bodies.

    In particular, this class is designed for measuring distances to objects
    in simulations. As such, it requires an origin point for those distance
    measurements, which defaults to `BODY_EARTH`.

    Parameters:
    *args: Variable length argument list to be passed to the
        rebound.Simulation constructor.
    **kwargs: Arbitrary keyword arguments. It includes an optional 'earth'
        parameter to set a specific Earth object. If 'earth' is not provided,
        a default value (BODY_EARTH) is used. Earth is used as the origin for
        distance measurements.

    Attributes: earth (object): An object representing Earth in the
        simulation, set to the value provided in 'earth' keyword argument or
        a default value (BODY_EARTH).

    Example:
    >>> sim = SolarSystemSimulation(earth=my_custom_earth_object)

    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the SolarSystemSimulation instance.

        It sets up the Earth object for the simulation, either using the
        provided 'earth' keyword argument or a default value, and then
        initializes the parent rebound.Simulation class with any additional
        arguments provided.

        """
        # Set Earth object for the simulation
        self.earth = kwargs.pop('earth', BODY_EARTH)

        # Initialize the parent class
        super().__init__(*args, **kwargs)

    def distances(self, bodies: BodyGroup, times):
        """
        Calculates the distances of specified bodies from Earth at given times
        in the simulation.

        This method computes the Euclidean distances of each specified
        celestial body from Earth at different times throughout the
        simulation. It integrates the positions of the bodies in the
        simulation to the specified times and then calculates the distances.

        Parameters:
        bodies (BodyGroup): A group of body objects whose distances from Earth
            are to be calculated.
        times (list or array): A list or array of times at which the distances
            are to be calculated.

        Returns:
        ndarray: A 2D numpy array containing the distances. The array has a
            shape of (len(times), len(bodies)), where each element [i, j]
            represents the distance of the j-th body from Earth at the i-th
            time.

        Example:
        >>> simulation = SolarSystemSimulation()
        >>> times = [0, 1, 2]  # Example times
        >>> bodies = BodyGroup(mars, venus)  # Example celestial bodies
        >>> distances = simulation.distances(bodies, times)

        """

        # Initialize arrays to store positions
        xyz = np.zeros((len(times), len(bodies), 3))
        earth_xyz = np.zeros((len(times), 3))

        # Loop over each time point
        for i, t in enumerate(times):
            # Integrate the simulation to time t
            self.integrate(t)
            # Get Earth's position at time t
            earth_xyz[i] = self.particles[self.earth.hash].xyz
            # Loop over each body to calculate its position at time t
            for j, body in enumerate(bodies):
                xyz[i, j] = self.particles[body.hash].xyz
        # Calculate the position deltas between Earth and each body
        deltas = earth_xyz[:, np.newaxis, :] - xyz
        # Calculate the Euclidean distance from the deltas
        dists = np.linalg.norm(deltas, axis=-1)
        return dists

    def positions(self, bodies: BodyGroup, times):
        """Like masked_distances, but just gives barycentric positions.

        Parameters:
        bodies (BodyGroup): A BodyGroup of body objects whose distances from
            Earth are to be calculated.
        times (MaskedArray): A 2D masked numpy array of times. The shape of
            the array is (number of bodies, number of times).

        Returns:
        MaskedArray: A 2D masked numpy array containing the distances. The
            shape of the array is the same as `times`, with each unmasked
            element indicating the distance of the corresponding body from
            Earth at that time.

        Raises: RuntimeError: If there are NaN values in the initial state or
        at any time step during the integration.

        """
        # Initialize arrays for storing positions
        xyz = np.zeros((times.shape[0], times.shape[1], 3))
        earth_xyz = np.zeros(xyz.shape[1:])
        all_times = np.sort(np.unique(times[~times.mask]))

        # Check the shape consistency of times array
        assert times.shape[1] == all_times.shape[0]

        # Validate the initial state
        for particle in self.particles:
            if np.any(~np.isfinite(particle.xyz)):
                raise BadMassError("nan in initial state")

        # Integrate and compute positions at each time step
        for i, t in enumerate(all_times):
            self.integrate(t)
            earth_xyz[i] = self.particles[self.earth.hash].xyz
            if np.any(~np.isfinite(earth_xyz[i])):
                raise BadMassError("nan at time", i)
            for j, body in enumerate(bodies):
                xyz[j, i] = self.particles[body.hash].xyz

        return xyz, earth_xyz

    def masked_distances(self, bodies: BodyGroup, times):
        """
        Calculates the distances of specified bodies from Earth at given
        times, generally different for each body, using masked arrays.

        This method is similar to `distances`, but it handles a 2D masked
        array of times. It computes the Euclidean distances from Earth to
        each specified celestial body for the unmasked time values. The
        result is a masked array where the distances corresponding to the
        masked times are also masked.

        The idea is that each body may be observed at different times. To keep
        track of all these observations in one data structure, we use a masked
        array with shape (n_bodies, n_times), where `n_times` includes all of
        the observing times for all bodies. The only unmasked values in the
        array are at indices (i, j) where body `i` is observed at time `j`.

        Parameters:
        bodies (BodyGroup): A BodyGroup of body objects whose distances from
            Earth are to be calculated.
        times (MaskedArray): A 2D masked numpy array of times. The shape of
            the array is (number of bodies, number of times).

        Returns:
        MaskedArray: A 2D masked numpy array containing the distances. The
            shape of the array is the same as `times`, with each unmasked
            element indicating the distance of the corresponding body from
            Earth at that time.

        Raises: RuntimeError: If there are NaN values in the initial state or
        at any time step during the integration.

        Example:
        >>> simulation = SolarSystemSimulation()
        >>> times = np.ma.array([[0, 1], [1, 2]], mask=[[False, True],
            [True, False]])  # Example masked times
        >>> bodies = BodyGroup(mars, venus)  # Example celestial bodies
        >>> distances = simulation.masked_distances(bodies, times)

        """
        xyz, earth_xyz = self.positions(bodies, times)

        # Calculate the distance vectors and distances
        deltas = earth_xyz[np.newaxis, :, :] - xyz
        dists = np.linalg.norm(deltas, axis=-1)

        # Check for NaN values in distances
        if np.any(~np.isfinite(dists)):
            raise BadMassError("nan in distance")

        # Mask the distances array using the original mask from times
        dists = np.ma.masked_array(dists, mask=times.mask)
        return dists

    def sample(self, body_group: BodyGroup, end_time: float):
        """
        Generates observing times for a group of bodies and calculates their
        distances from Earth at those times.

        This method uses a `BodyGroup` object to determine a set of observing
        times up to a specified end time. It then calculates the distances of
        the bodies in the group from Earth at these times using the
        `masked_distances` method. This is useful for simulations where
        observations of a group of celestial bodies are taken at various
        times until a certain end time.

        Parameters:
        body_group (BodyGroup): The group of celestial bodies to be observed.
        end_time (float): The end time up to which observations are taken.

        Returns:
        tuple: A tuple containing two elements. The first is an array
            of observing times, and the second is a masked array of distances
            of each body in the group from Earth at those times.

        Example:
        >>> simulation = SolarSystemSimulation()
        >>> body_group = BodyGroup(...)  # Example group of celestial bodies
        >>> end_time = 10.0  # Example end time
        >>> times, distances = simulation.sample(body_group, end_time)

        """
        # Generate observing times for the given body group and end time
        times = body_group.observing_times(end_time)

        # Calculate distances of the bodies from Earth at the observing times
        return times, self.masked_distances(body_group, times)


class SolarSystemGenerator(object):
    """
    A class for generating a solar system simulation.

    This class sets up and configures a simulation environment for the solar
    system. It allows for customization of various simulation attributes,
    such as the choice of integrator, the inclusion of specific celestial
    bodies, and other simulation properties.

    Parameters:
    *args: Variable length argument list, currently not used.
    **kwargs: Keyword arguments for setting up the solar system generator:
        - earth (object, optional): An object representing Earth in the
            simulation. Defaults to BODY_EARTH.
        - bodies (list, optional): A list of celestial bodies to include in
            the simulation. Defaults to SOLAR_SYSTEM_SIMPLIFIED.
        - sim_attrs (dict, optional): Additional attributes to be set on the
            simulation. Defaults to an empty dictionary.
        - epsilon (float, optional): The precision parameter for the
            integrator. Defaults to 1e-8.
        - G (float, optional): The gravitational constant to use in the
            simulation. Defaults to GRAVITATIONAL_CONSTANT.

    Attributes:
    earth (object): Earth object for the simulation.
    bodies (list): List of celestial bodies in the simulation.
    sim_attrs (dict): Additional attributes for the simulation.
    integrator (str): The integrator algorithm for the simulation.
    epsilon (float): The precision parameter for the integrator.
    G (float): The gravitational constant for the simulation.

    Raises:
    ValueError: If the specified integrator is not one of the allowed values.

    Example:
    >>> generator = SolarSystemGenerator(
            earth=my_earth, bodies=my_bodies,
            sim_attrs={'additional_attr': value})

    """
    def __init__(self, *args, **kwargs):
        self.earth = kwargs.get('earth', BODY_EARTH)
        self.bodies = kwargs.get('bodies', SOLAR_SYSTEM_SIMPLIFIED)
        self.sim_attrs = kwargs.get('sim_attrs', {})
        self.sim_attrs['gravity'] = \
            self.sim_attrs.get('gravity', 'compensated')
        self.integrator = self.sim_attrs.get('integrator', 'ias15')
        if self.integrator not in ("mercurius", "whfast", "ias15"):
            raise ValueError("integrator not found")
        self.epsilon = kwargs.get('epsilon', 1e-8)  # Low for testing
        self.G = GRAVITATIONAL_CONSTANT

    def make_simulation(self, pbh=None):
        """
        Creates and returns a SolarSystemSimulation instance based on the
        current generator configuration.

        This method sets up a simulation with the specified Earth object,
        celestial bodies, and simulation attributes. It adds bodies to the
        simulation in an order based on their initial position vectors.

        Parameters:
        pbh (Body, optional): An optional additional body (e.g., a primordial
            black hole) to include in the simulation.

        Returns:
        SolarSystemSimulation: An instance of the SolarSystemSimulation class
            configured according to the generator's settings.

        Example:
        >>> sim = generator.make_simulation(pbh=my_primordial_black_hole)

        """
        # Create a new SolarSystemSimulation instance
        sim = SolarSystemSimulation(earth=self.earth)
        sim.G = self.G
        sim.ri_ias15.epsilon = self.epsilon
        sim.ri_whfast.safe_mode = False
        sim.ri_whfast.corrector = 11
        sim.integrator = self.integrator

        # Apply additional attributes to the simulation
        for key, val in self.sim_attrs.items():
            setattr(sim, key, val)

        sim_bodies = self.bodies
        # Add optional primordial black hole (pbh) if provided
        if pbh is not None:
            sim_bodies += BodyGroup(pbh)

        # Determine the order of adding bodies based on their position vectors
        add_order = np.argsort([
            np.linalg.norm(b.ephemeris.xyz[0])
            for b in sim_bodies
        ])

        # Add bodies to the simulation
        for index in add_order:
            body = sim_bodies[index]
            x, y, z = body.ephemeris.xyz[0]
            vx, vy, vz = body.ephemeris.vxyz[0]
            sim.add(
                m=body.mass, x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz, hash=body.hash)

        # Move the simulation to the center of mass frame
        sim.move_to_com()

        return sim


class DistanceGenerator(object):
    """
    A class for generating distance measurements in a solar system simulation,
    including the effect of a primordial black hole (PBH).

    This class sets up a solar system simulation and calculates the distances
    of specified bodies from Earth. It can also simulate the effect of a
    primordial black hole (PBH) on these distances. When initialized, it runs
    the simulation without the PBH (the "base case") as a reference point.

    Parameters:
    *args: Variable length argument list, currently not used.
    **kwargs: Keyword arguments for setting up the distance generator:
        - bodies (BodyGroup, optional): A list of celestial bodies to include
            in the simulation. Defaults to SOLAR_SYSTEM_SIMPLIFIED.
        - sim_kwargs (dict, optional): Additional keyword arguments for the
            solar system simulation generator. Defaults to an empty dictionary.
        - dist_bodies (BodyGroup, required): An object representing the group
            of bodies whose distances from Earth are to be tracked.
        - track_years (float, optional): The number of years for which the
            simulation will track distances. Defaults to 20 years.
        - observing_offset (bool, optional): Whether to include an offset in
            observing times. Defaults to True.
        - times (array, optional): An array of observing times. If not
            provided, it's generated based on `dist_bodies` and `track_years`.

    Attributes:
    bodies (BodyGroup): List of celestial bodies in the simulation.
    sim_generator (SolarSystemGenerator): The solar system sim generator.
    dist_bodies (BodyGroup): The group of bodies to track in the simulation.
    track_years (float): The number of years to track in the simulation.
    observing_offset (bool): Flag to include random offsets in the observing
        times. Defaults to True.
    times (array): Array of observing times.
    base_sim (SolarSystemSimulation): The base case solar system simulation,
        i.e., with no PBH.
    base_dists (array): Base distances of the bodies from Earth in the
        simulation, in the absence of a PBH.
    base_masses (array): Array of the base masses of the celestial bodies.

    Example:
    >>> distance_generator = DistanceGenerator(
            bodies=my_bodies, dist_bodies=my_dist_bodies)

    """
    def __init__(self, *args, **kwargs):
        # Initialize attributes for the distance generator
        self.bodies = kwargs.get('bodies', SOLAR_SYSTEM_SIMPLIFIED)
        self.sim_kwargs = kwargs.get('sim_kwargs', {})
        self.sim_generator = \
            SolarSystemGenerator(bodies=self.bodies, **self.sim_kwargs)
        self.dist_bodies = kwargs.get('dist_bodies')
        self.track_years = kwargs.get('track_years', 20)
        self.observing_offset = kwargs.get('observing_offset', True)

        # Set observing time offsets
        if self.observing_offset:
            offsets = None
        else:
            offsets = 0.

        # Generate the base case simulation (with no PBH)
        self.times = kwargs.get(
            'times',
            self.dist_bodies.observing_times(
                self.track_years * YEAR, offsets=offsets)
        )
        self.base_sim = self.sim_generator.make_simulation(pbh=None)
        self.base_dists = \
            self.base_sim.masked_distances(self.dist_bodies, self.times)
        self.base_masses = np.array([body.mass for body in self.bodies])

    def pbh_body(
        self,
        pbh_mass: float,
        r: float,
        theta: float,
        phi: float,
        s: float,
        alpha: float,
        beta: float,
    ):
        """
        Creates a PBH (primordial black hole) body with specified mass and
        position and velocity in spherical coordinates.

        Parameters:
        pbh_mass (float): The mass of the PBH.
        r, theta, phi (float): Spherical coordinates for the PBH position.
        s, alpha, beta (float): Spherical coordinates for the PBH velocity,
            specified with respect to the the negative of the position vector.
            See below for details.

        Returns:
        PbhBody: A PBH body object with the specified properties.

        We'll write xyz and vxyz in angular coordinates as follows. We'll use
        (r, theta, phi) for the position. For velocity, we'll use (s, alpha,
        beta), where alpha is the angle from -x, and beta is the angle around
        that axis. We will say beta=0 corresponds to a displacement in the
        (v, z) plane.

        Here's how to translate (s, alpha, beta) to vxyz. We start with -x and
        rescale. Then we have to rotate. First find the rotation R that goes
        from z to v. Then write vxyz in spherical coordinates about v, and
        apply R^-1.
        
            - To go from z to -x, (1) rotate in the xz-plane by theta+pi (R1),
            - then (2) rotate in the xy-plane by phi (R2). Now R = R2.R1.
            - Invert this transformation, obtaining T = (R2.R1)^-1.
            - Write u = s*(
                sin(alpha) cos(beta),
                sin(alpha) sin(beta),
                cos(alpha)),
              in the coordinate system about x.
            - Write v = T.u. This gives the expression below.

        """
        pbh_xyz = r * np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        pbh_vxyz = s * np.array([
            -np.cos(theta)*np.cos(beta-phi)*np.sin(alpha) +
            np.cos(alpha)*np.sin(theta),
            #
            np.sin(alpha)*np.sin(beta-phi),
            #
            -np.cos(alpha)*np.cos(theta) -
            np.cos(beta-phi)*np.sin(beta)*np.sin(theta)
        ])
        pbh = PbhBody(
            mass=pbh_mass, ephemeris=Ephemeris(
                t=[0], x=[pbh_xyz[0]], y=[pbh_xyz[1]], z=[pbh_xyz[2]],
                vx=[pbh_vxyz[0]], vy=[pbh_vxyz[1]], vz=[pbh_vxyz[2]],
            )
        )
        return pbh

    def positions(self, *args, dm=None, **kwargs):
        # Create a PBH with specified parameters
        pbh = self.pbh_body(*args, **kwargs)

        # Adjust the masses of the bodies if necessary
        if dm is not None:
            body_mass_dm = zip(self.sim_generator.bodies, self.base_masses, dm)
            for body, mass, dm in body_mass_dm:
                body.mass = mass + dm

        # Create the simulation with the PBH
        sim = self.sim_generator.make_simulation(pbh=pbh)
        # Run the simulation and sample positions
        pos = sim.positions(self.dist_bodies, self.times)
        return pos

    def distances(self, *args, dm=None, **kwargs):
        """
        Calculates distances of specified bodies from Earth in a simulation
        that includes a PBH, optionally modifying body masses.

        This method creates a PBH with specified parameters and runs a
        simulation to calculate the distances of specified bodies from Earth.
        It allows for an optional adjustment in the masses of the bodies in
        the simulation.

        Parameters:
        *args: Arguments to pass to the `pbh_body` method for creating the PBH.
        dm (array, optional): Array of mass adjustments for the bodies.
        **kwargs: Keyword arguments to pass to the `pbh_body` method.

        Returns:
        array: A 2D array of distances of the specified bodies from Earth in
            the simulation, as returned by the `masked_distances` method of
            `SolarSystemSimulation`.

        If `dm` is provided, the masses of the bodies in the simulation are
        adjusted accordingly before running the simulation.

        """
        # Create a PBH with specified parameters
        pbh = self.pbh_body(*args, **kwargs)

        # Adjust the masses of the bodies if necessary
        if dm is not None:
            body_mass_dm = zip(self.sim_generator.bodies, self.base_masses, dm)
            for body, mass, dm in body_mass_dm:
                body.mass = mass + dm

        # Create the simulation with the PBH
        sim = self.sim_generator.make_simulation(pbh=pbh)
        # Run the simulation and sample distances
        dists = sim.masked_distances(self.dist_bodies, self.times)
        return dists

    def delta_distances(self, *args, **kwargs):
        """
        Calculates the difference in distances from the base simulation to
        the simulation including a PBH.

        This method computes the distances of specified bodies from Earth in
        a simulation that includes a PBH and subtracts these distances from
        those in the base case simulation.

        Parameters:
        *args: Arguments to pass to the `distances` method.
        **kwargs: Keyword arguments to pass to the `distances` method.

        Returns:
        array: A 2D array of the difference in distances from the base
        simulation to the simulation including a PBH.

        This method is useful for analyzing the impact of a PBH on the
        positions of celestial bodies in the solar system.

        """
        return self.distances(*args, **kwargs) - self.base_dists

    def delta_linear(self, pbh_mass: float, *args, **kwargs):
        """
        Calculates the distance deltas in a regime where the signal scales
        linearly with the PBH mass.

        This method evaluates the distances to objects for different multiples
        of the PBH mass. It checks if the maximum signal scales linearly with
        the mass. If not within a specified tolerance, it progressively
        scales up the mass to find the linear regime and adjusts the result
        accordingly.

        Parameters:
        pbh_mass (float): The mass of the PBH.
        *args: Additional arguments passed to the `distances` method.
        **kwargs: Keyword arguments, including:
            - shift_factor (float, optional): the factor by which to shift in
                order to test linearity. Defaults to 1.1.
            - move_factor (float, optional): the factor by which to move if
                retrying at a different mass is necessary. Defaults to 1e2.
            - tol (float, optional): The tolerance for testing linearity.
                Defaults to 0.05.
            - tries_left (int, optional): The number of attempts to find the
                linear regime. Defaults to 5.

        Returns:
        numpy.ndarray: The distance deltas calculated in the linear regime of
            the PBH mass and scaled back down.

        Raises:
        RuntimeError: If the linear regime is not found within the specified
            number of tries.

        This method is essential for ensuring accurate and meaningful distance
        calculations, especially in scenarios where the PBH mass is very low
        and the simulations are noise-dominated.

        """
        # Tolerance and number of attempts for finding the linear regime
        shift_factor = kwargs.pop('shift_factor', 1.1)
        move_factor = kwargs.pop('move_factor', 1e2)
        tol = kwargs.pop('tol', 0.05)
        tries_left = kwargs.pop('tries_left', 50)

        # Multiples of PBH mass to test for linearity
        multiples = np.array([1.0, shift_factor, 1/shift_factor])
        deltas = []
        delta_peaks = np.zeros_like(multiples)

        # Calculate deltas for each multiple of PBH mass
        for i, mass in enumerate(multiples * pbh_mass):
            deltas.append(self.delta_distances(mass, *args, **kwargs))
            delta_peaks[i] = np.amax(deltas[i])

        # Test if the signal is linear with respect to the PBH mass
        linear_signal = delta_peaks[0] * multiples
        linear_deviation = np.abs(linear_signal - delta_peaks) / linear_signal

        # If the signal is linear within the tolerance, return the dists
        if np.amax(linear_deviation) < tol:
            return deltas[0]
        # If not linear and there are tries left, scale up mass and try again
        elif tries_left > 0:
            #print("Rescaling", tries_left)
            next_kwargs = dict(kwargs)
            next_kwargs['tries_left'] = tries_left - 1
            shifted_result = self.delta_linear(
                move_factor*pbh_mass, *args, **kwargs,
                tries_left=tries_left-1, tol=tol,
                move_factor=move_factor, shift_factor=shift_factor,
            )
            return shifted_result / move_factor
        else:
            # If the linear regime is not found within the specified number of
            # tries, raise an error
            raise RuntimeError("Linear regime not found")

    def delta_pivot(self, pivot: float, pbh_mass: float, *args, **kwargs):
        rescale = pivot / pbh_mass
        return self.delta_distances(pivot, *args, **kwargs) / rescale
