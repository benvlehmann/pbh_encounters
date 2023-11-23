"""
Module: ephemeris.py

This module provides classes and functionality for creating and managing
ephemerides of celestial bodies within solar system simulations. It includes
classes for defining bodies, both standard and primordial black holes
(PBHs), and groups of bodies. Additionally, it offers utilities for fetching
ephemeris data from NASA's JPL Horizons system and calculating distances
between celestial objects.

Classes:
- Ephemeris: Represents the ephemeris of a celestial body, storing position
  and velocity data. It provides functionality to calculate distances to
  other objects and manage time-ordered ephemeris data.

- Body: A data class representing a celestial body with attributes like ID,
  name, mass, and ephemeris. It includes methods to fetch ephemeris data from
  JPL Horizons, calculate distances to other bodies, and update its
  ephemeris.

- PbhBody: A subclass of Body, specifically representing a primordial black
  hole. It overrides certain functionalities of the Body class to cater to
  the unique nature of PBHs.

- BodyGroup: Manages groups of bodies, allowing for collective operations such
  as updating ephemerides, generating observing times, and handling
  collections of bodies.

The module is structured to support a variety of solar system simulation
scenarios, including the incorporation of primordial black holes and the
study of their potential effects on other celestial bodies. It integrates
closely with the functionality provided in the `jpl.py` module for retrieving
necessary ephemeris data.

Example Usage:
>>> from ephemeris import Body, PbhBody, BodyGroup
>>> mars = Body.fetch(499)
>>> pbh = PbhBody(mass=1e-5)
>>> bodies = BodyGroup(mars, pbh)
>>> print(bodies)

Dependencies:
- numpy: Used for numerical operations and handling arrays.

"""

from copy import copy

from dataclasses import dataclass
import numpy as np

from .jpl import get_new_horizons
from .common import METER


class Ephemeris(object):
    """
    A class representing the ephemeris of a celestial body in a solar system
    simulation.

    This class stores and manages the time-ordered ephemeris data of a
    celestial body, including its position and velocity. It provides
    functionality to calculate distances to other objects based on their
    ephemeris data and to handle the ephemeris data efficiently.

    Parameters:
    *args: Variable length argument list, not currently used in this class.
    **kwargs: Keyword arguments that should include the ephemeris data keys:
        - t (array): Array of time points.
        - x, y, z (arrays): Arrays of x, y, z coordinates of the body's
            position.
        - vx, vy, vz (arrays): Arrays of vx, vy, vz components of the body's
            velocity.
        - data (array, optional): Full data array from an Ephemeris instance,
            which can be supplied in place of the other keywords.

    Attributes: data (numpy.ndarray): A 2D array of shape (n, 7) where 'n' is
    the number of time points. This array contains concatenated time,
    position (x, y, z), and velocity (vx, vy, vz) data. xyz(numpy.ndarray): A
    subset of 'data' containing only the position coordinates (x, y, z).
    vxyz (numpy.ndarray): A subset of 'data' containing only the velocity
    components (vx, vy, vz).

    Methods: dist(other): Calculates the Euclidean distance to another
    Ephemeris object at each time point.

    Example Usage:
    >>> eph_data = {'t': [0, 1], 'x': [1, 2], 'y': [2, 3], 'z': [3, 4],
                    'vx': [0, 0], 'vy': [1, 1], 'vz': [2, 2]}
    >>> eph = Ephemeris(**eph_data)
    >>> print(eph)

    Note: The Ephemeris object assumes that the input data arrays are of the
    same length and correspond to the same time points.

    """
    def __init__(self, *args, **kwargs):
        data = kwargs.get('data')
        if data is not None:
            self.data = data
        else:
            keys = ('t', 'x', 'y', 'z', 'vx', 'vy', 'vz')
            for key in keys:
                setattr(self, key, np.asarray(kwargs[key]))
            t_order = np.argsort(self.t)
            for key in keys:
                setattr(self, key, getattr(self, key)[t_order])
            self.data = np.vstack([getattr(self, key) for key in keys]).T
        self.xyz = self.data[:, 1:4]
        self.vxyz = self.data[:, 4:]

    def __repr__(self):
        return f'Ephemeris(t={self.t.min()}...{self.t.max()} n={self.t.size})'

    def dist(self, other):
        """
        Calculates the Euclidean distance to another Ephemeris object at each
        time point.

        Parameters:
        other (Ephemeris): Another Ephemeris object to which to calculate the
            distance.

        Returns:
        numpy.ndarray: An array of distances corresponding to each time point
            in the ephemeris data.

        Raises:
        AssertionError: If the time points in the two Ephemeris objects do not
            match.

        Note: It's crucial that both Ephemeris objects have the same time
        points for accurate distance calculation.

        """
        assert np.all(self.t == other.t), "Times must match"
        distances = np.linalg.norm(self.xyz - other.xyz, axis=1)
        return distances


@dataclass
class Body(object):
    """
    A data class representing a celestial body in a solar system simulation.

    This class encapsulates the properties of a celestial body, such as its
    identifier, name, mass, and ephemeris. It includes methods to fetch
    ephemeris data from NASA's JPL Horizons system, calculate distances to
    other bodies, update its properties, and generate observing times.

    Attributes:
    id (int): The unique identifier of the celestial body in the JPL database.
    name (str): The name of the celestial body.
    mass (float): The mass of the celestial body.
    ephemeris (Ephemeris): The ephemeris of the celestial body.
    obs_cadence (float): The observing cadence for the body, used in
        generating observing times.
    obs_noise (float): The observational noise level for the body.
    fetchable (bool): whether this object can be fetched from JPL Horizons.
        Defaults to True.

    Class Methods:
    fetch(cls, id, *args, **kwargs): Class method to create a Body instance
        by fetching ephemeris data from JPL Horizons.

    Instance Methods:
    dist(other): Calculates the distance to another Body instance based on
        their ephemerides.
    update(*args, **kwargs): Updates the Body's properties by re-fetching data
        from JPL Horizons.
    observing_times(end_time, offset): Generates an array of observing times
        up to a specified end time.
    hash: A property that returns a hash representation of the Body's ID.

    Example Usage:
    >>> mars = Body.fetch(499)
    >>> earth = Body.fetch(399)
    >>> distance = mars.dist(earth)
    >>> print(distance)

    Note:
    The fetch method requires an active internet connection to access the JPL
    Horizons system.

    """
    id: int = None
    name: str = None
    mass: float = None
    ephemeris: Ephemeris = None
    obs_cadence: float = 20
    obs_noise: float = 1e-1*METER
    fetchable: bool = True

    @classmethod
    def fetch(cls, id, *args, **kwargs):
        """
        Fetches ephemeris data for a celestial body from the JPL Horizons
        system and creates a Body instance.

        Parameters:
        id (int): The identifier of the celestial body in the JPL database.
        *args, **kwargs: Additional arguments and keyword arguments to be
            passed to the `get_new_horizons` function.

        Returns:
        Body: A Body instance with the fetched ephemeris data.

        Note:
        This method requires an active internet connection to access the JPL
        Horizons system.

        """
        _, name, mass, eph_data = get_new_horizons(id, *args, **kwargs)
        eph = Ephemeris(**eph_data)
        return Body(id=id, name=name, mass=mass, ephemeris=eph)

    def dist(self, other):
        """
        Calculates the distance to another Body instance based on ephemeris.

        Parameters:
        other (Body): Another Body instance to calculate the distance to.

        Returns:
        numpy.ndarray: An array of distances corresponding to each time point
            in the ephemeris data.

        Note:
        Both Body instances should have ephemeris data for the same time
            points for accurate distance calculation.

        """
        return self.ephemeris.dist(other.ephemeris)

    def update(self, *args, **kwargs):
        """
        Updates the Body's properties by re-fetching data from JPL Horizons.

        *args, **kwargs: Additional arguments and keyword arguments to be
            passed to the `fetch` method.

        This method updates the Body's name, mass, and ephemeris based on the
        latest data from JPL Horizons.

        """
        if not self.fetchable:
            return

        body = self.__class__.fetch(self.id, *args, **kwargs)
        if body.name:
            self.name = body.name
        if body.mass:
            self.mass = body.mass
        if body.ephemeris:
            self.ephemeris = body.ephemeris

    def __copy__(self):
        return Body(id=self.id, name=self.name, mass=self.mass,
                    ephemeris=self.ephemeris, obs_cadence=self.obs_cadence,
                    obs_noise=self.obs_noise, fetchable=self.fetchable)

    def observing_times(self, end_time, offset=None):
        """
        Generates an array of observing times up to a specified end time.

        Parameters:
        end_time (float): The end time up to which observing times are to be
            generated.
        offset (float, optional): An optional offset to start the observing
            times. If not specified, it's randomized based on obs_cadence.

        Returns:
        numpy.ndarray: An array of observing times.

        This method is useful for scheduling observations of the celestial
        body over a given time period.

        """
        if offset is None:
            offset = self.obs_cadence * np.random.uniform()
        return np.arange(offset, end_time, self.obs_cadence)

    @property
    def hash(self):
        """
        Returns a hash representation of the Body's ID.

        This property can be used as a unique identifier for the Body
        instance in rebound simulations.

        """
        return str(self.id)


@dataclass
class PbhBody(Body):
    """
    A subclass of the Body class, representing a Primordial Black Hole (PBH)
    in the solar system simulation.

    This class is tailored to handle the unique properties and behaviors of a
    Primordial Black Hole, such as its distinct ID and name. It inherits
    attributes and methods from the Body class but overrides certain
    functionalities to cater to the nature of PBHs. This class does not
    require observational noise or cadence as PBHs are theoretical entities.

    Attributes:
    id (int): The unique identifier for the PBH, defaulted to -1.
    name (str): The name of the PBH, defaulted to "PBH".
    obs_cadence (float): The observing cadence for the PBH, set to None as
        it's not applicable.
    obs_noise (float): The observational noise level for the PBH, set to None
        as it's not applicable.
    fetchable (bool): whether this object can be fetched from JPL Horizons.
        Defaults to False.

    The hash and name attributes are overridden to reflect the PBH's unique
    properties. The update method is also overridden to prevent modification
    as PBH properties are not expected to change like standard celestial
    bodies.

    Example Usage:
    >>> pbh = PbhBody(mass=1e-5)

    Note:
    PBHs are theoretical entities and their properties are not derived from
    observational data, hence some attributes like obs_noise and obs_cadence
    are not applicable.

    """
    id: int = -1
    name: str = "PBH"
    obs_cadence: float = None
    obs_noise: float = None
    fetchable: bool = False

    hash = -1
    name = "PBH"

    def update(self, *args, **kwargs):
        """
        Overrides and disables the update method from the Body class.

        For a PbhBody, the update method is not applicable as the properties
        of a theoretical PBH are not expected to change over time or with new
        observational data.

        """
        pass


class BodyGroup(object):
    """
    A class for managing groups of celestial bodies in a solar system
    simulation.

    This class provides functionality to handle collections of Body objects
    (including PBHs). It allows for collective operations such as adding
    bodies to the group, updating their ephemerides, generating observing
    times, and accessing bodies by their attributes. It also supports
    operations like iteration, length querying, and concatenation with other
    BodyGroup instances.

    Attributes:
    all (list): A list containing all Body instances in the group.
    id (dict): A dictionary mapping body IDs to Body instances.
    n (dict): A dictionary mapping body names to Body instances.
    h (dict): A dictionary mapping body hashes to Body instances.

    Methods:
    add(body): Adds a Body instance to the group.
    update(*args, **kwargs): Updates the ephemerides of all bodies in the
        group.
    observing_times(end_time, offsets): Generates observing times for the
        bodies in the group.
    ids: A property returning a list of Body instances based on their IDs.
    names: A property returning a list of Body instances based on their names.

    Special Methods:
    __getitem__, __iter__, __len__, __add__, __repr__: Special methods to
    support indexing, iteration, length querying, concatenation, and string
    representation of the BodyGroup instance.

    Example Usage:
    >>> mars = Body.fetch(499)
    >>> earth = Body.fetch(399)
    >>> bodies = BodyGroup(mars, earth)
    >>> for body in bodies:
    >>>     print(body.name)

    This class is useful for scenarios where multiple celestial bodies need
    to be managed and operated on collectively in a simulation environment.

    """
    def __init__(self, *args, **kwargs):
        self.all = []
        self.id = {}
        self.n = {}
        self.h = {}
        for body in args:
            self.add(body)

    def add(self, body):
        """
        Adds a Body instance to the group.

        Parameters:
        body (Body): The Body instance to be added to the group.

        """
        self.all.append(body)
        self.id[body.id] = body
        self.n[body.name] = body
        self.h[body.hash] = body

    @property
    def ids(self):
        """
        A property returning a list of IDs of Body instances in the group.
        """
        return list(self.id.values())

    @property
    def names(self):
        """
        A property returning a list of names of Body instances in the group.
        """
        return list(self.n.values())

    def update(self, *args, **kwargs):
        """
        Updates the ephemerides of all bodies in the group.

        *args, **kwargs: Additional arguments and keyword arguments for
            updating each body's ephemeris.

        """
        for body in self.all:
            # Delete the name and hash indexes
            del self.n[body.name]
            del self.h[body.hash]
            # Update the data
            body.update(*args, **kwargs)
            # Repopulate the indexes
            self.n[body.name] = body
            self.h[body.hash] = body

    def observing_times(self, end_time, offsets=None):
        """
        Generates observing times for the bodies in the group up to a
        specified end time.

        Parameters:
        end_time (float): The end time up to which observing times are to be
            generated.
        offsets (list or float, optional): Specific offsets for each body or
            a uniform offset for all bodies. If not provided, observing times
            are generated based on each body's obs_cadence.

        Returns:
        numpy.ma.MaskedArray: A 2D masked array of observing times for the
            bodies in the group.

        This method is useful for scheduling observations of multiple
        celestial bodies over a given time period.

        """
        # Generate observing times as a masked array
        if offsets is None:
            body_times = [body.observing_times(end_time) for body in self.all]
        elif offsets == 0:
            body_times = [body.observing_times(end_time, 0)
                          for body in self.all]
        else:
            body_times = [body.observing_times(end_time, offset)
                          for body, offset in zip(self.all, offsets)]
        # Collect and mask all times that don't apply to a given object
        all_times = np.sort(np.unique(np.hstack(body_times)))
        times = np.ma.masked_array(
            np.zeros((len(self), len(all_times)))
        )
        for i, body in enumerate(self):
            for j, time in enumerate(all_times):
                times[i, j] = time
                if time not in body_times[i]:
                    times[i, j] = np.ma.masked
        return times

    def __getitem__(self, *args, **kwargs):
        return self.all.__getitem__(*args, **kwargs)

    def __iter__(self):
        return iter(self.all)

    def __len__(self):
        return len(self.all)

    def __add__(self, other):
        return BodyGroup(*(self.all + other.all))

    def combine(self, a, b):
        """Remove object `a` and add its mass to a copy of object `b`."""
        if a.hash == b.hash:
            raise ValueError("Can't combine an object into itself")
        removal_index = None
        mass_index = None
        for i, body in enumerate(self.all):
            if body.hash == a.hash:
                removal_index = i
            if body.hash == b.hash:
                mass_index = i
        if removal_index is None:
            raise ValueError("Object `a` not contained in this group")
        if mass_index is None:
            raise ValueError("Object `b` not contained in this group")
        self.all[mass_index] = copy(b)
        self.all[mass_index].mass += a.mass
        self.all.pop(removal_index)

    def __repr__(self):
        if not self.all:
            return 'BodyGroup(empty)'
        rep = 'BodyGroup(\n'
        body_reps = [f'  {i}: Body({b.name})' for i, b in enumerate(self.all)]
        rep += ',\n'.join(body_reps)
        rep += '\n)'
        return rep
