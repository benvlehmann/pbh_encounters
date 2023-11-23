# pbh_encounters

The pbh_encounters package is a Python library for simulating and analyzing the impact of Primordial Black Holes (PBHs) on the solar system. It leverages advanced computational methods like Monte Carlo simulations and Markov Chain Monte Carlo (MCMC) techniques to explore and infer celestial dynamics.

## Modules

The package consists of the following key modules:

* simulation.py: Contains classes for simulating the solar system with functionalities to include additional celestial objects like PBHs.
* jpl.py: Provides functions for querying NASA's JPL Horizons system for ephemeris data and processing the data.
* data.py: Provides cached solar system object data obtained from Horizons.
* ephemeris.py: Manages ephemerides of celestial bodies, including special handling for PBHs.
* mcmc.py: Implements Monte Carlo simulations for parameter estimation and uncertainty quantification within the solar system models.
* common.py: Defines a consistent system of units and essential constants for astronomical and physical calculations used throughout the package.

## Usage

Import the necessary classes from the package and create instances as required:

```
from pbh_encounters.simulation import SolarSystemSimulation
from pbh_encounters.ephemeris import Body, PbhBody
from pbh_encounters.mcmc import SolarSystemMonteCarlo

# Example usage
simulation = SolarSystemSimulation(...)
pbh = PbhBody(...)
mc_simulation = SolarSystemMonteCarlo(...)
```

For detailed usage of each module and class, refer to the inline documentation and docstrings provided within each file.

## License

pbh_encounters is licensed under the MIT License.
