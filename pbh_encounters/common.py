"""
Module: common.py

This module establishes a system of units and defines constants for use in the
pbh_encounters package. It provides definitions for units of length, mass,
and time, along with the gravitational constant in these units. These
definitions are essential for maintaining consistency and accuracy in the
simulations and calculations throughout the package.

The chosen system of units is:
    Length unit: Astronomical Unit (AU)
    Mass unit: Earth Mass
    Time unit: Year

Constants:
- TIME_ORIGIN: A datetime object representing the origin time for the
    simulations.
- Gravitational constants in various unit systems.
- Unit conversions for mass, length, and time.

The constants and units defined in this module are used across the
pbh_encounters package to ensure that all calculations are consistent and
based on the same unit system.
"""


import datetime


class BadMassError(ValueError):
    """An error indicating that a simulation failed in a way that likely 
    depends on the specific PBH mass used."""


# Define the origin time for the simulations
TIME_ORIGIN = datetime.datetime(2023, 10, 14)

# Gravitational constant in different unit systems
G_METER_KILOGRAM_SECOND = 6.6743015e-11
G_KILOMETER_KILOGRAM_SECOND = G_METER_KILOGRAM_SECOND * 1e-9
G_AU_EARTHMASS_YEAR = 1.184e-4
G_AU_SOLARMASS_YEAR = 39.42
G_KILOMETER_SOLARMASS_SECOND = 1.327e11

# Masses
EARTHMASS = 1  # Earth mass as the unit
GRAM = 1.674e-28  # Gram in Earth mass units
KILOGRAM = GRAM * 1e3  # Kilogram in Earth mass units
SOLARMASS = 3.33e5  # Solar mass in Earth mass units

# Lengths
AU = 1  # Astronomical Unit as the unit
METER = 6.685e-12  # Meter in Astronomical Unit units
KILOMETER = METER * 1e3  # Kilometer in Astronomical Unit units

# Times
YEAR = 1  # Year as the unit
DAY = YEAR / 365.  # Day in Year units
SECOND = 3.171e-8  # Second in Year units

# Gravitational constant in the chosen system of units
GRAVITATIONAL_CONSTANT = G_AU_EARTHMASS_YEAR

# Density
DM_DENSITY = 5.35e-19 * GRAM/METER**3
