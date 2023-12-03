"""
Module: data.py

This module provides definitions of solar system objects as obtained from the
JPL Horizons database. This functions as a cache of Horizons data to avoid
repeated requests.

Dependencies:
- numpy: Used for numerical operations and handling arrays.

"""

import numpy as np

from .ephemeris import Body, BodyGroup, Ephemeris
from .common import DAY, METER


BODY_SUN = Body(
    id=10, name='Sun', mass=333021.04748507333,
    ephemeris=Ephemeris(
        data=np.array([
            [0.000000e+00, -8.327029e-03, -2.320555e-03, 2.133706e-04,
             1.514922e-03, -2.833245e-03, -1.077822e-05],
            [6.740363e+03, -8.322873e-03, -2.328313e-03, 2.133410e-04,
             1.518946e-03, -2.830252e-03, -1.087128e-05],
        ])
    )
)
BODY_MERCURY = Body(
    id=199, name='Mercury', mass=0.055285517621828095,
    ephemeris=Ephemeris(
        data=np.array([
            [+0.000000e+00, -3.994252e-01, -2.994370e-02, 3.382917e-02,
             -1.411363e+00, -9.805853e+00, -6.715283e-01],
            [+6.740363e+03, -4.023527e-01, -5.672181e-02, 3.191034e-02,
             -7.295813e-01, -9.734790e+00, -7.282568e-01],
        ])
    )
)
BODY_VENUS = Body(
    id=299, name='Venus', mass=0.815181670672153,
    obs_cadence=20*DAY,
    ephemeris=Ephemeris(
        data=np.array([
            [+0.000000e+00, 3.723337e-01, 6.109895e-01, -1.332819e-02,
             -6.294667e+00, 3.857946e+00, 4.163042e-01],
            [+6.740363e+03, 3.549405e-01, 6.213167e-01, -1.218245e-02,
             -6.401475e+00, 3.680344e+00, 4.200282e-01],
        ])
    )
)
BODY_EARTH = Body(
    id=399, name='Earth', mass=1.0002252576695465,
    ephemeris=Ephemeris(
        data=np.array([
            [+0.000000e+00, 9.289155e-01, +3.399458e-01, 1.875687e-04,
             -2.255612e+00, 5.874910e+00, -9.072860e-05],
            [+6.740363e+03, 9.225956e-01, +3.559896e-01, 1.873347e-04,
             -2.357771e+00, 5.836631e+00, -8.193405e-05],
        ])
    )
)
BODY_MOON = Body(
    id=301, name='Moon / (Earth)', mass=0.012302807582116902,
    obs_cadence=40*DAY, obs_noise=1e-3*METER,
    ephemeris=Ephemeris(
        data=np.array([
            [+0.000000e+00, 9.263020e-01, +3.394087e-01, 2.428204e-04,
             -2.207528e+00, 5.672763e+00, -1.892887e-02],
            [+6.740363e+03, 9.201742e-01, +3.549154e-01, 1.900676e-04,
             -2.266072e+00, 5.648417e+00, -1.943012e-02],
        ])
    )
)
BODY_MARS = Body(
    id=499, name='Mars', mass=0.10747108840243433,
    obs_cadence=10*DAY, obs_noise=1e-1*METER,
    ephemeris=Ephemeris(
        data=np.array([
            [0.000000e+00, -1.258192e+00, -9.699281e-01, 1.059330e-02,
             3.319687e+00, -3.605438e+00, -1.569051e-01],
            [6.740363e+03, -1.249050e+00, -9.797697e-01,
             1.016303e-02, +3.353789e+00, -3.578806e+00, -1.571835e-01],
        ])
    )
)
BODY_JUPITER = Body(
    id=599, name='Jupiter', mass=317.89997638696593,
    ephemeris=Ephemeris(
        data=np.array([
            [+0.000000e+00, 3.890415e+00, 3.081144e+00, -9.982226e-02,
             -1.740940e+00, 2.289419e+00, 2.943860e-02],
            [+6.740363e+03, 3.885641e+00, 3.087412e+00, -9.974148e-02,
             -1.744470e+00, 2.286691e+00, 2.952451e-02],
        ])
    )
)
BODY_SATURN = Body(
    id=699, name='Saturn', mass=95.18241114719262,
    ephemeris=Ephemeris(
        data=np.array([
            [0.000000e+00, 8.835815e+00, -4.121848e+00, -2.801266e-01,
             7.473577e-01, 1.841362e+00, -6.171430e-02],
            [6.740363e+03, 8.837861e+00, -4.116802e+00, -2.802957e-01,
             7.463619e-01, 1.841925e+00, -6.172812e-02],
        ])
    )
)
BODY_URANUS = Body(
    id=799, name='Uranus', mass=14.539011683763928,
    ephemeris=Ephemeris(
        data=np.array([
            [+0.000000e+00, +1.250700e+01, 1.511659e+01, -1.058872e-01,
             -1.116688e+00, +8.482974e-01, 1.761874e-02],
            [+6.740363e+03, +1.250394e+01, 1.511891e+01,
             -1.058389e-01, -1.116866e+00, 8.480824e-01, 1.763416e-02],
        ])
    )
)
BODY_NEPTUNE = Body(
    id=899, name='Neptune', mass=17.151611039291158,
    ephemeris=Ephemeris(
        data=np.array([
            [0.000000e+00, 2.982258e+01, -2.042888e+00, -6.452207e-01,
             7.074588e-02, 1.149964e+00, -2.532351e-02],
            [6.740363e+03, 2.982277e+01, -2.039738e+00, -6.452900e-01,
             7.064056e-02, 1.149975e+00, -2.533087e-02],
        ])
    )
)
BODY_PLUTO = Body(
    id=999, name='Pluto', mass=0.0024478642071461207,
    ephemeris=Ephemeris(
        data=np.array([
            [0.000000e+00, 1.698053e+01, -3.041068e+01, -1.657644e+00,
             1.033384e+00, 3.094407e-01, -3.341201e-01],
            [6.740363e+03, 1.698336e+01, -3.040983e+01, -1.658562e+00,
             1.030642e+00, 3.057515e-01, -3.354485e-01],
        ])
    )
)
BODY_CERES = Body(
    id=2000001, name='Ceres', mass=0.00015715739891638514,
    ephemeris=Ephemeris(
        data=np.array([
            [0.000000e+00, -1.753688e+00, -2.041360e+00, 2.572188e-01,
             2.673840e+00, -2.745596e+00, -5.790683e-01],
            [6.740363e+03, -1.746349e+00, -2.048867e+00, 2.556303e-01,
             2.683435e+00, -2.734342e+00, -5.804798e-01],
        ])
    )
)
BODY_VESTA = Body(
    id=2000004, name='Vesta', mass=4.338213853395478e-05,
    ephemeris=Ephemeris(
        data=np.array([
            [+0.000000e+00, +7.127524e-01, 2.462550e+00, -1.607669e-01,
             -3.552178e+00, +1.075457e+00, 4.006045e-01],
            [+6.740363e+03, +7.030141e-01, 2.465475e+00,
             -1.596679e-01, -3.556709e+00, 1.059822e+00, +4.016219e-01],
        ])
    )
)


BODY_PHOBOS = Body(
    id=401, name='Phobos', mass=1.8067270416660248e-09, fetchable=False)
BODY_DEIMOS = Body(
    id=402, name='Deimos', mass=3.0112117361100410e-10, fetchable=False)
BODY_IO = Body(
    id=501, name='Io', mass=1.4955472916520120e-02, fetchable=False)
BODY_EUROPA = Body(
    id=502, name='Europa', mass=8.0367035524180300e-03, fetchable=False)
BODY_GANYMEDE = Body(
    id=503, name='Ganymede', mass=2.4811965143378172e-02, fetchable=False)
BODY_CALLISTO = Body(
    id=504, name='Callisto', mass=1.8015299913306724e-02, fetchable=False)
BODY_TITAN = Body(
    id=606, name='Titan',    mass=2.2528882472329956e-02, fetchable=False)

SOLAR_SYSTEM = BodyGroup(
    BODY_SUN, BODY_MERCURY, BODY_VENUS, BODY_EARTH, BODY_MARS, BODY_JUPITER,
    BODY_SATURN, BODY_URANUS, BODY_NEPTUNE, BODY_PLUTO,
    BODY_CERES, BODY_VESTA,
    BODY_MOON, BODY_PHOBOS, BODY_DEIMOS, BODY_IO, BODY_EUROPA, BODY_GANYMEDE,
    BODY_CALLISTO, BODY_TITAN
)

SOLAR_SYSTEM_SIMPLIFIED = BodyGroup(
    BODY_SUN, BODY_MERCURY, BODY_VENUS, BODY_EARTH, BODY_MARS, BODY_JUPITER,
    BODY_SATURN, BODY_URANUS, BODY_NEPTUNE, BODY_PLUTO,
    BODY_CERES, BODY_VESTA,
    BODY_MOON, BODY_PHOBOS, BODY_DEIMOS, BODY_IO, BODY_EUROPA, BODY_GANYMEDE,
    BODY_CALLISTO, BODY_TITAN
)
SOLAR_SYSTEM_SIMPLIFIED.combine(BODY_MOON, BODY_EARTH)
SOLAR_SYSTEM_SIMPLIFIED.combine(BODY_PHOBOS, BODY_MARS)
SOLAR_SYSTEM_SIMPLIFIED.combine(BODY_DEIMOS, BODY_MARS)
SOLAR_SYSTEM_SIMPLIFIED.combine(BODY_IO, BODY_JUPITER)
SOLAR_SYSTEM_SIMPLIFIED.combine(BODY_EUROPA, BODY_JUPITER)
SOLAR_SYSTEM_SIMPLIFIED.combine(BODY_GANYMEDE, BODY_JUPITER)
SOLAR_SYSTEM_SIMPLIFIED.combine(BODY_CALLISTO, BODY_JUPITER)
SOLAR_SYSTEM_SIMPLIFIED.combine(BODY_TITAN, BODY_SATURN)
