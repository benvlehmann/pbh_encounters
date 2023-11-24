"""
Module: jpl.py

This module contains functions for interacting with NASA's JPL Horizons system
to retrieve and process astronomical data. It provides utilities for querying
the Horizons system for ephemeris data of celestial objects, parsing the
obtained data, and tokenizing text strings for data extraction.

Functions:
- query_jpl(obj_id, **kwargs): Queries the JPL Horizons system for ephemeris
  data of a specified celestial object. It supports customizing the query
  with parameters like start and stop times, step size, and other options.

- tokenize(line, delimiter): Splits a given line into tokens based on a
  specified delimiter and the equal sign. It is designed to parse lines of
  text into a list of tokens for further processing.

- parse_new_horizons(response_text): Parses the text response from the JPL
  Horizons system, extracting key information about celestial objects such as
  their ID, name, mass, and ephemeris data.

- get_new_horizons(*args, **kwargs): A convenience function that combines the
  functionality of querying and parsing data from the JPL Horizons system. It
  takes the same arguments as `query_jpl` and returns structured data about
  the queried object.

These functions are essential for astronomers and researchers who need to
retrieve and process data from the JPL Horizons system for studies in
celestial mechanics, solar system dynamics, and other astrophysical
research.

Example Usage:
>>> from jpl import query_jpl, parse_new_horizons, get_new_horizons
>>> ephemeris_data = query_jpl(
        599, start_time=datetime.datetime(2023, 1, 1),
        stop_time=datetime.datetime(2023, 1, 2))
>>> parsed_data = parse_new_horizons(ephemeris_data)
>>> id, name, mass, data = get_new_horizons(
        599, start_time=datetime.datetime(2023, 1, 1),
        stop_time=datetime.datetime(2023, 1, 2))

Dependencies:
- datetime: Used for handling date and time objects.
- requests: For sending HTTP requests to the JPL Horizons system.
- numpy: Used in processing and handling numerical data arrays.

"""


import io
import re
import datetime

import numpy as np
import requests

from .common import (
    TIME_ORIGIN, GRAVITATIONAL_CONSTANT, KILOMETER, SECOND, DAY
)


def query_jpl(obj_id: int, **kwargs):
    """
    ### Docstring written by ChatGPT Queries the JPL New Horizons API for
        information about a celestial object.

    This function retrieves data about a specified object from NASA's Jet
    Propulsion Laboratory (JPL) database. It uses the Horizons system to get
    ephemeris data, which includes object information and its position and
    velocity vectors over a specified time range.

    Parameters: obj_id (int): The unique identifier of the celestial object in
        the JPL database.
    **kwargs: Keyword arguments for specifying the query parameters:
        - start_time (datetime, optional): The start time for the ephemeris
            data retrieval. Defaults to TIME_ORIGIN if not specified.
        - stop_time (datetime, optional): The end time for the ephemeris data
            retrieval. Defaults to one day after start_time if not specified.
        - step_size (str, optional): The interval between each data point in
            the ephemeris data. Defaults to '1 DAY' if not specified.

    Returns:
    str: The ephemeris data of the specified object as a text string.

    Raises:
        AssertionError: If the specified start_time is not earlier than the
            stop_time.

    Example:
    >>> ephemeris_data = query_jpl(
            599, start_time=datetime.datetime(2023, 1, 1),
            stop_time=datetime.datetime(2023, 1, 2), step_size='1 DAY')

    """
    start_time = kwargs.get('start_time', TIME_ORIGIN)
    stop_time = kwargs.get(
        'stop_time', start_time + datetime.timedelta(days=1))
    step_size = kwargs.get('step_size', '1 DAY')
    assert start_time < stop_time, "start time must be before end time"
    start_time = f"{start_time.year}-{start_time.month}-{start_time.day}"
    stop_time = f"{stop_time.year}-{stop_time.month}-{stop_time.day}"
    request_data = io.StringIO(
        f"!$$SOF\n"
        f"COMMAND='{obj_id}'\n"
        "OBJ_DATA='YES'\n"
        "MAKE_EPHEM='YES'\n"
        "TABLE_TYPE='VECTOR'\n"
        "CENTER='@0'\n"
        f"START_TIME='{start_time}'\n"
        f"STOP_TIME='{stop_time}'\n"
        f"STEP_SIZE='{step_size}'\n"
    )
    # use request to get file
    url = "https://ssd.jpl.nasa.gov/api/horizons_file.api"
    r = requests.post(
        url, data={"format": "text"}, files={"input": request_data})

    return r.text


def tokenize(line, delimiter):
    """
    Splits a given line into tokens based on a specified delimiter and the
    equal sign.

    This function is designed to parse a line of text and tokenize it into a
    list. It first splits the line based on the provided delimiter. Each
    resulting segment is further split at the equal sign. The function then
    flattens this list of lists and removes any empty or whitespace-only
    elements. The final output is a list of tokens, with each token being a
    trimmed string.

    Parameters:
        line (str): The line of text to be tokenized.
        delimiter (str): The delimiter used to initially split the line.

    Returns:
        list: A list of string tokens extracted and cleaned from the input.

    Example:
    >>> tokenize("name = John == age = 30", "==")
        ['name', 'John', 'age', '30']

    """
    # first split based on equal sign and '  '
    tokens = [token.split("=") for token in line.split(delimiter) if token]
    # flatten and remove empty or ' ' values
    tokens = [item.strip() for sublist in tokens
              for item in sublist if item.strip()]
    return tokens


def parse_new_horizons(response_text: str):
    """
    Parses the response text from the NASA JPL Horizons system into a
    structured data format.

    This function processes the text obtained from a query to the NASA JPL
    Horizons system, extracting key information about a celestial body. The
    information includes the body's ID, name, mass, and ephemeris data such
    as position and velocity over time.

    Parameters: response_text (str): The response text obtained from the NASA
    JPL Horizons system.

    Returns: tuple: A tuple containing the ID (int or None), name (str or
    None), mass (float or None), and a dictionary of ephemeris data.

    The ephemeris data dictionary contains time ('t'), positions
    ('x', 'y', 'z'), and velocities ('vx', 'vy', 'vz') as numpy arrays.

    Example:
    >>> response_text = "..." # response text from NASA JPL Horizons
    >>> id, name, mass, data = parse_new_horizons(response_text)

    """
    # Initialize variables
    mass = None
    name = None
    id = None
    data = dict()

    # Initialize data keys
    for key in ('T', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ'):
        data[key] = []

    # Split the response text into sections
    sections = re.split(r"\*{77,}", response_text)

    for idx, section in enumerate(sections):
        # Processing the header section for mass, name, and ID
        if idx == 1:
            units = ["GM, km^3/s^2", "GM (km^3/s^2)"]
            lines = section.split("\n")
            _, name, id = [s for s in lines[1].split('     ') if s]
            name = name.strip()
            id = id.strip()
            if '/' in id:
                id = id.split('/')[0].strip()
            try:
                id = int(id)
            except ValueError:
                id = None
            lines = lines[4:]
            for line in lines:
                if any(unit in line for unit in units):
                    tokens = tokenize(line, delimiter="  ")
                    for idx, token in enumerate(tokens):
                        if any(unit in token for unit in units):
                            try:
                                mass = float(tokens[idx + 1])
                            except TypeError:
                                mass = None

        # Processing the ephemeris data section
        if "$$SOE" in section:
            lines = section.split("\n")[2:-2]
            blocks = [lines[i: i+4] for i in range(0, len(lines), 4)]
            for block in blocks:
                time = float(block[0].split()[0])
                # The time is specified in days
                data['T'].append(time)
                # Process position and velocity data
                for line in block[1:3]:
                    tokens = tokenize(line, delimiter=" ")
                    for key, value in zip(tokens[::2], tokens[1::2]):
                        data[key].append(float(value))

    # Convert keys to lowercase and values to numpy arrays
    for key in list(data.keys()):
        data[key.lower()] = np.array(data[key])
        del data[key]

    # Convert units to code units and remove offset
    try:
        data['t'] *= DAY
        for i in range(len(data['t'])):
            data['t'][i] -= np.min(data['t'])
    except KeyError:
        pass
    for key in ('x', 'y', 'z'):
        try:
            data[key] *= KILOMETER
        except KeyError:
            pass
    for key in ('vx', 'vy', 'vz'):
        try:
            data[key] *= KILOMETER/SECOND
        except KeyError:
            pass
    if mass is not None:
        # The mass is specified as GM [km^3/s^2]
        mass *= KILOMETER**3 / SECOND**2 / GRAVITATIONAL_CONSTANT

    return id, name, mass, data


def get_new_horizons(*args, **kwargs):
    """
    Retrieves and parses data from NASA's JPL Horizons system for a specified
    celestial object.

    This convenience function combines querying the JPL Horizons system for a
    celestial object and parsing the returned data. It first calls
    `query_jpl` with the provided arguments to retrieve the ephemeris data
    from JPL. Then, it parses this data using `parse_new_horizons` to extract
    meaningful information about the object, such as its ID, name, mass, and
    ephemeris data.

    The arguments and keyword arguments passed to this function are directly
    forwarded to `query_jpl`.

    Returns:
    tuple: A tuple containing the ID (int or None), name (str or None),
        mass (float or None), and a dictionary of ephemeris data.

    The ephemeris data dictionary contains time ('t'), positions
    ('x', 'y', 'z'), and velocities ('vx', 'vy', 'vz') as numpy arrays.

    Example:
    >>> id, name, mass, data = get_new_horizons(
            599, start_time=datetime.datetime(2023, 1, 1),
            stop_time=datetime.datetime(2023, 1, 2), step_size='1 DAY')

    """
    # Call query_jpl with the given arguments and parse the result
    return parse_new_horizons(query_jpl(*args, **kwargs))
