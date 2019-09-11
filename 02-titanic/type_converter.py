# coding: utf-8

import numpy as np
import pandas as pd
import logging

NA_VALUES = ['""', "''", 'NaN', 'nan', None, 'None', np.nan]
ALLOWED_TYPES = ['number', 'bool', 'id', 'date', 'object', 'constant']

LOGGER = logging.getLogger(__name__)


class TypeConverter():
    """Type Converter class with fit=transform.

    Parameters
    ----------
    detected_type : string
        Output of the detect module, this will change the conversion to match the detected type.

    """

    def __init__(self, detected_type=None):
        self.detected_type = detected_type

        if self.detected_type not in ALLOWED_TYPES:
            raise ValueError("Unknown type -> {0} received from detect module, "
                             "allowed types are {1}".format(self.detected_type,
                                                            ALLOWED_TYPES))

    def fit(self, X, y=None):
        """Type converters don't need a fit method. This is necessary for scikit pipeline."""
        return self

    def transform(self, X, y=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like, list, Series}, shape (n_samples)
            Input data, where ``n_samples`` is the number of samples.

        Returns
        -------
        self : Custom_fillna
            Returns self.

        """
        serie = pd.Series(X)

        if self.detected_type == 'number':
            conversion_func = float_converter
        elif self.detected_type in ['object', 'bool']:
            conversion_func = str_converter
        elif self.detected_type == 'date':
            conversion_func = date_converter
        else:
            conversion_func = (lambda x: x)
        transformed = np.array(serie.map(lambda x: conversion_func(x))).reshape(-1)

        return transformed


def float_converter(element, logger=LOGGER):
    """Convert an input to a float as hard as you can, if it fails returns a NaN.

    Parameters
    ----------
    element: object
        Value present in a dataset

    Returns
    -------
    float_value: float
        The conversion to float, if failed returns a NaN

    """
    warning_msg = "Unknown input for float conversion during transformations"
    # try converting to float
    try:
        if element not in NA_VALUES:
            float_value = float(element)
        else:
            return np.nan
    except (ValueError, TypeError):
        try:
            float_value = float(element.replace(',', '.').replace(' ', ''))
        except (AttributeError, ValueError, TypeError):
            logger.warning(warning_msg)
            return np.nan

    return float_value


def str_converter(element, logger=LOGGER):
    """Convert an input to a string as hard as you can, if it fails returns a NaN.

    Parameters
    ----------
    element: object
        Value present in a dataset

    Returns
    -------
    str_value: float
        The conversion to str, if failed returns a NaN

    """
    warning_msg = "Unknown input for str conversion during transformations"
    # try converting to float
    try:
        if element not in NA_VALUES:
            str_value = str(element)
        else:
            return np.nan
    except (ValueError, TypeError, AssertionError):
        logger.warning(warning_msg)
        return np.nan

    return str_value


def date_converter(element, logger=LOGGER):
    """Convert an input to a string as hard as you can, if it fails returns a NaN.

    Parameters
    ----------
    element: object
        Value present in a dataset

    Returns
    -------
    date_value: pd.datetime
        The conversion to date, if failed returns a NaT or NaN

    """
    warning_msg = "Unknown input for date conversion during transformations"
    # try converting to date
    try:
        if element not in NA_VALUES:
            date_value = pd.to_datetime(element, errors='coerce')
        else:
            return np.nan

    except (ValueError, TypeError):
        logger.warning(warning_msg)
        return np.nan

    return date_value
