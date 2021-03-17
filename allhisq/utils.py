"""
Helpful utility functions
"""
import time
import functools
import logging

LOGGER = logging.getLogger(__name__)


def timing(fcn):
    """Time the execution of fcn. Use as decorator."""
    @functools.wraps(fcn)
    def wrap(*args, **kwargs):
        """Wrapped version of the function."""
        t_initial = time.time()
        result = fcn(*args, **kwargs)
        t_final = time.time()
        LOGGER.info(
            "TIMING: %s took: %.4f sec",
            fcn.__name__,
            t_final - t_initial
        )
        return result
    return wrap


def bundle_mask(dataframe, adict):
    """
    Bundles together several boolean masks using the "bitwise AND" operator to match several column-
    value pairs.
    Args:
        dataframe: pandas.DataFrame
        adict: dict specifying the masks
    Returns:
        mask: boolean mask for the dataframe
    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame([
            ['a', 1, 'foo'],
            ['a', 2, 'bar'],
            ['b', 1, 'spam'],
            ['b', 2, 'eggs']],
            columns=['letter','number','word'])
    >>> mask = bundle_mask(df, {'letter':'a', 'number':1})
    >>> print(df[mask])
    letter  number word
    0      a       1  foo
    """
    return functools.reduce(lambda a, b: a&b,
                  [dataframe[key] == value for key, value in adict.items()])


def extract_unique(dataframe, column):
    """
    Extracts the unique value of a DataFrame column.
    Args:
        dataframe: pandas.DataFrame
        column: str, name of a column
    Returns:
        the unique value of the column
    """
    if dataframe.empty:
        return None
    if len(dataframe[column].unique()) > 1:
        raise ValueError(f"Non-unique {column} encountered.")
    return dataframe[column].unique().item()
