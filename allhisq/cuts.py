"""
doc here
"""
import numpy as np
import pandas as pd
from . import hdf5_cache as hdf5


def get_correlator(engine, basename, sources='even'):
    """
    Reads raw data, applying a cut to include only TSM solves using the
    specified sources.
    Args:
        engine: database connection engine
        basename: str, the correlator name without suffix 'loose' or 'fine'
        sources: str, 'even', 'odd', or 'mixed'
    Returns:
        data_raw: array, the raw data
    """
    if sources not in ['even', 'odd', 'mixed']:
        raise ValueError((
            "sources must be 'even', 'odd', or 'mixed. "
            f"sources={sources}"))
    df_raw = get_data_with_tsm_status(engine, basename)
    mask = (df_raw['status:full'] == sources)
    data_raw = np.array(list(df_raw[mask]['data'].values))
    return data_raw


def get_data_with_tsm_status(engine, basename):
    """
    Gets raw post-TSM data from the cache, together with information about
    whether a given configuration used even, odd, mixed sources for the TSM.
    Note: getting the source information requires reading raw, pre-TSM data
        from the data base, which makes this method unavoidably slow.
    Args:
        basename: str, the correlator name without suffix 'fine' or 'loose'
        engine: database connection engine
    Returns:
        data: DataFrame
    """
    cols = ['series', 'trajectory']
    # Grab pre-TSM raw data from database
    pre_tsm = hdf5.get_pre_tsm_raw_data(basename, engine)
    # Infer status of TSM sources (even, odd, mixed)
    status = hdf5.check_tsm_sources(pre_tsm).sort_values(by=cols)
    # Grab post-TSM raw data from cache
    post_tsm = hdf5.get_correlator(engine, basename)
    # Align data with status codes
    data = pd.DataFrame(post_tsm.attrs)
    data['data'] = [config for config in post_tsm]
    data = pd.merge(data, status, on=cols)
    return data
