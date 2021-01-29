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
    status = check_tsm_sources(pre_tsm).sort_values(by=cols)
    # Grab post-TSM raw data from cache
    post_tsm = hdf5.get_correlator(engine, basename)
    # Align data with status codes
    data = pd.DataFrame(post_tsm.attrs)
    data['data'] = [config for config in post_tsm]
    data = pd.merge(data, status, on=cols)
    return data


def check_tsm_sources(pre_tsm):
    """
    Checks the sources used for the truncated solver method (TSM), which are
    either even or odd.
    Full (fine+loose) solves:
        * even: all sources, both fine and loose, are even
        * odd: all sources, both fine and loose, are odd
        * mixed: a matching even fine-loose pair exists; all of the remaining
          loose solves are odd
        * missing match: no matching fine-loose pair exists
        * unknown: any case not in 'even', 'odd', 'mixed', or 'missing match'
    Fine solves / loose solves:
        * even: all fine/loose solves use even sources
        * odd: all fine/loose solves use odd sources
        * mixed: the fine/loose solves use a mixture of even and odd sources
    Args:
        pre_tsm: DataFrame <more description...>
    Returns:
        count: DataFrame <more description...>
    """
    counts = []
    groups = pre_tsm.groupby(['series', 'trajectory'])
    for (series, trajectory), group in groups:
        fine = (group['solve_type'] == 'fine')
        loose = ~fine
        # match = (group[loose]['tsrc'] == group[fine]['tsrc'].item())
        match = group[loose]['tsrc'].isin(group[fine]['tsrc'])
        n_match = bool(len(group[loose][match]))

        # Full (fine + loose) solves
        n_odd = np.sum(group['tsrc'].values % 2)
        n_even = len(group) - n_odd
        if n_odd == 0:
            status_full = 'even'
        elif n_odd == len(group):
            status_full = 'odd'
        elif n_match > 0:
            if n_even == 2:
                status_full = 'mixed'
            else:
                status_full = 'unknown'
        else:
            status_full = 'missing match'

        # Fine solves only
        n_odd = np.sum(group[fine]['tsrc'].values % 2)
        if n_odd == 0:
            status_fine = 'even'
        elif n_odd == len(group[fine]):
            status_fine = 'odd'
        else:
            status_fine = 'mixed'

        # Loose solves only
        n_odd = np.sum(group[loose]['tsrc'].values % 2)
        if n_odd == 0:
            status_loose = 'even'
        elif n_odd == len(group[loose]):
            status_loose = 'odd'
        else:
            status_loose = 'mixed'

        counts.append({
            'series': series,
            'trajectory': trajectory,
            'status:full': status_full,
            'status:fine': status_fine,
            'status:loose': status_loose,
        })

    return pd.DataFrame(counts)
 