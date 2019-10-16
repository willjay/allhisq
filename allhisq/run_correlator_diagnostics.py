"""
Runs diagnostic on all correlators, checking the
mean and median noise-to-signal ratio across timeslices.
"""
import logging
from logging.config import fileConfig
import collections
import pandas as pd
import numpy as np
from sqlalchemy.exc import OperationalError
# local imports
import db_connection as db
import db_io
import hdf5_cache
import utils



fileConfig('logging_config.ini')
LOGGER = logging.getLogger()


def main():
    """Runs the diagnostics on all the correlators."""

    engines = db.get_engines()
    pending = get_pending(engines['postgres'])

    groups = pending.groupby('ens_id')
    print(groups.size())

    for ens_id, dataframe in groups:
        LOGGER.info('Starting ens_id: %d', ens_id)
        try:
            records = compute_diagnostics(engines[ens_id], dataframe)
        except OperationalError as error:
            LOGGER.error('Error reading ens_id: %d', ens_id)
            LOGGER.error(str(error))
            continue
        write_diagnostics(engines['postgres'], records)
        LOGGER.info('Finished ens_id: %d', ens_id)

# -- end main -- #


Stats = collections.namedtuple("Stats", ['average', 'median'])


@utils.timing
def get_pending(engine):
    """
    Gets the pending correlators which lack diagnostic information.
    Args:
        engine: db connection engine
    Returns:
        DataFrame with columns 'basename', 'corr_id', and 'ens_id'
    """
    query = (
        "SELECT "
        "DISTINCT(RTRIM(name, '-_losefin')) AS basename, corr_id, ens_id "
        "FROM correlator_n_point "
        "WHERE NOT (EXISTS "
        "("
        "SELECT FROM corr_diagnostic "
        "WHERE corr_diagnostic.corr_id = correlator_n_point.corr_id"
        "));"
    )
    return pd.read_sql_query(query, engine)


@utils.timing
def compute_diagnostics(engine, dataframe):
    """
    Computes the diagnostics for all the correlators specified in
    'dataframe', which should have columns 'basename' and 'corr_id'.
    """
    records = []
    for _, row in dataframe.iterrows():
        basename = row['basename']
        LOGGER.info(basename)
        data = hdf5_cache.get_correlator(engine, basename)
        diagnostic = check_noise_to_signal(data)
        records.append({
            'corr_id': row['corr_id'],
            'average_noise_to_signal': diagnostic.average,
            'median_noise_to_signal': diagnostic.median,
        })
    return pd.DataFrame(records)


@utils.timing
def write_diagnostics(engine, records):
    """Writes the diagnostics to the table 'corr_diagnostic' in the DB."""
    db_io.upsert(engine, 'corr_diagnostic', records)


def check_noise_to_signal(data):
    """Checks the average and median of the noise-to-signal ratio."""

    nconfigs, _ = data.shape
    err = np.std(data, axis=0) / np.sqrt(nconfigs)
    central_value = np.mean(data, axis=0)
    noise_to_signal = np.abs(err / central_value)
    mask = ~np.isnan(noise_to_signal)
    noise_to_signal = noise_to_signal[mask]
    average = np.mean(noise_to_signal)
    median = np.median(noise_to_signal)
    return Stats(float(average), float(median))


if __name__ == '__main__':
    main()
