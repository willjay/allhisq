"""
A simple utility for caching lattice correlator data in hdf5 files from an
sqlite database (of a particular fixed format). The cached data combines data
from different "t_source" times and applies the "truncated solver method" to
combine "loose and fine solves."
"""
import datetime
import bz2
import logging
import os
import argparse
import numpy as np
import pandas as pd
import sqlalchemy as sqla
import h5py
from . import utils

LOGGER = logging.getLogger(__name__)


def main():
    """ Runs caching from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="input database including full path")
    parser.add_argument("--fout", type=str,
                        help="output filename including full path")
    parser.add_argument("--log", type=str,
                        help="log filename including full path")
    parser.add_argument("--test", type=_str2bool, nargs='?',
                        const=True, default=False,
                        help="Whether to run in test mode (no data is read).")

    args = parser.parse_args()
    input_db = args.db
    if args.fout:
        if (not args.fout.endswith(".h5")) and (
                not args.fout.endswith(".hdf5")):
            raise ValueError(
                "Output file must have '.h5' or '.hdf5' file stem")
        output_fname = args.fout
        _, tail = os.path.split(output_fname)
        stem = tail.rstrip('sqlite')
    else:
        _, tail = os.path.split(input_db)
        stem = tail.rstrip('sqlite')
        output_fname = "{stem}hdf5".format(stem=stem)

    if args.log:
        log_fname = args.log
    else:
        log_fname = "cache_{stem}log".format(stem=stem)

    test_only = args.test
    run_reduction(input_db, output_fname, log_fname, test_only)


def _str2bool(val):
    """ Parses string to bool """
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run_reduction(input_db, output_fname, log_fname, test_only=False):
    """ Runs the reduction from the sqlite db to HDF5 """
    logging.basicConfig(filename=log_fname,
                        format='%(levelname)s:%(message)s',
                        level=logging.INFO)
    # Carry out the work
    LOGGER.info('FNAME:input database name: %s', input_db)
    LOGGER.info('FNAME:output hdf5 name: %s', output_fname)
    LOGGER.info('FNAME:log name: %s', log_fname)
    LOGGER.info('TIMESTAMP:start, %s', datetime.datetime.now().isoformat())
    interface = ReductionInterface(input_db, output_fname)
    if test_only:
        LOGGER.info('TEST: Skipping reading actual data from db.')
    else:
        interface.process_data()
    LOGGER.info('TIMESTAMP:finish, %s', datetime.datetime.now().isoformat())


def make_engine(db_choice):
    """ Makes a db connection 'engine'. """
    if not db_choice.endswith('sqlite'):
        raise ValueError("Only connection to sqlite databases supported.")
    settings = {'database': db_choice}
    # unix/mac seems to need 3, 4, or 5 leading slashes in total
    # no idea why this is the case
    url = 'sqlite:///{database}'
    return sqla.create_engine(url.format(**settings), echo=False)


def tsm_single_config(dataframe):
    """
    Combine "loose" and "fine" solves for a single configuration
    according to the so-called truncated solver method (TSM):
    1) Compute the bias from the difference of a fine and a loose solve
       for some fixed "tsrc"
    2) Average loose results with different "tsrc" values on each timeslice
    3) Correct this average using the bias from step 1
    Usage:
    Assumes that that the fine and loose solves have already been aligned
    using something like
    >>> input_df = df.pivot(index='tsrc', columns='solve_type', values='data')
    >>> tsm_single_config(input_df)
    Args:
        df: DataFrame with columns 'fine' and 'loose'.
    Returns:
        array, the combined data
    """
    for col in ['fine', 'loose']:
        if col not in dataframe.columns:
            msg = (
                "DataFrame must have column {0} for tsm. \n"
                "Columns present: {1}".format(col, dataframe.columns)
            )
            raise KeyError(msg)

    # Isolate the single measurement with a fine solve
    mask = ~dataframe['fine'].isna()

    # Compare the difference between the fine and loose solve
    # to yield a measurement of the bias introduced by the
    # loose solve
    fine_control = np.array(dataframe[mask]['fine'].item())
    loose_control = np.array(dataframe[mask]['loose'].item())
    bias_correction = fine_control - loose_control

    # Average results of the loose solve from
    # different "tsrc" on each timeslice
    loose_full = np.array(list(dataframe['loose'].apply(np.array)))
    average = np.mean(loose_full, axis=0)

    # Correct for the bias
    return average + bias_correction, bias_correction


@utils.timing
def tsm(data):
    """ Computes the average correlator via truncated solver method """
    for col in ['tsrc', 'solve_type', 'data']:
        if col not in data.columns:
            msg = "DataFrame must have column {0} for tsm.".format(col)
            raise KeyError(msg)

    # Group according to trajectory
    cols = ['series', 'trajectory']
    group = data.groupby(cols)

    # Conduct the "TSM" average
    # There must be some elegant way to do this with an aggregate,
    # group.agg(...)
    reduced = []
    for (series, trajectory), subdf in group:
        # Combine 'fine' and 'loose' solves using tsm
        to_combine = subdf.pivot(
            index='tsrc', columns='solve_type', values='data')
        data_tsm, bias_correction = tsm_single_config(to_combine)
        reduced.append({
            'series': series,
            'trajectory': trajectory,
            'data': data_tsm,
            'bias_correction': bias_correction,
            'n_tsrc': len(subdf['tsrc'].unique()),
        })
    return pd.DataFrame(reduced)


def infer_solve_type(name):
    """ Gets the suffix 'loose' or 'fine'. """
    for suffix in ['loose', 'fine']:
        if name.endswith(suffix):
            return suffix
    return None


def to_floats(string_list, delim='\n'):
    """ Recovers list of floats from a string representation. """
    if isinstance(string_list, bytes):
        string_list = string_list.decode()  # <type 'bytes'> to <type 'str'>
    values = string_list.split(delim)
    return [float(val) for val in values]


@utils.timing
def unpackage(dataframe):
    """ Decompresses column 'data' and infer 'solve_type' from basename """
    dataframe['data'] = dataframe['dataBZ2'].apply(bz2.decompress)
    dataframe['data'] = dataframe['data'].apply(to_floats)
    dataframe['solve_type'] = dataframe['name'].apply(infer_solve_type)
    return dataframe


@utils.timing
def write_hdf5(h5file, data, postgres_attrs=None, sqlite_attrs=None):
    """
    Writes the averaged correlator data to hdf5 file.
    Remarks:
        data must contain the following columns:
        'basename','data','n_tsrc','series', and 'trajectory'.
    Args:
        h5file: h5py.File
        data: DataFrame with averaged correlator data
        postgres_attrs: dict, optional postgres-level attributes
    Returns:
        None
    """
    for col in ['basename', 'data', 'n_tsrc', 'series', 'trajectory']:
        if col not in data.columns:
            msg = "Error: data must have column {0}".format(col)
            raise ValueError(msg)

    basename = data['basename'].unique().item()

    # load into hdf5 dataset
    dpath = 'data/{0}'.format(basename)
    values = np.array(list(data['data'].values))
    dset = h5file.create_dataset(name=dpath, data=values,
                                 compression='gzip', shuffle=True)

    # Store meta data as dataset "attributes"
    # basic attributes
    dset.attrs['basename'] = basename
    dset.attrs['calcdate'] = datetime.datetime.now().isoformat()
    dset.attrs['truncated_solver_method'] = True

    # post-tsm summary information
    dset.attrs['n_tsrc'] = [int(n) for n in data['n_tsrc'].values]
    dset.attrs['series'] = [str(s) for s in data['series'].values]
    dset.attrs['trajectory'] = [int(n) for n in data['trajectory'].values]

    # optional attributes
    if postgres_attrs is None:
        postgres_attrs = {}
    if sqlite_attrs is None:
        sqlite_attrs = {}
    for attrs in [postgres_attrs, sqlite_attrs]:
        for key, val in attrs.items():
            dset.attrs[key] = val

    return None

class ReductionInterface(object):
    """Interface for reading/processing data: sqlite to hdf5."""
    def __init__(self, db_choice, h5fname):
        self.h5fname = h5fname
        self.db_choice = db_choice
        with h5py.File(h5fname) as ifile:
            try:
                self.existing = list(ifile['data'].keys())
            except KeyError:
                self.existing = []
        LOGGER.info('Total existing entries in HDF5 file: %s',
                    len(self.existing))
        self.engine = make_engine(db_choice)
        self.pending = self.get_pending()

    def get_pending(self):
        """ Gets a DataFrame of pending basenames and correlator_id(s) """
        query = """
            SELECT id as correlator_id, RTRIM(name,'_-finelos') AS basename
            FROM correlators;"""
        pending = pd.read_sql_query(query, self.engine)
        LOGGER.info('Total basenames in db: %s',
                    len(pending['basename'].unique()))
        # Restrict to correlators which are NOT yet in the HDF5 file.
        # This is what really makes them "pending" and ensures that
        # the script is safe to run twice without clobbering data.
        mask = ~pending['basename'].isin(self.existing)
        pending = pending[mask]
        LOGGER.info('Total pending basenames: %s',
                    len(pending['basename'].unique()))
        return pending

    @utils.timing
    def read_data(self, correlator_ids):
        """ Gets correlator data from db corresponding to correlator_id(s) """
        def _handle_ids(correlator_ids):
            list_of_ids = [str(elt) for elt in correlator_ids]
            return '({0})'.format(','.join(list_of_ids))
        query = """
            SELECT data.*, correlators.name FROM data
            JOIN correlators ON correlators.id = data.correlator_id
            WHERE correlator_id IN {0};""".\
            format(_handle_ids(correlator_ids))
        return pd.read_sql_query(query, self.engine)

    def __iter__(self):
        """ Reads and reduces pending data """
        for basename, subdf in self.pending.groupby('basename'):
            LOGGER.info('INFO: starting %s', basename)
            LOGGER.info('TIMESTAMP: %s', datetime.datetime.now().isoformat())
            data = self.read_data(subdf['correlator_id'].values)
            data = unpackage(data)
            reduced = tsm(data)
            reduced['basename'] = basename
            yield reduced

    def process_basename(self, basename):
        """Reads and reduces a single basename """
        mask = (self.pending['basename'] == basename)
        subdf = self.pending[mask]
        if subdf.empty:
            msg = "Error processing basename '{0}'".format(basename)
            raise ValueError(msg)
        LOGGER.info('INFO: starting %s', basename)
        LOGGER.info('TIMESTAMP: %s', datetime.datetime.now().isoformat())
        data = self.read_data(subdf['correlator_id'].values)
        data = unpackage(data)
        reduced = tsm(data)
        reduced['basename'] = basename
        return reduced

    @utils.timing
    def process_data(self, basename=None):
        """Reads the data, processes it, and writes to the HDF5 file."""
        # Run through all the data
        if basename is None:
            for data in self.__iter__():
                with h5py.File(self.h5fname) as ofile:
                    write_hdf5(ofile, data)
                    ofile.flush()
                    ofile.close()
        else:
            data = self.process_basename(basename)
            with h5py.File(self.h5fname) as ofile:
                write_hdf5(ofile, data)
                ofile.flush()
                ofile.close()


def default_cache_name(engine):
    """ Constructs the default cache name from the engine """
    db_name = engine.url.database
    stem = db_name.rstrip(engine.name)
    return "{stem}hdf5".format(stem=stem)


def cache_exists(engine):
    """ Checks if the cache file exists """
    return os.path.isfile(default_cache_name(engine))


def basename_cached(engine, basename):
    """ Checks if the basename is cached """
    h5fname = default_cache_name(engine)
    try:
        with h5py.File(h5fname, 'r') as ifile:
            keys = list(ifile['data'].keys())
        is_cached = (basename in keys)
    except IOError:
        is_cached = False
    except KeyError:
        is_cached = False
    return is_cached


class CachedData(np.ndarray):
    """
    Minimal wrapper to add attributes to numpy array.
    Basically copied from the docs:
    https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
    """
    def __new__(cls, dset):
        obj = np.asarray(dset).view(cls)
        obj.attrs = dict(dset.attrs)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.attrs = getattr(obj, 'attrs', None)


def get_correlator(engine, basename):
    """ Gets a correlator from the cache."""
    h5fname = default_cache_name(engine)
    if basename.endswith('loose') or basename.endswith('fine'):
        raise ValueError(
            "'basename' cannot end with the suffixes 'loose' or 'fine'.")
    if (not cache_exists(engine)) or (not basename_cached(engine, basename)):
        # Process a missing correlator into the cache
        LOGGER.error('missing correlator %s', basename)
        input_db = engine.url.database
        interface = ReductionInterface(input_db, h5fname)
        interface.process_data(basename=basename)
    # Grab the correaltor
    with h5py.File(h5fname, 'r') as ifile:
        dset = ifile['data'][basename]
        arr = CachedData(dset)
    return arr


if __name__ == '__main__':
    main()
