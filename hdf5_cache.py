"""
module doc here
"""
import time
import datetime
import bz2
import logging
import os
import argparse
from functools import wraps
import numpy as np
import pandas as pd
import sqlalchemy as sqla
import h5py

logger = logging.getLogger(__name__)


def timing(fcn):
    """Time the execution of fcn. Use as decorator."""
    @wraps(fcn)
    def wrap(*args, **kwargs):
        """Wrapped version of the function."""
        t_initial = time.time()
        result = fcn(*args, **kwargs)
        t_final = time.time()
        logger.info("TIMING: {0} took: {1:.4f} sec".format(
            fcn.__name__, t_final - t_initial))
        return result
    return wrap

def str2bool(v):
    """ parse string to bool """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """run from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="input database including full path")
    parser.add_argument("--fout", type=str,
                        help="output filename including full path")
    parser.add_argument("--log", type=str,
                        help="log filename including full path")
    parser.add_argument("--test", type=str2bool, nargs='?',
                        const=True, default=False, 
                        help="Whether to run in test mode (no data is read).")
    
    args = parser.parse_args()
    input_db = args.db
    if args.fout:
        if (not args.fout.endswith(".h5")) and (not args.fout.endswith(".hdf5")):
            raise ValueError("Output file must have '.h5' or '.hdf5' file stem")
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


def run_reduction(input_db, output_fname, log_fname, test_only=False):
    """Run the reduction from the sqlite db to HDF5."""
    logging.basicConfig(filename=log_fname,
                        format='%(levelname)s:%(message)s',
                        level=logging.INFO)
    # Carry out the work
    logger.info('FNAME:input database name: %s', input_db)
    logger.info('FNAME:output hdf5 name: %s', output_fname)
    logger.info('FNAME:log name: %s', log_fname)
    logger.info('TIMESTAMP:start, %s', datetime.datetime.now().isoformat())
    interface = ReductionInterface(input_db, output_fname)
    if test_only:
        logger.info('TEST: Skipping reading actual data from db.')
    else:
        interface.process_data()
    logger.info('TIMESTAMP:finish, %s', datetime.datetime.now().isoformat())


def make_engine(db_choice):
    """Make a db connection 'engine'."""
    if not db_choice.endswith('sqlite'):
        raise ValueError("Only connection to sqlite databases supported.")
    settings = {'database': db_choice}
    # unix/mac seems to need 3, 4, or 5 leading slashes in total
    # no idea why this is the case
    url = 'sqlite:///{database}'
    return sqla.create_engine(url.format(**settings), echo=False)


def tsm_single_config(df):
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
        if col not in df.columns:
            msg = "DataFrame must have column {0} for tsm.".format(col)
            raise KeyError(msg)

    # Isolate the single measurement with a fine solve
    mask = ~df['fine'].isna()

    # Compare the difference between the fine and loose solve
    # to yield a measurement of the bias introduced by the
    # loose solve
    fine_control = np.array(df[mask]['fine'].item())
    loose_control = np.array(df[mask]['loose'].item())
    bias_correction = fine_control - loose_control

    # Average results of the loose solve from
    # different "tsrc" on each timeslice
    loose_full = np.array(list(df['loose'].apply(np.array)))
    average = np.mean(loose_full, axis=0)

    # Correct for the bias
    return average + bias_correction, bias_correction


@timing
def tsm(data):
    """Compute average correlator via truncated solver method."""
    for col in ['tsrc', 'solve_type','data']:
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


class ReductionInterface(object):
    """Interface for reading/processing data: sqlite to hdf5."""

    def __init__(self, db_choice, h5fname):
        self.h5fname = h5fname
        self.db_choice = db_choice
        with h5py.File(h5fname) as ifile:
            try:
                self.existing = ifile['data'].keys()
            except KeyError:
                self.existing = []
        logger.info('Total existing entries in HDF5 file: %s',
                    len(self.existing))
        self.engine = make_engine(db_choice)
        self.pending = self.get_pending()

    def get_pending(self):
        """Get DataFrame of pending basenames and correlator_id(s)."""
        query = """
            SELECT id as correlator_id, RTRIM(name,'_-finelos') AS basename
            FROM correlators;"""
        pending = pd.read_sql_query(query, self.engine)
        logger.info('Total basenames in db: %s',
                    len(pending['basename'].unique()))
        # Restrict to correlators which are NOT yet in the HDF5 file.
        # This is what really makes them "pending" and ensures that
        # the script is safe to run twice without clobbering data.
        mask = ~pending['basename'].isin(self.existing)
        pending = pending[mask]
        logger.info('Total pending basenames: %s',
                    len(pending['basename'].unique()))
        return pending

    @timing
    def read_data(self, correlator_ids):
        """Get correlator data from db corresponding to correlator_id(s)."""
        def _handle_ids(correlator_ids):
            s = [str(elt) for elt in correlator_ids]
            return '({0})'.format(','.join(s))
        query = """
            SELECT data.*, correlators.name FROM data
            JOIN correlators ON correlators.id = data.correlator_id
            WHERE correlator_id IN {0};""".\
            format(_handle_ids(correlator_ids))
        return pd.read_sql_query(query, self.engine)

    @staticmethod
    def infer_solve_type(name):
        """ Get the suffix 'loose' or 'fine'."""
        for suffix in ['loose', 'fine']:
            if name.endswith(suffix):
                return suffix
        return None

    @staticmethod
    def to_floats(string_list, delim='\n'):
        """Recover list of floats from a string representation."""
        return [float(val) for val in string_list.split(delim)]

    @staticmethod
    @timing
    def unpackage(df):
        """Decompress column 'data' and infer 'solve_type' from basename."""
        df['data'] = df['dataBZ2'].apply(bz2.decompress)
        df['data'] = df['data'].apply(str).apply(ReductionInterface.to_floats)
        df['solve_type'] = df['name'].\
            apply(ReductionInterface.infer_solve_type)
        return df

    def __iter__(self):
        """Read and reduce pending data."""
        for basename, subdf in self.pending.groupby('basename'):
            logger.info('INFO: starting %s', basename)
            logger.info('TIMESTAMP: %s', datetime.datetime.now().isoformat())
            data = self.read_data(subdf['correlator_id'].values)
            data = ReductionInterface.unpackage(data)
            reduced = tsm(data)
            reduced['basename'] = basename
            yield reduced

    def process_basename(self, basename):
        """Read and reduce a single basename."""
        mask = (self.pending['basename'] == basename)
        subdf = self.pending[mask]
        if subdf.empty:
            msg = "Error processing basename '{0}'".format(basename)
            raise ValueError(msg)
        logger.info('INFO: starting %s', basename)
        logger.info('TIMESTAMP: %s', datetime.datetime.now().isoformat())
        data = self.read_data(subdf['correlator_id'].values)
        data = ReductionInterface.unpackage(data)
        reduced = tsm(data)
        reduced['basename'] = basename
        return reduced

    @timing
    def process_data(self, basename=None):
        """Read the data, process it, and write to the HDF5 file."""
        # Run through all the data
        if basename is None:
            for data in self.__iter__():
                with h5py.File(self.h5fname) as ofile:
                    ReductionInterface.write_hdf5(ofile, data)
                    ofile.flush()
                    ofile.close()
        else:
            data = self.process_basename(basename)
            with h5py.File(self.h5fname) as ofile:
                ReductionInterface.write_hdf5(ofile, data)
                ofile.flush()
                ofile.close()

    @staticmethod
    @timing
    def write_hdf5(h5file, data, postgres_attrs=None, sqlite_attrs=None):
        """
        Write averaged correlator data to hdf5 file.
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
            for key, val in attrs.iteritems():
                dset.attrs[key] = val

        return None


def default_cache_name(engine):
    """Construct the default cache name from the engine."""
    db_name = engine.url.database
    stem = db_name.rstrip(engine.name)
    return "{stem}hdf5".format(stem=stem)


def cache_exists(engine):
    """Check if the cache file exists."""
    return os.path.isfile(default_cache_name(engine))


def basename_cached(engine, basename):
    """Check if the basename is cached."""
    h5fname = default_cache_name(engine)
    try:
        with h5py.File(h5fname, 'r') as ifile:
            keys = ifile['data'].keys()
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
        if obj is None: return
        self.attrs = getattr(obj, 'attrs', None)


def get_correlator(engine, basename):
    """ Get a correlator from the cache."""
    h5fname = default_cache_name(engine)
    if basename.endswith('loose') or basename.endswith('fine'):
        raise ValueError("'basename' cannot end with the suffixes 'loose' or 'fine'.")
    if (not cache_exists(engine)) or (not basename_cached(engine, basename)):
        # Process a missing correlator into the cache
        logger.error('missing correlator %s', basename)
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
