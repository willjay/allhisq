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
import shutil
import numpy as np
import pandas as pd
import sqlalchemy as sqla
import h5py
from . import utils
from . import alias

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
    parser.add_argument("--tsm", type=str, default=None,
                        help="Possible restriction of tsm to even/odd/mixed solves")
    parser.add_argument("--do_tsm", type=_str2bool, default=True,
                        help="Whether or not the data supports using the TSM")

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

    run_reduction(input_db, output_fname, log_fname, args.test, args.tsm, args.do_tsm)


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


def run_reduction(input_db, output_fname, log_fname, test_only=False, tsm_source=None, do_tsm=True):
    """ Runs the reduction from the sqlite db to HDF5 """
    logging.basicConfig(filename=log_fname,
                        format='%(levelname)s:%(message)s',
                        level=logging.INFO)
    # Carry out the work
    LOGGER.info('FNAME:input database name: %s', input_db)
    LOGGER.info('FNAME:output hdf5 name: %s', output_fname)
    LOGGER.info('FNAME:log name: %s', log_fname)
    LOGGER.info('TIMESTAMP:start, %s', datetime.datetime.now().isoformat())
    interface = ReductionInterface(input_db, output_fname, tsm_source, do_tsm)
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


def tsm_single_config(dataframe, loose_only=False):
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
        loose_only: bool, whether to get the average value of the loose solves
             only. Default is False.
    Returns:
        array, the combined data
    """
    for col in ['fine', 'loose']:
        if col not in dataframe.columns:
            msg = (
                "TSM_SINGLE_CONFIG: DataFrame must have column {0} for tsm. \n"
                "Columns present: {1}".format(col, dataframe.columns)
            )
            raise KeyError(msg)

    if loose_only:
        loose_full = np.array(list(dataframe['loose'].dropna().apply(np.array)))
        average = np.mean(loose_full, axis=0)
        return average, None

    # Isolate the single measurement with a fine solve
    mask = ~dataframe['fine'].isna()

    # Compare the difference between the fine and loose solve
    # to yield a measurement of the bias introduced by the
    # loose solve
    nfine = len(dataframe[mask]['fine'])
    nloose = len(dataframe[mask]['loose'])
    if nfine != nloose:
        LOGGER.error("TSM: Invalid solves nfine %s, nloose %s", nfine, nloose)
        raise ValueError("Invalide solves encountered.")
    if nfine > 1:
        LOGGER.info("TSM: averaging bias from nfine %s solves.", nfine)
        fine_control = dataframe[mask]['fine'].apply(np.asarray)
        loose_control = dataframe[mask]['loose'].apply(np.asarray)
        bias_correction = (fine_control - loose_control).values
        bias_correction = np.mean(np.vstack(bias_correction), axis=0)
    else:
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
        try:
            data_tsm, bias_correction = tsm_single_config(to_combine)
            reduced.append({
                'series': series,
                'trajectory': trajectory,
                'data': data_tsm,
                'bias_correction': bias_correction,
                'n_tsrc': len(subdf['tsrc'].unique()),
            })
        except KeyError as err:
            if "TSM_SINGLE_CONFIG" in str(err):
                LOGGER.info(
                    "TSM: missing fine/loose pair series %s trajectory %s",
                    series, trajectory)
            else:
                raise err
    return pd.DataFrame(reduced)


def combine_without_tsm(data):
    """
    Computes the averaged correlator by averaging over data with different source times.
    This function does NOT apply the truncated solver method.
    Args:
        data: pandas.DataFrame with columns 'name', 'series', 'trajectory', 'data', 'tsrc'
    Returns:
        reduced: pandas.DataFrame with columns 'series', 'trajectory', 'data', 'n_tsrc'
    """
    name = data['name'].unique().item()
    if name.endswith('loose'):
        raise ValueError("Found {0} with suffix 'loose'. Shouldn't this use TSM?".format(name))
    group = data.groupby(['series','trajectory'])
    reduced = []
    for (series, trajectory), subdf in group:
        datum = np.vstack(subdf['data'].apply(np.asarray))  # full data with shape (tsrc, nt)
        reduced.append({
            'series': series,
            'trajectory': trajectory,
            'data': np.mean(datum, axis=0),
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
    def __init__(self, db_choice, h5fname, tsm_source=None, do_tsm=True):
        if tsm_source not in ['even', 'odd', 'mixed', None]:
            raise ValueError((
                "Invalid tsm_source. "
                "Please choose 'even', 'odd', 'mixed' or None"))
        LOGGER.info('Applying TSM? %s', str(do_tsm))
        self.do_tsm = do_tsm
        self.tsm_source = tsm_source
        self.h5fname = h5fname
        self.db_choice = db_choice
        with h5py.File(h5fname, mode='a') as ifile:
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
        for idx, (basename, subdf) in enumerate(self.pending.groupby('basename')):
            LOGGER.info('INFO: starting %s %s', idx, basename)
            LOGGER.info('TIMESTAMP: %s', datetime.datetime.now().isoformat())
            data = self.read_data(subdf['correlator_id'].values)
            data = unpackage(data)
            if self.do_tsm:
                # Restrict to certain tsm solves (even/odd)?
                # None takes all solves; this should be the usual preferred default
                if self.tsm_source is not None:
                    status = check_tsm_sources(data)
                    data = pd.merge(data, status, on=["series", "trajectory"])
                    LOGGER.info('INFO: Restricting to tsm: %s', self.tsm_source)
                    LOGGER.info('INFO: size of data before: %d', len(data))
                    data = data[data['status:full'] == self.tsm_source]
                    LOGGER.info('INFO: size of data after: %d', len(data))
                reduced = tsm(data)
            else:
                # Skip tsm. Just combine data from different source times
                reduced = combine_without_tsm(data)
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
            for idx, data in enumerate(self.__iter__()):
                with h5py.File(self.h5fname, mode='a') as ofile:
                    if data.empty:
                        print("Empty data frame at", idx)
                        continue
                    write_hdf5(ofile, data)
                    ofile.flush()
                    ofile.close()
                # To aid recovery from failures (and possible file corruptions)
                # make a backup version every so often.
                # 500 is arbitrary.
                if (idx % 500) == 0:
                    backup = f"{self.h5fname}.bak"
                    LOGGER.info('BACKUP: copying results to %s', backup)
                    shutil.copy(self.h5fname, backup)
        else:
            data = self.process_basename(basename)
            with h5py.File(self.h5fname, mode='a') as ofile:
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

def _get_correlator(h5fname, basename):
    "Gets a correlator (basename) from the cache (h5fname)."
    with h5py.File(h5fname, 'r') as ifile:
        if 'data' in ifile:
            dset = ifile['data'][basename]
        else:
            dset = ifile[basename]
        arr = CachedData(dset)
    return arr

def get_correlator(engine, basename):
    """
    Gets a correlator from the cache.
    Args:
        engine: database connetion engine
        basename: str, the name of the correlator (no suffix 'loose' or 'fine')
    Returns:
        arr: CachedData, i.e., a np.array-like object containing the correlator
            and with shape (nconfigs, nt)
    """
    h5fname = default_cache_name(engine)
    if basename.endswith('loose') or basename.endswith('fine'):
        raise ValueError(
            "'basename' cannot end with the suffixes 'loose' or 'fine'.")
    # if (not cache_exists(engine)) or (not basename_cached(engine, basename)):
    #     # Process a missing correlator into the cache
    #     LOGGER.error('missing correlator %s', basename)
    #     input_db = engine.url.database
    #     interface = ReductionInterface(input_db, h5fname)
    #     interface.process_data(basename=basename)
    # Grab the correlator from the cache
    return _get_correlator(h5fname, basename)

@utils.timing
def get_pre_tsm_raw_data(basename, engine):
    """
    Gets "pre-tsm" raw data from a database.
    Args:
        basename: str, correlator name without suffix 'fine' or 'loose'
        engine: database connection engine
    """
    # Verify input basename
    if basename.endswith('loose') or basename.endswith('fine'):
        raise ValueError((
            "basename must not end with suffix 'loose' or 'fine'. "
            f"basename: {basename}"))
    if (not alias.match_2pt(basename)) and (not alias.match_3pt(basename)):
        raise ValueError((
            "basename is unrecognized as a valid 2pt or 3pt function. "
            f"basename: {basename}"))

    query = """
        SELECT correlators.name, data.*  FROM correlators
        JOIN data ON correlators.id = data.correlator_id
        WHERE correlator_id IN
        (SELECT id FROM correlators WHERE name LIKE '{basename}%%');
        """.format(basename=basename)
    data_raw = pd.read_sql_query(query, engine)
    data_raw = unpackage(data_raw)
    return data_raw


def check_tsm_sources(pre_tsm):
    """
    Checks the sources used for the truncated solver method (TSM), which are
    either even or odd. The meaning of the status codes are the following:
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
        pre_tsm: DataFrame with data before having applied the truncated solver
            method. Often applied to the output of the function unpackage(...)
    Returns:
        count: DataFrame with columns 'series', 'trajectory', 'status:fine',
            'status:loose', 'status:full'.
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


if __name__ == '__main__':
    main()
