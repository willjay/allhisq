"""
Tools for parsing MILC output files.
"""
import os
import re
import logging
import argparse
import bz2
import hashlib
import sqlite3
import functools
import time
import numpy as np
import pandas as pd
from sqlalchemy import Column, Sequence, ForeignKey, UniqueConstraint
from sqlalchemy import Integer, Float, String, BLOB
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy.orm
Base = declarative_base()

# Correct for obscure Python3 issue with saving integers in databases
sqlite3.register_adapter(np.int64, int)

# Set up logging
# logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S')
LOGGER = logging.getLogger("parse_milc")
# LOGGER.setLevel(level=logging.INFO)

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

@timing
def main():
    """
    Command-line utility for parsing MILC spectroscopy output into a SQLite
    database.
    """
    args = parse_args()
    LOGGER.info("Parsing datafiles in %s", args.base)

    # Match file names like 'fpi3296f211b630m0074m0222m440a.lat.654'
    valid_fname = re.compile(
        r"^fpi(\d+)f(\d+)b(\d+)(m\d+)(m\d+)(m\d+)?([a-z]).lat.(\d+)$")

    for root, _, files in os.walk(args.base):
        files = [fname for fname in files if re.match(valid_fname, fname)]
        nfiles = len(files)
        for idx, fname in enumerate(files):
            LOGGER.info(f" Starting file {idx+1}/{nfiles} ".center(80, '#'))
            # Parse file names
            match = re.match(valid_fname, fname)
            if not match:
                continue
            tokens = match.groups()
            trajectory = int(tokens[-1])
            series = tokens[-2]
            LOGGER.info("series=%s, trajectory=%d", series, trajectory)

            # Read data
            fullpath = os.path.join(root, fname)
            LOGGER.info("Reading %s", fullpath)
            data = parse_file(fullpath)
            data['series'] = series
            data['trajectory'] = trajectory
            LOGGER.info("Read %d correlators", len(data))

            # Write result
            dbname = fname.replace(f"{series}.lat.{trajectory}", ".sqlite")
            database = os.path.join(os.getcwd(), dbname)
            engine = sqlalchemy.create_engine('sqlite:///' + database)
            LOGGER.info("Writing %s", database)
            # os.remove(database)
            if not os.path.isfile(database):
                # Create tables within database
                LOGGER.info("Creating tables in database")
                Base.metadata.create_all(engine)

            write_records(engine, data)

# ----- end main ----- #

def parse_args():
    """
    Parses command-line arguments.
    Returns:
        args: namedtuple, attributes corresponding to command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, help="base directory for search")
    parser.add_argument("--logfile", type=str, default=None,
                        help="base directory for search")
    args = parser.parse_args()
    if args.base is None:
        raise ValueError("Please specify a base path")
    if not os.path.exists(args.base):
        LOGGER.error("Unrecognized base path, %s", args.base)
        raise ValueError("Invalid base path")

    if args.logfile:
        handler = logging.FileHandler(args.logfile)
    else:
        handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    return args


@timing
def parse_file(fname):
    """
    Parses a MILC output file.
    Args:
        fname: str, the full path to the output file
    Returns:
        corrs: pandas.DataFrame containing the correlator data
    """
    data = []
    tsrc = None
    with open(fname) as ifile:
        for line in ifile:
            # Grab source time from a comment line
            if "# source time " in line:
                tsrc = int(line.split()[-1].strip())
            # Read correlator data
            if "STARTPROP" in line:
                if tsrc is None:
                    raise ValueError("Unspecified source time")
                # The header block with meta data contains precisely five lines
                head = [next(ifile) for _ in range(5)]
                datum = parse_header(head)
                t, real, imag = parse_datablock(ifile)
                datum['t'] = t
                datum['real'] = real
                datum['imag'] = imag
                datum['tsrc'] = tsrc
                data.append(datum)
    # Format results
    data = pd.DataFrame(data)
    data['data_real_bz2'] = data['real'].apply(compress)
    data['data_imag_bz2'] = data['imag'].apply(compress)
    cols = ['momentum', 'masses', 'sinks', 'source', 'sinkops']
    data['correlator_hash'] = data[cols].\
        apply(lambda args: Correlator(*args).correlator_hash, axis=1)
    return data


def compress(data):
    """Compresses "data" list for easy of writing to a database."""
    return memoryview(bz2.compress("\n".join(data).encode('utf-8'), 9))


def parse_header(head):
    """
    Parses a list of header lines into a dictionary.
    Args:
        head: list of strings
    Returns:
        result: dict
    Notes:
    In the raw output files, the header lines have the following form:
    MOMENTUM: p000
    MASSES: 0.0037 0.0037
    SOURCE: even_and_odd_wall even_and_odd_wall
    SINKOPS: identity identity
    SINKS: POINT_KAON_5
    """
    result = {}
    for line in head:
        tokens = line.rstrip().split()
        if tokens[0] == "MOMENTUM:":
            result["momentum"] = tokens[1]
        elif tokens[0] == 'MASSES:':
            masses = (float(tokens[1]), float(tokens[2]))
            result["masses"] = masses
            result["mass1"] = min(masses)
            result["mass2"] = max(masses)
        elif tokens[0] == 'SOURCE:':
            result["source"] = " ".join(tokens[1:])
        elif tokens[0] == 'SINKOPS:':
            result["sinkops"] = " ".join(tokens[1:])
        elif tokens[0] == 'SINKS:':
            result["sinks"] = tokens[1]
        else:
            raise ValueError(f"Unrecognized header line {line}")
    return result


def parse_datablock(ifile):
    """
    Parses a correlator data block into lists of t, Re(C), Im(C).
    Args:
        ifile:
    Returns:
        t, real, imag: three lists with the data
    Notes:
    The data block is assumed to consist of three columns like the following:
    0 3.103021e-06 0.000000e+00
    1 8.273227e-07 0.000000e+00
    ...
    ENDPROP
    The datablock ends with the flag "ENDPROP".
    """
    # Use regex to be picky about what constitutes data and to fail
    # early and noisily if something unexpected appears
    scientific = r"(-?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?)"
    datum = re.compile(r"^(\d+)" + f" {scientific} {scientific}$")
    t, real, imag = [], [], []
    data_block = True
    while data_block:
        line = next(ifile)
        match = re.match(datum, line)
        if match:
            tokens = match.groups() # t, real, imag
            t.append(tokens[0])
            real.append(tokens[1])
            imag.append(tokens[2])
        elif "ENDPROP" in line:
            data_block = False
        else:
            LOGGER.error("ERROR: Unrecognized line in data block %s", line)
            raise ValueError(f"Unrecognized line in data block, {line}")
    return t, real, imag


class Data(Base):
    """
    Database table "Data" for storing correlator data
    Args:
        correlator:
        series: str, the series label. Often a letter like 'a', 'b', etc...
        trajectory: int, the trajectory number
        tsrc: int, the source time
        real: list, data for the real part of the correlator Re(C(t))
        imag: list, data for the imaginary part of the correlator Im(C(t))
    """
    __tablename__ = "data"
    data_id = Column(Integer, Sequence('data_id_seq'), primary_key=True)
    correlator_id = Column(Integer, ForeignKey('correlator.correlator_id'))
    series = Column(String, nullable=False)
    trajectory = Column(Integer, nullable=False)
    tsrc = Column(Integer, nullable=False)
    data_real_bz2 = Column(BLOB, nullable=True)
    data_imag_bz2 = Column(BLOB, nullable=True)
    __table_args__ = (
        UniqueConstraint('correlator_id', 'series', 'trajectory', 'tsrc'),
    )
    def __init__(self, correlator, series, trajectory, tsrc, real, imag):
        self.correlator_id = correlator.correlator_id
        self.series = series
        self.trajectory = trajectory
        self.tsrc = tsrc
        self.data_real_bz2 = bz2.compress(real, 9)
        self.data_imag_bz2 = bz2.compress(imag, 9)


class Correlator(Base):
    """
    Database table "Correlator" for registering the existence of a correlator.
    Args:
        masses: list / tuple, a pair of masses
        sinks: str, e.g., "POINT_KAON_5"
        source: str, e.g., "random_color_wall, random_color_wall"
    """
    __tablename__ = 'correlator'
    correlator_id = Column(Integer, Sequence('correlator_id_seq'), primary_key=True)
    correlator_hash = Column(String, nullable=False)
    momentum = Column(String, nullable=False)
    mass1 = Column(Float, nullable=False)
    mass2 = Column(Float, nullable=False)
    sinks = Column(String, nullable=False)
    source = Column(String, nullable=False)
    sinkops = Column(String, nullable=False)
    data = sqlalchemy.orm.relation(
        Data,
        primaryjoin=(correlator_id == Data.correlator_id))
    __table_args__ = (
        UniqueConstraint('momentum', 'mass1', 'mass2', 'sinks', 'source',
                         'sinkops', 'correlator_hash'),
    )
    def __init__(self, momentum, masses, sinks, source, sinkops):
        if len(masses) != 2:
            raise ValueError("Expected exactly two masses")
        self.momentum = momentum
        self.mass1 = min(masses)
        self.mass2 = max(masses)
        self.sinks = sinks
        self.source = source
        self.sinkops = sinkops
        self.correlator_hash = self.hash()

    def hash(self):
        """Computes a hash representation of the correlator."""
        tag = (
            f"{self.momentum}_{self.mass1}_{self.mass2}_"
            f"{self.sinks}_{self.source}_{self.sinkops}")
        return hashlib.md5(tag.encode('utf-8')).hexdigest()


@timing
def write_records(engine, records):
    """
    Writes the contents of the DataFrame "records" to database tables,
    'correlator', 'data' using bulk inserts and handling relevant foreign keys.
    """
    def _build_values_clause(dataframe):
        """
        Converts dataframe to "payload" for use with parameterized SQL queries.
        """
        # Careful! For use with engine.execute, the payload *must* be a tuple
        # of tuples. Other iterables seem mysteriously not to work.
        return tuple(tuple(val) for val in dataframe.to_records(index=False))

    # Find missing correlator entries
    existing = pd.read_sql_query("select correlator_hash from correlator;", engine)
    mask = ~records['correlator_hash'].isin(existing['correlator_hash'].values)
    pending = records[mask]

    if not pending.empty:
        # Write new entries to 'correlator'
        cols = ['momentum', 'mass1', 'mass2', 'sinks', 'source', 'sinkops',
                'correlator_hash']
        values = _build_values_clause(pending[cols])
        query = (
            "WITH ins (momentum, mass1, mass2, sinks, source, sinkops, correlator_hash) "
            "AS ( VALUES (?,?,?,?,?,?,?) ) "
            "INSERT OR IGNORE INTO correlator "
            "(momentum, mass1, mass2, sinks, source, sinkops, correlator_hash) "
            "SELECT * FROM ins;"
        )
        engine.execute(query, values)

    # Finally write all the data columns, handling foreign keys on the DB side
    cols = ['data_real_bz2', 'data_imag_bz2',
            'series', 'trajectory', 'tsrc', 'correlator_hash']
    values = _build_values_clause(records[cols])
    query = (
        "WITH ins (data_real_bz2, data_imag_bz2, series, trajectory, tsrc, correlator_hash)"
        " AS ( VALUES (?, ?, ?, ?, ?, ?) ) "
        "INSERT OR IGNORE INTO data "
        "(data_real_bz2, data_imag_bz2, series, trajectory, tsrc, correlator_id) "
        "SELECT "
        "ins.data_real_bz2, ins.data_imag_bz2, "
        "ins.series, ins.trajectory, ins.tsrc, correlator.correlator_id "
        "FROM ins "
        "JOIN correlator USING(correlator_hash);"
    )
    engine.execute(query, values)


if __name__ == '__main__':
    main()
