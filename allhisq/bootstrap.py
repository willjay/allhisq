"""
A simple utility for parsing databases and recording their locations and the
correlators they contain. Now deprecated in favor of 'parse_database.py'
"""
import os
import sys
import logging
import pandas as pd
import sqlalchemy as sqla
from . import db_connection as db
from . import db_io


LOGGER = logging.getLogger(__name__)


def main():
    """
    Runs through a hard-coded base directory and processes databases, recording
    the location and names of correlators it finds.
    """
    assert False, "This script is deprecated in favor of parse_database.py."
    home = "/home/wjay/dbs/allHISQ/"
    LOGGER.info("[+] Looking for sqlite databases in %s", home)
    for root, dirs, files in os.walk(home):
        dirs.sort()  # modify in place to search in order
        LOGGER.info("[+] root %s", root)
        for fname in files:
            # Fermilab-MILC stores results in sqlite databases
            # Each ensemble has its own database
            # Filenames start with "l{ns}{nt}f..."
            if fname.startswith("l") and fname.endswith("sqlite"):
                process_database(fname, root)


def process_database(fname, location):
    """
    Processes the database, recording its location and the names of its
    correlation functions.
    """
    LOGGER.info("[+] Processing the database %s", fname)
    engine = db.make_engine()
    full_path = os.path.join(location, fname)
    sqlite_engine = db.make_engine(sqlite=True, db_choice=full_path)

    # Record ensemble in analysis database
    name = fname.rstrip('.sqlite')
    ns, nt = extract_nsnt(name)

    # Check for existing ensembles
    existing = fetch_existing(engine)
    if (location not in existing['location'].values) and (
            name not in existing['name'].values):
        ens = {
            'name': name,
            'ns': ns,
            'nt': nt,
            'location': location,
            'type': 'sqlite'}
        query = db_io.build_upsert_query(engine, 'ensemble', ens)
        engine.execute(query)
        ens_id = fetch_ens_id(engine, ens)

        # Record location of database
        ens['ens_id'] = ens_id
        query = db_io.build_upsert_query(engine, 'external_database', ens)
        engine.execute(query)

        if ens_id is None:
            msg = "Missing ensemble? ens ={0}".format(ens)
            raise ValueError(msg)

        # Record the names of correlators in analysis database
        corr_names = fetch_corr_names(sqlite_engine)
        write_corr_names(engine, corr_names, ens_id)
    else:
        LOGGER.info("[+] Skipping the database %s (already processed).", fname)


def fetch_existing(engine):
    """ Fetches existing ensembles and external databases """
    query = (
        "SELECT ensemble.*, external_database.location, external_database.type "
        "FROM ensemble "
        "JOIN external_database ON ensemble.ens_id = external_database.ens_id;"
    )
    LOGGER.debug(query)
    return pd.read_sql_query(query, engine)


def fetch_ens_id(engine, src):
    """ Fetches the ensemble id associated with 'src' """
    return db_io.fetch_id(engine, 'ensemble', src)


def fetch_corr_names(engine):
    """ Fetches all the correlator names from the specified database """
    dataframe = pd.read_sql_query("SELECT name FROM correlators;", engine)
    return dataframe['name'].unique()


def write_corr_names(engine, corr_names, ens_id):
    """ Writes the correlator names to the analysis database."""
    LOGGER.info("[+] Writing correlator names to analysis database.")
    src = {'ens_id': ens_id}
    total = len(corr_names)
    for count, corr_name in enumerate(corr_names):
        progress(count, total, corr_name)
        src['name'] = corr_name
        query = db_io.build_upsert_query(engine, 'correlator_n_point', src)
        engine.execute(query)


def extract_nsnt(name):
    """ Extracts 'ns' and 'nt' from a string 'name' """
    nsnt = name.split('f')[0].lstrip('l')
    if len(nsnt) == 4:
        ns = int(nsnt[:2])
        nt = int(nsnt[2:])
    else:
        raise ValueError(f"Unrecognized 'nsnt' with length {nsnt}")
    return ns, nt


def progress(count, total, status=''):
    """
    Displays progress bar of the form:
    [========================================--------------------] 66.6% status
    Args:
        count: numeric, progress counter
        total: total, the end value of the counter
        status: str, name of the status to appear on the right
    Returns:
        None

    As suggested by Rom Ruben
    (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
    """

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
