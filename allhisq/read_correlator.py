"""
doc here
"""
import pandas as pd
from . import hdf5_cache as hdf5
from . import alias


def get_tsm_data(engine, basename):
    """
    Gets data from database and applies truncated solver method to combine the
    loose and fine solves. If matching pairs of fine+loose solves are missing,
    the resulting array will contain NaNs.
    """
    if not (alias.match_2pt(basename) or alias.match_3pt(basename)):
        raise ValueError(
            "Specified basename is not a recognized 2pt or 3pt function.",
            basename)
    corr_ids = get_corr_ids(engine, basename)
    data = read_data(engine, corr_ids)
    data = hdf5.unpackage(data)
    reduced = hdf5.tsm(data)
    return reduced


def read_data(engine, correlator_ids):
    """ Gets correlator data from db corresponding to correlator_id(s) """
    def _handle_ids(correlator_ids):
        list_of_ids = [str(elt) for elt in correlator_ids]
        return '({0})'.format(','.join(list_of_ids))
    query = (
        "SELECT data.*, correlators.name FROM data "
        "JOIN correlators ON correlators.id = data.correlator_id "
        f"WHERE correlator_id IN {_handle_ids(correlator_ids)};")
    return pd.read_sql_query(query, engine)


def get_corr_ids(engine, basename):
    """
    Gets correlator_id values corresponding to the loose and fine solves for
    a given basename.
    """
    query = (
        "SELECT id AS correlator_id "
        "FROM correlators "
        f"WHERE name LIKE '{basename}%%';")
    corr_ids = pd.read_sql_query(query, engine)['correlator_id']
    return corr_ids
