"""
Read and write functions for interacting with a database.
"""
import logging
import os
import pathlib
import collections
import ast
from functools import reduce
import pandas as pd
import gvar as gv
import numpy as np
import sqlalchemy as sqla
from psycopg2.extensions import register_adapter, AsIs
import aiosql
import re
from . import alias
from . import hdf5_cache
from . import db_connection as db
from . import utils
from . import conventions


LOGGER = logging.getLogger(__name__)

# The cludgy hack seems to be necessary in python 3.7.
# With it, the code below works. Without it, it bombs.
# Previously it just worked...


def adapt_numpy_int64(numpy_int64):
    """
    Adapting numpy.int64 type to SQL-conform int type using psycopg extension,
    see [1]_ for more info.
    References:
    http://initd.org/psycopg/docs/advanced.html#adapting-new-python-types-to-sql-syntax
    """
    return AsIs(numpy_int64)


register_adapter(np.int64, adapt_numpy_int64)


def reshape_params(params, nstates):
    """
    Reshapes arrays of amplitudes which may have been flattened.
    Args:
        params: dict with keys 'Vnn', 'Vno', 'Von', 'Voo'
        nstates: namedtuple with field names 'n', 'no', 'm', 'mo'
    Returns:
        params: dict with reshaped values
    """
    params['Vnn'] = params['Vnn'].reshape(nstates.n, nstates.m)
    params['Vno'] = params['Vno'].reshape(nstates.n, nstates.mo)
    params['Von'] = params['Von'].reshape(nstates.no, nstates.m)
    params['Voo'] = params['Voo'].reshape(nstates.no, nstates.mo)
    for key in ['fluctuation', 'log(fluctuation)']:
        if key in params:
            # unpack singleton arrays
            params[key] = params[key].item()
    return params


def get_best_fit_information(engine, form_factor_id):
    """
    Gets a dict of best-fit information from the database regarding
    the specified form factor. Also gets some "meta" information like
    the associated momentum, current, and lattice size.
    Args:
        form_factor_id: int, the unique id for the form factor
    Returns:
        best_fit: dict with the best-fit information
    """
    Nstates = collections.namedtuple(
        'NStates', ['n', 'no', 'm', 'mo'], defaults=(1, 0, 0, 0)
    )

    def _float_or_none(astr):
        if astr is None:
            return None
        if (astr.lower() == 'nan') or (astr.lower() == 'none'):
            return None
        return float(astr)

    query = f"""
        select
            campaign.form_factor_id,
            ens.ens_id,
            ens.ns,
            form_factor.momentum,
            form_factor.spin_taste_current,
            result_id,
            n_decay_ll as n,
            n_oscillating_ll as "no",
            n_decay_hl as m,
            n_oscillating_hl as mo,
            tmin_ll as tmin_src,
            tmax_ll as tmax_src,
            tmin_hl as tmin_snk,
            tmax_hl as tmax_snk,
            binsize,
            shrinkage,
            fold as do_fold,
            sign,
            pedestal,
            params
        from campaign_form_factor as campaign
        join form_factor using(form_factor_id)
        join ensemble as ens using(ens_id)
        join sign_form_factor using(ens_id, spin_taste_current)
        join result_form_factor as result
        on  (result.form_factor_id = campaign.form_factor_id) and
            (result.id = campaign.result_id)
        join analysis_form_factor as analysis
        on  (analysis.analysis_id = result.analysis_id)
        join reduction_form_factor as reduction
        on  (reduction.reduction_id = result.reduction_id)
        where
        campaign.form_factor_id in ({form_factor_id});"""

    best_fit = pd.read_sql_query(query, engine)
    best_fit['params'] = best_fit['params'].apply(parse_string_dict)
    best_fit['pedestal'] = best_fit['pedestal'].apply(_float_or_none)
    best_fit['nstates'] = best_fit[['n', 'no', 'm', 'mo']].apply(
        lambda args: Nstates(*args), axis=1)
    best_fit['params'] = best_fit[['params', 'nstates']].apply(
        lambda pair: reshape_params(*pair), axis=1)
    best_fit, = best_fit.to_dict('records')  # Unpack single entry
    return best_fit


def get_glance_data(form_factor_id, engine, apply_alias=True):
    """
    Reads all the required correlators for analyzing the specified
    form factor from the table "glance_correlator_n_point". The data
    from this table lacks any information about correlations and usually
    only employs partial statistics (often having restricted to 'fine'
    solves only). However, such data serves a useful cache of partial
    results for quick, informal analyses.
    Args:
        form_factor_id: int, the id from the database
        engines: the database connection engine
    Returns:
        data: dict with the data as arrays
    """
    query = f"""
        select
        form_factor.form_factor_id,
        rtrim(name, '-_fine') as basename,
        glance_correlator_n_point.data
        from form_factor
        join junction_form_factor using(form_factor_id)
        join correlator_n_point using(corr_id)
        join glance_correlator_n_point using(corr_id)
        where
        (nconfigs > 100) and
        not ((corr_type != 'three-point') and (name like 'A4-A4%%'))
        and (form_factor_id = {form_factor_id});"""
    dataframe = pd.read_sql_query(query, engine)
    basenames = dataframe['basename'].values
    data = {}
    for _, row in dataframe.iterrows():
        key = row['basename']
        data[key] = parse_string(row['data'])
    if apply_alias:
        aliases = alias.get_aliases(basenames)
        aliases = alias.apply_naming_convention(aliases)
        for basename in basenames:
            data[aliases[basename]] = data.pop(basename)
    return data

def fix_signs(data):
    """
    For reasons which remain mysterious, sometimes 3pt functions
    have a funny minus-sign issue where the global sign of the correlator
    seems to flip on every other configuration [1, -1, 1, -1, ...].
    This problem seems mostly to afflict the currents S-S and V4-V4.
    for a=0.057 fm.

    As a quick fix, this function simply flips all the signs to be positive.
    """
    for key in data.keys():
        if not isinstance(key, int):
            continue
        data[key] = np.sign(data[key]) * data[key]
    return data



def sanitize_data(data):
    """
    Sanitizes data, removing NaNs.
    Args:
        data: dict, with array-like values
    Returns:
        data: dict, with array-like values with all rows containing NaNs removed
        nan_configs: set, the rows numbers where NaNs were encountered
    """
    # Make sure the same number of configurations appear everywhere
    nconfigs = np.inf
    for datum in data.values():
        nconfigs = min(nconfigs, datum.shape[0])
    for key, datum in data.items():
        data[key] = datum[:nconfigs, :]

    # Locate rows with NaNs
    nan_rows = {locate_nan_rows(data[key]) for key in data}

    # When no NaNs are found, the data are already sanitized
    if not nan_rows:
        return data, nan_rows

    # Multiple distinct sets of rows with NaNs encountered
    use_nans = False
    if len(nan_rows) > 1:
        LOGGER.warning("Found NaNs in different rows; taking the union of all such rows.")
        use_nans = True
        nan_rows = [reduce(lambda a, b: a | b, nan_rows), ]

    # Remove the NaNs
    keys = list(data.keys())
    for key in keys:
        if use_nans:
            data[key] = remove_nans(data.pop(key), nan_rows[0])
        else:
            data[key] = remove_nans(data.pop(key))

    # Verify that resulting data are consistenly shaped
    distinct_shapes = {val.shape for val in data.values()}
    if len(distinct_shapes) != 1:
        raise ValueError("Removing NaNs produced inconsistenly shaped data.", distinct_shapes)

    nan_rows, = nan_rows
    return data, nan_rows


def locate_nan_rows(arr):
    """Locates rows in which NaNs appear."""
    # Count the number of NaNs in each row
    nan_counts = np.sum(~np.isfinite(arr), axis=1)
    # Trigger on a NaN appearing anywhere in a line/row
    nans, = np.where(nan_counts > 1)
    return frozenset(nans)


def remove_nans(arr, nan_rows=None):
    """Removes NaNs from an array."""
    # Remove NaNs
    nconfigs, nt = arr.shape
    if nan_rows is None:
        mask = np.isfinite(arr)
    else:
        mask = np.array([n for n in np.arange(nconfigs) if n not in nan_rows])
    return arr[mask].reshape(-1, nt)


def read_data(basenames, engine, apply_alias=True, sanitize=True):
    """
    Reads all the required correlators for analyzing the specified form factor.
    Args:
        basename: list of correlator basenames
        engines: connection engine
        apply_alias: bool, whether to rename the correlators descriptively
        sanitize: bool, whether or not to remove nans from the data
    Returns:
        data: dict with the data as arrays
    """
    if apply_alias:
        # Map to descriptive names like 'source' or 'sink'
        aliases = alias.get_aliases(basenames)
        # Further map to conventional names like 'sink' --> 'heavy-light'
        name_map = alias.apply_naming_convention(aliases)
    data = {}
    for basename in basenames:
        key = name_map.get(basename, None) if apply_alias else basename
        if key is None:
            continue
        data[key] = hdf5_cache.get_correlator(engine, basename)
    if sanitize:
        data, nan_rows = sanitize_data(data)
        if nan_rows:
            LOGGER.warning("WARNING: NaNs found while sanitizing: %s", nan_rows)
    return data


def get_form_factor_data(form_factor_id, engines, apply_alias=True, sanitize=True):
    """
    Reads all the required correlators for analyzing the specified form factor.
    Args:
        form_factor_id: int, the id from the database
        engines: dict of database connection engiens
    Returns:
        data: dict with the data as arrays
    """
    query = (
        "SELECT ens_id, RTRIM(name, '-_fine') as BASENAME, corr_type "
        "FROM junction_form_factor AS junction "
        "JOIN correlator_n_point AS corr ON (corr.corr_id = junction.corr_id) "
        "WHERE (form_factor_id = {form_factor_id}) AND (name LIKE '%%fine');"
    )
    query = query.format(form_factor_id=form_factor_id)
    dataframe = pd.read_sql_query(query, engines['postgres'])
    ens_id = dataframe['ens_id'].unique().item()
    basenames = dataframe['basename'].values

    # Grab a list of necessary correlators, in particular identifying the
    # source and sink 2pt functions. This line gives a map from the full
    # basename to a name like 'source' or 'sink'.
    aliases = alias.get_aliases(basenames)
    # Apply any further renaming, e.g., 'sink' --> 'heavy-light'
    name_map = alias.apply_naming_convention(aliases)
    data = {}
    for basename in aliases:
        key = name_map[basename] if apply_alias else basename
        try:
            data[key] = hdf5_cache.get_correlator(engines[ens_id], basename)
        except ValueError as err:
            LOGGER.warning("WARNING: Unable to load %s", key)
    if sanitize:
        data, nan_rows = sanitize_data(data)
        if nan_rows:
            LOGGER.warning("WARNING: NaNs found while sanitizing: %s", nan_rows)
    return data


def get_form_factor_data_elvira(form_factor_id, engines, sanitize=True):

    # Get location of hdf5 file
    query = """
        SELECT ext.name, CONCAT(ext.location, '/', ext.name, '.hdf5') AS h5fname
        FROM form_factor JOIN ensemble USING(ens_id)
        JOIN external_database as ext USING(ens_id)
        WHERE form_factor_id = %(form_factor_id)s"""
    df = pd.read_sql(query, engines['postgres'], params={'form_factor_id': form_factor_id})
    if not df['name'].item().startswith('egamiz'):
        raise ValueError("Please use get_form_factor_data(...) instead.")
    h5fname = df['h5fname'].item()

    # Get the correlator basenames
    query = """
        SELECT ens_id, name AS basename, corr_type
        FROM junction_form_factor AS junction
        JOIN correlator_n_point USING(corr_id)
        WHERE (form_factor_id = %(form_factor_id)s)"""
    query = query.format(form_factor_id=form_factor_id)
    dataframe = pd.read_sql(query, engines['postgres'], params={'form_factor_id': form_factor_id})
    basenames = dataframe['basename'].values

    # Apply relevant aliases
    aliases = {}
    regex_light_light = re.compile(r"^2pt(pi|K)mom(\d)$")
    regex_heavy_light = re.compile(r"^2ptD(s?)mom(\d)$")
    regex_3pt = re.compile(r"^3pt(D|K)(P|K)((m0)?)T(\d\d?)$")
    for _, row in dataframe.iterrows():
        basename = row['basename']
        corr_type = row['corr_type']
        if (corr_type == 'light-light') or regex_light_light.match(basename):
            aliases[basename] = 'light-light'
        elif (corr_type == 'heavy-light') or regex_heavy_light.match(basename):
            aliases[basename] = 'heavy-light'
        elif regex_3pt.match(basename):
            aliases[basename] = int(regex_3pt.match(basename).groups()[-1])
        else:
            raise ValueError("Unrecognized basename", basename)

    # Read the data
    data = {}
    for basename in basenames:
        data[aliases[basename]] = hdf5_cache._get_correlator(h5fname, basename)
    if sanitize:
        data, nan_rows = sanitize_data(data)
        if nan_rows:
            LOGGER.warning("WARNING: NaNs found while sanitizing: %s", nan_rows)
    return data


def sanitize_record(record, table):
    """
    Sanitizes the dict 'record' for writing to 'table',
    i.e., restricts to keys which appear as columns of table.
    """
    try:
        columns = table.columns
    except AttributeError:
        columns = vars(table)
    return {key: value for key, value in record.items() if key in columns}


def to_text(adict):
    """ Wrapper for converting dicts to text for postgres"""
    new_dict = {}
    for key, val in sorted(adict.items()):
        new_dict[key] = str(val)
    return '$delim${{{0}}}$delim$'.format(str(new_dict))


def rebuild_params(params, nstates):
    """
    Reshapes arrays of amplitudes which may have been flattened.
    Args:
        params: dict with keys 'Vnn', 'Vno', 'Von', 'Voo'
        nstates: namedtuple with field names 'n', 'no', 'm', 'mo'
    Returns:
        params: dict with reshaped values
    """
    params['Vnn'] = params['Vnn'].reshape(nstates.n, nstates.m)
    params['Vno'] = params['Vno'].reshape(nstates.n, nstates.mo)
    params['Von'] = params['Von'].reshape(nstates.no, nstates.m)
    params['Voo'] = params['Voo'].reshape(nstates.no, nstates.mo)
    return params

def parse_string_dict(dict_as_string):
    """
    Parse a string representation of a dictionary, e.g.,
    "{{'key1': 'val1', 'key2': 'val2', ...}}"
    """
    new_dict = ast.literal_eval(dict_as_string[1:-1])
    new_dict = {key: parse_string(val) for key, val in new_dict.items()}
    return new_dict


def parse_string(str_arr):
    """
    Parse a string representation of an array of gvars into
    an array of gvars. This operation arises frequently, for
    example, when reading from the various "glance*" tables,
    which store preprocessed data.
    """
    def to_arr(str_arr):
        """ Switch to list. """
        row = str_arr.replace(']', '').\
            replace('[', '').\
            replace('{', '').\
            replace('}', '').\
            replace('\n', '').split()

        if '+-' in row:
            row = kludge_gvars(row)
        row = [gv.gvar(str(elt)) for elt in row]
        return np.array(row)

    def kludge_gvars(mangled):
        """
        Occasionally, gvars get rendered to strings as, e.g.,
        -4e-06 +- 1 instead of -0.000006(1.0). This makes a
        complete mess of trying to parse the a list of gvar
        which has been turned into a string, e.g.,
        '[1(2) 1 +- 2 0.003(2)]', since the usual str.split()
        separates '1 +- 2' --> ['1','+-','2']. This function is
        a kludge which works around this difficulty.
        """
        # Loop in reverse looking for '+-', but don't run off the end
        for idx in range(len(mangled) - 1)[::-1]:
            if mangled[idx + 1] == '+-':
                reunited = ' '.join(mangled[idx:idx + 3])
                # Throw away the used elements...
                for _ in range(3):
                    mangled.pop(idx)
                # Repair the list with reunited gvar string
                mangled.insert(idx, reunited)
        return mangled

    return to_arr(str_arr)


def upsert(engine, table_name, dataframe):
    """
    Upsert the content of a DataFrame into a db table

    Credit: User Ryan Tuck
    https://stackoverflow.com/questions/41724658/how-to-do-a-proper-upsert-using-sqlalchemy-on-postgresql

    TODO: implement insert_ignore.
    Assume that insert_ignore() is a function defined elsewhere ignores instead
    of updating conflicting rows.
    """
    # Reflect table from db
    metadata = sqla.MetaData(bind=engine)
    table = sqla.Table(
        table_name,
        metadata,
        autoload=True,
        autoload_with=engine)

    # Unpackage DataFrame
    records = []
    for _, row in dataframe.iterrows():
        # Edge case: serial primary keys, e.g., may not be in the row yet
        records.append({col.name: row[col.name] for col in table.columns if col.name in row})

    # get list of fields making up primary key
    primary_keys = [
        key.name for key in sqla.inspection.inspect(table).primary_key]
    assert len(primary_keys) == 1

    # assemble base statement
    stmt = sqla.dialects.postgresql.insert(table).values(records)

    # Isolate non-primary keys for updating
    update_dict = {
        col.name: col for col in stmt.excluded if not col.primary_key}

    # Edge case: all columns make up a primary key
    # Then upsert <--> 'on conflict do nothing'
    if update_dict == {}:
        LOGGER.warning('No updateable columns found for table %s. Skipping upsert.', table_name)
        # Still want to upsert without error.
        # TODO: implement insert_ignore()
        # insert_ignore(table_name, records)
        return None

    # Assemble statement with 'on conflict do update' clause
    update_stmt = stmt.on_conflict_do_update(
        index_elements=primary_keys,
        set_=update_dict,
    )
    LOGGER.debug(update_stmt)
    # execute
    with engine.connect() as conn:
        result = conn.execute(update_stmt)
        return result


def write(engine, table_name, src, return_id=True, do_update=False):
    """
    Writes the dictionary 'src' to the table named 'table_name'.
    """
    query = build_upsert_query(engine, table_name, src, do_update=do_update)
    LOGGER.debug(query)
    engine.execute(query)
    if return_id:
        return fetch_id(engine, table_name, src)


def fetch_id(engine, table_name, src):
    """
    Fetch id from database given a query with error handling.
    Args:
        engine: database connection "engine"
        query: str, containing a "raw" SQL query which will return a single id.
    Returns:
        int, containing the id or None if not found
    """

    # Helper functions
    def get_unique_columns(table):
        """ Gets the unique columns from the table's contraints. """
        for constraint in table.constraints:
            if isinstance(constraint, sqla.UniqueConstraint):
                return constraint.columns
        # We should never get this far.
        # All tables in my db should have unique constraints
        assert False

    def get_id_name(table):
        """ Gets the name of the primary key column. """
        primary_key_columns = table.primary_key.columns.items()
        if len(primary_key_columns) == 1:
            name, _ = primary_key_columns[0]
            return name
        # We should never get this far.
        # All tables in my db should have a single primary key column
        assert False

    # Reflect table from db
    meta = sqla.MetaData()
    table = sqla.Table(table_name, meta, autoload=True, autoload_with=engine)

    unique_cols = get_unique_columns(table)
    id_name = get_id_name(table)

    # Build the SQL query by hand
    query = """SELECT {0} from {1} WHERE """.format(id_name, table_name)
    constraints = []
    for col in unique_cols:
        val = src[col.name]
        if 'TEXT' in str(col.type):
            template = "({col}='{val}')"
        else:
            template = "({col}={val})"
        constraints.append(template.format(col=col, val=val))
    constraints = ' AND '.join(constraints) + ';'
    query = query + constraints

    # Fetch the id
    LOGGER.debug(query)
    dataframe = pd.read_sql_query(query, engine)
    if len(dataframe) == 1:
        return dataframe[id_name].item()
    elif dataframe.empty == 0:
        return None
    else:
        raise ValueError("Non-unique id encountered.")


def build_upsert_query(engine, table_name, src_dict, do_update=False):
    """
    Builds a raw SQL query for "upserting" data into the database.
    Args:
        engine: database connection "engine"
        table_name: str, name of the database table for the upsert
        srs_dict: dict, containing the column information for the upsert
        do_update: bool, whether or not to update the row when a conflict is
            encountered. Default is False.
    Returns:
        query: str, containing the raw SQL query.

    Remarks:
        Intended use is, e.g.,
        >>> engine = db.make_engine()
        >>> query = build_upsert_query(engine, 'table_name', src_dict)
        >>> engine.execute(query)
    """
    def _for_pgsql(value, dtype):
        """
        Converts a python datatype to the appropriate string (including, e.g., \
        the necessary single quotes and/or brackets ) for use in a raw \
        postgresql query.
        Args:
            value: (various datatypes) the value in question
            dtype: str, the datatype
        Returns:
            str, with the necessary formatting
        """
        if dtype.startswith(('int', 'float', 'double', 'numeric')):
            if value is None:
                return "Null"
            elif str(value).lower() == 'nan':
                return "'nan'"
            elif dtype.endswith('[]'):
                value = ', '.join([str(v) for v in value])
                value = "'{" + value + "}'"
                return value
            else:
                return str(value)
        elif dtype.startswith('time'):
            if value is None:
                return "Null"
            else:
                return "'" + str(value) + "'"
        elif dtype.startswith('bool'):
            if value is None:
                raise ValueError("Error: bool should not be None.")
            else:
                if str(value).startswith(('t', 'T')):
                    return str(True)
                else:
                    return str(False)
        elif dtype.startswith('json'):
            # In this case, value itself should be a dict
            value = ','.join(['"{k}":"{v}"'.format(k=k, v=v)
                              for k, v in value.items()])
            value = "'{" + value + "}'"
            return value
        elif dtype == 'text[]':
            value = ', '.join(['"' + str(v) + '"' for v in value])
            value = "'{" + str(value) + "}'"
            return value
        else:
            if str(value).startswith('$delim$') and\
               str(value).endswith('$delim$'):
                return str(value)
            if '::' in str(value):
                value = str(value).split("::")[0].strip("'")
            return "'" + str(value) + "'"

    def _get_values(uprow, types):
        """
        Gets a list of values for use in a raw SQL query, e.g.,

        INSERT INTO table_name
        (column1, column2, ...)
        VALUES
        (value1, value2, ...);

        This function returns a string "value1, value2, ..."
        Args:
            uprow: dict, containing the values
            types: dict, containing the data types of the values
        Return:
            str, containing the values as described above.
        """
        tmp_uprow = {k: _for_pgsql(v, types[k]) for k, v in uprow.items()}
        mappable = ",".join(["{" + str(k) + "}" for k in uprow.keys()])
        values = mappable.format(**tmp_uprow)
        return values

    def _get_set_pairs(uprow, types):
        """
        Gets a list of "set pairs" for use in a raw SQL query, e.g.,

        INSERT INTO table_name
        (column1, column2, ...)
        VALUES
        (value1, value2, ...)
        ON CONFLOCT (column1) DO UPDATE
        SET
        column1=value1,
        column2=value2

        This function returns a string "column1=value1, column=value2
        Args:
            uprow: dict, containing the values
            types: dict, containing the data types of the values
        Return:
            str, containing the "set pairs" as described above.
        """
        pairs = []
        for key, val in uprow.items():
            pairs.append("{0}={1}".format(key, _for_pgsql(val, types[key])))
        return ", ".join(pairs)

    # Mirror table from DB
    meta = sqla.MetaData(bind=engine)
    insp = sqla.inspect(engine)
    table = sqla.Table(table_name, meta, autoload=True, autoload_with=engine)
    table_cols = [str(col).split('.')[1] for col in table.columns]

    # Collect dict entries that also appear in the table as a "row"
    uprow = {key: src_dict[key] for key in src_dict if key in table_cols}

    # Load defaults and collect types
    types = {}
    for column in insp.get_columns(table_name, default=True):
        name = column['name']
        if (name not in uprow) and (name != 'id'):
            uprow[name] = column['default']
        types[name] = str(column['type']).lower()

    # Build base query
    columns = "{keylist}".format(keylist=', '.join(uprow.keys()))
    values = _get_values(uprow, types)

    query = "INSERT INTO {table_name}\n".format(table_name=table_name)
    query += "({columns})\n".format(columns=columns)
    query += "VALUES\n"
    query += "({values})\n".format(values=values)

    # Fetch unique columns
    unique_constraints = insp.get_unique_constraints(table_name)

    # Handle potential conflicts
    if len(unique_constraints) > 0:
        unique_cols = insp.get_unique_constraints(table_name)[
            0]['column_names']
        if len(unique_cols) > 1:
            unique_cols = ", ".join([str(col) for col in list(unique_cols)])
        else:
            unique_cols = ', '.join(unique_cols)
        if do_update:
            set_clause = "ON CONFLICT ({unique_cols}) DO UPDATE SET\n".\
                format(unique_cols=unique_cols)
            set_clause += _get_set_pairs(uprow, types)
            query += set_clause
        else:
            query += "ON CONFLICT ({unique_cols}) DO NOTHING\n".\
                format(unique_cols=unique_cols)
    else:
        # No unique constraints, look for primary key instead
        primary_key = [c for c in table.columns if c.primary_key]
        if len(primary_key) == 1:
            primary_key, = primary_key
            # Ditch reference to foreign table
            if '.' in str(primary_key):
                primary_key = str(primary_key).split('.')[-1]
        else:
            tmp = []
            for col in primary_key:
                # Ditch reference to foreign table
                if '.' in str(col):
                    col = str(col).split('.')[-1]
                tmp.append(col)
            primary_key = ", ".join(tmp)
        if do_update:
            set_clause = "ON CONFLICT ({primary_key}) DO UPDATE SET\n".\
                format(primary_key=primary_key)
            set_clause += _get_set_pairs(uprow, types)
            query += set_clause
        else:
            query += "ON CONFLICT ({primary_key}) DO NOTHING\n".\
                format(primary_key=primary_key)

    query += ';'

    return query

def fetch_basenames(engine, form_factor):
    """
    Fetches a list of correlators matching the specified form factor.
    Args:
        engine: sqlalchemy.Engine
        form_factor: dict with keys 'current', 'mother', 'daughter', 'spectator', and 'momentum'
    Returns:
        numpy.array, the matching correlators
    """
    for key in ['current', 'm_mother', 'm_daughter', 'm_spectator', 'momentum']:
        if key not in form_factor:
            raise KeyError(f"Required key '{key}' is missing.")

    def abspath(dirname):
        return os.path.join(pathlib.Path(__file__).parent.absolute(), dirname)

    # 2pt correlators like 'P5-P5_RW_RW_d_d_m0.002426_m0.002426_p000'
    mother = "%_RW_RW_d_d_m{m_mother}_m{m_spectator}_p000%fine"
    daughter = "%_RW_RW_d_d_m{m_daughter}_m{m_spectator}_{momentum}%fine"
    if form_factor['m_daughter'] < form_factor['m_spectator']:
        daughter = "%_RW_RW_d_d_m{m_spectator}_m{m_daughter}_{momentum}%fine"

    # 3pt correlators like 'P5-P5_RW_RW_d_d_m0.002426_m0.002426_p000',
    corr3 = "%_{current}_T%_m{m_mother}_RW_RW_x_d_m{m_spectator}_m{m_daughter}_{momentum}%fine"

    params = {
        'mother': mother.format(**form_factor),
        'daughter': daughter.format(**form_factor),
        'corr3': corr3.format(**form_factor)}
    queries = aiosql.from_path(abspath("sql/"), "sqlite3")
    with db.connection_scope(engine) as conn:
        corrs = queries.postgres.get_correlator_names(conn, **params)
    
    return np.squeeze(np.array(corrs))


def get_ns(name):
    """
    Gets the spatial size of the lattice in the configuration
    Args:
        name: str, the name of the ensemble (e.g., 'l3248f211b580m002426m06730m8447-allHISQ')
    Returns:
        int, the spatial size of the lattice
    """
    ensembles = conventions.ensembles
    mask = (ensembles['name'] == name)
    return utils.extract_unique(ensembles[mask], 'ns')


def get_nt(name):
    """
    Gets the temporal size of the lattice in the configuration
    Args:
        name: str, the name of the ensemble (e.g., 'l3248f211b580m002426m06730m8447-allHISQ')
    Returns:
        int, the temporal size of the lattice
    """
    ensembles = conventions.ensembles
    mask = (ensembles['name'] == name)
    return utils.extract_unique(ensembles[mask], 'nt')


def get_sign(current):
    """
    Gets the conventional sign associated with a matrix element / form factor.
    """
    signs = conventions.form_factor_signs
    mask = (signs['spin_taste_current'] == current)
    return utils.extract_unique(signs[mask], 'sign')


def get_mq(a_fm, description, quark_alias):
    """
    Gets the bare quark mass from a table given an alias (e.g., '1.0 m_light').
    Args:
        a_fm: float, the lattice spacing in fm
        description: str, a description of the light quarks. Usually '1/27', '1/10', or '1/5'
        quark_alias: str, the alias for the quark mass
    Returns:
        float, the bare quark mass
    """
    quark = conventions.quark_masses
    mask = utils.bundle_mask(quark, a_fm=a_fm, description=description, alias=quark_alias)
    return utils.extract_unique(quark[mask], 'mq')


def get_alias(a_fm, description, quark_mass):
    """
    Gets an alias for a quark mass (e.g., '1.0 m_light') from a table.
    Args:
        a_fm: float, the lattice spacing in fm
        description: str, a description of the light quarks. Usually '1/27', '1/10', or '1/5'
        mq: float, the bare quark mass
    Returns:
        str, the alias for the quark mass
    """
    quark = conventions.quark_masses
    mask = utils.bundle_mask(quark, a_fm=a_fm, description=description, mq=quark_mass)
    return utils.extract_unique(quark[mask], 'alias')


def get_ensemble(a_fm, description):
    """
    Gets an ensemble name (e.g., 'l3248f211b580m002426m06730m8447-allHISQ') from a table.
    Args:
        a_fm: float, the lattice spacing in fm
        description: str, a description of the light quarks. Usually '1/27', '1/10', or '1/5'
    Returns:
        str, the ensemble name
    """
    ens = conventions.ensembles
    mask = utils.bundle_mask(ens, a_fm=a_fm, description=description)
    return utils.extract_unique(ens[mask], 'name')
