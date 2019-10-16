"""
A simple utility for parsing correlator metadata out of sqlite databases into
the central analysis database. Typically this metadata is NOT normalized in the
sqlite databases. This utility normalizes the metadata before writing to the
central database. For example, see the function 'wrangle' below.
"""
import os
import re
import ast
import argparse
import pandas as pd
from sqlalchemy.ext.automap import automap_base
from . import db_connection as db


def main():
    """ Parses the metadata out of a hard-coded database."""
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=str,
                        help="input database including full path")
    args = parser.parse_args()
    input_db = args.db
    db_info = parse_db_name(input_db)

    # Reflect schema from central database
    engines = db.get_engines()
    engine_pg = engines['postgres']
    CentralBase = automap_base()
    CentralBase.prepare(engine_pg, reflect=True)

    # Locate ens_id, registering a new ensemble if necessary
    ens_id = find_engine(input_db, engines)
    if ens_id is None:
        ens_id = register_ensemble(engine_pg, db_info, CentralBase)
        # Need a fresh set of engines after registering
        engines = db.get_engines()

    # Find existing correlator entries in central database
    Corr = CentralBase.classes.correlator_n_point
    with db.session_scope(engines['postgres']) as session:
        query = session.query(Corr.name).filter(Corr.ens_id == ens_id)
        existing = pd.read_sql(query.statement, session.bind)

    # Read correlator names and meta data from embedded database
    ExtBase = automap_base()
    ExtBase.prepare(engines[ens_id], reflect=True)
    ExtCorr = ExtBase.classes.correlators
    Param = ExtBase.classes.parameters
    with db.session_scope(engines[ens_id]) as session:
        # Get correlators and meta parameters from embedded database
        query = session.query(ExtCorr.name, Param.param).join(Param)
        print(query.statement)
        df_meta = pd.read_sql(query.statement, session.bind)

    # Wrangle parameters into normalized metadata
    df_meta = wrangle(df_meta)
    mask = ~df_meta['name'].isin(existing['name'].values)
    new = df_meta[mask]
    new['ens_id'] = ens_id
    records = new.to_dict(orient='records')

    print('Total new entries: {}'.format(len(records)))

    # Write new correlators to database
    for record in records:
        register_correlator(engine_pg, record, CentralBase)


def extract_nsnt(nsnt):
    """ Extracts the values of ns and nt from a string """
    if (len(nsnt) == 4) or (len(nsnt) == 5):
        ns = int(nsnt[:2])
        nt = int(nsnt[2:])
    else:
        raise ValueError(
            "Unrecognized 'nsnt' with length {0}".format(
                len(nsnt)))
    return ns, nt


def sanitize_record(record, table):
    """
    Sanitize the dict 'record', making sure it only has keys corresponding
    to the colums in 'table'.
    """
    try:
        columns = table.columns
    except AttributeError:
        columns = vars(table)
    return {key: value for key, value in record.items() if key in columns}


def register_correlator(engine_pg, record, CentralBase):
    """ Registers a correlator in the central db """
    Corr = CentralBase.classes.correlator_n_point
    MetaDataCorr = CentralBase.classes.meta_data_correlator
    MetaDataSeq = CentralBase.classes.meta_data_sequential

    with db.session_scope(engine_pg) as session:
        corr = Corr(**sanitize_record(record, Corr))
        session.add(corr)
        session.flush()
        record['corr_id'] = corr.corr_id

        meta_corr = MetaDataCorr(**sanitize_record(record, MetaDataCorr))
        session.add(meta_corr)
        session.flush()

        if record['has_sequential']:
            record['meta_correlator_id'] = meta_corr.meta_correlator_id
            meta_seq = MetaDataSeq(**sanitize_record(record, MetaDataSeq))
            session.add(meta_seq)


def register_ensemble(engine_pg, vals, CentralBase):
    """Registers a new external db in the central db."""

    Ensemble = CentralBase.classes.ensemble
    ExternalDatabase = CentralBase.classes.external_database
    LatticeSpacing = CentralBase.classes.lattice_spacing

    with db.session_scope(engine_pg) as session:
        ens = Ensemble(**sanitize_record(vals, Ensemble))
        session.add(ens)
        session.flush()
        ens_id = ens.ens_id
        vals['ens_id'] = ens_id
        ext = ExternalDatabase(**sanitize_record(vals, ExternalDatabase))
        session.add(ext)
        vals['type'] = 'nominal'
        a_fm = LatticeSpacing(**sanitize_record(vals, LatticeSpacing))
        session.add(a_fm)

    return ens_id


def find_engine(db_name, engines):
    """Find the engine corresponding to db_name."""
    _, database = os.path.split(db_name)
    for ens_id, engine in engines.items():
        _, compare = os.path.split(engine.url.database)
        if database == compare:
            return ens_id


def parse_db_name(db_full_path):
    """
    Parses database names with regex, making sure they follow
    the expected format, for example:
    'l3248f211b580m002426m06730m8447-allHISQ.sqlite'
    In other words:
    'l{nsnt}b{beta}f{fermion_count}b{beta}m{m_light}m{m_strange}m{m_charm}-{campaign}.{suffix}'
    """
    location, name = os.path.split(db_full_path)

    # Extract lattice spacing from the path
    regex = re.compile(r'[a](\d+\.\d+)$')
    match = re.search(regex, location)
    if not match:
        raise ValueError("Invalid location? Unable to match lattice spacing.")
    a_fm = float(match.group(1))

    # Verify the rest of the database name
    regex = re.compile(
        r'^(l\d+)(f\d+)(b\d+)(m\d+)(m\d+)(m\d+)\-(allHISQ)\.(sqlite)$')
    match = re.search(regex, name)
    suffix_idx = 8
    if not match:
        # try again with for alternative file structure include 'redo'
        regex = re.compile(
            r'^(l\d+)(f\d+)(b\d+)(m\d+)(m\d+)(m\d+)\-(allHISQ)\-(redo)\.(sqlite)$')
        match = re.search(regex, name)
        suffix_idx = 9
        if not match:
            raise ValueError("Invalid db_name")

    name = name.rstrip('.sqlite')
    nsnt = match.group(1).lstrip('l')
    ns, nt = extract_nsnt(nsnt)
    return {
        'a_fm': a_fm,
        'type': 'sqlite',
        'location': location,
        'name': name,
        'nsnt': nsnt,
        'ns': ns,
        'nt': nt,
        'fermion_count': match.group(2).lstrip('f'),
        'beta': match.group(3).lstrip('b'),
        'm_light': match.group(4).lstrip('m'),
        'm_strange': match.group(5).lstrip('m'),
        'm_charm': match.group(6).lstrip('m'),
        'campaign': match.group(7),
        'suffix': match.group(suffix_idx),
    }


def wrangle(df_meta, two_point_prefix='P5-P5_RW'):
    """Wrangles and unpacks correlator data."""
    # conversion: str --> dict
    df_meta['param'] = df_meta['param'].apply(ast.literal_eval)

    # Each row in the column 'param' is now a dict.
    # However, the data is not normalized.
    # The keys appearing in this dict depend on whether or not the
    # corresponding correlation function was a two- or three-point
    # function. The three-point functions contain an extra key
    # 'antiquark_sink_ops', which we need and want to process.
    # Therefore, when unpackaging the keys from the dictionary,
    # we should be sure to take a three-point function as an example.
    # Names for two-point functions are assumed to begin with the
    # the specified argument 'two_point_prefix'.
    mask = ~df_meta['name'].str.startswith(two_point_prefix)
    example_keys = df_meta[mask]['param'].values[0].keys()
    for key in example_keys:
        df_meta[key] = df_meta['param'].apply(lambda d: d.get(key))

    # E.g., '64,64,64,96' --> [64, 64, 64, 96]
    df_meta['lattice_size'] = df_meta['lattice_size'].apply(parse_lattice_size)

    # Enforce that 'correlator_key' agree with 'name'
    # This condition is probably always true anyway.
    # But enforcing it explicitly demonstrates our expectation
    # and prevents surprises further down the analysis pipeline.
    df_meta['correlator_key'] = df_meta['name']
    df_meta = unpackage_antiquark_sink_ops(df_meta)
    return df_meta


def parse_lattice_size(str_of_ints):
    """ Converts str to list of ints """
    return [int(val) for val in str_of_ints.split(',')]


def unpackage_antiquark_sink_ops(df_meta):
    """
    Unpackages the column 'antiquark_sink_ops', which contains
    a list of dicts. Concretely, it looks something like this:
    [{'mom': [0, 0, 0],
      'operation': 'ext_src_ks',
      'spin_taste_extend': 'G5-G5',
      't0': 87},
     {'eps_naik': -0.2368,
      'mass': 0.648,
      'momentum_twist': [0.0, 0.0, 0.0],
      'operation': 'ks_inverse'}]
    """
    # Check for presence of a sequential propagator
    df_meta['has_sequential'] = df_meta['antiquark_sink_ops'].apply(bool)
    mask = df_meta['has_sequential']

    # Take the first entry as an example.
    # See the docstring above for the precise expected
    # form of the list of dictionaries.
    example_list_of_dicts = df_meta[mask]['antiquark_sink_ops'].values[0]

    # Extract each value to a new column in the DataFrame
    # Note that the key 'operation' appears in both dictionaries,
    # so the data is not properly normalized. The present (kludgy)
    # solution simply appends a counting index to distinguish them
    for idx, dct in enumerate(example_list_of_dicts):
        for key in dct:
            if key == 'operation':
                col_name = '{key}_{idx}'.format(key=key, idx=idx)
            else:
                col_name = key
            df_meta.loc[mask, col_name] = df_meta.loc[mask, 'antiquark_sink_ops'].\
                apply(lambda list_of_dicts: list_of_dicts[idx].get(key))
    return df_meta


if __name__ == '__main__':
    main()
