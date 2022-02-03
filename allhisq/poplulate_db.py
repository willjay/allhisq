"""
TODO: doc here
"""
import os
import sys
import yaml
import re
import pandas as pd
import ast
from tqdm import tqdm
from db_tools import SQLWrapper
from conventions import quark_masses
from conventions import form_factor_signs
import utils

def main():

    if len(sys.argv) != 4:
        raise ValueError("Usage: create_schema.py /path/to/credentials /path/to/database <light-quark description>")
    path = sys.argv[1]
    database = sys.argv[2]
    description = sys.argv[3]
    if description not in ['1/27', '1/10', '1/5']:
        raise ValueError("Expected light-quark description '1/27', '1/10', or '1/5'. Found", description)

    # database = sys.argv[2]
    # database = "/home/wjay/dbs/allHISQ/a0.12/l4864f211b600m001907m05252m6382-allHISQ.sqlite"
    # database = "/home/wjay/dbs/allHISQ/a0.088/l6496f211b630m0012m0363m432-allHISQ-run3.sqlite"

    # Build database connections
    with open(path) as ifile:
        db_settings = yaml.load(ifile, yaml.SafeLoader)
    sql = SQLWrapper(db_settings)
    sqlite = SQLWrapper({'database': database}, driver='sqlite3')

    # Register ensemble
    db_info = parse_db_name(database)
    db_info['description'] = description
    with sql.connection as conn:
        ens_id,  = sql.queries.write_ensemble(conn, **db_info)
        sql.queries.write_external_database(conn, **db_info)
        db_info['type'] = 'nominal'  # 'nominal' lattice spacing inferred from file name
        sql.queries.write_lattice_spacing(conn, **db_info)
    print("Located ens_id", ens_id)

    # Locate existing correlators in central database
    existing = pd.read_sql(
        sql.queries.get_existing_correlators.sql,
        sql.engine,
        params={'ens_id': ens_id})

    print("Located", len(existing), "existing correlators in central database.")

    # Read and wrangle data from embedded databases
    df_meta = pd.read_sql(sqlite.queries.get_correlators.sql, sqlite.engine)
    df_meta = wrangle(df_meta)
    df_meta['ens_id'] = ens_id
    mask = ~df_meta['name'].isin(existing['name'].values)
    records = df_meta[mask].to_dict(orient='records')
    # TODO: switch to logging statement
    print("Located", len(records), "new correlators in embedded database.")

    # Write new correlators
    if len(records) > 0:
        print("Writing new correlators.")
        with sql.connection as conn:
            for record in tqdm(records):
                sql.queries.write_correlator(conn, **record)
                sql.queries.write_meta_data_correlator(conn, **record)
                if record['has_sequential']:
                    sql.queries.write_meta_data_sequential(conn, **record)

    #########################
    # Register form factors #
    #########################

    with sql.connection as conn:
        sql.queries.refresh_two_point_materialized(conn)
        sql.queries.refresh_three_point_materialized(conn)

    form_factors = get_form_factors(sql, ens_id=ens_id)
    # Explicitly repeat the spin-taste combination from the sink at the source
    # This information isn't handy in the meta data, so add it by hand here
    form_factors['spin_taste_source'] = form_factors['spin_taste_sink']

    # Decide which form factors are new and must be registered
    existing = pd.read_sql(sql.queries.get_existing_form_factors.sql, sql.engine, params={'ens_id':ens_id})
    possible = form_factors[existing.columns].to_dict(orient='records')
    existing = existing.to_dict(orient='records')
    pending_mask = [(record not in existing) for record in possible]
    print("Possible form factors", len(possible))
    print("Pending form factors to register:", len(form_factors[pending_mask]))

    # Write the form factors
    if len(form_factors[pending_mask]) > 0:
        print("Writing form factors.")
        for record in tqdm(form_factors[pending_mask].to_dict(orient='records')):
            corrs = pd.read_sql(sql.queries.get_correlators.sql, sql.engine, params=record)
            with sql.connection as conn:
                corrs['form_factor_id'] = sql.queries.write_form_factor(conn, **record)
                sql.queries.write_junction_form_factor(conn, corrs.to_dict(orient='records'))

    #############################
    # Fill in incidental tables #
    #############################

    # Write alias_quark mass
    mask = utils.bundle_mask(quark_masses, a_fm=db_info['a_fm'], description=db_info['description'])
    quark_masses.loc[mask, 'ens_id'] = ens_id
    with sql.connection as conn:
        sql.queries.write_alias_quark_mass(conn, quark_masses[mask].to_dict(orient='records'))
        sql.queries.refresh_materialized_views(conn)

    # Write signs_form_factor
    form_factor_signs['ens_id'] = ens_id
    with sql.connection as conn:
        sql.queries.write_sign_form_factor(conn, form_factor_signs.to_dict(orient='records'))

    # Write transition_name
    aliases = pd.read_sql(sql.queries.get_alias_form_factor.sql, sql.engine, params={'ens_id':ens_id})
    transition_names = infer_transition_names(aliases)
    existing = pd.read_sql(sql.queries.get_existing_transition_name.sql, sql.engine)
    mask = ~transition_names['form_factor_id'].isin(existing['form_factor_id'].values)
    with sql.connection as conn:
        sql.queries.write_transition_name(conn, transition_names[mask].to_dict(orient='records'))

    # Populate Vi-S form factors
    with sql.connection as conn:
        sql.queries.write_Vi_form_factors(conn)


# ----- end main ----- ##

###########################################################
# Functions for registering new ensembles and correlators #
###########################################################

def parse_db_name(db_full_path):
    """
    Parses database names with regex, making sure they follow
    the expected format, for example:
    'l3248f211b580m002426m06730m8447-allHISQ.sqlite'
    In other words:
    'l{nsnt}b{beta}f{fermion_count}b{beta}m{m_light}m{m_strange}m{m_charm}-{campaign}.{suffix}'
    """
    def extract_nsnt(nsnt):
        """ Extracts the values of ns and nt from a string """
        if (len(nsnt) == 4) or (len(nsnt) == 5):
            ns = int(nsnt[:2])
            nt = int(nsnt[2:])
        elif len(nsnt) == 6:
            ns = int(nsnt[:3])
            nt = int(nsnt[3:])
        else:
            raise ValueError(
                "Unrecognized 'nsnt' with length {0}".format(
                    len(nsnt)))
        return ns, nt

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
            r'^(l\d+)(f\d+)(b\d+)(m\d+)(m\d+)(m\d+)\-(allHISQ)\-(fix|build|redo|run\d+)\.(sqlite)$')
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

def wrangle(df_meta):
    """
    Wrangles and unpacks correlator data.
    """

    def parse_lattice_size(str_of_ints):
        """ e.g., '64,64,64,96' --> [64, 64, 64, 96] """
        return [int(val) for val in str_of_ints.split(',')]

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
    mask = ~df_meta['name'].str.startswith(('P5-P5_RW', 'A4-A4_RW_RW'))
    example_keys = df_meta[mask]['param'].values[0].keys()
    for key in example_keys:
        df_meta[key] = df_meta['param'].apply(lambda d: d.get(key))

    df_meta['lattice_size'] = df_meta['lattice_size'].apply(parse_lattice_size)

    # Enforce that 'correlator_key' agree with 'name'
    # This condition is probably always true anyway.
    # But enforcing it explicitly demonstrates our expectation
    # and prevents surprises further down the analysis pipeline.
    df_meta['correlator_key'] = df_meta['name']
    df_meta = unpackage_antiquark_sink_ops(df_meta)
    return df_meta


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
    # Note: if the code bombs here, make sure the correct value for
    # "two_point_prefix" is chosen when calling wrangle. Otherwise the example
    # dictionary might be two-point function, which indeed does not contain the
    # needed 'antiquark_sink_ops'.
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

##########################################
# Functions for registering form factors #
##########################################

def get_form_factors(sql, ens_id):
    """
    Gets a DataFrame, where each row correpsonds to a different
    form factor analysis.
    """
    df_c3 = pd.read_sql(sql.queries.get_3pt_combinations.sql, sql.engine)
    df_src = pd.read_sql(sql.queries.get_source_combinations.sql, sql.engine)
    df_snk = pd.read_sql(sql.queries.get_sink_combinations.sql, sql.engine)

    df = combine_combinations(df_c3, df_src, df_snk)
    verify_uniqueness(df)
    return df[df['ens_id']==ens_id]

def combine_combinations(df_c3, df_src, df_snk):
    """
    Merges source 2pt functions, sink 2pt functions, and 3pt function according
    to masses and momentum.
    """
    # Isolate sets of masses in immutable, hashable form
    df_src['masses_src'] = df_src[['m_light_src', 'm_heavy_src']].\
        apply(lambda args: frozenset(args), axis=1)
    df_snk['masses_snk'] = df_snk[['m_light_snk', 'm_heavy_snk']].\
        apply(lambda args: frozenset(args), axis=1)

    df_c3['masses_src'] = df_c3[['m_source_to_current', 'm_spectator']].\
        apply(lambda args: frozenset(args), axis=1)
    df_c3['masses_snk'] = df_c3[['m_sink_to_current', 'm_spectator']].\
        apply(lambda args: frozenset(args), axis=1)

    # Merge combinations on pairs of masses (and momentum, for the source)
    df = pd.merge(df_c3, df_src, on=['ens_id', 'masses_src', 'momentum'])
    df = pd.merge(df, df_snk, on=['ens_id', 'masses_snk'])

    # Include the physical restriction that the heavy mass (and corresponding
    # heavy meson) be on the sink leg of the 3pt function
    mask = (df['m_sink_to_current'] == df['m_heavy'])
    df = df[mask]
    return df

def verify_uniqueness(df):
    """
    Verfies that each row in df corresponds to a unique form factor.
    """
    # Check 1: Make sure the groupby is trivial
    cols = [
        'ens_id', 'momentum',
        'spin_taste_current', 'spin_taste_sink',
        'm_spectator', 'm_sink_to_current', 'm_source_to_current'
    ]
    groups = df.groupby(cols)
    for key, subdf in groups:
        if len(subdf) != 1:
            print("Bad key:")
            print(key)
            raise ValueError("Non-unique form factor encountered.")
    # Check 2: This check should be redundant, but I'm nervous
    if len(groups) != len(df):
        raise ValueError("Mismatched form factors.")

###############################################
# Functions for constructing transition names #
###############################################

def infer_transition_names(aliases):
    """
    Infer transition names given values for 'alias_light', 'alias_heavy', and 'alias_spectator'.
    Args:
        aliases: pd.DataFrame
    Returns:
        pd.DataFrame with columns 'daughter', 'mother', and 'process'
    """
    def identify_state(states, m1, m2):
        """Idenfity state based on input aliases for the masses m1 and m2."""
        # Try first ordering
        state = states.get((m1, m2))
        if state is not None:
            return state
        # Try second ordering
        state = states.get((m2, m1))
        if states is not None:
            return state
        raise ValueError("Unrecognized masses", m1, m2)

    def name_process(mother, daughter):
        """ e.g., ('D', 'pi) --> 'D to pi' """
        return "{0} to {1}".format(mother, daughter)

    # Define the quark masses in our dataset
    light_quarks = ['1.0 m_light', '0.1 m_strange', '0.2 m_strange']
    heavy_ratios = ['0.9', '1.0', '1.1', '1.4', '1.5', '2.0', '2.2', '2.5', '3.0', '3.3', '4.0', '4.2', '4.4']
    heavy_quarks = ['{0} m_charm'.format(rat) for rat in heavy_ratios]

    # Build up a dictionary of states (m1,m2) : state
    states = {}
    for light in light_quarks:
        states[(light, light)] = 'pi'
        states[(light, '1.0 m_strange')] = 'K'
        for heavy in heavy_quarks:
            states[(light, heavy)] = 'H'
    for heavy in heavy_quarks:
        states[('1.0 m_strange', heavy)] = 'Hs'

    # Compute the names of the mother and daughter particles and the process
    aliases['daughter'] = aliases[['alias_light', 'alias_spectator']].\
                            apply(lambda args: identify_state(states, *args), axis=1)
    aliases['mother'] = aliases[['alias_heavy', 'alias_spectator']].\
                            apply(lambda args: identify_state(states, *args), axis=1)
    aliases['process'] = aliases[['mother', 'daughter']].\
                            apply(lambda args: name_process(*args), axis=1)
    return aliases


if __name__ == '__main__':
    main()