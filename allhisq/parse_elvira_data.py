"""
Ref: https://arxiv.org/pdf/1901.08989.pdf

Bare quark masses for input from
Ref: https://arxiv.org/pdf/1411.1651.pdf


Need entries in tables:
DONE: ensemble
DONE: lattice_spacing
strong_coupling --> can probably just copy the table with appropriate tweaks
DONE: form_factor
DONE: alias_form_factor
DONE: sign_form_factor
campaign_results_two_point --> Must be done at the level of fits
    Hmm. Actually we need the ids:
        * result_id serial PRIMARY KEY,
        * ens_id integer REFERENCES ensemble(ens_id),
        * corr_id integer REFERENCES correlator_n_point(corr_id),
    --> at the very least, we'll need to write entires for correlator_n_point
    Fortunately, these should be easy, since all we need to supply are
        * ens_id
        * name






for db_io.get_form_factor_data, we need entries in tables:
junction_form_factor
correlator_n_point
Hmm. Since this function also aliases the correlators, it might make sense
simply to write a new function. Perhaps something like 'get_form_factor_data_elvira'
which would return a data dictionary to match the original function.
This would mean that...??

"""
import pandas as pd
import numpy as np
import re
import os
import yaml
import sqlalchemy as sqla
import aiosql
import conventions
import h5py
import datetime
from db_tools import abspath
from db_tools import connection_scope

def main():

    with open("db_settings.yaml") as ifile:
        db_settings = yaml.load(ifile, yaml.SafeLoader)
    sql = SQLWrapper(db_settings)

    # Location for Elvira's data files
    base = "/home/egamiz/CorrDtoP"

    # Location for saving HDF5 files
    obase = "/home/wjay/dbs/allHISQ/tmp_cache/Elvira/FromElvira"

    ######################
    # Register ensembles #
    ######################

    # ens_id should be understood as a "nominal" ens_id for the corresponding data generated in the allhisq campaign
    # The actual details (like the number of configurations or the precise values of the bare masses) can differ.
    ensembles = pd.DataFrame([
        [36,   0.042, "1/5",  "egamiz:l64192f211b700m00316m0158m188", 64, 192, "a045m00316b1.dat"],
        [28,   0.057, "1/27", "egamiz:l96192f211b672m0008m022m260",   96, 192, "a06m0008b1.dat"],
        [35,   0.057, "1/5",  "egamiz:l48144f211b672m0048m024m286",   48, 144, "a06m0048b1.dat"],
        [15,   0.088, "1/27", "egamiz:l6496f211b630m0012m0363m432",   64, 96, "a09m0012b1.dat"],
        [13,   0.088, "1/10", "egamiz:l4896f211b630m00363m0363m430",  48, 96, "a09m00363b1.dat"],
        [None, 0.088, "1/5",  "egamiz:l3296f211b630m0074m037m440",    32, 96, "a09m0074b1.dat"],
        [25,   0.12,  "1/27", "egamiz:l4864f211b600m00184m0507m628",  48, 64, "a12m00184b1.dat"],
        [14,   0.12,  "1/10", "egamiz:l3264f211b600m00507m0507m628",  32, 64, "a12m00507b1.dat"], # corresponding allhisq data never analyzed?
        [None, 0.12,  "1/10", "egamiz:l2464f211b600m00507m0507m628",  24, 64, "a12m00507_small_b1.dat"],
        [None, 0.12,  "1/10", "egamiz:l4064f211b600m00507m0507m628",  40, 64, "a12m00507_large_b1.dat"],
        [None, 0.12,  "1/5",  "egamiz:l2464f211b600m0102m0509m635",   24, 64, "a12m0102b1.dat"],],
        columns=["nominal_ens_id", "a_fm", "description", "name", "ns", "nt", "fname"]
    )

    with sql.connection as conn:
        for record in ensembles.to_dict(orient='records'):
            record['ens_id'] = sql.queries.write_ensemble(conn, **record)
            sql.queries.write_lattice_spacing(conn, **record)

    #########################
    # Register form_factors #
    #########################

    form_factors = pd.DataFrame([
        # "l64192f211b700m00316m0158m188" -- 1/5
        ["egamiz:l64192f211b700m00316m0158m188", "q2=0", "S-S", "P5-P5", "P5-P5", 0.00311, 0.1827, 0.00311, 'D to pi'], # D2pi
        ["egamiz:l64192f211b700m00316m0158m188", "q2=0", "S-S", "P5-P5", "P5-P5", 0.00311, 0.1827, 0.01555, 'D to K'], # D2K
        # "l96192f211b672m0008m022m260" -- 1/27
        # TODO: Check these lines!
        ["egamiz:l96192f211b672m0008m022m260", "q2=0", "S-S", "P5-P5", "P5-P5", 0.00077, 0.26, 0.0008, 'D to pi'], # D2pi
        ["egamiz:l96192f211b672m0008m022m260", "q2=0", "S-S", "P5-P5", "P5-P5", 0.00077, 0.26, 0.022, 'D to K'], # D2K
        # "l48144f211b672m0048m024m286" -- 1/5
        ["egamiz:l48144f211b672m0048m024m286", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0048, 0.286, 0.0048, 'D to pi'], # D2pi
        ["egamiz:l48144f211b672m0048m024m286", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0048, 0.286, 0.024, 'D to K'], # D2K
        # "l6496f211b630m0012m0363m432" -- 1/27
        ["egamiz:l6496f211b630m0012m0363m432", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0012705, 0.432, 0.0012705, 'D to pi'], # D2pi
        ["egamiz:l6496f211b630m0012m0363m432", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0012705, 0.432, 0.0363, 'D to K'], # D2K
        # "l4896f211b630m00363m0363m430" -- 1/10
        ["egamiz:l4896f211b630m00363m0363m430", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0038, 0.43, 0.0038, 'D to pi'], # D2pi
        ["egamiz:l4896f211b630m00363m0363m430", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0038, 0.43, 0.038, 'D to K'], # D2K
        # "l3296f211b630m0074m037m440" -- "1/5"
        ["egamiz:l3296f211b630m0074m037m440", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0076, 0.44, 0.0076, 'D to pi'], # D2pi
        ["egamiz:l3296f211b630m0074m037m440", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0076, 0.44, 0.038, 'D to K'], # D2K
        # "l4864f211b600m00184m0507m628" -- "1/27"
        ["egamiz:l4864f211b600m00184m0507m628", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0018585, 0.6269, 0.0018585, 'D to pi'], # D2pi
        ["egamiz:l4864f211b600m00184m0507m628", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0018585, 0.6269, 0.0531, 'D to K'], # D2K
        # "l3264f211b600m00507m0507m628" -- "1/10"
        ["egamiz:l3264f211b600m00507m0507m628", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0053, 0.650, 0.0053, 'D to pi'], # D2pi
        ["egamiz:l3264f211b600m00507m0507m628", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0053, 0.650, 0.053, 'D to K'], # D2K
        # "l2464f211b600m00507m0507m628" -- "1/10" small
        ["egamiz:l2464f211b600m00507m0507m628", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0053, 0.650, 0.0053, 'D to pi'], # D2pi
        ["egamiz:l2464f211b600m00507m0507m628", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0053, 0.650, 0.053, 'D to K'], # D2K
        # "l4064f211b600m00507m0507m628" -- "1/10" big
        ["egamiz:l4064f211b600m00507m0507m628", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0053, 0.650, 0.0053, 'D to pi'], # D2pi
        ["egamiz:l4064f211b600m00507m0507m628", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0053, 0.650, 0.053, 'D to K'], # D2K
        # "l2464f211b600m0102m0509m635" -- "1/5"
        ["egamiz:l2464f211b600m0102m0509m635", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0107, 0.6363, 0.0107, 'D to pi'], # D2pi
        ["egamiz:l2464f211b600m0102m0509m635", "q2=0", "S-S", "P5-P5", "P5-P5", 0.0107, 0.6363, 0.0535, 'D to K'], # D2K
        ],
        columns=[
            "name", "momentum", "spin_taste_current", "spin_taste_sink", "spin_taste_source",
            "m_spectator", "m_heavy", "m_light", "process"]
    )
    form_factors['m_sink_to_current'] = form_factors['m_heavy']
    form_factors['mother'] = form_factors['process'].apply(lambda astr: astr.split()[0])
    form_factors['daughter'] = form_factors['process'].apply(lambda astr: astr.split()[2])
    form_factors = pd.merge(form_factors, ensembles, on="name")
    with sql.connection as conn:
        records = form_factors.to_dict(orient='records')
        sql.queries.write_form_factor(conn, records)
        sql.queries.write_transition_name(conn, records)

    ##################################
    # Write aliases for quark masses #
    ##################################

    aliases = pd.DataFrame([
        # "l64192f211b700m00316m0158m188" -- 1/5
        ["egamiz:l64192f211b700m00316m0158m188", 0.00311, "0.2 m_strange"],
        ["egamiz:l64192f211b700m00316m0158m188", 0.01555, "1.0 m_strange"],
        ["egamiz:l64192f211b700m00316m0158m188", 0.1827,  "1.0 m_charm"],
        # "l96192f211b672m0008m022m260" -- 1/27
        ["egamiz:l96192f211b672m0008m022m260",  0.00077, "1.0 m_light"], # TODO: check!
        ["egamiz:l96192f211b672m0008m022m260",  0.0008, "1.0 m_light"], # TODO: check!
        ["egamiz:l96192f211b672m0008m022m260",  0.022, "1.0 m_strange"],
        ["egamiz:l96192f211b672m0008m022m260",  0.26, "1.0 m_charm"],
        # "l48144f211b672m0048m024m286" -- 1/5
        ["egamiz:l48144f211b672m0048m024m286", 0.0048, "0.2 m_strange"],
        ["egamiz:l48144f211b672m0048m024m286", 0.024, "1.0 m_strange"],
        ["egamiz:l48144f211b672m0048m024m286", 0.286, "1.0 m_charm"],
        # "l6496f211b630m0012m0363m432" -- 1/27
        ["egamiz:l6496f211b630m0012m0363m432",  0.0012705, "1.0 m_light"],
        ["egamiz:l6496f211b630m0012m0363m432",  0.0363, "1.0 m_strange"],
        ["egamiz:l6496f211b630m0012m0363m432",  0.432, "1.0 m_charm"],
        # "l4896f211b630m00363m0363m430" -- 1/10
        ["egamiz:l4896f211b630m00363m0363m430",  0.0038, "0.1 m_strange"],
        ["egamiz:l4896f211b630m00363m0363m430",  0.038,  "1.0 m_strange"],
        ["egamiz:l4896f211b630m00363m0363m430",  0.43,   "1.0 m_charm"],
        # "l3296f211b630m0074m037m440" -- "1/5"
        ["egamiz:l3296f211b630m0074m037m440",  0.0076, "0.2 m_strange"],
        ["egamiz:l3296f211b630m0074m037m440",  0.038, "1.0 m_strange"],
        ["egamiz:l3296f211b630m0074m037m440",  0.44, "1.0 m_charm"],
        # "l4864f211b600m00184m0507m628" -- "1/27"
        ["egamiz:l4864f211b600m00184m0507m628",  0.0018585, "1.0 m_light"],
        ["egamiz:l4864f211b600m00184m0507m628",  0.0531, "1.0 m_strange"],
        ["egamiz:l4864f211b600m00184m0507m628",  0.6269, "1.0 m_charm"],
        # "l3264f211b600m00507m0507m628" -- "1/10"
        ["egamiz:l3264f211b600m00507m0507m628",  0.0053, "0.1 m_strange"],
        ["egamiz:l3264f211b600m00507m0507m628",  0.053, "1.0 m_strange"],
        ["egamiz:l3264f211b600m00507m0507m628",  0.650, "1.0 m_charm"],
        # "l2464f211b600m00507m0507m628" -- "1/10" small
        ["egamiz:l2464f211b600m00507m0507m628",  0.0053, "0.1 m_strange"],
        ["egamiz:l2464f211b600m00507m0507m628",  0.053, "1.0 m_strange"],
        ["egamiz:l2464f211b600m00507m0507m628",  0.650, "1.0 m_charm"],
        # "l4064f211b600m00507m0507m628" -- "1/10" big
        ["egamiz:l4064f211b600m00507m0507m628",  0.0053, "0.1 m_strange"],
        ["egamiz:l4064f211b600m00507m0507m628",  0.053, "1.0 m_strange"],
        ["egamiz:l4064f211b600m00507m0507m628",  0.650, "1.0 m_charm"],
        # "l2464f211b600m0102m0509m635" -- "1/5"
        ["egamiz:l2464f211b600m0102m0509m635",  0.0107, "0.2 m_strange"],
        ["egamiz:l2464f211b600m0102m0509m635",  0.0535, "1.0 m_strange"],
        ["egamiz:l2464f211b600m0102m0509m635",  0.6363, "1.0 m_charm"],
        ],
        columns=['name', 'mq', 'alias']
    )
    with sql.connection as conn:
        sql.queries.write_alias_quark_mass(conn, aliases.to_dict(orient='records'))

    ##########################
    # Write sign form factor #
    ##########################

    # All of Elvira's data is for the scalar current only
    form_factor_signs = conventions.form_factor_signs
    mask = form_factor_signs['spin_taste_current'] == 'S-S'
    record, = form_factor_signs[mask].to_dict(orient='records')  # Note that this is a single row!
    with sql.connection as conn:
        for name in ensembles['name']:
            record['name'] = name
            sql.queries.write_sign_form_factor(conn, **record)

    ############################
    # Write correlator_n_point #
    ############################

    corrs = []
    for _, row in ensembles.iterrows():
        name = row['name']
        fname = os.path.join(base, row['fname'])
        data = read_data(fname)
        for corr_name in data.keys():
            corrs.append({
                'ens_name': row['name'],
                'corr_name': corr_name,
            })
    with sql.connection as conn:
        sql.queries.write_correlator_n_point(conn, corrs)

    ##############################
    # Write junction_form_factor #
    ##############################

    corrs = pd.DataFrame(corrs)
    corrs['corr_type'] = corrs['corr_name'].apply(infer_corr_type)

    # Isolate the 2pt functions necessary for studying q2=0
    mask = corrs['corr_type'].isin(('light-light', 'heavy-light'))
    corr2 = pd.DataFrame(corrs[mask])
    corr2['state'] = corr2['corr_name'].apply(identify_state)
    corr2['q2=0'] =\
        (corr2['state'].isin(('K', 'pi')) & corr2['corr_name'].str.contains('mom2')) |\
        (corr2['state'].isin(('D', 'Ds')) & corr2['corr_name'].str.contains('mom0'))
    corr2 = corr2[corr2['q2=0']]

    # Isolate the 3pt functions necessary for studyign q2=0
    mask = (corrs['corr_type'] == 'three-point')
    corr3 = pd.DataFrame(corrs[mask])
    corr3['tokens'] = corr3['corr_name'].apply(identify_transition)
    for key in ['mother', 'daughter', 'q2=0']:
        corr3[key] = corr3['tokens'].apply(lambda adict: adict[key])
    corr3 = corr3[corr3['q2=0']]
    corr3.drop(columns='tokens', inplace=True)

    # Combine the 2pt and 3pt functions belonging together in the junction
    for (ens_name, mother, daughter), subdf in corr3.groupby(['ens_name', 'mother', 'daughter']):
        print("Writing", ens_name, mother, daughter)
        mask = (corr2['ens_name'] == ens_name) & corr2['state'].isin((mother, daughter))
        cols = ['corr_type', 'corr_name']
        payload = pd.concat([subdf[cols], corr2[mask][cols]])
        payload['process'] = f"{mother} to {daughter}"
        payload['ens_name'] = ens_name
        with sql.connection as conn:
            sql.queries.write_junction_form_factor(conn, payload.to_dict(orient='records'))

    ########################
    # Convert data to HDF5 #
    ########################

    for _, row in ensembles.iterrows():
        ens_name = row['name'].replace(":", "_")
        fname = os.path.join(base, row['fname'])
        h5fname = os.path.join(obase, f"{ens_name}.hdf5")
        for basename, data in read_data(fname).items():
            h5write(h5fname, basename, data)

        with sql.connection as conn:
            record = {
                "ens_name": row['name'],
                "name": row['name'].replace(":", "_"),
                "location": obase,
            }
            sql.queries.write_external_database(conn, **record)




def h5write(h5fname, basename, data):

    with h5py.File(h5fname, mode='a') as h5file:
        # keys = list(h5fname['data'].keys())
        # if basename not in keys:
        # Load into hdf5 dataset
        dpath = 'data/{0}'.format(basename)
        try:
            dset = h5file.create_dataset(name=dpath, data=data, compression='gzip', shuffle=True)
            # Store meta data as dataset "attributes"
            dset.attrs['basename'] = basename
            dset.attrs['calcdate'] = datetime.datetime.now().isoformat()
        except ValueError:
            pass

    # corr2 = pd.DataFrame(corrs[corrs['corr_type'].isin(('light-light','heavy-light'))])
    # corr3 = pd.DataFrame(corrs[corrs['corr_type']=='three-point'])

    # print(corr2)

    # corr2['state'] = corrs['corr_name'].apply(identify_state)

    # print(corr2['state'].unique())

    # form_factor_id integer REFERENCES form_factor(form_factor_id),
    # corr_id integer REFERENCES correlator_n_point(corr_id),
    # corr_type text NOT NULL,
    # UNIQUE(form_factor_id, corr_id, corr_type)





# ----- end main ----- #

def make_engine(settings, echo=False):
    url = 'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
    return sqla.create_engine(url.format(**dict(settings)), echo=echo)


class SQLWrapper:
    """Wrapper class for basic database I/O operations."""
    def __init__(self, settings, echo=False):
        self.engine = make_engine(settings)
        self.queries = aiosql.from_path(abspath("sql/wjay/"), "psycopg2")

    @property
    def connection(self):
        """Wrapper for accessing the context manager for database connection."""
        return connection_scope(self.engine)



# 2ptDmom0
# 2ptpimom2
# 2ptpimom1
# 2ptpimom0
# 2ptKmom0
# 2ptKmom1
# 2ptKmom2

# 3ptDKT15
# 3ptDKT18
# 3ptDKT20
# 3ptDKT21
# 3ptDKT22

# 3ptDPT15
# 3ptDPT18
# 3ptDPT20
# 3ptDPT21
# 3ptDPT22

# 3ptDPm0T15
# 3ptDPm0T18
# 3ptDPm0T20
# 3ptDPm0T21
# 3ptDPm0T22

# 3ptDKm0T15
# 3ptDKm0T18
# 3ptDKm0T20
# 3ptDKm0T21
# 3ptDKm0T22

# Infer 3pt functions
# regex = re.compile(r"^3ptDPT(\d\d?)$")
# for key in data:
#     match = regex.match(key)
#     if match:
#         T, = match.groups()
#         key_map[key] = int(T)

# tmp = {}

def infer_corr_type(key):
    valid_names = {
        'light-light': re.compile(r"^2pt(pi|K)mom(\d)$"),
        'heavy-light': re.compile(r"^2ptD(s?)mom(\d)$"),
        'three-point': re.compile(r"^3pt(D|K)(P|K)((m0)?)T(\d\d?)$")
    }
    for name, regex in valid_names.items():
        if regex.match(key):
            return name
    raise ValueError("Unrecognized correlator name", key)

def identify_state(key):
    regex = re.compile(r"^2pt(pi|K|D|Ds)mom(\d)$")
    match = regex.match(key)
    if not match:
        raise ValueError("Unable to identify state from", key)
    state = match.groups()[0]
    return state

def identify_transition(key):
    regex = re.compile(r"^3pt(D|K)(P|K)((m0)?)T(\d\d?)$")
    match = regex.match(key)
    if not match:
        raise ValueError("Unable to identify transition from", key)
    tokens = match.groups()
    mother = tokens[0]
    daughter = tokens[1]
    if daughter == 'P':
        daughter = 'pi'
    if tokens[2] == 'm0':
        # this is daughter at rest, so q2=q2max
        keep = False
    else:
        keep = True
    return {
        'mother': mother,
        'daughter': daughter,
        'q2=0': keep,
    }



    # if light_light.match(key):
    #     return light_light



    # pion_2pt = re.compile(r"^2ptpimom(\d)$")
    # kaon_2pt = re.compile(r"^2ptKmom(\d)$")
    # d_2pt = re.compile(r"^2ptDmom(\d)$")

    # if pion_2pt.match(key) or kaon_2pt.match(key):
    #     return 'light-light'

    # if d_2pt.match(key):
    #     return 'heavy_light'

    # # 3pt functions
    # d2pi_3pt_q2zero = re.compile(r"^3ptD(P|K)(m0?)T(\d\d?)$")

    # d2pi_3pt_q2zero = re.compile(r"^3ptDPT(\d\d?)$")
    # d2k_3pt_q2zero = re.compile(r"^3ptDKT(\d\d?)$")
    # regex = re.compile(r"^3ptDKm0T(\d\d?)$")
    # regex = re.compile(r"^3ptDPm0T(\d\d?)$")

    # if

    # , d_2pt]:
    #     if regex.match(key):
    #         return
    #     if match:




    # 3pt functions for d2pi and d2k at q2=0

    # # D2pi 3pt functions, q2=q2max
    # regex = re.compile(r"^3ptDKm0T(\d\d?)$")

    # # D2K 3pt functions, q2=q2max
    # regex = re.compile(r"^3ptDPm0T(\d\d?)$")


def rename(key, masses):
    """
    2pt functions have names like
    A4-A4_RW_RW_d_d_m0.389_m0.0363_p000-fine
    P5-P5_RW_RW_d_d_m0.0012_m0.0012_p000-fine
    Note that

    For the pion and kaon 2pt functions, the following tags are used
    * mom0 = pion at rest
    * mom1 = pion with momentum such that q2=0 for K2pi
    * mom2 = pion with momentum such that q2=0 for D2pi


    """
    # template_2pt = "P5-P5_RW_RW_d_d_m{mass1}_mass{2}_q2=0"


    # 2pt functions
    pion_2pt = re.compile(r"^2ptpimom(\d)$")
    kaon_2pt = re.compile(r"^2ptKmom(\d)$")
    d_2pt = re.compile(r"^2ptDmom(\d)$")


    # 3pt functions for d2pi and d2k at q2=0
    d2pi_3pt = re.compile(r"^3ptDPT(\d\d?)$")
    d2k_3pt = re.compile(r"^3ptDKT(\d\d?)$")





    # OK to skip data at q2=q2max?

    # # kaon 2pt functions
    # # mom0 = kaon at rest
    # # mom1 = kaon with momentum such that q2=0 for ???
    # # mom2 = kaon with momentum such that q2=0 for D2K
    # kaon_2pt = re.compile(r"^2ptKmom(\d)$")


    # # D2pi 3pt functions, q2=q2max
    # regex = re.compile(r"^3ptDKm0T(\d\d?)$")

    # # D2K 3pt functions, q2=q2max
    # regex = re.compile(r"^3ptDPm0T(\d\d?)$")


def read_data(fname):
    """
    Reads data from file. Assumes data appears in space-deliminated rows
    with the first column identifying the data datum with a "key".
    Args:
        fname: str, the full path to the file
    Returns:
        data: dict
    """
    data = {}
    with open(fname, 'r') as ifile:
        for line in ifile:
            tokens = line.split()
            key = tokens[0]
            datum = np.array(tokens[1:], dtype=float)
            if key in data:
                data[key].append(datum)
            else:
                data[key] = [datum]

    for key in data:
        data[key] = np.array(data[key])
    return data




if __name__ == '__main__':
    main()

