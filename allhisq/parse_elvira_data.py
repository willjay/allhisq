"""
Parses Elvira's 2pt and 3pt data at q2=0 into the database and automation pipeline.
Ref: https://arxiv.org/pdf/1901.08989.pdf
Bare quark masses for input from
Ref: https://arxiv.org/pdf/1411.1651.pdf
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

    ##############################
    # Write meta_data_correlator #
    ##############################

    meta_data_correlator = pd.DataFrame([
        ["egamiz:l64192f211b700m00316m0158m188", "2ptDmom0", 0.1827, 0.00311, 'p000', False],
        ["egamiz:l64192f211b700m00316m0158m188", "2ptDsmom0", 0.1827, 0.01555, 'p000', False],
        ["egamiz:l64192f211b700m00316m0158m188", "2ptKmom0", 0.01555, 0.00311, 'p000', False],
        ["egamiz:l64192f211b700m00316m0158m188", "2ptKmom2", 0.01555, 0.00311, 'q2=0', False],
        ["egamiz:l64192f211b700m00316m0158m188", "2ptpimom0", 0.00311, 0.00311, 'p000', False],
        ["egamiz:l64192f211b700m00316m0158m188", "2ptpimom1", 0.00311, 0.00311, 'unknown', False],
        ["egamiz:l64192f211b700m00316m0158m188", "2ptpimom2", 0.00311, 0.00311, 'q2=0', False],
        ["egamiz:l96192f211b672m0008m022m260", "2ptDmom0", 0.26, 0.00077, 'p000', False],
        ["egamiz:l96192f211b672m0008m022m260", "2ptDsmom0", 0.26, 0.022, 'p000', False],
        ["egamiz:l96192f211b672m0008m022m260", "2ptKmom0", 0.022, 0.00077, 'p000', False],
        ["egamiz:l96192f211b672m0008m022m260", "2ptKmom2", 0.022, 0.00077, 'q2=0', False],
        ["egamiz:l96192f211b672m0008m022m260", "2ptpimom0", 0.00077, 0.00077, 'p000', False],
        ["egamiz:l96192f211b672m0008m022m260", "2ptpimom1", 0.00077, 0.00077, 'unknown', False],
        ["egamiz:l96192f211b672m0008m022m260", "2ptpimom2", 0.00077, 0.00077, 'q2=0', False],
        ["egamiz:l48144f211b672m0048m024m286", "2ptDmom0", 0.286, 0.0048, 'p000', False],
        ["egamiz:l48144f211b672m0048m024m286", "2ptKmom0", 0.024, 0.0048, 'p000', False],
        ["egamiz:l48144f211b672m0048m024m286", "2ptKmom2", 0.024, 0.0048, 'q2=0', False],
        ["egamiz:l48144f211b672m0048m024m286", "2ptpimom0", 0.0048, 0.0048, 'p000', False],
        ["egamiz:l48144f211b672m0048m024m286", "2ptpimom2", 0.0048, 0.0048, 'q2=0', False],
        ["egamiz:l6496f211b630m0012m0363m432", "2ptDmom0", 0.432, 0.0012705, 'p000', False],
        ["egamiz:l6496f211b630m0012m0363m432", "2ptKmom0", 0.0363, 0.0012705, 'p000', False],
        ["egamiz:l6496f211b630m0012m0363m432", "2ptKmom2", 0.0363, 0.0012705, 'q2=0', False],
        ["egamiz:l6496f211b630m0012m0363m432", "2ptpimom0", 0.0012705, 0.0012705, 'p000', False],
        ["egamiz:l6496f211b630m0012m0363m432", "2ptpimom1", 0.0012705, 0.0012705, 'unknown', False],
        ["egamiz:l6496f211b630m0012m0363m432", "2ptpimom2", 0.0012705, 0.0012705, 'q2=0', False],
        ["egamiz:l4896f211b630m00363m0363m430", "2ptDmom0", 0.43, 0.0038, 'p000', False],
        ["egamiz:l4896f211b630m00363m0363m430", "2ptKmom0", 0.038, 0.0038, 'p000', False],
        ["egamiz:l4896f211b630m00363m0363m430", "2ptKmom2", 0.038, 0.0038, 'q2=0', False],
        ["egamiz:l4896f211b630m00363m0363m430", "2ptpimom0", 0.0038, 0.0038, 'p000', False],
        ["egamiz:l4896f211b630m00363m0363m430", "2ptpimom2", 0.0038, 0.0038, 'q2=0', False],
        ["egamiz:l3296f211b630m0074m037m440", "2ptDmom0", 0.44, 0.0076, 'p000', False],
        ["egamiz:l3296f211b630m0074m037m440", "2ptKmom0", 0.038, 0.0076, 'p000', False],
        ["egamiz:l3296f211b630m0074m037m440", "2ptKmom1", 0.038, 0.0076, 'unknown', False],
        ["egamiz:l3296f211b630m0074m037m440", "2ptKmom2", 0.038, 0.0076, 'q2=0', False],
        ["egamiz:l3296f211b630m0074m037m440", "2ptpimom0", 0.0076, 0.0076, 'p000', False],
        ["egamiz:l3296f211b630m0074m037m440", "2ptpimom1", 0.0076, 0.0076, 'unknown', False],
        ["egamiz:l3296f211b630m0074m037m440", "2ptpimom2", 0.0076, 0.0076, 'q2=0', False],
        ["egamiz:l4864f211b600m00184m0507m628", "2ptDmom0", 0.6269, 0.0018585, 'p000', False],
        ["egamiz:l4864f211b600m00184m0507m628", "2ptKmom0", 0.0531, 0.0018585, 'p000', False],
        ["egamiz:l4864f211b600m00184m0507m628", "2ptKmom2", 0.0531, 0.0018585, 'q2=0', False],
        ["egamiz:l4864f211b600m00184m0507m628", "2ptpimom0", 0.0018585, 0.0018585, 'p000', False],
        ["egamiz:l4864f211b600m00184m0507m628", "2ptpimom2", 0.0018585, 0.0018585, 'q2=0', False],
        ["egamiz:l3264f211b600m00507m0507m628", "2ptDmom0", 0.650, 0.0053, 'p000', False],
        ["egamiz:l3264f211b600m00507m0507m628", "2ptKmom0", 0.053, 0.0053, 'p000', False],
        ["egamiz:l3264f211b600m00507m0507m628", "2ptKmom1", 0.053, 0.0053, 'unknown', False],
        ["egamiz:l3264f211b600m00507m0507m628", "2ptKmom2", 0.053, 0.0053, 'q2=0', False],
        ["egamiz:l3264f211b600m00507m0507m628", "2ptpimom0", 0.0053, 0.0053, 'p000', False],
        ["egamiz:l3264f211b600m00507m0507m628", "2ptpimom2", 0.0053, 0.0053, 'q2=0', False],
        ["egamiz:l2464f211b600m00507m0507m628", "2ptDmom0", 0.650, 0.0053, 'p000', False],
        ["egamiz:l2464f211b600m00507m0507m628", "2ptKmom0", 0.053, 0.0053, 'p000', False],
        ["egamiz:l2464f211b600m00507m0507m628", "2ptKmom2", 0.053, 0.0053, 'q2=0', False],
        ["egamiz:l2464f211b600m00507m0507m628", "2ptpimom0", 0.0053, 0.0053, 'p000', False],
        ["egamiz:l2464f211b600m00507m0507m628", "2ptpimom1", 0.0053, 0.0053, 'unknown', False],
        ["egamiz:l2464f211b600m00507m0507m628", "2ptpimom2", 0.0053, 0.0053, 'q2=0', False],
        ["egamiz:l4064f211b600m00507m0507m628", "2ptDmom0", 0.650, 0.0053, 'p000', False],
        ["egamiz:l4064f211b600m00507m0507m628", "2ptKmom0", 0.053, 0.0053, 'p000', False],
        ["egamiz:l4064f211b600m00507m0507m628", "2ptKmom2", 0.053, 0.0053, 'q2=0', False],
        ["egamiz:l4064f211b600m00507m0507m628", "2ptpimom0", 0.0053, 0.0053, 'p000', False],
        ["egamiz:l4064f211b600m00507m0507m628", "2ptpimom1", 0.0053, 0.0053, 'unknown', False],
        ["egamiz:l4064f211b600m00507m0507m628", "2ptpimom2", 0.0053, 0.0053, 'q2=0', False],
        ["egamiz:l2464f211b600m0102m0509m635", "2ptDmom0", 0.6363, 0.0107, 'p000', False],
        ["egamiz:l2464f211b600m0102m0509m635", "2ptKmom0", 0.0535, 0.0107, 'p000', False],
        ["egamiz:l2464f211b600m0102m0509m635", "2ptKmom1", 0.0535, 0.0107, 'unknown', False],
        ["egamiz:l2464f211b600m0102m0509m635", "2ptKmom2", 0.0535, 0.0107, 'q2=0', False],
        ["egamiz:l2464f211b600m0102m0509m635", "2ptpimom0", 0.0107, 0.0107, 'p000', False],
        ["egamiz:l2464f211b600m0102m0509m635", "2ptpimom1", 0.0107, 0.0107, 'unknown', False],
        ["egamiz:l2464f211b600m0102m0509m635", "2ptpimom2", 0.0107, 0.0107, 'q2=0', False],],
        columns=['ens_name', 'corr_name', 'quark_mass', 'antiquark_mass', 'momentum', 'has_sequential']
    )

    with sql.connection as conn:
        records = meta_data_correlator.to_dict(orient='records')
        sql.queries.write_meta_data_correlator(conn, records)

    alpha = pd.DataFrame([
        ["egamiz:l64192f211b700m00316m0158m188",'0.205(1)'],    # 0.042 fm
        ["egamiz:l96192f211b672m0008m022m260",  '0.2235(30)'],  # 0.057 fm
        ["egamiz:l48144f211b672m0048m024m286",  '0.2235(30)'],  # 0.057 fm
        ["egamiz:l6496f211b630m0012m0363m432",  '0.2641(43)'],  # 0.088 fm
        ["egamiz:l4896f211b630m00363m0363m430", '0.2641(43)'],  # 0.088 fm
        ["egamiz:l3296f211b630m0074m037m440",   '0.2641(43)'],  # 0.088 fm
        ["egamiz:l4864f211b600m00184m0507m628", '0.3068(61)'],  # 0.12 fm
        ["egamiz:l3264f211b600m00507m0507m628", '0.3068(61)'],  # 0.12 fm
        ["egamiz:l2464f211b600m00507m0507m628", '0.3068(61)'],  # 0.12 fm
        ["egamiz:l4064f211b600m00507m0507m628", '0.3068(61)'],  # 0.12 fm
        ["egamiz:l2464f211b600m0102m0509m635",  '0.3068(61)'],],# 0.12 fm
        columns=['ens_name', 'coupling_value']
    )
    with sql.connection as conn:
        records = alpha.to_dict(orient='records')
        sql.queries.write_strong_coupling(conn, records)

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

# ----- end main ----- #


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
            # This entry (probably) already exists.
            pass


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

