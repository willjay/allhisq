"""
Makes a minimal set of tables for an analysis database.
"""
import db_connection as db

def main():

    # For specifying ensembles, whose configuratiosn may reside externally
    # Meta data concerning the action, bare parameters, etc can be added in 
    # a new table later
    
    ensemble = """
    CREATE TABLE IF NOT EXISTS ensemble
    (
      ens_id serial PRIMARY KEY,
      name   text,    -- e.g., "HISQ 2+1+1"
      ns     integer, 
      nt     integer,
      UNIQUE(name, ns, nt)
    );"""

    # For specifying a correlator, whose data may reside externally
    # Meta data concerning sources, sinks, etc can be added in a new 
    # table later

    correlator_n_point = """
    CREATE TABLE IF NOT EXISTS correlator_n_point
    (
      corr_id   SERIAL PRIMARY KEY,
      ens_id    integer REFERENCES ensemble (ens_id),
      name      text NOT NULL DEFAULT '',
      UNIQUE(ens_id, name)
    );"""
        
    # For specifying the name of a particular analysis on a given ensemble
    # A given analysis typically includes several correlators, whose number
    # may vary. Correlators themselves are joined with the analysis below.
        
    correlator_analysis = """
    CREATE TABLE IF NOT EXISTS correlator_analysis
    (
      analysis_id   SERIAL PRIMARY KEY,
      ens_id        integer REFERENCES ensemble (ens_id),
      analysis_name text,
      UNIQUE(ens_id, analysis_name)
    );"""
    
    # Junction for linking correlators with a given analysis
    # Example: combining '-loose' and '-fine' solves in a single analysis
    
    junction_n_point_analysis = """
    CREATE TABLE IF NOT EXISTS junction_n_point_analysis
    (
      corr_analysis_id SERIAL PRIMARY KEY,
      corr_id          integer REFERENCES correlator_n_point (corr_id),
      analysis_id      integer REFERENCES correlator_analysis (analysis_id),
      UNIQUE(corr_id, analysis_id)
    );"""
    
    # The reduction from configuration-level to ensemble-level information
    # in a two-point correlation function
    
    reduction_two_point = """
    CREATE TABLE IF NOT EXISTS reduction_two_point
    (
      reduction_id   SERIAL PRIMARY KEY,
      ens_id integer REFERENCES ensemble (ens_id),
      fold           boolean NOT NULL, 
      block_length   integer NOT NULL,
      nsamples       integer NOT NULL,
      n_time_sources integer NOT NULL,
      log            text DEFAULT '',
      UNIQUE(fold, block_length, nsamples, n_time_sources)
    );"""

    rangefit_two_point = """
    CREATE TABLE IF NOT EXISTS rangefit_two_point
    (
      rangefit_id SERIAL PRIMARY KEY,
      tmin        integer,         -- Inclusive
      tmax        integer,         -- Exclusive
      prior_alias text DEFAULT '', -- Some quick nickname
      prior       text DEFAULT '', -- The prior itself
      UNIQUE(tmin, tmax, prior_alias)
    );"""
    
    result_two_point = """
    CREATE TABLE IF NOT EXISTS result_two_point
    ( 
      result_id    SERIAL PRIMARY KEY,
      ens_id       integer REFERENCES ensemble (ens_id),
      reduction_id integer REFERENCES reduction_two_point (reduction_id),
      analysis_id  integer REFERENCES correlator_analysis (analysis_id),
      rangefit_id  integer REFERENCES rangefit_two_point (rangefit_id),
      chi2    float,
      dof     integer,
      npoints integer,
      q       float,
      params  text,
      quality_factor float,
      log     text DEFAULT '',
      calcdate timestamp with time zone,
      UNIQUE(ens_id, reduction_id, analysis_id, rangefit_id)
    );"""

    campaign_two_point = """
    CREATE TABLE IF NOT EXISTS campaign_two_point
    (
      campaign_id SERIAL PRIMARY KEY, 
      result_id   integer REFERENCES result_two_point (result_id),
      ens_id      integer REFERENCES ensemble (ens_id), 
      analysis_id integer REFERENCES correlator_analysis (analysis_id),
      UNIQUE(ens_id, analysis_id)
    );"""

    systematic_two_point = """
    CREATE TABLE IF NOT EXISTS systematic_two_point
    (
      id SERIAL PRIMARY KEY,
      analysis_id integer REFERENCES correlator_analysis (analysis_id),
      systematic text NOT NULL,
      UNIQUE(analysis_id)
    );"""

    analysis_form_factor = """
    CREATE TABLE IF NOT EXISTS analysis_form_factor
    (
      analysis_id SERIAL PRIMARY KEY,
      ens_id integer REFERENCES ensemble (ens_id),
      form_factor_id integer REFERENCES form_factor(form_factor_id),
      -- nstates --
      n_decay_ll integer NOT NULL,
      n_decay_hl integer NOT NULL,
      n_oscillating_ll integer NOT NULL,
      n_oscillating_hl integer NOT NULL,
      -- tmin / tmax --
      tmin_ll integer NOT NULL,
      tmin_hl integer NOT NULL,
      tmax_ll integer NOT NULL,
      tmax_hl integer NOT NULL,
      UNIQUE(
        ens_id, form_factor_id,
        n_decay_ll, n_decay_hl,              
        n_oscillating_ll, n_oscillating_hl,
        tmin_ll, tmin_hl, 
        tmax_ll, tmax_hl)
    );"""
        
    form_factor = """
    CREATE TABLE IF NOT EXISTS form_factor
    (
      form_factor_id SERIAL PRIMARY KEY,
      ens_id integer REFERENCES ensemble(ens_id),
      momentum             text NOT NULL,
      spin_taste_current   text NOT NULL,
      spin_taste_sink      text NOT NULL,
      spin_taste_source    text NOT NULL,
      spectator_quark_mass float NOT NULL,
      antiquark_mass       float NOT NULL,
      heavy_mass           float NOT NULL,
      UNIQUE(ens_id, momentum,
             spin_taste_current, spin_taste_sink, spin_taste_source,
             spectator_quark_mass, antiquark_mass, heavy_mass)
    );
    """
   
    junction_form_factor = """
    CREATE TABLE IF NOT EXISTS junction_form_factor
    (
      form_factor_corr_id SERIAL PRIMARY KEY,
      form_factor_id integer REFERENCES form_factor(form_factor_id),
      corr_id integer REFERENCES correlator_n_point(corr_id),
      corr_type text NOT NULL,
      UNIQUE(form_factor_id, corr_id)
    );
    """
    
    result_form_factor = """
    CREATE TABLE IF NOT EXISTS result_form_factor
    (
      id SERIAL PRIMARY KEY,
      ens_id      integer REFERENCES ensemble (ens_id),
      analysis_id integer REFERENCES analysis_form_factor (analysis_id),
      prior  text,
      params text,
      nparams  integer,
      ndata    integer,
      chi2     float,
      chi2_aug float,
      p_value  float,
      Q        float,
      svdcut   float,
      calcdate timestamp with time zone,
      UNIQUE(ens_id, analysis_id)
    );"""
    
    external_database = """
    CREATE TABLE IF NOT EXISTS external_database
    (
      id SERIAL PRIMARY KEY,
      ens_id   integer REFERENCES ensemble (ens_id),
      name     text,
      location text,
      type     text,
      UNIQUE(ens_id, name)
    );"""
    
    analysis_queue = """
    CREATE TABLE IF NOT EXISTS analysis_queue
    (
      analysis_id integer PRIMARY KEY REFERENCES correlator_analysis (analysis_id),
      status text,
      UNIQUE(analysis_id)
    );"""
    
    queue_form_factor = """
    CREATE TABLE IF NOT EXISTS queue_form_factor
    (
      analysis_id integer PRIMARY KEY REFERENCES analysis_form_factor,
      status text,
      UNIQUE(analysis_id)
    );
    """
    
    # Meta data about the correlation functions
    # Typically stored in the external databases, but in a format which is hard to query
    
    meta_data_correlator = """
    CREATE TABLE IF NOT EXISTS meta_data_correlator
    (
        meta_correlator_id      SERIAL PRIMARY KEY,
        ens_id                  integer REFERENCES ensemble (ens_id),
        has_sequential          boolean,   -- Whether or not to look for sequential information
        -- Columns coming in pairs, quarks and antiquarks
        quark_type              text,      -- 'staggered'
        antiquark_type          text, 
        quark_mass              float,     -- 0.0012
        antiquark_mass          float,
        quark_source_ncolor     integer,   -- 1
        antiquark_source_ncolor integer,
        quark_source_type       text,      -- 'vector_field'
        antiquark_source_type   text, 
        quark_source_label      text,      -- 'RW', 'd'
        antiquark_source_label  text, 
        quark_source_origin     integer[], -- [0,0,0,72]
        antiquark_source_origin integer[], 
        quark_sink_label        text,      -- 'RW', 'd'
        antiquark_sink_label    text,
        quark_source_subset     text,      -- 'corner'
        antiquark_source_subset text,
        quark_source_file       text,      -- '/full/path/to/the/source'
        antiquark_source_file   text,
        quark_source_mom        integer[], -- [0,0,0]
        antiquark_source_mom    integer[],
        quark_epsilon           float,     -- '-0.2368'
        antiquark_epsilon       float,     
        -- Singleton columns 
        lattice_size            int[],     -- [64,64,64,96], but must convert!
        file_sequence           text,      -- What is this supposed to be?
        correlator              text,      -- 'P5-P5'
        momentum                text,      -- 'p000-fine'
        correlator_key          text,      -- 'P5-P5_RW_RW_d_d_m0.648_m0.0012_p000-fine'
        spin_taste_sink         text,      -- 'G5-G5'    
        UNIQUE(ens_id, correlator_key)
    );"""

    meta_data_sequential = """
    CREATE TABLE IF NOT EXISTS meta_data_sequential
    (
        meta_sequential_id      SERIAL PRIMARY KEY,
        meta_correlator_id      integer REFERENCES meta_data_correlator (meta_correlator_id),
        -- 'antiquark_sink_ops' for three-point functions
        mom                     integer[], -- [0,0,0]
        operation_0             text,      -- 'ext_src_ks'
        spin_taste_extend       text,      -- 'G5-G5',
        t0                      integer,   -- 87
        eps_naik                float,     -- -0.2368
        mass                    float,     -- 0.648
        momentum_twist          float[],   -- [0.0, 0.0, 0.0]
        operation_1             text,      -- 'ks_inverse'
        UNIQUE(meta_correlator_id)
    );"""
    
    glance_dataset_form_factor = """
    CREATE TABLE IF NOT EXISTS glance_dataset_form_factor
    (
        ds_id SERIAL PRIMARY KEY,
        -- unique columns --
        ens_id               integer REFERENCES ensemble(ens_id),
        p                    text  NOT NULL, -- e.g., 'p000'
        spectator_quark_mass float NOT NULL, 
        antiquark_mass       float NOT NULL,
        heavy_mass           float NOT NULL,
        -- data columns -- 
        T13 text,
        T14 text,
        T15 text,
        T16 text,
        hl  text,
        ll  text,
        UNIQUE(ens_id, p, spectator_quark_mass, antiquark_mass, heavy_mass)
    );"""
    
    glance_correlator_n_point = """
    CREATE TABLE IF NOT EXISTS glance_correlator_n_point
    (
        corr_id integer REFERENCES correlator_n_point(corr_id),
        data text,
        UNIQUE(corr_id)
    );"""
        
    pgd_mesons = """
    CREATE TABLE IF NOT EXISTS pdg_mesons
    (
        p_id SERIAL PRIMARY KEY,
        name text NOT NULL,
        mass_mev float NOT NULL,
        quantum_numbers text,
        quark_content text NOT NULL,
        i float, -- isospin 
        g text,  -- g-parity (+/-)
        j float, -- spin
        p text,  -- parity (+/-)
        c text,  -- charge conjugation (+/-_
        UNIQUE(name, mass_MeV)
    );
    """

    pdg_quarks = """
    CREATE TABLE IF NOT EXISTS pdg_quarks
    (
        p_id SERIAL PRIMARY KEY,
        name text NOT NULL,
        mass_mev float NOT NULL,
        UNIQUE(name, mass_mev)
    );
    """
        
    lattice_spacing = """
    CREATE TABLE IF NOT EXISTS lattice_spacing
    (
        id SERIAL PRIMARY KEY,
        ens_id integer REFERENCES ensemble(ens_id),
        a_fm   float NOT NULL,
        type   text  NOT NULL, -- e.g., 'nominal' or 'r1' or 'wilson flow'
        UNIQUE(ens_id, type)
    );
    """
          
    pdg_prior = """
    CREATE TABLE IF NOT EXISTS pdg_prior
    (
        pdg_prior_id SERIAL PRIMARY KEY,
        ens_id     integer REFERENCES ensemble(ens_id),
        corr_id    integer REFERENCES two_point(corr_id)
        -- 
        momentum   text  NOT NULL,
        m1a        float NOT NULL, -- quark mass, lattice units
        m2a        float NOT NULL, -- quark mass, lattice units
        m1_mev     float NOT NULL, -- quark mass, MeV
        m2_mev     float NOT NULL, -- quark mass, MeV
        m1_type    text REFERENCES pdg_quarks(name), -- quark type 
        m2_type    text REFERENCES pdg_quarks(name), -- quark type
        meson_name text REFERENCES pdg_mesons(name), -- quark type
        meson_ma  float NOT NULL, -- meson mass, lattice units
        meson_mev float NOT NULL, -- meson mass, MeV
        is_boosted      boolean NOT NULL, -- is the state boosted?
        is_physical     boolean NOT NULL, -- is the state at the physical point?
        CONSTRAINT ordered_masses CHECK (m1a <= m2a),
        UNIQUE(corr_id)
    );
    """
        
    engine = db.make_engine()

    queries = [
#         ensemble,
#         correlator_n_point,
#         correlator_analysis,
#         junction_n_point_analysis,
#         reduction_two_point,
#         rangefit_two_point,
#         result_two_point,
#         campaign_two_point,
#         systematic_two_point,
#         external_database,
#         analysis_queue,
#         meta_data_correlator,
#         meta_data_sequential,
#         analysis_form_factor,
#         result_form_factor,
#         queue_form_factor,
#         glance_dataset_form_factor
#         form_factor,
#         glance_correlator_n_point
    ]
    for query in queries:
        print(query)
        engine.execute(query)


if __name__ == '__main__':
    main()
