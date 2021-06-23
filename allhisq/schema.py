"""
Makes a minimal set of tables for an analysis database.
"""
from . import db_connection as db

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
      corr_analysis_id SERIAL PRIMARY KEY,
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
      analysis_id      integer REFERENCES correlator_analysis (corr_analysis_id),
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
      binsize        integer NOT NULL,
      nsamples       integer NOT NULL,
      n_time_sources integer NOT NULL,
      shrinkage      text DEFAULT 'None',
      log            text DEFAULT '',
      UNIQUE(fold, binsize, nsamples, n_time_sources, shrinkage)
    );"""

    analysis_two_point = """
    CREATE TABLE IF NOT EXISTS analysis_two_point
    (
      analysis_id SERIAL PRIMARY KEY,
      tmin        integer,         -- Inclusive
      tmax        integer,         -- Exclusive
      n           integer,         -- Number of decaying states
      no          integer,         -- Number of oscillating states
      prior_alias text DEFAULT '', -- Some quick nickname
      UNIQUE(tmin, tmax, n, no, prior_alias)
    );"""

    result_two_point = """
    CREATE TABLE IF NOT EXISTS result_two_point
    (
      result_id        SERIAL PRIMARY KEY,
      ens_id           integer REFERENCES ensemble (ens_id),
      reduction_id     integer REFERENCES reduction_two_point (reduction_id),
      analysis_id      integer REFERENCES analysis_two_point (analysis_id),
      corr_analysis_id integer REFERENCES correlator_analysis (corr_analysis_id),
      chi2    float,
      dof     integer,
      npoints integer,
      q       float,
      params  text,
      prior   text,
      quality_factor float,
      log     text DEFAULT '',
      calcdate timestamp with time zone,
      UNIQUE(ens_id, reduction_id, analysis_id, corr_analysis_id)
    );"""

    campaign_two_point = """
    CREATE TABLE IF NOT EXISTS campaign_two_point
    (
      campaign_two_point_id SERIAL PRIMARY KEY,
      corr_analysis_id integer REFERENCES correlator_analysis (corr_analysis_id),
      result_id   integer REFERENCES result_two_point (result_id),
      criterion   text DEFAULT '',
      automated   boolean NOT NULL,
      UNIQUE(corr_analysis_id)
    );"""

    systematic_two_point = """
    CREATE TABLE IF NOT EXISTS systematic_two_point
    (
      id SERIAL PRIMARY KEY,
      corr_analysis_id integer REFERENCES correlator_analysis (corr_analysis_id),
      systematic text NOT NULL,
      UNIQUE(corr_analysis_id)
    );"""

    reduction_form_factor = """
    CREATE TABLE IF NOT EXISTS reduction_form_factor
    (
      reduction_id   SERIAL PRIMARY KEY,
      ens_id integer REFERENCES ensemble (ens_id),
      fold           boolean NOT NULL,
      binsize        integer NOT NULL,
      nsamples       integer NOT NULL,
      n_time_sources integer NOT NULL,
      shrinkage      text DEFAULT 'None',
      log            text DEFAULT '',
      UNIQUE(fold, binsize, nsamples, n_time_sources, shrinkage)
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
      prior_alias text NOT NULL,
      UNIQUE(
        ens_id, form_factor_id,
        n_decay_ll, n_decay_hl,
        n_oscillating_ll, n_oscillating_hl,
        tmin_ll, tmin_hl,
        tmax_ll, tmax_hl,
        prior_alias)
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
      m_spectator          float NOT NULL,
      m_heavy              float NOT NULL,
      m_light              float NOT NULL,
      m_sink_to_current    float NOT NULL,
      UNIQUE(ens_id, momentum,
             spin_taste_current, spin_taste_sink, spin_taste_source,
             m_spectator, m_heavy, m_light, m_sink_to_current)
    );
    """

    junction_form_factor = """
    CREATE TABLE IF NOT EXISTS junction_form_factor
    (
      form_factor_corr_id SERIAL PRIMARY KEY,
      form_factor_id integer REFERENCES form_factor(form_factor_id),
      corr_id integer REFERENCES correlator_n_point(corr_id),
      corr_type text NOT NULL,
      UNIQUE(form_factor_id, corr_id, corr_type)
    );
    """

    result_form_factor = """
    CREATE TABLE IF NOT EXISTS result_form_factor
    (
      id SERIAL PRIMARY KEY,
      ens_id         integer REFERENCES ensemble (ens_id),
      analysis_id    integer REFERENCES analysis_form_factor (analysis_id),
      form_factor_id integer REFERENCES form_factor (form_factor_id),
      reduction_id   integer REFERENCES reduction_form_factor (reduction_id),
      prior  text,
      params text,
      nparams  integer,
      npoints  integer,
      chi2     float,
      q        float,
      log      text,
      r        text,
      r_guess  text,
      normalization text,
      calcdate timestamp with time zone,
      UNIQUE(ens_id, analysis_id, form_factor_id, reduction_id)
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
      corr_analysis_id integer PRIMARY KEY REFERENCES correlator_analysis (corr_analysis_id),
      status text,
      UNIQUE(corr_analysis_id)
    );"""

    queue_form_factor = """
    CREATE TABLE IF NOT EXISTS queue_form_factor
    (
      queue_id SERIAL PRIMARY KEY,
      form_factor_id integer REFERENCES form_factor (form_factor_id),
      status text,
      UNIQUE(form_factor_id)
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
        form_factor_id       integer REFERENCES form_factor(form_factor_id),
        -- data columns --
        T13 text,
        T14 text,
        T15 text,
        T16 text,
        hl  text,
        ll  text,
        UNIQUE(ens_id, form_factor_id)
    );"""

    glance_correlator_n_point = """
    CREATE TABLE IF NOT EXISTS glance_correlator_n_point
    (
        corr_id integer REFERENCES correlator_n_point(corr_id),
        data text,
        nconfigs         int  NOT NULL, -- the total number of configurations used
        fine_only        bool NOT NULL, -- whether or not this data just used the fine solves
        tsrc_combination text NOT NULL, -- e.g., "tsm", if applicable
        nfine_per_config int  NOT NULL, -- the number of fine solves per configuration
        UNIQUE(corr_id, nconfigs, fine_only)
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
        c text,  -- charge conjugation (+/-)
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

    three_point = """
    CREATE VIEW three_point AS (
        SELECT
            corr.ens_id,
            corr.corr_id,
            corr.name,
            -- Currents --
            split_part(meta.correlator, '_'::text, 1) AS spin_taste_sink,
            split_part(meta.correlator, '_'::text, 2) AS spin_taste_current,
            -- Masses --
            GREATEST(meta.quark_mass, meta_seq.mass) AS m_heavy,
            LEAST(meta.quark_mass, meta_seq.mass)    AS m_light,
            meta.antiquark_mass                      AS m_spectator,
            meta_seq.mass                            AS m_sink_to_current,
            meta.quark_mass                          AS m_source_to_current,
            -- Momentum --
            split_part(meta.momentum, '-'::text, 1) AS momentum,
            -- Sink time T --
            CAST(TRIM(LEADING 'T' FROM split_part(corr.name, '_'::text, 3)) AS integer) AS t_sink,
            -- Checking the sink time T from the meta data --
            CAST(TRIM(LEADING 'T' FROM split_part(corr.name, '_'::text, 3)) AS integer) = meta_seq.t0 - meta.quark_source_origin[4] AS t_sink_verify
        FROM correlator_n_point AS corr
        JOIN meta_data_correlator AS meta ON meta.correlator_key = corr.name AND meta.ens_id = corr.ens_id
        JOIN meta_data_sequential AS meta_seq ON meta_seq.meta_correlator_id = meta.meta_correlator_id
        WHERE meta.has_sequential = true
    );"""

    two_point = """
    CREATE VIEW two_point AS (
        SELECT
            corr.ens_id,
            corr.corr_id,
            corr.name,
            GREATEST(meta.quark_mass, meta.antiquark_mass) AS m_heavy,
            LEAST(meta.quark_mass, meta.antiquark_mass) AS m_light,
            split_part(meta.momentum, '-'::text, 1) AS momentum
        FROM correlator_n_point corr
        JOIN meta_data_correlator meta ON (corr.name = meta.correlator_key) AND (corr.ens_id = meta.ens_id)
        WHERE meta.has_sequential = false
    );"""

    two_point_state_content = """
    CREATE TABLE IF NOT EXISTS two_point_state_content
    (
        corr_id integer PRIMARY KEY REFERENCES correlator_n_point(corr_id),
        n   integer,
        no  integer,
        log text,
        UNIQUE(corr_id)
    );"""

    alias_quark_mass = """
    CREATE TABLE IF NOT EXISTS alias_quark_mass
    (
        alias_mq_id SERIAL PRIMARY KEY,
        ens_id integer REFERENCES ensemble(ens_id),
        mq float NOT NULL,   -- the bare quark mass
        alias text NOT NULL, -- the alias, e.g., '0.1 mc'
        style text NOT NULL default 'default',
        UNIQUE(ens_id, mq, style)
    );"""

    alias_form_factor = """
    CREATE VIEW alias_form_factor AS (
        SELECT
            ff.form_factor_id,
            alias_l.alias as alias_light,
            alias_h.alias as alias_heavy,
            alias_s.alias as alias_spectator
        FROM form_factor as ff
        -- alias for light quark
        JOIN alias_quark_mass AS alias_l
        ON (alias_l.ens_id = ff.ens_id) AND (alias_l.mq = ff.m_light)
        -- alias for heavy quark
        JOIN alias_quark_mass AS alias_h
        ON (alias_h.ens_id = ff.ens_id) AND (alias_h.mq = ff.m_heavy)
        -- alias for spectator quark
        JOIN alias_quark_mass AS alias_s
        ON (alias_s.ens_id = ff.ens_id) AND (alias_s.mq = ff.m_spectator)
    );"""

    alias_two_point = """
    CREATE VIEW alias_two_point AS (
        SELECT
            two_point.corr_id,
            alias_l.alias AS alias_light,
            alias_h.alias AS alias_heavy
        FROM two_point
        -- alias for light quark
        JOIN alias_quark_mass alias_l
        ON (alias_l.ens_id = two_point.ens_id) AND (alias_l.mq = two_point.m_light)
        -- alias for heavy quark
        JOIN alias_quark_mass alias_h
        ON (alias_h.ens_id = two_point.ens_id) AND (alias_h.mq = two_point.m_heavy)
    );"""

    campaign_form_factor = """
    CREATE TABLE IF NOT EXISTS campaign_form_factor
    (
        campaign_ff_id SERIAL PRIMARY KEY,
        form_factor_id integer REFERENCES form_factor(form_factor_id),
        result_id      integer REFERENCES result_form_factor(id),
        criterion      text DEFAULT '',
        automated      boolean NOT NULL,
        UNIQUE(form_factor_id)
    );
    """

    missing_campaign_two_point = """
    CREATE VIEW missing_campaign_two_point AS
    -- Isolate two-points NOT appearing in the campaign table
    (
        SELECT *
        FROM correlator_analysis
        WHERE NOT EXISTS (
            SELECT -- intentionally empty
            FROM campaign_two_point
            WHERE correlator_analysis.corr_analysis_id = campaign_two_point.corr_analysis_id
        )
    );
    """

    missing_campaign_form_factor = """
    CREATE VIEW missing_campaign_form_factor AS
    -- Isolate form factors NOT appearing in the campaign table
    (
        SELECT *
        FROM form_factor
        WHERE NOT EXISTS (
            SELECT -- intentionally empty
            FROM campaign_form_factor
            WHERE form_factor.form_factor_id = campaign_form_factor.form_factor_id
        )
    );
    """

    missing_result_form_factor = """
    CREATE VIEW missing_result_form_factor AS
    -- Isolate form factors with missing results
    (
        SELECT *
        FROM form_factor
        WHERE NOT EXISTS (
            SELECT -- intentionally empty
            FROM result_form_factor
            WHERE form_factor.form_factor_id = result_form_factor.form_factor_id
        )
    );
    """

    # Note: the materialized view is convenient since the query is
    # somewhat slow (~60 seconds). The basic problem is the somewhat
    # clumsy handling of the correlator identification in the table
    # two_point, which requires some string parsing to join with correlator_analysis
    summary_two_point = """
    CREATE MATERIALIZED VIEW summary_two_point AS
    (
        SELECT
            result.ens_id,
            campaign.*,
            result.params,
            two_point.momentum,
            two_point.m_light,
            two_point.m_heavy,
            alias.alias_light,
            alias.alias_heavy
        FROM campaign_two_point AS campaign
        JOIN result_two_point AS result
        ON campaign.result_id = result.result_id
        JOIN correlator_analysis
        ON correlator_analysis.corr_analysis_id = campaign.corr_analysis_id
        JOIN two_point
        ON (RTRIM(two_point.name,'fine-_') = correlator_analysis.analysis_name)
            AND (two_point.ens_id = correlator_analysis.ens_id)
        JOIN alias_two_point AS alias
        ON alias.corr_id = two_point.corr_id
    );"""

    summary_form_factor = """
    CREATE VIEW summary_form_factor AS
    (
        SELECT
            result.ens_id,
            campaign.*,
            result.r,
            result.normalization,
            result.q,
            form_factor.momentum,
            form_factor.m_light,
            form_factor.m_heavy,
            form_factor.m_spectator,
            form_factor.spin_taste_current,
            alias.alias_light,
            alias.alias_heavy,
            alias.alias_spectator
        FROM campaign_form_factor AS campaign
        JOIN result_form_factor AS result
        ON campaign.result_id = result.id
        JOIN form_factor
        ON campaign.form_factor_id = form_factor.form_factor_id
        JOIN alias_form_factor AS alias
        ON campaign.form_factor_id = alias.form_factor_id
    );
    """

    tough_form_factor = """
    CREATE TABLE IF NOT EXISTS tough_form_factor
    (
        tough_id SERIAL PRIMARY KEY,
        form_factor_id integer REFERENCES form_factor(form_factor_id),
        log text default '',
        UNIQUE(form_factor_id)
    );"""

    tough_two_point = """
    CREATE TABLE IF NOT EXISTS tough_two_point
    (
        tough_id SERIAL PRIMARY KEY,
        corr_analysis_id integer REFERENCES correlator_analysis(corr_analysis_id),
        log text default '',
        UNIQUE(corr_analysis_id)
    );"""

    strong_coupling = """
    CREATE TABLE IF NOT EXISTS strong_coupling
    (
        coupling_id SERIAL PRIMARY KEY,
        coupling_name  text NOT NULL,    -- e.g., 'alphaV'
        coupling_scale text NOT NULL,    -- e.g., '2/a'
        coupling_value text NOT NULL,    -- e.g., '0.3478(81)'
        lattice_spacing_id int REFERENCES lattice_spacing(id),
        log            text default '',  -- e.g., 'Zech private communication'
        UNIQUE(lattice_spacing_id, coupling_name, coupling_scale)
    );"""

    missing_result_two_point = """
    CREATE VIEW missing_result_two_point AS
    -- Isolate missing two-point results
    (
    SELECT
        missing.corr_analysis_id,
        missing.analysis_name,
        missing.ens_id,
        two_point.m_light,
        two_point.m_heavy,
        two_point.momentum
    FROM two_point
    JOIN (
        SELECT * FROM correlator_analysis
        WHERE NOT EXISTS (
            SELECT FROM result_two_point
            WHERE result_two_point.corr_analysis_id = correlator_analysis.corr_analysis_id
            )
    ) AS missing
    ON (missing.analysis_name = RTRIM(two_point.name, 'fine-_')) AND (missing.ens_id = two_point.ens_id)
    WHERE two_point.name like '%%fine'
    );"""

    summary_status_form_factor = """
    CREATE VIEW summary_status_form_factor AS
    -- Summary of the status of form factor fits
    (
    SELECT
        present.ens_id,
        present.present,
        COALESCE(campaign.campaign, 0) AS campaign,
        COALESCE(missing_campaign.missing_campaign, 0) AS missing_campaign,
        COALESCE(results.results,0)    AS results,
        COALESCE(missing.missing,0)    AS missing,
        present.present - (COALESCE(results.results,0) + COALESCE(missing.missing,0))
                                       AS unaccounted,
        COALESCE(pending.pending, 0)   AS status_pending,
        COALESCE(complete.complete, 0) AS status_complete,
        COALESCE(error.error, 0)       AS status_error,
        COALESCE(failed.failed, 0)     AS status_failed,
        COALESCE(pending.pending, 0) + COALESCE(complete.complete, 0) +
            COALESCE(error.error, 0) + COALESCE(failed.failed, 0)
                                       AS status_total,
        present.present - (
            COALESCE(pending.pending, 0) + COALESCE(complete.complete, 0) +
            COALESCE(error.error, 0) + COALESCE(failed.failed, 0))
                                       AS not_in_queue
    FROM (
        SELECT ens_id, count(distinct(form_factor_id)) AS present
        FROM form_factor
        GROUP BY(ens_id)
        ) AS present
    LEFT OUTER JOIN (
        SELECT ens_id, count(distinct(camp.form_factor_id)) AS campaign FROM campaign_form_factor AS camp
        JOIN form_factor AS ff
        ON camp.form_factor_id = ff.form_factor_id
        GROUP BY(ens_id)
        ) AS campaign
    ON campaign.ens_id = present.ens_id
    LEFT OUTER JOIN (
        SELECT ens_id, count(DISTINCT(form_factor_id)) AS missing_campaign FROM missing_campaign_form_factor
        GROUP BY(ens_id)
        ) AS missing_campaign
    ON (present.ens_id = missing_campaign.ens_id)
    LEFT OUTER JOIN (
         SELECT ens_id, count(distinct(form_factor_id)) AS results
         FROM result_form_factor
         GROUP BY(ens_id)
         ) AS results
    ON (present.ens_id = results.ens_id)
    LEFT OUTER JOIN (
        SELECT ens_id, count(distinct(form_factor_id)) AS missing
        FROM missing_result_form_factor
        GROUP BY(ens_id)
        ) AS missing
    ON (present.ens_id = missing.ens_id)
    LEFT OUTER JOIN (
        SELECT ens_id, count(queue.form_factor_id) AS pending
        FROM queue_form_factor as queue
        JOIN form_factor AS ff ON (ff.form_factor_id = queue.form_factor_id)
        WHERE status = 'pending'
        GROUP BY(ens_id)
        ) AS pending
    ON pending.ens_id = present.ens_id
    LEFT OUTER JOIN (
        SELECT ens_id, count(queue.form_factor_id) AS complete
        FROM queue_form_factor AS queue
        JOIN form_factor AS ff ON (ff.form_factor_id = queue.form_factor_id)
        WHERE status = 'complete'
        GROUP BY(ens_id)
        ) AS complete
    on complete.ens_id = present.ens_id
    LEFT OUTER JOIN (
        SELECT ens_id, count(queue.form_factor_id) as error
        FROM queue_form_factor as queue
        JOIN form_factor AS ff ON (ff.form_factor_id = queue.form_factor_id)
        WHERE status = 'error'
        GROUP BY(ens_id)
        ) AS error
    on error.ens_id = present.ens_id
    LEFT OUTER JOIN (
        SELECT ens_id, count(queue.form_factor_id) as failed
        FROM queue_form_factor as queue
        JOIN form_factor AS ff ON (ff.form_factor_id = queue.form_factor_id)
        WHERE status = 'failed'
        GROUP BY(ens_id)
        ) as failed
    ON failed.ens_id = present.ens_id
    );"""

    summary_status_two_point = """
    CREATE VIEW summary_status_two_point AS
    -- Summary of the status of two_point fits
    (
    SELECT
        two_point.ens_id,
        two_point.present,
        COALESCE(results.results, 0) AS results,
        COALESCE(missing.missing, 0) AS missing,
        two_point.present - (COALESCE(results.results, 0) + COALESCE(missing.missing, 0)) AS unaccounted,
        COALESCE(campaign.campaign, 0) AS campaign,
        COALESCE(missing_campaign.missing_campaign) AS missing_campaign
    FROM (
        SELECT
        ens_id,
        count(corr_analysis_id) AS present
        FROM correlator_analysis
        GROUP BY(ens_id)
        ) AS two_point
    LEFT OUTER JOIN(
        SELECT ens_id, count(*) AS campaign
        FROM campaign_two_point AS camp
        JOIN correlator_analysis AS corr ON corr.corr_analysis_id = camp.corr_analysis_id
        GROUP BY(ens_id)
        ) AS campaign
    ON campaign.ens_id = two_point.ens_id
    LEFT OUTER JOIN(
        SELECT ens_id, count(DISTINCT(corr_analysis_id)) AS results
        FROM result_two_point
        GROUP BY (ens_id)
        ) AS results
    ON results.ens_id = two_point.ens_id
    LEFT OUTER JOIN(
        SELECT ens_id, count(*) as missing
        FROM missing_result_two_point
        GROUP BY(ens_id)
        ) AS missing
    ON missing.ens_id = two_point.ens_id
    LEFT OUTER JOIN(
        SELECT ens_id, count(distinct(corr_analysis_id)) as missing_campaign
        FROM missing_campaign_two_point
        GROUP BY(ens_id)
        ) AS missing_campaign
    ON missing_campaign.ens_id = two_point.ens_id
    );"""

    campaign_two_point_alt_q = """
    CREATE VIEW campaign_two_point_alt_q AS
    (
    SELECT
        result_two_point.corr_analysis_id,
        result_two_point.result_id,
        'q' as criterion,
        true as automated
    FROM result_two_point
    JOIN (
        SELECT result.corr_analysis_id, max(q) AS q
        FROM result_two_point AS result
        JOIN analysis_two_point AS analysis
        ON result.analysis_id = analysis.analysis_id
        JOIN (SELECT * FROM junction_two_point WHERE solve_type = 'fine') AS junction
        ON junction.corr_analysis_id = result.corr_analysis_id
        JOIN two_point ON two_point.corr_id = junction.corr_id
        WHERE
            ((prior_alias = 'boosted prior, 1.0 discretization') AND (momentum != 'p000'))
        GROUP BY(result.corr_analysis_id)
    ) AS maxes
    ON (result_two_point.corr_analysis_id = maxes.corr_analysis_id) AND
       (result_two_point.q = maxes.q)
    UNION
    -- Grab the zero-momentum fits from the real campaign table
    SELECT
        camp.corr_analysis_id,
        camp.result_id,
        camp.criterion,
        camp.automated
    FROM campaign_two_point as camp
    JOIN (SELECT * FROM junction_two_point WHERE solve_type = 'fine') AS junction
    ON junction.corr_analysis_id = camp.corr_analysis_id
    JOIN two_point ON two_point.corr_id = junction.corr_id
    WHERE (momentum = 'p000')
    );"""

    campaign_two_point_alt_quality_factor = """
    CREATE VIEW campaign_two_point_alt_quality_factor AS
    (
    SELECT
        result_two_point.corr_analysis_id,
        result_two_point.result_id,
        'quality_factor' as criterion,
        true as automated
    FROM result_two_point
    JOIN (
        SELECT result.corr_analysis_id, max(quality_factor) AS quality_factor
        FROM result_two_point AS result
        JOIN analysis_two_point AS analysis
        ON result.analysis_id = analysis.analysis_id
        JOIN (SELECT * FROM junction_two_point WHERE solve_type = 'fine') AS junction
        ON junction.corr_analysis_id = result.corr_analysis_id
        JOIN two_point ON two_point.corr_id = junction.corr_id
        WHERE
            ((prior_alias = 'boosted prior, 1.0 discretization') AND (momentum != 'p000'))
        GROUP BY(result.corr_analysis_id)
    ) AS maxes
    ON (result_two_point.corr_analysis_id = maxes.corr_analysis_id) AND
       (result_two_point.quality_factor = maxes.quality_factor)
    UNION
    -- Grab the zero-momentum fits from the real campaign table
    SELECT
        camp.corr_analysis_id,
        camp.result_id,
        camp.criterion,
        camp.automated
    FROM campaign_two_point as camp
    JOIN (SELECT * FROM junction_two_point WHERE solve_type = 'fine') AS junction
    ON junction.corr_analysis_id = camp.corr_analysis_id
    JOIN two_point ON two_point.corr_id = junction.corr_id
    WHERE (momentum = 'p000')
    );"""

    summary_two_point_alt_q = """
    CREATE VIEW summary_two_point_alt_q AS
    (
    SELECT
        result.ens_id,
        camp.corr_analysis_id,
        camp.result_id,
        camp.criterion,
        camp.automated,
        result.params,
        result.prior,
        two_point.momentum,
        two_point.m_light,
        two_point.m_heavy,
        alias.alias_light,
        alias.alias_heavy
    FROM campaign_two_point_alt_q AS camp
    JOIN result_two_point AS result
    ON (result.result_id = camp.result_id)
    AND (result.corr_analysis_id = camp.corr_analysis_id)
    JOIN (SELECT * FROM junction_two_point WHERE solve_type = 'fine') AS junction
    ON junction.corr_analysis_id = camp.corr_analysis_id
    JOIN two_point
    ON two_point.corr_id = junction.corr_id
    JOIN alias_two_point AS alias ON alias.corr_id = two_point.corr_id
    );"""

    summary_two_point_alt_quality_factor = """
    CREATE VIEW summary_two_point_alt_quality_factor AS
    (
    SELECT
        result.ens_id,
        camp.corr_analysis_id,
        camp.result_id,
        camp.criterion,
        camp.automated,
        result.params,
        result.prior,
        two_point.momentum,
        two_point.m_light,
        two_point.m_heavy,
        alias.alias_light,
        alias.alias_heavy
    FROM campaign_two_point_alt_quality_factor AS camp
    JOIN result_two_point AS result
    ON (result.result_id = camp.result_id)
    AND (result.corr_analysis_id = camp.corr_analysis_id)
    JOIN (SELECT * FROM junction_two_point WHERE solve_type = 'fine') AS junction
    ON junction.corr_analysis_id = camp.corr_analysis_id
    JOIN two_point
    ON two_point.corr_id = junction.corr_id
    JOIN alias_two_point AS alias ON alias.corr_id = two_point.corr_id
    );"""

    junction_two_point = """
    CREATE MATERIALIZED VIEW junction_two_point AS
    (
    SELECT
        two_point.ens_id,
        corr_analysis_id,
        corr_id,
        analysis_name,
        CASE
            WHEN two_point.name LIKE '%%fine' THEN 'fine'
            WHEN two_point.name LIKE '%%loose' THEN 'loose'
            ELSE 'unknown'
        END AS solve_type
    FROM
    correlator_analysis
    JOIN two_point ON
    (RTRIM(two_point.name,'losefin-_') = correlator_analysis.analysis_name) AND
    (two_point.ens_id = correlator_analysis.ens_id)
    );
    """

    status_form_factor = """
    CREATE VIEW status_form_factor AS
    (
    -- Grab all the statuses...
    SELECT form_factor_id, status
    FROM (
        -- Here are completed results, with entry in campaign
        SELECT form_factor_id, 'complete' AS status FROM campaign_form_factor
        UNION
        -- Here are partial results, with some fits but no campaign choice
        SELECT DISTINCT(form_factor_id), 'partial_result' AS status FROM result_form_factor AS result
        WHERE NOT (EXISTS (
            SELECT FROM campaign_form_factor AS camp
            WHERE camp.form_factor_id = result.form_factor_id
            ))
        UNION
        -- Here are the missing results, where we don't have any fits
        SELECT form_factor_id, 'missing_result' AS status FROM missing_result_form_factor
    ) AS status
    -- ...but exclude results where we marked a problem.
    WHERE NOT (EXISTS (
        SELECT FROM tough_form_factor as tough
        WHERE tough.form_factor_id = status.form_factor_id
        ))
    UNION
    -- Finally, grab the results we marked as a problem.
    SELECT form_factor_id, 'problem' FROM tough_form_factor
    );"""

    transition_name = """
    CREATE TABLE IF NOT EXISTS transition_name
    (
        transition_id SERIAL PRIMARY KEY,
        form_factor_id integer REFERENCES form_factor(form_factor_id),
        mother   text NOT NULL,
        daughter text NOT NULL,
        process  text NOT NULL,
        log      text default '',
        UNIQUE(form_factor_id)
    );"""

    meson_name = (
        "CREATE TABLE IF NOT EXISTS meson_name"
        "("
        "meson_id SERIAL PRIMARY KEY, "
        "corr_id integer REFERENCES correlator_n_point(corr_id), "
        "name text NOT NULL, "
        "log text default '', "
        "UNIQUE(corr_id)"
        ");"
    )

    fastfit = (
        "CREATE TABLE IF NOT EXISTS fastfit "
        "("
        "fastfit_id SERIAL PRIMARY KEY, "
        "corr_analysis_id integer REFERENCES correlator_analysis(corr_analysis_id), "
        "energy text NOT NULL, "
        "ampl text NOT NULL, "
        "tmin integer NOT NULL, "
        "tmax integer, "
        "nterm integer NOT NULL, "
        "osc boolean NOT NULL, "
        "UNIQUE(corr_analysis_id)"
        ");"
    )

    corr_diagnostic = (
        "CREATE TABLE IF NOT EXISTS corr_diagnostic "
        "("
        "corr_diagnostic_id SERIAL PRIMARY KEY, "
        "corr_id integer REFERENCES correlator_n_point(corr_id), "
        "average_noise_to_signal float NOT NULL, "
        "median_noise_to_signal float NOT NULL, "
        "UNIQUE(corr_id)"
        ");"
    )

    corr_sign_flip = (
        "CREATE TABLE IF NOT EXISTS corr_sign_flip "
        "("
        "corr_sign_flip_id SERIAL PRIMARY KEY, "
        "corr_id integer REFERENCES correlator_n_point(corr_id), "
        "tmin integer NOT NULL, "
        "tmax integer NOT NULL, "
        "nconfig integer NOT NULL, "
        "total integer NOT NULL, "
        "nflips integer NOT NULL, "
        "UNIQUE(corr_id, tmin, tmax, nconfig)"
        ");"
    )

    form_factor_r_guess = (
        "CREATE TABLE IF NOT EXISTS form_factor_r_guess "
        "( "
        "r_guess_id SERIAL PRIMARY KEY, "
        "form_factor_id integer REFERENCES form_factor(form_factor_id), "
        "r_guess text NOT NULL, "
        "calcdate timestamp with time zone, "
        "UNIQUE(form_factor_id)"
        ");"
    )

    taste_splittings = """
        CREATE TABLE IF NOT EXISTS taste_splitting
        (
        taste_splitting_id SERIAL PRIMARY KEY,
        ens_id integer REFERENCES ensemble(ens_id),
        splitting float NOT NULL,     -- e.g., 0.12345
        splitting_name text NOT NULL, -- e.g., 'Delta5'
        units text NOT NULL,            -- e.g., 'lattice' or 'p4s'
        UNIQUE(ens_id, splitting_name, units)
        );"""

    bootstrap = """
        CREATE TABLE IF NOT EXISTS bootstrap
        (
        boot_id serial PRIMARY KEY,
        ens_id integer REFERENCES ensemble (ens_id),
        seed integer NOT NULL,
        nconfigs integer NOT NULL,
        nresample integer NOT NULL,
        UNIQUE(ens_id, seed, nconfigs, nresample)
        );"""

    bootstrap_draw = """
        CREATE TABLE IF NOT EXISTS bootstrap_draw
        (
        boot_draw_id serial PRIMARY KEY,
        boot_id integer REFERENCES bootstrap (boot_id),
        draw_number integer NOT NULL,
        checksum text NOT NULL,
        UNIQUE(boot_id, draw_number, checksum)
        );"""

    bootstrap_result_form_factor = """
        CREATE TABLE IF NOT EXISTS bootstrap_result_form_factor
        (
        boot_result_form_factor_id serial PRIMARY KEY,
        boot_draw_id integer REFERENCES bootstrap_draw (boot_draw_id),
        form_factor_id integer REFERENCES form_factor (form_factor_id),
        result_id integer REFERENCES result_form_factor (id),
        params text,
        prior text,
        nparams integer,
        npoints integer,
        chi2 float,
        q float,
        r text,
        normalization text,
        calcdate timestamp with time zone,
        UNIQUE(boot_draw_id, form_factor_id, result_id)
        );"""

    form_factor_signs = """
        CREATE TABLE IF NOT EXISTS sign_form_factor
        (
        sign_id serial PRIMARY KEY,
        ens_id integer REFERENCES ensemble (ens_id),
        spin_taste_current text,
        sign integer,
        UNIQUE(ens_id, spin_taste_current)
        );"""

    bootstrap_status_form_factor = """
        CREATE TABLE IF NOT EXISTS bootstrap_status_form_factor
        (
        boot_status_form_factor_id serial PRIMARY KEY,
        form_factor_id integer REFERENCES form_factor (form_factor_id),
        boot_id integer REFERENCES bootstrap (boot_id),
        boot_draw_id integer REFERENCES bootstrap_draw (boot_draw_id),
        status text NOT NULL default 'pending', -- pending / success / failure
        UNIQUE(form_factor_id, boot_id, boot_draw_id)
        );"""

    bootstrap_summary_form_factor = """
        create view bootstrap_summary_form_factor as
        (
        select
            form_factor_id,
            boot_id,
            COALESCE(nsuccess, 0) as nsucess,
            COALESCE(nfailure, 0) as nfailure,
            COALESCE(npending, 0) as npending,
            COALESCE(nsuccess, 0)
            + COALESCE(nfailure, 0)
            + COALESCE(npending, 0) as ntotal
        from
        (
            select form_factor_id, boot_id, count(*) as nsuccess
            from bootstrap_status_form_factor
            where status = 'success'
            group by(form_factor_id, boot_id)
        ) as success
        full join
        (
            select form_factor_id, boot_id, count(*) as nfailure
            from bootstrap_status_form_factor
            where status = 'failure'
            group by(form_factor_id, boot_id)
        ) as failure using(form_factor_id, boot_id)
        full join
        (
            select form_factor_id, boot_id, count(*) as npending
            from bootstrap_status_form_factor
            where status = 'pending'
            group by(form_factor_id, boot_id)
        ) as pending using(form_factor_id, boot_id)
        );"""

    form_factor_name = """
        CREATE TABLE IF NOT EXISTS form_factor_name
        (
        form_factor_name_id serial primary key,
        spin_taste_current text not null,
        name text not null,
        UNIQUE(spin_taste_current, name)
        );"""

    form_factor_grouping = """
        CREATE VIEW form_factor_grouping AS (
        SELECT
        ens_id, form_factor_id, form_factor_name.name, momentum,
        process, alias_light, alias_spectator, alias_heavy
        FROM form_factor
        JOIN form_factor_name USING(spin_taste_current)
        JOIN transition_name USING(form_factor_id)
        JOIN alias_form_factor USING(form_factor_id)
        JOIN ensemble USING(ens_id)
        );"""

    noise_threshy = """
        CREATE TABLE IF NOT EXISTS form_factor_noise_threshy (
        noise_id serial primary key,
        form_factor_id integer REFERENCES form_factor (form_factor_id),
        noise_threshy float,
        UNIQUE(form_factor_id)
        );"""

    lec_slope_mu = """
        CREATE TABLE IF NOT EXISTS lec_slope_mu (
        lec_id serial primary key,
        ens_id integer REFERENCES ensemble (ens_id),
        mu text NOT NULL,
        UNIQUE(ens_id)
        );"""

    form_factor_systematic = """
        CREATE TABLE IF NOT EXISTS form_factor_systematic (
        systematic_id serial primary key,
        form_factor_id integer REFERENCES form_factor (form_factor_id),
        systematic float NOT NULL,
        UNIQUE(form_factor_id)
        );"""

    two_point_materialized = """
        CREATE MATERIALIZED VIEW two_point_materialized AS (
        SELECT
            corr.ens_id,
            corr.corr_id,
            corr.name,
            GREATEST(meta.quark_mass, meta.antiquark_mass) AS m_heavy,
            LEAST(meta.quark_mass, meta.antiquark_mass) AS m_light,
            split_part(meta.momentum, '-'::text, 1) AS momentum
        FROM correlator_n_point corr
        JOIN meta_data_correlator meta ON (corr.name = meta.correlator_key) AND (corr.ens_id = meta.ens_id)
        WHERE meta.has_sequential = false);"""

    three_point_materialized = """
        CREATE MATERIALIZED VIEW three_point_materialized AS (
            SELECT
                corr.ens_id,
                corr.corr_id,
                corr.name,
                -- Currents --
                split_part(meta.correlator, '_'::text, 1) AS spin_taste_sink,
                split_part(meta.correlator, '_'::text, 2) AS spin_taste_current,
                -- Masses --
                GREATEST(meta.quark_mass, meta_seq.mass) AS m_heavy,
                LEAST(meta.quark_mass, meta_seq.mass)    AS m_light,
                meta.antiquark_mass                      AS m_spectator,
                meta_seq.mass                            AS m_sink_to_current,
                meta.quark_mass                          AS m_source_to_current,
                -- Momentum --
                split_part(meta.momentum, '-'::text, 1) AS momentum,
                -- Sink time T --
                CAST(TRIM(LEADING 'T' FROM split_part(corr.name, '_'::text, 3)) AS integer) AS t_sink,
                -- Checking the sink time T from the meta data --
                CAST(TRIM(LEADING 'T' FROM split_part(corr.name, '_'::text, 3)) AS integer) = meta_seq.t0 - meta.quark_source_origin[4] AS t_sink_verify
            FROM correlator_n_point AS corr
            JOIN meta_data_correlator AS meta ON meta.correlator_key = corr.name AND meta.ens_id = corr.ens_id
            JOIN meta_data_sequential AS meta_seq ON meta_seq.meta_correlator_id = meta.meta_correlator_id
            WHERE meta.has_sequential = true
        );"""

    model_averaging_two_point_result = """
        CREATE TABLE IF NOT EXISTS model_averaging_two_point_result
        (
            fit_id serial PRIMARY KEY,
            ens_id integer REFERENCES ensemble(ens_id),
            basename text not NULL,
            n_decay integer not NULL,
            n_oscillating integer not NULL,
            tmin integer not NULL,
            tmax integer not NULL,
            energy text,
            amp text,
            aic float,
            chi2_aug float,
            chi2 float,
            chi2_per_dof float,
            model_probability float,
            dof integer,
            nparams integer,
            npoints integer,
            p_value float,
            params text,
            prior text,
            prior_alias text,
            q_value float,
            calcdate timestamp with time zone,
            UNIQUE(ens_id, basename, n_decay, n_oscillating, tmin, tmax)
        );"""

    model_averaging_two_point_best = """
        CREATE VIEW model_averaging_two_point_best_fit AS(
        SELECT DISTINCT ON (ens_id, basename) *
        FROM model_averaging_two_point_result
        ORDER BY ens_id, basename, model_probability DESC
        );"""

    model_averaging_ratio_result = """
        CREATE TABLE IF NOT EXISTS model_averaging_ratio_result
        (
            fit_id serial PRIMARY KEY,
            form_factor_id integer REFERENCES form_factor(form_factor_id),
            -- priors / posteriors
            prior text,
            params text,
            r text not NULL,
            -- statistics
            q_value float,
            p_value float,
            chi2_aug float,
            chi2 float,
            chi2_per_dof float,
            dof integer,
            nparams integer,
            npoints integer,
            aic float,
            model_probability float,
            -- analysis inputs
            m_src text not NULL,
            m_snk text not NULL,
            tmin_src integer not NULL,
            tmin_snk integer not NULL,
            t_step integer not NULL,
            n_decay_src integer not NULL,
            n_decay_snk integer not NULL,
            calcdate timestamp with time zone,
            UNIQUE(form_factor_id, tmin_src, tmin_snk, t_step, n_decay_src, n_decay_snk)
        );"""

    model_averaging_ratio_best_fit = """
        CREATE VIEW model_averaging_ratio_best_fit AS(
        SELECT DISTINCT ON (form_factor_id) *
        FROM model_averaging_ratio_result
        ORDER BY form_factor_id, model_probability DESC);"""

    result_form_factor_variations = """
        CREATE TABLE IF NOT EXISTS result_form_factor_variations
        (
            fit_id serial PRIMARY KEY,
            ens_id integer REFERENCES ensemble(ens_id),
            form_factor_id, interger REFERENCES form_factor(form_factor_id),
            -- statistics
            q_value float,
            p_value float,
            chi2_aug float,
            chi2 float,
            chi2_per_dof float,
            dof integer,
            nparams integer,
            npoints integer,
            aic float,
            model_probability float,
            -- analysis inputs
            tmin_src integer not NULL,
            tmin_snk integer not NULL,
            tmax_src integer not NULL,
            tmax_snk integer not NULL,
            n_decay_src integer not NULL,
            n_decay_snk integer not NULL,
            n_oscillating_src inteter not NULL,
            n_oscillating_snk integer not NULL,
            -- priors
            prior text,
            r_guess text,
            prior_alias text,
            -- posteriors
            params text,
            energy_src text,
            energy_snk text,
            amp_src text,
            amp_snk text,
            r text,
            form_factor text,
            calcdate timestamp with time zone,
            UNIQUE(ens_id, form_factor_id, tmin_src, tmin_snk, tmax_src, tmax_snk,
                   n_decay_src, n_decay_snk, n_oscillating_src, n_oscillating_snk)
        );"""

    campaign_results_form_two_point = """
        CREATE TABLE IF NOT EXISTS campaign_results_two_point (
        result_id serial PRIMARY KEY,
        ens_id integer REFERENCES ensemble(ens_id),
        corr_id integer REFERENCES correlator_n_point(corr_id),
        -- meta information
        a_fm float not NULL,
        basename text not NULL,
        momentum text not NULL,
        m_heavy float not NULL,
        m_light float not NULL,
        alias_heavy text not NULL,
        alias_light text not NULL,
        -- analysis inputs
        n_decay integer not NULL,
        n_oscillating  integer not NULL,
        tmin integer not NULL,
        tmax integer not NULL,
        -- priors
        prior text,
        -- posters
        params text,
        energy text,
        amp text,
        -- statistics
        q_value float,
        p_value float,
        chi2_aug float,
        chi2 float,
        chi2_per_dof float,
        dof integer,
        nparams integer,
        npoints integer,
        aic float,
        model_probability float,
        calcdate timestamp with time zone,
        UNIQUE(ens_id, corr_id, a_fm, basename, momentum, m_heavy, m_light)
    );"""

    rbar = """
        CREATE TABLE IF NOT EXISTS rbar (
        rbar_id serial PRIMARY KEY,
        form_factor_id interger REFERENCES form_factor(form_factor_id),
        t_snk integer NOT NULL,
        rbar text NOT NULL,
        m_src text NOT NULL,
        m_snk text NOT NULL,
        calcdate timestamp with time zone,
        UNIQUE(form_factor_id, t_snk)
        );"""

    engine = db.make_engine()

    queries = [
#         ensemble,
#         correlator_n_point,
#         correlator_analysis,
#         junction_n_point_analysis,
#         reduction_two_point,
#         analysis_two_point,
#         result_two_point,
#         campaign_two_point,
#         systematic_two_point,
#         external_database,
#         analysis_queue,
#         meta_data_correlator,
#         meta_data_sequential,
#         glance_dataset_form_factor
#         form_factor,
#         junction_form_factor,
#         analysis_form_factor,
#         reduction_form_factor,
#         result_form_factor,
#         queue_form_factor,
#         glance_correlator_n_point,
#         three_point,
#         two_point,
#         two_point_state_content
#         alias_quark_mass,
#         alias_form_factor,
#         alias_two_point,
#         campaign_form_factor
#         campaign_two_point,
#         missing_campaign_two_point,
#         missing_campaign_form_factor,
#         missing_result_form_factor,
#        summary_form_factor,
#         summary_two_point,
#        tough_form_factor,
#        tough_two_point
#         strong_coupling,
#        missing_result_two_point,
#         summary_status_form_factor,
#         summary_status_two_point
#         junction_two_point,
#         campaign_two_point_alt_q,
#         campaign_two_point_alt_quality_factor,
#         summary_two_point_alt_q,
#         summary_two_point_alt_quality_factor,
#         status_form_factor,
#        transition_name,
#         meson_name,
#        fastfit,
#         corr_diagnostic,
#         corr_sign_flip,
#         form_factor_r_guess,
#         taste_splitting,
#         bootstrap,
#         bootstrap_draw,
#         bootstrap_result_form_factor,
#         form_factor_signs,
#         bootstrap_status_form_factor
#         lec_slope_mu,
        # form_factor_systematic

    ]
    for query in queries:
        print(query)
        engine.execute(query)


if __name__ == '__main__':
    main()
