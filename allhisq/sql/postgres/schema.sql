-- name: create_schema#
-- Creates the schema for the database
CREATE TABLE IF NOT EXISTS ensemble
(
    ens_id      serial PRIMARY KEY,
    name        text NOT NULL,    -- e.g., "HISQ 2+1+1"
    ns          integer NOT NULL,
    nt          integer NOT NULL,
    description text DEFAULT '',
    UNIQUE(name, ns, nt)
);

CREATE TABLE IF NOT EXISTS correlator_n_point
(
    corr_id   SERIAL PRIMARY KEY,
    ens_id    integer REFERENCES ensemble (ens_id),
    name      text NOT NULL DEFAULT '',
    UNIQUE(ens_id, name)
);

CREATE TABLE IF NOT EXISTS form_factor
(
    form_factor_id      SERIAL PRIMARY KEY,
    ens_id              integer REFERENCES ensemble(ens_id),
    momentum            text NOT NULL,
    spin_taste_current  text NOT NULL,
    spin_taste_sink     text NOT NULL,
    spin_taste_source   text NOT NULL,
    m_spectator         float NOT NULL,
    m_heavy             float NOT NULL,
    m_light             float NOT NULL,
    m_sink_to_current   float NOT NULL,
    UNIQUE(ens_id, momentum,
           spin_taste_current, spin_taste_sink, spin_taste_source,
           m_spectator, m_heavy, m_light, m_sink_to_current)
);

CREATE TABLE IF NOT EXISTS junction_form_factor
(
    form_factor_corr_id SERIAL PRIMARY KEY,
    form_factor_id integer REFERENCES form_factor(form_factor_id),
    corr_id integer REFERENCES correlator_n_point(corr_id),
    corr_type text NOT NULL,
    UNIQUE(form_factor_id, corr_id, corr_type)
);

CREATE TABLE IF NOT EXISTS external_database
(
    id SERIAL PRIMARY KEY,
    ens_id   integer REFERENCES ensemble (ens_id),
    name     text,
    location text,
    type     text,
    UNIQUE(ens_id, name)
);

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
);


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
);

CREATE TABLE IF NOT EXISTS lattice_spacing
(
    id SERIAL PRIMARY KEY,
    ens_id integer REFERENCES ensemble(ens_id),
    a_fm   float NOT NULL,
    type   text  NOT NULL, -- e.g., 'nominal' or 'r1' or 'wilson flow'
    UNIQUE(ens_id, type)
);

CREATE TABLE IF NOT EXISTS alias_quark_mass
(
    alias_mq_id SERIAL PRIMARY KEY,
    ens_id integer REFERENCES ensemble(ens_id),
    mq float NOT NULL,   -- the bare quark mass
    alias text NOT NULL, -- the alias, e.g., '0.1 mc'
    style text NOT NULL default 'default',
    UNIQUE(ens_id, mq, style)
);

CREATE OR REPLACE VIEW alias_form_factor AS (
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
);

CREATE OR REPLACE VIEW two_point AS (
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
);

CREATE OR REPLACE VIEW alias_two_point AS (
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
);

CREATE TABLE IF NOT EXISTS strong_coupling
(
    coupling_id SERIAL PRIMARY KEY,
    coupling_name  text NOT NULL,    -- e.g., 'alphaV'
    coupling_scale text NOT NULL,    -- e.g., '2/a'
    coupling_value text NOT NULL,    -- e.g., '0.3478(81)'
    lattice_spacing_id int REFERENCES lattice_spacing(id),
    log            text default '',  -- e.g., 'Zech private communication'
    UNIQUE(lattice_spacing_id, coupling_name, coupling_scale)
);

CREATE TABLE IF NOT EXISTS transition_name
(
    transition_id SERIAL PRIMARY KEY,
    form_factor_id integer REFERENCES form_factor(form_factor_id),
    mother   text NOT NULL,
    daughter text NOT NULL,
    process  text NOT NULL,
    log      text default '',
    UNIQUE(form_factor_id)
);

CREATE TABLE IF NOT EXISTS meson_name
(
meson_id SERIAL PRIMARY KEY,
corr_id integer REFERENCES correlator_n_point(corr_id),
name text NOT NULL,
log text default '',
UNIQUE(corr_id)
);

CREATE TABLE IF NOT EXISTS bootstrap
(
boot_id serial PRIMARY KEY,
ens_id integer REFERENCES ensemble (ens_id),
seed integer NOT NULL,
nconfigs integer NOT NULL,
nresample integer NOT NULL,
UNIQUE(ens_id, seed, nconfigs, nresample)
);

CREATE TABLE IF NOT EXISTS sign_form_factor
(
sign_id serial PRIMARY KEY,
ens_id integer REFERENCES ensemble (ens_id),
spin_taste_current text,
sign integer,
UNIQUE(ens_id, spin_taste_current)
);

CREATE MATERIALIZED VIEW IF NOT EXISTS two_point_materialized AS (
SELECT
    corr.ens_id,
    corr.corr_id,
    corr.name,
    GREATEST(meta.quark_mass, meta.antiquark_mass) AS m_heavy,
    LEAST(meta.quark_mass, meta.antiquark_mass) AS m_light,
    split_part(meta.momentum, '-'::text, 1) AS momentum
FROM correlator_n_point corr
JOIN meta_data_correlator meta ON (corr.name = meta.correlator_key) AND (corr.ens_id = meta.ens_id)
WHERE meta.has_sequential = false);

CREATE MATERIALIZED VIEW IF NOT EXISTS three_point_materialized AS (
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
);

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
);

CREATE TABLE IF NOT EXISTS result_form_factor
(
    fit_id serial PRIMARY KEY,
    ens_id integer REFERENCES ensemble(ens_id),
    form_factor_id integer REFERENCES form_factor(form_factor_id),
    seed_str text NOT NULL,
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
    tmin_src integer NOT NULL,
    tmin_snk integer NOT NULL,
    tmax_src integer NOT NULL,
    tmax_snk integer NOT NULL,
    n_decay_src integer NOT NULL,
    n_decay_snk integer NOT NULL,
    n_oscillating_src integer NOT NULL,
    n_oscillating_snk integer NOT NULL,
    -- priors
    prior text,
    r_guess text,
    prior_alias text NOT NULL,
    -- posteriors
    params text,
    energy_src text,
    energy_snk text,
    amp_src text,
    amp_snk text,
    r text,
    form_factor text,
    calcdate timestamp with time zone,
    UNIQUE(ens_id, form_factor_id, seed_str,
           tmin_src, tmin_snk, tmax_src, tmax_snk,
           n_decay_src, n_decay_snk, n_oscillating_src, n_oscillating_snk,
           prior_alias)
);

CREATE TABLE IF NOT EXISTS bootstrap (
    boot_id integer PRIMARY KEY,
    ens_id integer REFERENCES ensemble(ens_id),
    seed integer NOT NULL,
    nconfigs integer NOT NULL,
    nresample integer NOT NULL,
    UNIQUE(ens_id, seed, nconfigs, nresample)
);

CREATE TABLE IF NOT EXISTS result_form_factor_bootstrap (
    fit_form_factor_boot_id SERIAL PRIMARY KEY,
    form_factor_id integer REFERENCES form_factor(form_factor_id),
    boot_id integer REFERENCES bootstrap(boot_id),
    draw_number integer NOT NULL,
    checksum text NOT NULL,
    seed_str text NOT NULL,
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
    n_oscillating_src integer not NULL,
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
    UNIQUE(
        form_factor_id, boot_id, draw_number, checksum, seed_str,
        tmin_src, tmin_snk, tmax_src, tmax_snk,
        n_decay_src, n_decay_snk, n_oscillating_src, n_oscillating_snk)
);

CREATE MATERIALIZED VIEW IF NOT EXISTS pion AS
(
    SELECT ens_id, energy
    FROM campaign_results_two_point
    WHERE
    (momentum = 'p000') AND
    (basename like 'P5-P5%%') AND
    (alias_light, alias_heavy) IN (
        ('1.0 m_light', '1.0 m_light'),
        ('0.1 m_strange', '0.1 m_strange'),
        ('0.2 m_strange', '0.2 m_strange')
    )
);

CREATE MATERIALIZED VIEW IF NOT EXISTS kaon AS
(
    SELECT ens_id, energy
    FROM campaign_results_two_point
    WHERE
    (momentum = 'p000') AND
    (basename like 'P5-P5%%') AND
    (alias_heavy = '1.0 m_strange')
);

CREATE MATERIALIZED VIEW IF NOT EXISTS d_meson AS (
    SELECT ens_id, alias_heavy, m_heavy, energy
    FROM campaign_results_two_point
    WHERE
    (momentum = 'p000') AND
    (basename like 'P5-P5%%') AND
    (alias_light in ('1.0 m_light', '0.1 m_strange', '0.2 m_strange')) AND
    (alias_heavy like '% m_charm')
);

CREATE MATERIALIZED VIEW IF NOT EXISTS ds_meson AS (
    SELECT ens_id, alias_heavy, m_heavy, energy
    FROM campaign_results_two_point
    WHERE
    (momentum = 'p000') AND
    (basename like 'P5-P5%%') AND
    (alias_light = '1.0 m_strange') AND
    (alias_heavy like '% m_charm')
);

CREATE OR REPLACE VIEW hadron_masses AS
(
    SELECT ens_id, m_heavy, alias_heavy,
    pion.energy AS pion,
    kaon.energy AS kaon,
    d_meson.energy AS d,
    ds_meson.energy AS ds
    FROM pion
    JOIN kaon USING(ens_id)
    JOIN d_meson USING(ens_id)
    JOIN ds_meson USING(ens_id, m_heavy, alias_heavy)
);