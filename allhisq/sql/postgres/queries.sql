-- name: get_correlator_names
-- Fetch the names of correlators for extracting a form factor.
SELECT RTRIM(name, '_-fine') AS basename FROM correlators WHERE
(name LIKE :corr3) OR (name LIKE :mother) OR (name LIKE :daughter);

-- name: write_ensemble^
INSERT INTO ensemble (name, ns, nt, description)
VALUES      (:name, :ns, :nt, :description)
ON CONFLICT (name, ns, nt)
DO NOTHING;
SELECT  ens_id
FROM    ensemble
WHERE   (name, ns, nt) = (:name, :ns, :nt);

-- name: write_external_database!
INSERT INTO external_database(ens_id, name, location, type)
SELECT      ens_id, :name, :location, :type
FROM        ensemble
WHERE       (name, ns, nt) = (:name, :ns, :nt)
ON CONFLICT (ens_id, name)
DO NOTHING;

-- name: write_lattice_spacing!
INSERT INTO lattice_spacing (ens_id, a_fm, type)
SELECT      ens_id, :a_fm, :type
FROM        ensemble
WHERE       (name, ns, nt) = (:name, :ns, :nt)
ON CONFLICT (ens_id, type)
DO NOTHING;

--name: get_lattice_spacing$
SELECT  a_fm
FROM    lattice_spacing
WHERE   ens_id = :ens_id
AND     type = 'nominal';

-- name: get_existing_correlators
SELECT  name
FROM    correlator_n_point
WHERE   ens_id = :ens_id;

-- name: write_correlator!
INSERT INTO correlator_n_point(ens_id, name)
VALUES      (:ens_id, :name)
ON CONFLICT (ens_id, name)
DO NOTHING;

-- name: write_meta_data_correlator!
INSERT INTO meta_data_correlator (
    ens_id,
    has_sequential,
    quark_type,
    antiquark_type,
    quark_mass,
    antiquark_mass,
    quark_source_ncolor,
    antiquark_source_ncolor,
    quark_source_type,
    antiquark_source_type,
    quark_source_label,
    antiquark_source_label,
    quark_source_origin,
    antiquark_source_origin,
    quark_sink_label,
    antiquark_sink_label,
    quark_source_subset,
    antiquark_source_subset,
    quark_source_file,
    antiquark_source_file,
    quark_source_mom,
    antiquark_source_mom,
    quark_epsilon,
    antiquark_epsilon,
    lattice_size,
    file_sequence,
    correlator,
    momentum,
    correlator_key,
    spin_taste_sink)
VALUES (
    :ens_id,
    :has_sequential,
    :quark_type,
    :antiquark_type,
    :quark_mass,
    :antiquark_mass,
    :quark_source_ncolor,
    :antiquark_source_ncolor,
    :quark_source_type,
    :antiquark_source_type,
    :quark_source_label,
    :antiquark_source_label,
    :quark_source_origin,
    :antiquark_source_origin,
    :quark_sink_label,
    :antiquark_sink_label,
    :quark_source_subset,
    :antiquark_source_subset,
    :quark_source_file,
    :antiquark_source_file,
    :quark_source_mom,
    :antiquark_source_mom,
    :quark_epsilon,
    :antiquark_epsilon,
    :lattice_size,
    :file_sequence,
    :correlator,
    :momentum,
    :correlator_key,
    :spin_taste_sink)
ON CONFLICT (ens_id, correlator_key)
DO NOTHING;

-- name: write_meta_data_sequential!
INSERT INTO meta_data_sequential (
    meta_correlator_id,
    mom,
    operation_0,
    spin_taste_extend,
    t0,
    eps_naik,
    mass,
    momentum_twist,
    operation_1)
SELECT
    meta_correlator_id,
    :mom,
    :operation_0,
    :spin_taste_extend,
    :t0,
    :eps_naik,
    :mass,
    :momentum_twist,
    :operation_1
FROM    meta_data_correlator
WHERE   (ens_id = :ens_id)
AND     (correlator_key = :correlator_key)
ON CONFLICT (meta_correlator_id)
DO NOTHING;

-- name: refresh_materialized_views!
REFRESH MATERIALIZED VIEW two_point_materialized;
REFRESH MATERIALIZED VIEW three_point_materialized;
REFRESH MATERIALIZED VIEW pion;
REFRESH MATERIALIZED VIEW kaon;
REFRESH MATERIALIZED VIEW d_meson;
REFRESH MATERIALIZED VIEW ds_meson;

-- name: refresh_two_point_materialized!
REFRESH MATERIALIZED VIEW two_point_materialized;

-- name: refresh_three_point_materialized!
REFRESH MATERIALIZED VIEW three_point_materialized;

-- name: get_existing_form_factors
SELECT
    ens_id,
    momentum,
    spin_taste_current,
    spin_taste_sink,
    m_spectator,
    m_heavy,
    m_light,
    m_sink_to_current
FROM form_factor
WHERE ens_id = :ens_id;

-- name: get_3pt_combinations
-- Get all possible combinations of currents, masses, and momentum
-- for 3pt functions used to extract a form factor
SELECT
    ens_id,
    spin_taste_current,
    spin_taste_sink,
    momentum,
    m_spectator,
    m_heavy,
    m_light,
    m_sink_to_current,
    m_source_to_current,
    SPLIT_PART(name, '_', 1) as prefix
FROM three_point_materialized
GROUP BY(
    ens_id,
    spin_taste_current, spin_taste_sink,
    momentum,
    m_heavy, m_light, m_spectator,
    m_source_to_current,
    m_sink_to_current,
    SPLIT_PART(name, '_', 1)
);

-- name: get_source_combinations
-- Get all possible combinations of ensemble, momentum, and masses
-- for source 2pt functions.
SELECT
    ens_id,
    m_light as m_light_src,
    m_heavy as m_heavy_src,
    momentum
FROM two_point_materialized
GROUP BY(ens_id, m_light, m_heavy, momentum);

-- name: get_sink_combinations
-- Get all possible combinations of ensemble, momentum, and masses
-- for sink 2pt functions.
-- Note: Sink two-point functions are restricted to zero momentum.
SELECT
    ens_id,
    m_light as m_light_snk,
    m_heavy as m_heavy_snk
FROM two_point_materialized
WHERE (momentum = 'p000')
GROUP BY(ens_id, m_light, m_heavy);

-- name: get_correlators
-- Get the correlators associated with the form factor
-- specified by the given
SELECT
    corr_id,
    name,
    'light-light' AS corr_type,
    0 as t_sink
FROM two_point_materialized
WHERE
    (ens_id = :ens_id)
    AND (momentum = :momentum)
    AND (m_light = :m_light_src)
    AND (m_heavy = :m_heavy_src)
UNION
SELECT
    corr_id,
    name,
    'heavy-light' AS corr_type,
    0 as t_sink
FROM two_point_materialized
WHERE
    (ens_id = :ens_id)
    AND (momentum = 'p000')
    AND (m_light = :m_light_snk)
    AND (m_heavy = :m_heavy_snk)
UNION
SELECT
    corr_id,
    name,
    'three-point' AS corr_type,
    t_sink
FROM three_point_materialized
WHERE
    (ens_id = :ens_id)
    AND (momentum = :momentum)
    AND (spin_taste_current = :spin_taste_current)
    AND (
        (m_source_to_current, m_sink_to_current, m_spectator)
        =(:m_source_to_current, :m_sink_to_current, :m_spectator)
);

-- name: write_form_factor$
INSERT INTO form_factor (
    ens_id,
    momentum,
    spin_taste_current,
    spin_taste_sink,
    spin_taste_source,
    m_spectator,
    m_heavy,
    m_light,
    m_sink_to_current)
VALUES (
    :ens_id,
    :momentum,
    :spin_taste_current,
    :spin_taste_sink,
    :spin_taste_source,
    :m_spectator,
    :m_heavy,
    :m_light,
    :m_sink_to_current)
ON CONFLICT DO NOTHING;
SELECT form_factor_id
FROM form_factor
WHERE (
    ens_id,
    momentum,
    spin_taste_current,
    spin_taste_sink,
    spin_taste_source,
    m_spectator,
    m_heavy,
    m_light,
    m_sink_to_current)
    =(
    :ens_id,
    :momentum,
    :spin_taste_current,
    :spin_taste_sink,
    :spin_taste_source,
    :m_spectator,
    :m_heavy,
    :m_light,
    :m_sink_to_current);

-- name: write_junction_form_factor*!
INSERT INTO junction_form_factor (form_factor_id, corr_id, corr_type)
VALUES (:form_factor_id, :corr_id, :corr_type);

-- name: write_alias_quark_mass*!
INSERT INTO alias_quark_mass (ens_id, mq, alias)
VALUES (:ens_id, :mq, :alias)
ON CONFLICT DO NOTHING;

-- name: write_sign_form_factor*!
INSERT INTO sign_form_factor (ens_id, spin_taste_current, sign)
VALUES (:ens_id, :spin_taste_current, :sign)
ON CONFLICT DO NOTHING;

-- name: get_alias_form_factor
SELECT  form_factor_id,
        alias_light,
        alias_heavy,
        alias_spectator
FROM    alias_form_factor
JOIN    form_factor USING(form_factor_id)
WHERE   ens_id = :ens_id;

-- name: get_existing_transition_name
SELECT form_factor_id FROM transition_name;

-- name: write_transition_name*!
INSERT INTO transition_name (form_factor_id, mother, daughter, process)
VALUES (:form_factor_id, :mother, :daughter, :process);

--name: write_Vi_form_factors!
INSERT INTO form_factor(
    ens_id, momentum, spin_taste_current,
    spin_taste_sink, spin_taste_source,
    m_spectator, m_heavy, m_light, m_sink_to_current
)
SELECT
    ens_id, momentum,
    'Vi-S' as spin_taste_current,
    spin_taste_sink, spin_taste_source,
    m_spectator, m_heavy, m_light, m_sink_to_current
FROM form_factor
WHERE (spin_taste_current = 'V1-S')
ON CONFLICT DO NOTHING;