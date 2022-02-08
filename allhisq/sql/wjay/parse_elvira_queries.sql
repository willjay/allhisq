-- name: write_ensemble$
INSERT INTO ensemble (name, ns, nt, description)
VALUES  (:name, :ns, :nt, :description)
ON CONFLICT DO NOTHING;
SELECT  ens_id
FROM    ensemble
WHERE (name, ns, nt) = (:name, :ns, :nt);

-- name: write_lattice_spacing!
INSERT INTO lattice_spacing (ens_id, a_fm, type)
VALUES (:ens_id, :a_fm, 'nominal')
ON CONFLICT DO NOTHING;

-- name: write_form_factor*!
INSERT INTO form_factor(
    ens_id,
    momentum,
    spin_taste_current,
    spin_taste_sink,
    spin_taste_source,
    m_spectator,
    m_heavy,
    m_light,
    m_sink_to_current)
SELECT
    ens_id,
    :momentum,
    :spin_taste_current,
    :spin_taste_sink,
    :spin_taste_source,
    :m_spectator,
    :m_heavy,
    :m_light,
    :m_sink_to_current
FROM ensemble WHERE :name = ensemble.name
ON CONFLICT DO NOTHING;

-- name: write_transition_name*!
INSERT INTO transition_name(
    form_factor_id,
    mother,
    daughter,
    process)
SELECT
    form_factor_id,
    :mother,
    :daughter,
    :process
FROM form_factor
JOIN ensemble USING(ens_id)
WHERE
    ( name,  momentum,  spin_taste_current,  m_spectator,  m_heavy,  m_light)=
    (:name, :momentum, :spin_taste_current, :m_spectator, :m_heavy, :m_light)
ON CONFLICT DO NOTHING;

-- name: write_alias_quark_mass*!
INSERT INTO alias_quark_mass(
    ens_id,
    mq,
    alias)
SELECT
    ens_id,
    :mq,
    :alias
FROM ensemble WHERE :name = ensemble.name
ON CONFLICT DO NOTHING;

-- name: write_sign_form_factor!
INSERT INTO sign_form_factor(
    ens_id,
    spin_taste_current,
    sign)
SELECT
    ens_id,
    :spin_taste_current,
    :sign
FROM ensemble WHERE :name = ensemble.name
ON CONFLICT DO NOTHING;

-- name: write_correlator_n_point*!
INSERT INTO correlator_n_point(
    ens_id,
    name)
SELECT
    ens_id,
    :corr_name
FROM ensemble WHERE ensemble.name = :ens_name
ON CONFLICT DO NOTHING;

-- name: write_junction_form_factor*!
INSERT INTO junction_form_factor(
        form_factor_id,
        corr_id,
        corr_type)
SELECT  form_factor_id,
        corr_id,
        :corr_type
FROM    ensemble
JOIN    form_factor USING(ens_id)
JOIN    transition_name USING(form_factor_id)
JOIN    correlator_n_point USING(ens_id)
WHERE   (ensemble.name = :ens_name)
    AND (process = :process)
    AND (correlator_n_point.name = :corr_name)
ON CONFLICT DO NOTHING;

-- name: write_external_database!
INSERT INTO external_database(
    ens_id,
    name,
    location,
    type)
SELECT
    ens_id,
    :name,
    :location,
    'hdf5'
FROM ensemble
WHERE ensemble.name = :ens_name
ON CONFLICT DO NOTHING;
;

-- name: write_meta_data_correlator*!
INSERT INTO meta_data_correlator(
    ens_id,
    correlator_key,
    quark_mass,
    antiquark_mass,
    momentum,
    has_sequential)
SELECT
    ens_id,
    :corr_name,
    :quark_mass,
    :antiquark_mass,
    :momentum,
    :has_sequential
FROM ensemble WHERE ensemble.name = :ens_name
ON CONFLICT DO NOTHING;

--name: write_strong_coupling*!
INSERT INTO strong_coupling(
    lattice_spacing_id,
    coupling_name,
    coupling_scale,
    coupling_value)
SELECT
    lattice_spacing.id,
    'alphaV',
    '2/a',
    :coupling_value
FROM    ensemble
JOIN    lattice_spacing USING(ens_id)
WHERE   (lattice_spacing.type = 'nominal')
        AND (ensemble.name = :ens_name)
ON CONFLICT DO NOTHING;




