-- name: get_correlator_names
-- Fetch the names of correlators for extracting a form factor.
SELECT RTRIM(name, '_-fine') AS basename FROM correlators WHERE
(name LIKE :corr3) OR (name LIKE :mother) OR (name LIKE :daughter);