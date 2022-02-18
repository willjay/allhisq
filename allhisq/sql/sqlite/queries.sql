-- name: get_correlators
SELECT name, param
FROM   correlators
JOIN   parameters
ON     correlators.parameter_id = parameters.id;