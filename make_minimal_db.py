"""
Makes a minimal set of tables for an analysis database.
"""
import db_connection as db

def main():

    ensemble = """
    CREATE TABLE IF NOT EXISTS ensemble
    (
      id serial PRIMARY KEY,
      name text,  -- e.g., "HISQ 2+1+1"
      ns integer, 
      nt integer,
      UNIQUE(name, ns, nt)
    );"""

    correlation_function = """
    CREATE TABLE IF NOT EXISTS correlation_function
    (
      id SERIAL PRIMARY KEY,
      ens_id integer REFERENCES ensemble (id),
      name   text NOT NULL DEFAULT '',
      UNIQUE(ens_id, name)
    );"""

    reduction_two_point = """
    CREATE TABLE IF NOT EXISTS reduction_two_point
    (
      id SERIAL PRIMARY KEY,
      ens_id integer REFERENCES ensemble (id),
      fold           boolean NOT NULL, 
      block_length   integer NOT NULL,
      nsamples       integer NOT NULL,
      n_time_sources integer NOT NULL,
      log            text DEFAULT '',
      UNIQUE(fold, block_length, nsamples, n_time_sources)
    );"""

    analysis_two_point = """
    CREATE TABLE IF NOT EXISTS analysis_two_point
    (
      id SERIAL PRIMARY KEY,
      tmin        integer,         -- Inclusive
      tmax        integer,         -- Exclusive
      prior_alias text DEFAULT '', -- Some quick nickname
      prior       text DEFAULT '', -- The prior itself
      UNIQUE(tmin, tmax, prior_alias)
    );"""

    results_two_point = """
    CREATE TABLE IF NOT EXISTS result_two_point
    ( 
      id SERIAL PRIMARY KEY,
      ens_id       integer REFERENCES ensemble (id),
      corr_id      integer REFERENCES correlation_function (id),
      reduction_id integer REFERENCES reduction_two_point (id),
      anal_id      integer REFERENCES analysis_two_point (id),
      chi2    float,
      npoints integer,
      q       float,
      params  text,
      log     text DEFAULT '',
      calcdate timestamp with time zone,
      UNIQUE(ens_id, corr_id, reduction_id, anal_id)
    );"""

    campaign_two_point = """
    CREATE TABLE IF NOT EXISTS campaign_two_point
    (
      id SERIAL PRIMARY KEY, 
      result_id integer REFERENCES result_two_point (id),
      ens_id    integer REFERENCES ensemble (id), 
      corr_id   integer REFERENCES correlation_function (id),
      UNIQUE(ens_id, corr_id)
    );"""

    systematic_two_point = """
    CREATE TABLE IF NOT EXISTS systematic_two_point
    (
      id SERIAL PRIMARY KEY,
      ens_id  integer REFERENCES ensemble (id),
      corr_id integer REFERENCES correlation_function (id),
      systematic text NOT NULL,
      UNIQUE(ens_id, corr_id)
    );"""

    engine = db.make_engine()

    queries = [
        ensemble,
        correlation_function,
        reduction_two_point,
        analysis_two_point,
        results_two_point,
        campaign_two_point,
        systematic_two_point
    ]
    for query in queries:
        print(query)
        engine.execute(query)


if __name__ == '__main__':
    main()
