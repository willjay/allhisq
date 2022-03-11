"""
Script to populate the table strong_coupling.
Various pieces in the analysis pipeline use rough values of the strong coupling constant
to estimate the size of expected discretization effects. Precise values are not typically
necessary.
"""
import sys
import yaml
import pandas as pd
from db_tools import SQLWrapper
import conventions

def main():

    if len(sys.argv) != 2:
        raise ValueError("Usage: python populate_strong_coupling /path/to/credentials")
    path = sys.argv[1]

    # Build database connections
    with open(path) as ifile:
        db_settings = yaml.load(ifile, yaml.SafeLoader)
    sql = SQLWrapper(db_settings)

    # Grab results and write
    df = pd.read_sql(sql.queries.get_lattice_spacing_id.sql, sql.engine)
    df = pd.merge(df, conventions.strong_coupling, on='a_fm')
    with sql.connection as conn:
        sql.queries.write_strong_coupling(conn, df.to_dict(orient='records'))

if __name__ == '__main__':
    main()
