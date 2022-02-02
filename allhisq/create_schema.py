"""
Creates the tables in an analysis database.
Input to this script is a single text file with credentials / database settings, assumed
to be stored as a dict / json.
For example:
{
    'host': <hostname>,
    'port': <port>,
    'user': <username>,
    'password': <password>,
    'database': <database>,
}
"""
import pathlib
from contextlib import contextmanager
import os
import sqlalchemy as sqla
import aiosql
import sys
import yaml

def main():

    if len(sys.argv) != 2:
        raise ValueError("Usage: create_schema.py /path/to/credentials")
    path = sys.argv[1]

    with open(path) as ifile:
        db_settings = yaml.load(ifile, yaml.SafeLoader)

    sql = SQLWrapper(db_settings)
    with sql.connection as conn:
        sql.queries.create_schema(conn)


def abspath(dirname):
    """
    Builds the absolute path of relative directories with respect to this file's location, i.e.,
    ./dir/  --> full/path/to/this/directory/dir/
    Args:
        dirname: str, directory name
    Returns:
        str: a full path
    """
    return os.path.join(pathlib.Path(__file__).parent.absolute(), dirname)


@contextmanager
def connection_scope(engine):
    """
    Context manager for working with raw low-level DBAPI connections.
    Args:
        engine: connection engine
    """
    connection = engine.raw_connection()
    try:
        yield connection
    except:
        print("Issue encountered in connection_scope")
        raise
    finally:
        connection.commit()
        connection.close()


def make_engine(db_settings, echo=False):
    settings = dict(db_settings)
    url = 'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
    if settings['database'] not in ('fermilab_milc_prd', 'atlytle_dev', 'atlytle_prd', 'atlytle'):
        raise ValueError("Invalid database")
    return sqla.create_engine(url.format(**settings), echo=echo)


class SQLWrapper:
    """Wrapper class for basic database I/O operations."""
    def __init__(self, db_settings, echo=False, driver="psycopg2"):
        if driver != "psycopg2":
            raise NotImplementedError("Only psycopg2 currently supported, not", driver)
        self.engine = make_engine(db_settings, echo=echo)
        self.queries = aiosql.from_path(abspath("sql/postgres"), driver)

    @property
    def connection(self):
        """Wrapper for accessing the context manager for database connection."""
        return connection_scope(self.engine)

if __name__ == '__main__':
    main()