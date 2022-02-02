"""
TODO: doc here
"""
import pathlib
from contextlib import contextmanager
import os
import sqlalchemy as sqla
import aiosql


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


def make_engine(settings, echo=False, driver='psycopg2'):
    if driver == 'psycopg2':
        url = 'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
        if settings['database'] not in ('fermilab_milc_prd', 'atlytle_dev', 'atlytle_prd', 'atlytle'):
            raise ValueError("Invalid database")
    elif driver == 'sqlite3':
        # unix/mac seems to need 3, 4, or 5 leading slashes in total
        # no idea why this is the case
        url = 'sqlite:///{database}'
    else:
        raise ValueError("Unsupport driver", driver)
    return sqla.create_engine(url.format(**dict(settings)), echo=echo)


class SQLWrapper:
    """Wrapper class for basic database I/O operations."""
    def __init__(self, db_settings, echo=False, driver="psycopg2"):
        if driver not in ("psycopg2", "sqlite3"):
            raise ValueError("Unrecognized driver", driver)
        self.engine = make_engine(db_settings, echo=echo, driver=driver)
        if driver == "psycopg2":
            self.queries = aiosql.from_path(abspath("sql/postgres/"), driver)
        elif driver == "sqlite3":
            self.queries = aiosql.from_path(abspath("sql/sqlite/"), driver)

    @property
    def connection(self):
        """Wrapper for accessing the context manager for database connection."""
        return connection_scope(self.engine)