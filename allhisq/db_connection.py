import os
from contextlib import contextmanager
import pandas as pd
import sqlalchemy as sqla
from sqlalchemy.orm import Session
from . import db_settings

def get_engines():
    """
    Gets a dictionary of engines.
    Keys are 'postgres' for the analysis database or ens_id (an integer) for the
    various external databases.
    """
    # Engine for analysis database
    engines = {'postgres': make_engine()}

    # Get locations of external databases
    query = """
        SELECT
            ext_db.ens_id,
            ens.ns,
            ens.nt,
            ext_db.name,
            ext_db.location,
            ext_db.type
        FROM ensemble as ens
        JOIN external_database AS ext_db
            on ens.ens_id = ext_db.ens_id;
        """
    df = pd.read_sql_query(query, engines['postgres'])

    # Create engines for external databases sqlite
    for _, row in df.iterrows():
        ens_id   = row['ens_id']
        name     = row['name']
        location = row['location']
        db_type  = row['type']
        db_name  = os.path.join(location,name+'.sqlite')

        if (db_type != 'sqlite'):
            raise ValueError("Expected sqlite databse")

        engines[ens_id] = make_engine(db_choice=db_name, sqlite=True)

    return engines

def make_engine(NEED_PASSWORD=True, PORT_FWD=False, port=8887, db_choice=None, sqlite=False, echo=False):
    """
    Makes an engine for interacting with SQL database directly with pandas.
    Args:
        NEED_PASSWORD: bool, whether a password is needed for the database
        PORT_FWD: bool, whether port forwarding is used for the database,
            e.g., when running remotely and port is *not* the default 5432
        port: int, port to use for forwarding; only used if PORT_FWD is True
    Returns:
	database engine
    """
    if sqlite:
        if db_choice is None:
            raise ValueError("Must specfiy db_choice for sqlite database.")
        settings = { 'database' : db_choice}
        # unix/mac seems to need 3, 4, or 5 leading slashes in total
        # no idea why this is the case
        url = 'sqlite:///{database}'
        return sqla.create_engine(url.format(**settings), echo=echo)

    else:
        settings = dict(db_settings.DATABASE)
        if db_choice is not None:
            settings['database'] = db_choice
        if PORT_FWD:
            settings['port'] = port
            settings['host'] = 'localhost'

        if NEED_PASSWORD:
            url = 'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
        else:
            url = 'postgresql+psycopg2://{user}@{host}:{port}/{database}'

        return sqla.create_engine(url.format(**settings), echo=echo)

@contextmanager
def session_scope(engine):
    """
    Provide a transactional scope around a series of operations.
    Recipe taken from the docs:
    [https://docs.sqlalchemy.org/en/13/orm/session_basics.html]
    See also the following thread for more context:
    [https://stackoverflow.com/questions/14799189/avoiding-boilerplate-session-handling-code-in-sqlalchemy-functions]
    """
    session = Session(engine)
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

@contextmanager
def connection_scope(engine):
    connection = engine.raw_connection()
    try:
        yield connection
    except:
        print("Issue encountered in connection_scope")
        raise
    finally:
        connection.commit()
        connection.close()
