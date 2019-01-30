import sqlalchemy as sqla
import db_settings

def make_engine(NEED_PASSWORD=True, PORT_FWD=False, port=8887, db_choice=None,                        sqlite=False):
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
        return sqla.create_engine(url.format(**settings), echo=False)

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
            
        return sqla.create_engine(url.format(**settings))    