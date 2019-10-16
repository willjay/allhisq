"""
 * Looks at existing databases
 * Loads ensemble-level information
 * Loads correlator names
 * 
"""

import db_connection as db    
import db_io as io
import pandas as pd
import os, sys
import sqlalchemy as sqla

def main():
    home = "/home/wjay/dbs/allHISQ/"
    print("[+] Looking for sqlite databases in {0}".format(home))
    for root, dirs, files in os.walk(home):
        dirs.sort() # modify in place to search in order
        print("[+] root {0}".format(root))
        for fname in files:
            # Fermilab-MILC stores results in sqlite databases
            # Each ensemble has its own database
            # Filenames start with "l{ns}{nt}f..."
            if fname.startswith("l") and fname.endswith("sqlite"):
                process_database(fname, root)
                
def process_database(fname, location):

    print("[+] Processing the database {0}".format(fname))
    engine = db.make_engine()
    full_path = os.path.join(location, fname)
    sqlite_engine = db.make_engine(sqlite=True, db_choice=full_path)

    # Record ensemble in analysis database
    name = fname.rstrip('.sqlite')
    ns, nt = extract_nsnt(name)
    
    # Check for existing ensembles
    existing = fetch_existing(engine)    
    if (location not in existing['location'].values) and (name not in existing['name'].values):
        ens = {'name':name, 'ns':ns, 'nt':nt, 'location':location, 'type': 'sqlite'}
        query = io.build_upsert_query(engine, 'ensemble', ens)
        engine.execute(query)
        ens_id = fetch_ens_id(engine, ens)
    
        # Record location of database
        ens['ens_id'] = ens_id
        query = io.build_upsert_query(engine, 'external_database', ens)
        engine.execute(query)
 
        if ens_id is None:
            msg = "Missing ensemble? ens ={0}".format(ens)
            raise ValueError(msg)

        # Record the names of correlators in analysis database
        corr_names = fetch_corr_names(sqlite_engine)
        write_corr_names(engine, corr_names, ens_id)                
    else:
        print("[+] Skipping the database {0} (already processed).".format(fname))

def fetch_existing(engine, verbose=False):

    query = """
        SELECT ensemble.*, external_database.location, external_database.type	
        FROM ensemble JOIN external_database ON ensemble.ens_id = external_database.ens_id;"""
    if verbose:
        print(query)
    return pd.read_sql_query(query, engine)

def fetch_ens_id(engine, src, verbose=False):
#    query = """
#        SELECT ens_id FROM ensemble where
#        (name='{name}') and (ns='{ns}') and (nt='{nt}')""".format(**src)
#    return io.fetch_id(engine, query, verbose, tag='ens_id')
    return io.fetch_id(engine, 'ensemble', src, verbose=False)

def fetch_corr_names(engine):
            
    query = """SELECT name FROM correlators"""
    df = pd.read_sql_query(query, engine)
    return df['name'].unique()

def write_corr_names(engine, corr_names, ens_id):
    print("[+] Writing correlator names to analysis database.")
    src = {'ens_id' : ens_id}
    total = len(corr_names)
    for count, corr_name in enumerate(corr_names):
        progress(count, total, corr_name)
        src['name'] = corr_name
        query = io.build_upsert_query(engine, 'correlator_n_point', src)
        engine.execute(query)

def extract_nsnt(name):
    nsnt = name.split('f')[0].lstrip('l')
    if len(nsnt) == 4:
        ns = int(nsnt[:2])
        nt = int(nsnt[2:])
    else:
        raise ValueError("Unrecognized 'nsnt' with length {0}".format(len(nsnt)))
    return ns,nt

def progress(count, total, status=''):
    """
    Displays progress bar of the form:
    [========================================--------------------] 66.6% status
    Args:
        count: numeric, progress counter
        total: total, the end value of the counter
        status: str, name of the status to appear on the right
    Returns:
        None

    As suggested by Rom Ruben 
    (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)    
    """

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  

if __name__ == '__main__':
    main()

