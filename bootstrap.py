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

def main():
    home = "/home/wjay/dbs/allHISQ/"
    for root, dirs, files in os.walk(home):
        dirs.sort() # modify in place to search in order
        for fname in files:
            # Fermilab-MILC stores results in sqlite databases
            # Each ensemble has its own database
            # Filenames start with "l{ns}{nt}f..."
            if fname.starswith("l") and fname.endswith("sqlite"):
                process_database(fname)
                
def process_database(fname):

    print("[+] Processing the database {0}".format(fname))
    engine = db.make_engine()
    sqlite_engine = db.make_engine(sqlite=True, db_choice=fname)
    
    # Record ensemble in analysis database
    name = fname.rstrip('.sqlite')
    ns, nt = extract_nsnt(name)
    ens = {'name':name, 'ns':ns, 'nt':nt}
    query = io.build_upsert_query(engine, 'ensemble', ens)
    engine.execute(query)
                
    # Grab ens_id for correlators
    ens_id = fetch_ens_id(engine, ens)
    if ens_id is None:
        msg = "Missing ensemble? ens ={0}".format(ens)
        raise ValueError(msg)

    # Record the names of correlators in analysis database
    corr_names = fetch_corr_names(sqlite_engine)
    write_corr_names(engine, corr_names, ens_id)                

def fetch_ens_id(engine, src, verbose=False):
    query = """
        SELECT id FROM ensemble where
        (name='{name}') and (ns='{ns}') and (nt='{nt}')""".format(**src)
    return io.fetch_id(engine, query, verbose)

def fetch_corr_names(engine):
    query = """SELECT name FROM correlators"""
    df = pd.read_sql_query(query, engine)
    return df['name'].unique().values

def write_corr_names(engine, corr_names, ens_id):
    print("[+] Writing correlator names to analysis database.")
    src = {'ens_id' : ens_id}
    total = len(corr_names)
    for count, corr_name in enumerate(corr_names):
        progress(count, total, corr_name)
        src['name'] = corr_name
        query = io.build_upsert_query(engine, 'correlation_function', src)
        engine.execute(query)

def extract_nsnt(name):
    nsnt = name.split('f')[0]
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

