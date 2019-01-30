import sqlalchemy as sqla
import pandas as pd

def fetch_id(engine, query, verbose=False):
    """ 
    Fetch id from database given a query with error handling.
    Args:
        engine: database connection "engine"
        query: str, containing a "raw" SQL query which will return a single id.
        verbose: bool whether to print diagnostic messages
    Returns:
        int, containing the id or None if not found
    """
    if verbose:
        print(query)
    tmp_df = pd.read_sql_query(query, engine)        
    if len(tmp_df) == 1:
        return tmp_df['id'].item()
    elif len(tmp_df) == 0:
        return None
    else:
        raise ValueError("Non-unique id encountered.")

def build_upsert_query(engine, table_name, src_dict, do_update=False):
    """
    Builds a raw SQL query for "upserting" data into the database.
    Args:
        engine: database connection "engine"
        table_name: str, name of the database table for the upsert
        srs_dict: dict, containing the column information for the upsert
        do_update: bool, whether or not to update the row when a conflict is
            encountered. Default is False.
    Returns:
        query: str, containing the raw SQL query.

    Remarks:
        Intended use is, e.g., 
        >>> engine = db.make_engine()
        >>> query = build_upsert_query(engine, 'table_name', src_dict)
        >>> engine.execute(query)
    """
    def _for_pgsql(value, dtype):
        """
        Converts a python datatype to the appropriate string (including, e.g., \
        the necessary single quotes and/or brackets ) for use in a raw \
        postgresql query.
        Args:
            value: (various datatypes) the value in question
            dtype: str, the datatype
        Returns:
            str, with the necessary formatting
        """
        if dtype.startswith(('int','float','double','numeric')): 
            if value is None:
                return "Null"
            elif str(value) == 'NaN':
                return "'NaN'"
            elif dtype.endswith('[]'):
                value = ', '.join([str(v) for v in value])
                value = "'{" + value + "}'"
                return value
            else:
                return str(value)
        elif dtype.startswith('time'):
            if value is None:
                return "Null"
            else:
                return "'" + str(value) + "'"
        elif dtype.startswith('bool'):
            if value is None:
                raise ValueError("Error: bool should not be None.")
            else:
                if str(value).startswith(('t','T')):
                    return str(True)
                else:
                    return str(False)
        elif dtype.startswith('json'):
            # In this case, value itself should be a dict
            value = ','.join(['"{k}":"{v}"'.format(k=k,v=v) for k,v in value.iteritems()])
            value = "'{" + value + "}'"
            return value
        elif dtype == 'text[]':
            value = ', '.join(['"'+str(v)+'"' for v in value])
            value = "'{" + str(value) + "}'"
            return value
        else:
            if str(value).startswith('$delim$') and\
               str(value).endswith('$delim$'):
               return str(value)
            if '::' in str(value):
                value = str(value).split("::")[0].strip("'")
            return "'" + str(value) + "'"

    def _get_values(uprow, types):
        """
        Gets a list of values for use in a raw SQL query, e.g., 
        
        INSERT INTO table_name 
        (column1, column2, ...)
        VALUES
        (value1, value2, ...);
            
        This function returns a string "value1, value2, ..."
        Args:
            uprow: dict, containing the values 
            types: dict, containing the data types of the values
        Return:
            str, containing the values as described above.
        """
        tmp_uprow = {k: _for_pgsql(v, types[k]) for k,v in uprow.iteritems()}
        mappable = ",".join(["{" + str(k) + "}" for k in uprow.keys()])
        values = mappable.format(**tmp_uprow)
        return values

    def _get_set_pairs(uprow, types):
        """
        Gets a list of "set pairs" for use in a raw SQL query, e.g., 
        
        INSERT INTO table_name 
        (column1, column2, ...)
        VALUES
        (value1, value2, ...)
        ON CONFLOCT (column1) DO UPDATE
        SET
        column1=value1,
        column2=value2

        This function returns a string "column1=value1, column=value2
        Args:
            uprow: dict, containing the values 
            types: dict, containing the data types of the values
        Return:
            str, containing the "set pairs" as described above.
        """
        pairs = []
        for k, v in uprow.iteritems():            
            pairs.append("{0}={1}".format(k, _for_pgsql(v, types[k])))
        return ", ".join(pairs)
        
    
    ## Mirror table from DB
    meta = sqla.MetaData(bind=engine)
    insp = sqla.inspect(engine)
    table = sqla.Table(table_name, meta, autoload=True, autoload_with=engine)
    table_cols = [str(col).split('.')[1] for col in table.columns]

    ## Collect dict entries that also appear in the table as a "row"
    uprow = {key: src_dict[key] for key in src_dict if key in table_cols}
    
    ## Load defaults and collect types
    types = {}
    for column in insp.get_columns(table_name, default=True):
        name = column['name']
        if (name not in uprow) and (name != 'id'):
            uprow[name] = column['default']
        types[name] = str(column['type']).lower()

    ## Build base query
    columns = "{keylist}".format(keylist=', '.join(uprow.keys()))
    values = _get_values(uprow, types)

    query = "INSERT INTO {table_name}\n".format(table_name=table_name)
    query+= "({columns})\n".format(columns=columns)
    query+= "VALUES\n"
    query+= "({values})\n".format(values=values)

    ## Fetch unique columns
    unique_constraints= insp.get_unique_constraints(table_name)
    
    ## Handle potential conflicts
    if len(unique_constraints) > 0:                
        unique_cols = insp.get_unique_constraints(table_name)[0]['column_names']
        if len(unique_cols) >1:
            unique_cols = ", ".join([str(col) for col in list(unique_cols)])
        else:
            unique_cols = ', '.join(unique_cols)
        if do_update:
            set_clause = "ON CONFLICT ({unique_cols}) DO UPDATE SET\n".\
                format(unique_cols=unique_cols)
            set_clause +=  _get_set_pairs(uprow, types)
            query += set_clause
        else:                                
            query+= "ON CONFLICT ({unique_cols}) DO NOTHING\n".\
                format(unique_cols=unique_cols)
    else:
        ## No unique constraints, look for primary key instead
        primary_key = [c for c in table.columns if c.primary_key]
        if len(primary_key) == 1:
            primary_key, = primary_key
            ## Ditch reference to foreign table
            if '.' in str(primary_key):
                primary_key = str(primary_key).split('.')[-1]
        else:
            tmp = []
            for c in primary_key:
            ## Ditch reference to foreign table
                if '.' in str(c):
                    c = str(c).split('.')[-1]
                tmp.append(c)
            primary_key = ", ".join(tmp)
        if do_update:
            set_clause = "ON CONFLICT ({primary_key}) DO UPDATE SET\n".\
                format(primary_key=primary_key)
            set_clause += _get_set_pairs(uprow, types)
            query += set_clause                
        else:
            query+= "ON CONFLICT ({primary_key}) DO NOTHING\n".\
                format(primary_key=primary_key)

    query += ';'
            
    return query