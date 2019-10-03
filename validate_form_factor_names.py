"""
Validates the correlator names appearing in the
Postgres analysis database for use with the existing
analysis framework.
"""
import pandas as pd
import db_connection as db
import alias


def main():
    """Runs the main function."""
    # Grab all possible form factors analyses and the correlator names
    # associated with them
    query = (
        "SELECT distinct(rtrim(corr.name,'-_finelos')) as basename, "
        "ff.ens_id, ff.form_factor_id, m_light, m_heavy, m_spectator, "
        "momentum, spin_taste_current, transition_name.process, "
        "junction.corr_type "
        "FROM form_factor AS ff "
        "JOIN junction_form_factor AS junction "
        "ON junction.form_factor_id = ff.form_factor_id "
        "JOIN correlator_n_point AS corr "
        "ON corr.corr_id = junction.corr_id "
        "JOIN transition_name "
        "ON transition_name.form_factor_id = ff.form_factor_id;"""
    )
    engines = db.get_engines()
    df = pd.read_sql_query(query, engines['postgres'])

    # Exlclude two-point functions starting with 'A4-A4'
    # These are new and tend to mess up my pipeline.
    # This is a known conflict, which I can work around.
    # Some adjustments and thinking will be necessary to
    # include them gracefully. What we want to validate
    # is that no *other* conflicts occur.

    exclude_prefix = 'A4-A4'
    # exclude_prefix = 'P5-P5'
    print(
        "Excluding two-point functions with the prefix '{}'.".
        format(exclude_prefix)
    )

    is_two_point =\
        (df['corr_type'] == 'light-light') |\
        (df['corr_type'] == 'heavy-light')
    has_prefix = df['basename'].str.startswith(exclude_prefix)
    mask = ~(is_two_point & has_prefix)
    df = df[mask]

    # Conduct the validation for each form factor analysis
    groups = df.groupby('form_factor_id')
    for form_factor_id, subdf in groups:
        basenames = subdf['basename'].values
        try:
            _ = alias.get_aliases(basenames)
        except ValueError as err:
            print("Invalid basenames on form_factor_id= %d" % form_factor_id)
            print(basenames)
            raise err

    print(
        "Validation was successful: no conflicts found. "
        "But be sure to exclude two-point functions starting with 'A4-A4' "
        "in your analysis scripts."
    )


if __name__ == '__main__':
    main()
