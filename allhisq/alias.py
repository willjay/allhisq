"""
Utility functions for aliasing correlation functions to standardize
their treatment in analyses of form factors.
"""
import re
import collections

MassTriplet = collections.namedtuple("MassTriplet", ['snk', 'spec', 'src'])


def match_3pt(basename):
    """
    Parses keys for three-point correlation functions using regex.
    Matches keys, for example, of the form:
    'P5-P5_S-S_T13_m0.389_RW_RW_x_d_m0.0012_m0.0012_p000'.
    The ordering of the mass tags is meaningful and corresponds to
    (sink, spectator, source).
    Args:
        basename: str
    Returns:
        dict or None: parameters extracted from basename.
            None if basename does not match.
    """
    regex_3pt = re.compile(
        r'^'
        r'(P5-P5|A4-A4)_'
        r'(S-S|V1-S|V2-S|V3-S|V4-V4|T14-V4|T24-V4|T34-V4)_'
        r'T(\d+)_'
        r'm(\d+.\d+)_RW_RW_x_d_m(\d+.\d+)_m(\d+.\d+)_'
        r'(p\d+)'
        r'$'
    )
    match = re.search(regex_3pt, basename)
    if match:
        return {
            'gamma_src_snk': match.group(1),
            'spin_taste_current': match.group(2),
            't_snk': int(match.group(3)),
            'mass_triplet': MassTriplet(
                float(match.group(4)),
                float(match.group(5)),
                float(match.group(6))
            ),
            'momentum': match.group(7),
        }
    return match


def match_2pt(basename):
    """
    Parses keys for three-point correlation functions using regex.
    Matches keys, for example, of the form:
    P5-P5_RW_RW_d_d_m0.0012_m0.0012_p000.
    The ordering of the mass tags is not meaningful for two-point
    correlation functions.
    Args:
        basename: str
    Returns:
        dict or None: parameters extracted from basename.
            None if basename does not match.
    """
    regex_2pt = re.compile(
        r'^'
        r'(P5-P5|A4-A4)_'
        r'RW_RW_d_d_'
        r'm(\d+.\d+)_m(\d+.\d+)_'
        r'(p\d+)'
        r'$'
    )
    match = re.search(regex_2pt, basename)
    if match:
        return {
            'basename': basename,
            'gamma_src_snk': match.group(1),
            'masses': set((float(match.group(2)), float(match.group(3)))),
            'momentum': match.group(4),
        }
    return match

def is_sink(corr2, corr3):
    """
    Checks whether or not corr2 corresponds to the sink correlator of corr3.
    By convention, the sink operator corresponds to a particle at rest
    and so must have zero momentum.
    Args:
        corr2, corr3: dicts
    Returns:
        bool
    """
    mass_triplet = corr3['mass_triplet']
    mass_agrees = (corr2['masses'] == set((mass_triplet.snk, mass_triplet.spec)))
    momentum_agrees = (corr2['momentum'] == 'p000')
    operator_agrees = corr2['gamma_src_snk'] == corr3['gamma_src_snk']
    return mass_agrees and momentum_agrees and operator_agrees


def is_source(corr2, corr3):
    """
    Checks whether or not c2 corresponds to the source correlator of c3.
    Args:
        c2, c3: dicts
    Returns:
        bool
    """
    mass_triplet = corr3['mass_triplet']
    mass_agrees = (corr2['masses'] == set((mass_triplet.src, mass_triplet.spec)))
    momentum_agrees = (corr2['momentum'] == corr3['momentum'])
    return mass_agrees and momentum_agrees


def all_equal(iterator):
    """Checks whether all entries in an interable are equal."""
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def get_aliases(basenames):
    """
    Gets aliases of the form 'source', 'sink', and t_snk for the correlators,
    which standardizes input for later analysis of form factors.
    Args:
        basenames: list of names (str)
    Returns:
        dict: the mapping of the aliases
    Raises:
        ValueError
    """
    aliases = {}
    # Isolate the three-point functions and extract their parameters
    three_points = []
    for basename in basenames:
        corr3 = match_3pt(basename)
        if corr3:
            aliases[basename] = corr3.pop('t_snk')
            three_points.append(corr3)

    # Verify that parameters match (aside from t_snk)
    if not all_equal(three_points):
        raise ValueError("Three-point correlation functions do not match.")
    corr3 = three_points[0]

    # Isolate the two-point functions and extract their parameters
    two_points = []
    for basename in basenames:
        corr2 = match_2pt(basename)
        if corr2:
            two_points.append(corr2)

    # Isoalte the two-point functions associated with the source and sink
    source, = [corr2 for corr2 in two_points if is_source(corr2, corr3)]
    sink, = [corr2 for corr2 in two_points if is_sink(corr2, corr3)]
    aliases[source['basename']] = 'source'
    aliases[sink['basename']] = 'sink'

    return aliases


def apply_naming_convention(aliases, convention=None):
    """
    Applies a naming convention to aliases.
    By default, the naming convention applies only to the source and sink
    correlators, which are aliased as'light-light' and 'heavy-light',
    respectively. This default is sensible for the common usage case of
    "heavy-to-light" transitions like D-to-pi, for example. The default
    behavior is the following:

    {'P5-P5_RW_RW_d_d_m0.389_m0.0012_p000': 'source',
     'P5-P5_RW_RW_d_d_m0.0012_m0.0012_p000': 'sink',
     ... <other keys related to three-point functions>
     }
    -->
    {'P5-P5_RW_RW_d_d_m0.389_m0.0012_p000': 'light-light',
     'P5-P5_RW_RW_d_d_m0.0012_m0.0012_p000': 'heavy-light',
     ... <other keys related to three-point functions>
     }
    """
    if convention is None:
        convention = {'source': 'light-light', 'sink': 'heavy-light'}
    for key in aliases.keys():
        val = aliases[key]
        if val in convention:
            aliases[key] = convention[val]
    return aliases


class MesonNames(object):
    """
    Constructs a quick look-up table for meson names based on the physical identify of the quarks.
    E.g., ('1.0 m_light', '1.0 m_light') corresponds to a pion.
    Typical usage:
    >>> table = MesonNames()
    >>> table.identify('1.0 m_light', '1.0 m_strange')
    'K'
    """
    def __init__(self):
        self.build_state_table()

    def identify(self, mass1, mass2):
        """
        Identifies the physical name for a state give a pair of masses, working in either order for
        masses. For example:
        ('1.0 m_light', '1.0 m_light') --> 'pi'
        ('1.0 m_light', '1.0 m_strange') --> 'K'
        """
        state = self.states.get((mass1, mass2), None)
        if not state:
            state = self.states.get((mass2, mass1), None)
        return state

    def build_state_table(self):
        """
        Builds a "table", i.e., a dictionary of the form {(mass1, mass2): 'state_name'}.
        """
        # Define the quark masses in our dataset
        light_quarks = ['1.0 m_light', '0.1 m_strange', '0.2 m_strange']
        charm_ratios = ['0.9', '1.0', '1.1', '1.4', '1.5', '2.0', '2.5']
        charm_quarks = ['{0} m_charm'.format(rat) for rat in charm_ratios]
        bottom_quarks = ['3.0 m_charm', '4.0 m_charm', '4.2 m_charm']

        # Build up a dictionary of states (m1, m2) : state
        states = {}
        for light in light_quarks:
            states[(light, light)] = 'pi'
            states[(light, '1.0 m_strange')] = 'K'
            for charm in charm_quarks:
                states[(light, charm)] = 'D'
            for bottom in bottom_quarks:
                states[(light, bottom)] = 'B'
        for charm in charm_quarks:
            states[('1.0 m_strange', charm)] = 'Ds'
        for bottom in bottom_quarks:
            states[('1.0 m_strange', bottom)] = 'Bs'
        self.states = states
