"""
doc here
"""

import pandas as pd

form_factor_signs = pd.DataFrame([
    ['S-S', +1],
    ['V4-V4', -1],
    ['V1-S', -1],
    ['V2-S', +1],
    ['V3-S', -1],
    ['T14-V4', +1],
    ['T24-V4', -1],
    ['T34-V4', +1]],
    columns=['spin_taste_current', 'sign'])

quark_masses = pd.DataFrame([
    ###########
    # 0.15 fm #
    ###########
    [0.15, '1/27', 0.9292, '1.1 m_charm'],
    [0.15, '1/27', 0.8447, '1.0 m_charm'],
    [0.15, '1/27', 0.7602, '0.9 m_charm'],
    [0.15, '1/27', 0.0673, '1.0 m_strange'],
    [0.15, '1/27', 0.002426, '1.0 m_light'],
    ###########
    # 0.12 fm #
    ###########
    # 1/5 ms
    [0.12, '1/5', 0.8935, '1.4 m_charm'],
    [0.12, '1/5', 0.6382, '1.0 m_charm'],
    [0.12, '1/5', 0.5744, '0.9 m_charm'],
    [0.12, '1/5', 0.05252, '1.0 m_strange'],
    [0.12, '1/5', 0.010504, '0.2 m_strange'],
    # 1/27 ms "physical-mass"
    [0.12, '1/27', 0.8935, '1.4 m_charm'],
    [0.12, '1/27', 0.6382, '1.0 m_charm'],
    [0.12, '1/27', 0.5744, '0.9 m_charm'],
    [0.12, '1/27', 0.05252, '1.0 m_strange'],
    [0.12, '1/27', 0.001907, '1.0 m_light'],
    # 1/10 ms
    [0.12, '1/10', 0.8935, '1.4 m_charm'],
    [0.12, '1/10', 0.6382, '1.0 m_charm'],
    [0.12, '1/10', 0.5744, '0.9 m_charm'],
    [0.12, '1/10', 0.05252, '1.0 m_strange'],
    [0.12, '1/10', 0.005252, '0.1 m_strange'],
    ###########
    # 0.09 fm #
    ###########
    # 1/27 ms "physical-mass"
    [0.088, '1/27', 1.08, '2.5 m_charm'],
    [0.088, '1/27', 0.864, '2.0 m_charm'],
    [0.088, '1/27', 0.648, '1.5 m_charm'],
    [0.088, '1/27', 0.432, '1.0 m_charm'],
    [0.088, '1/27', 0.389, '0.9 m_charm'],
    [0.088, '1/27', 0.0363, '1.0 m_strange'],
    [0.088, '1/27', 0.0012, '1.0 m_light'],
    # 1/10 ms
    [0.088, '1/10', 1.08, '2.5 m_charm'],
    [0.088, '1/10', 0.864, '2.0 m_charm'],
    [0.088, '1/10', 0.648, '1.5 m_charm'],
    [0.088, '1/10', 0.432, '1.0 m_charm'],
    [0.088, '1/10', 0.389, '0.9 m_charm'],
    [0.088, '1/10', 0.0363, '1.0 m_strange'],
    [0.088, '1/10', 0.00363, '0.1 m_strange'],
    ###########
    # 0.06 fm #
    ###########
    # 1/27 ms "physical-mass"
    [0.057, '1/27', 1.144, '4.4 m_charm'],
    [0.057, '1/27', 0.858, '3.3 m_charm'],
    [0.057, '1/27', 0.572, '2.2 m_charm'],
    [0.057, '1/27', 0.286, '1.1 m_charm'],
    [0.057, '1/27', 0.257, '1.0 m_charm'],
    [0.057, '1/27', 0.022, '1.0 m_strange'],
    [0.057, '1/27', 0.0008, '1.0 m_light'],
    # 1/10 ms
    [0.057, '1/10', 1.144, '4.0 m_charm'],
    [0.057, '1/10', 0.858, '3.0 m_charm'],
    [0.057, '1/10', 0.572, '2.0 m_charm'],
    [0.057, '1/10', 0.286, '1.0 m_charm'],
    [0.057, '1/10', 0.257, '0.9 m_charm'],
    [0.057, '1/10', 0.024, '1.0 m_strange'],
    [0.057, '1/10', 0.0024, '0.1 m_strange'],
    # 1/5 ms
    [0.057, '1/5', 1.144, '4.0 m_charm'],
    [0.057, '1/5', 0.858, '3.0 m_charm'],
    [0.057, '1/5', 0.572, '2.0 m_charm'],
    [0.057, '1/5', 0.286, '1.0 m_charm'],
    [0.057, '1/5', 0.257, '0.9 m_charm'],
    [0.057, '1/5', 0.024, '1.0 m_strange'],
    [0.057, '1/5', 0.0048, '0.2 m_strange'],
    ###########
    # 0.04 fm #
    ###########
    # 1/5 ms
    [0.042, '1/5', 0.767, '4.2 m_charm'],
    [0.042, '1/5', 0.731, '4.0 m_charm'],
    [0.042, '1/5', 0.548, '3.0 m_charm'],
    [0.042, '1/5', 0.365, '2.0 m_charm'],
    [0.042, '1/5', 0.1827, '1.0 m_charm'],
    [0.042, '1/5', 0.164, '0.9 m_charm'],
    [0.042, '1/5', 0.01555, '1.0 m_strange'],
    [0.042, '1/5', 0.00311, '0.2 m_strange'],
    # 1/27 ms "physical-mass"
    [0.042, '1/27', 0.767, '4.2 m_charm'],
    [0.042, '1/27', 0.731, '4.0 m_charm'],
    [0.042, '1/27', 0.548, '3.0 m_charm'],
    [0.042, '1/27', 0.365, '2.0 m_charm'],
    [0.042, '1/27', 0.1827, '1.0 m_charm'],
    [0.042, '1/27', 0.164, '0.9 m_charm'],
    [0.042, '1/27', 0.01555, '1.0 m_strange'],
    [0.042, '1/27', 0.00569, '1.0 m_light'],],
    columns=['a_fm', 'description', 'mq', 'alias'])

ensembles = pd.DataFrame([
    [0.15, '1/27', 32, 48, 'l3248f211b580m002426m06730m8447-allHISQ'],
    [0.12, '1/27', 48, 64, 'l4864f211b600m001907m05252m6382-allHISQ'],
    [0.088, '1/27', 64, 96, 'l6496f211b630m0012m0363m432-allHISQ'],
    [0.088, '1/10', 48, 96, 'l4896f211b630m00363m0363m430-allHISQ'],
    [0.057, '1/27', 96, 192, 'l96192f211b672m0008m022m260-allHISQ'],
    [0.057, '1/10', 64, 144, 'l64144f211b672m0024m024m286-allHISQ'],
    [0.057, '1/5', 48, 144, 'l48144f211b672m0048m024m286-allHISQ'],
    [0.042, '1/5', 64, 192, 'l64192f211b700m00316m0158m188-allHISQ'],],
    columns=['a_fm', 'description', 'ns', 'nt', 'name'])

states = pd.DataFrame([
    ["0.1 m_strange", "0.9 m_charm", "d"],
    ["0.1 m_strange", "1.0 m_charm", "d"],
    ["0.1 m_strange", "1.1 m_charm", "d"],
    ["0.1 m_strange", "1.0 m_strange", "k"],
    ["0.1 m_strange", "1.4 m_charm", "d"],
    ["0.1 m_strange", "1.5 m_charm", "d"],
    ["0.1 m_strange", "2.0 m_charm", "d"],
    ["0.1 m_strange", "2.2 m_charm", "b"],
    ["0.1 m_strange", "2.5 m_charm", "b"],
    ["0.1 m_strange", "3.0 m_charm", "b"],
    ["0.1 m_strange", "3.3 m_charm", "b"],
    ["0.1 m_strange", "4.0 m_charm", "b"],
    ["0.1 m_strange", "4.4 m_charm", "b"],
    ["0.2 m_strange", "0.2 m_strange", "pi"],
    ["0.2 m_strange", "0.9 m_charm", "d"],
    ["0.2 m_strange", "1.0 m_charm", "d"],
    ["0.2 m_strange", "1.1 m_charm", "d"],
    ["0.2 m_strange", "1.0 m_strange", "k"],
    ["0.2 m_strange", "1.4 m_charm", "d"],
    ["0.2 m_strange", "2.0 m_charm", "d"],
    ["0.2 m_strange", "2.2 m_charm", "b"],
    ["0.2 m_strange", "3.0 m_charm", "b"],
    ["0.2 m_strange", "3.3 m_charm", "b"],
    ["0.2 m_strange", "4.0 m_charm", "b"],
    ["0.2 m_strange", "4.2 m_charm", "b"],
    ["0.2 m_strange", "4.4 m_charm", "b"],
    ["1.0 m_light",   "0.9 m_charm", "d"],
    ["1.0 m_light",   "1.0 m_charm", "d"],
    ["1.0 m_light",   "1.0 m_light", "pi"],
    ["1.0 m_light",   "1.0 m_strange", "k"],
    ["1.0 m_light",   "1.1 m_charm", "d"],
    ["1.0 m_light",   "1.4 m_charm", "d"],
    ["1.0 m_light",   "1.5 m_charm", "d"],
    ["1.0 m_light",   "2.0 m_charm", "d"],
    ["1.0 m_light",   "2.2 m_charm", "b"],
    ["1.0 m_light",   "2.5 m_charm", "b"],
    ["1.0 m_light",   "3.0 m_charm", "b"],
    ["1.0 m_light",   "3.3 m_charm", "b"],
    ["1.0 m_light",   "4.0 m_charm", "b"],
    ["1.0 m_light",   "4.2 m_charm", "b"],
    ["1.0 m_light",   "4.4 m_charm", "b"],
    ["1.0 m_strange", "0.9 m_charm", "ds"],
    ["1.0 m_strange", "1.0 m_charm", "ds"],
    ["1.0 m_strange", "1.1 m_charm", "ds"],
    ["1.0 m_strange", "1.4 m_charm", "ds"],
    ["1.0 m_strange", "1.5 m_charm", "ds"],
    ["1.0 m_strange", "2.0 m_charm", "ds"],
    ["1.0 m_strange", "2.2 m_charm", "bs"],
    ["1.0 m_strange", "2.5 m_charm", "bs"],
    ["1.0 m_strange", "3.0 m_charm", "bs"],
    ["1.0 m_strange", "3.3 m_charm", "bs"],
    ["1.0 m_strange", "4.0 m_charm", "bs"],
    ["1.0 m_strange", "4.2 m_charm", "bs"],
    ["1.0 m_strange", "4.4 m_charm", "bs"],],
    columns=['alias_light', 'alias_heavy', 'state'])
