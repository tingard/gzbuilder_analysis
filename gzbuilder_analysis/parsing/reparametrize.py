# import numpy as np
# from scipy.special import gamma
# from gzbuilder_analysis.rendering.sersic import _b
#
#
# def sersic_ltot(I, Re, n):
#     return (
#         2 * np.pi * I * Re**2 * n
#         * np.exp(_b(n)) / _b(n)**(2 * n)
#         * np.exp(gamma(2.0 * n))
#     )
#
#
# def sersic_I(L, Re, n):
#     return L / (
#         2 * np.pi * Re**2 * n
#         * np.exp(_b(n)) / _b(n)**(2 * n)
#         * np.exp(gamma(2.0 * n))
#     )
#
#
# def get_new_params(p):
#     # go from original param specification to new
#     p_new = p.copy()
#     disk_l = sersic_ltot(p['disk'])
#     p_new[('disk', 'L')] = disk_l
#     p_new.drop(('disk', 'I'), inplace=True)
#     for c in ('bulge', 'bar'):
#         try:
#             comp_l = sersic_ltot(p[c])
#             p_new[(c, 'Re')] = p[(c, 'Re')] / p[('disk', 'Re')]
#             p_new[(c, 'mux')] = p[(c, 'mux')] - p[('disk', 'mux')]
#             p_new[(c, 'muy')] = p[(c, 'muy')] - p[('disk', 'muy')]
#             p_new[(c, 'L')] = sersic_ltot(p[c]) / (disk_l + comp_l)
#             p_new.drop((c, 'I'), inplace=True)
#         except KeyError:
#             pass
#     return p_new
#
#
# def get_original_params(p):
#     # go from new param specification to original
#     p_new = p.copy()
#     disk_I = sersic_I(p['disk'])
#     p_new[('disk', 'I')] = disk_I
#     p_new.drop(('disk', 'L'), inplace=True)
#     if p_new[('disk', 'Re')] == 0:
#         p_new[('disk', 'I')] = 0
#         p_new[('disk', 'Re')] = 0.01
#     for c in ('bulge', 'bar'):
#         try:
#             p_new[(c, 'Re')] = p[(c, 'Re')] * p[('disk', 'Re')]
#             p_new[(c, 'mux')] = p[(c, 'mux')] + p[('disk', 'mux')]
#             p_new[(c, 'muy')] = p[(c, 'muy')] + p[('disk', 'muy')]
#             if p_new[(c, 'Re')] == 0 or disk_I == 0:
#                 p_new[(c, 'Re')] = 0.01
#                 p_new[(c, 'I')] = 0.0
#             else:
#                 comp_l = p[('disk', 'L')] * p[(c, 'L')] / (1 - p[(c, 'L')])
#                 p_new[(c, 'L')] = comp_l
#                 p_new[(c, 'I')] = sersic_I(p_new[c])
#             p_new.drop((c, 'L'), inplace=True)
#         except KeyError:
#             pass
#     return p_new
