"""Cohort-calibrated prototypes (CCP-A) for few-shot open-set recognition.

Code of the paper "Cohort-calibrated prototypes for few-shot open-set recognition" (I. Adjei-Mensah, A. A. Baffour). Equation and section
numbers in the docstrings are those of the paper:

  Eq. 1  embedding z(x) = n(g(x) + g(flip x))                        ccp.embedding.unit_flip
  Eq. 2  prototype p_k = n(mean_i [g(x_ki) + g(flip x_ki)])          ccp.embedding.prototypes
  Eq. 3  cohort statistic m_K(v), K = 1000                           ccp.cohort.m_topk
  Eq. 4  CCP-A s_k(q) = c_k(q) - 1/2 [m_K(q) + m_K(p_k)]             ccp.cohort.ccp_a_matrix
  Eq. 5  unknown score u(q) = 1/2 m_K(q) - max_k [c_k - 1/2 m_K(p_k)] ccp.cohort.unknown_score
  Eq. 6  CCP-QW u_QW(q) = lambda m_K(q) - max_k [c_k - beta m_K(p_k)] ccp.cohort.ccp_qw_matrix
  Eq. 7  fixed-weight fusion H_fix = D + w u                         ccp.fusion.h_fix
  Eq. 8  SCG bin gap d_j                                             ccp.scg.scg_bins
CCP-S has no numbered equation (Section 3.3). Names follow the paper; docs/naming.md maps them to the names used in the registration
documents and the recorded result files.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cohort-calibrated-prototypes")   # single source: pyproject.toml
except PackageNotFoundError:                                # a source tree that was never installed
    __version__ = "2.0.0"
