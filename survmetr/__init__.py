"""survmetr: Advanced predictive metrics for survival analysis."""

from .antolini import (
    c_index_antolini_vector,
    c_index_antolini_sksurv,
    c_index_antolini_pycox,
    AntoliniCIndexVecScorer,
)
from .classification import make_survival_scorer
from .others import harrel_c_index_scorer
from .util import split_y

# Backward-compatible alias: default Antolini implementation
c_index_antolini = c_index_antolini_vector

__all__ = [
    # Antolini time-dependent concordance index
    'c_index_antolini',             # alias for c_index_antolini_vector
    'c_index_antolini_vector',      # vectorized NumPy implementation
    'c_index_antolini_sksurv',      # scikit-survival based implementation
    'c_index_antolini_pycox',       # numba-JIT parallel implementation

    # Scorers (take model, X, y)
    'AntoliniCIndexVecScorer',      # dataclass scorer for vector impl
    'harrel_c_index_scorer',        # Harrell's C-index scorer
    'make_survival_scorer',         # factory for time-dependent classification scorers

    # Utilities
    'split_y',                      # extract event/time from structured arrays
]
