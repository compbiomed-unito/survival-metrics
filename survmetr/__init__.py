"""survmetr: Advanced predictive metrics for survival analysis."""

from .antolini import AntoliniCIndexVecScorer
from .classification import make_survival_scorer, _create_default_classification_scorers
from .others import harrel_c_index_scorer
from .util import split_y

default_scorers = {
    'c_index_antolini': AntoliniCIndexVecScorer(),
    
}
default_scorers.update(_create_default_classification_scorers())

__all__ = [
    # Antolini time-dependent concordance index
    # Scorers (take model, X, y)
    'AntoliniCIndexVecScorer',      # dataclass scorer for vector impl
    'harrel_c_index_scorer',        # Harrell's C-index scorer
    'make_survival_scorer',         # factory for time-dependent classification scorers
    'default_scorers',

    # Utilities
    'split_y',                      # extract event/time from structured arrays
]
