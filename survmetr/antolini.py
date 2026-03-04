import numpy as np
from sksurv.util import check_y_survival
from .util import split_y
from dataclasses import dataclass
from typing import Literal, Union

__all__ = [
    'c_index_antolini_vector',
    'c_index_antolini_sksurv',
    'c_index_antolini_pycox',
    'AntoliniCIndexVecScorer',
]

# antolini vectorial implementation
def c_index_antolini_vector(estimate, y, time_ties: Literal['none', 'censored', 'all']='none', tie_weight=0.5, return_all=False) -> Union[float, dict]:
    '''time_ties: include time ties in the risk sets'''
    event, time = check_y_survival(y)
    if event.dtype != np.bool_:
        raise TypeError(f"event indicator must be boolean, got {event.dtype}")
    n_events = event.sum()

    if estimate.shape[1] == len(time):
        estimate = estimate[:,event]

    if estimate.shape != (len(time), n_events):
        raise ValueError(f"estimate shape {estimate.shape} does not match expected ({len(time)}, {n_events})")

    if np.isnan(estimate).all():
        raise ValueError('all risk values are nan')

    self_idx = (
        np.arange(event.sum()) + (~event).cumsum()[event],
        np.arange(n_events),
    )

    # compute risk sets

    time_t = time.reshape(-1, 1) # trasposed times
    e_time = time[event] # event times
    if time_ties == 'all' or time_ties == 'censored':

        if time_ties == 'censored':
            risk_sets = (time_t > e_time) | ((time_t == e_time) & ~event.reshape(-1, 1))
        elif time_ties == 'all':
            risk_sets = time_t >= e_time

        # remove self-comparisons from risk set
        risk_sets[*self_idx] = False
    elif time_ties == 'none':
        risk_sets = time_t > e_time
    else:
        raise ValueError(f"Invalid time_ties '{time_ties}'")

    if risk_sets.shape != estimate.shape:
        raise ValueError(f"risk_sets shape {risk_sets.shape} does not match estimate shape {estimate.shape}")

    # compute concordant pairs
    diag_events = estimate[*self_idx]
    concordant = risk_sets * (estimate < diag_events)
    ties = risk_sets * (estimate == diag_events)

    n_comparable = int(risk_sets.sum())
    n_concordant = int(concordant.sum())
    n_ties = int(ties.sum())

    num = n_concordant + n_ties * tie_weight
    if n_comparable > 0:
        c_index = num / n_comparable
    else:
        c_index = float('nan')

    if return_all:
        return {
            'c_index': c_index,
            'concordant': n_concordant,
            'comparable': n_comparable,
            'tied_risk': n_ties,
            'discordant': n_comparable - n_concordant - n_ties,
            'numerator': num,
        }
    else:
        return c_index


@dataclass
class AntoliniCIndexVecScorer:
    time_ties: str = 'none'

    def __call__(self, model, X, y, return_all=False):
        est = model.predict('survival', X, y['time'][y['event']])
        return c_index_antolini_vector(est, y, time_ties=self.time_ties, return_all=return_all)

# sksurv extension
from .antolini_sksurv import _estimate_concordance_index_antolini as sksurv_antolini
def c_index_antolini_sksurv(estimate, y, return_all=False) -> Union[float, dict]:
    ind, time = split_y(y)

    r = sksurv_antolini(
        event_indicator=ind,
        event_time=time,
        estimate=1.0 - estimate,  # use failure
        weights=np.full(len(y), 1.0),
        tied_tol=0.0,
    )
    return r if return_all else r['c_index']
 
# pycox implementation
from .antolini_pycox import concordance_td as pycox_antolini
def c_index_antolini_pycox(estimate, y, method: Literal['adj_antolini', 'antolini'] = 'antolini', return_all=False) -> Union[float, dict]:
    ind, time = split_y(y)
    indexes = np.arange(estimate.shape[0])
    r = pycox_antolini(time, ind, estimate.T, indexes, method=method)
    return r if return_all else r['c_index']

