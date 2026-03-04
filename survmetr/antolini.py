import numpy as np
from sksurv.util import check_y_survival
#event,
#from .util import split_y, predict_survival
from dataclasses import dataclass
from typing import Literal

# antolini vectorial implementation
def c_index_antolini_vector(est, y, time_ties: Literal['none', 'censored', 'all']='none', tie_weight=0.5, return_all=False):
    '''time_ties: include time ties in the risk sets'''
    event, time = check_y_survival(y)
    assert event.dtype == np.bool
    n_events = event.sum()

    if est.shape[1] == len(time):
        est = est[:,event]

    assert est.shape == (len(time), n_events)

    if np.isnan(est).all():
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

    assert risk_sets.shape == est.shape, f'{risk_sets.shape=} != {est.shape=}'

    # compute concordant pairs
    diag_events = est[*self_idx]
    assert all(diag_events == np.diag(est[event]))
    concordant = risk_sets * (
        (est > diag_events) + (est == diag_events) * tie_weight
    )
    concordant = risk_sets * (est > diag_events)
    ties = risk_sets * (est == diag_events)

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
            'c-index': c_index,
            'concordant': n_concordant,
            'comparable': n_comparable,
            'tied_risk': n_ties, # check that is the same order as survhive and sksurv cindex
            'numerator': num,
        }
    else:
        return c_index


@dataclass
class AntoliniCIndexVecScorer:
    time_ties: str = 'none'

    def __call__(self, model, X, y, return_all=False):
        est = model.predict('survival', X, y['time'][y['event']])
        return vec_anto(est, y, time_ties=self.time_ties, return_all=return_all)

# sksurv extension
from .antolini_sksurv import _estimate_concordance_index_antolini as sksurv_antolini
def c_index_antolini_sksurv(est, y, return_all=False):
    ind, time = split_y(y)

    r = sksurv_antolini(
        event_indicator=ind,
        event_time=time,
        estimate=1.0 - est,  # use failure
        weights=np.full(len(y), 1.0),
        tied_tol=0.0,
    )
    return r if return_all else r['c-index']
 
# pycox implementation
from .antolini_pycox import concordance_td as pycox_antolini
def c_index_antolini_pycox(est_sq, y, method: Literal['adj_antolini', 'antolini'] = 'antolini', return_all=False):
    #indexes = np.arange(est_sq.shape[1]) + np.cumsum(~y['event'])
    ind, time = split_y(y)
    #assert est_sq.shape[0] == est_sq.shape[1], 'pycox needs square' #FIXME is this true?
    indexes = np.arange(est_sq.shape[0])
    r = pycox_antolini(time, ind, est_sq.T, indexes, method=method)#, return_all=return_all)
    return r if return_all else r['c-index']


# scorer: can we make a single scorer for the various implementations?
if False:
    def concordance_index_antolini_scorer(estimator, X, y, return_all=False):
        """Antolini's extension of concordance index to time-dependent predictions.

        Implementation based on scikit-survival concordance_index_censored code.

        Parameters:
        - estimator: estimator object with a `predict_survival` method,
        - X: feature matrix for the `predict_survival` method
        - y: survival outcomes for evaluating the prediction
        - return_all: bool, set to true to return more information on the score

        Returns:
        - the concordance score (float) if return_all is false or a tuple (score: float, concordant: int, discordant: int, tied_risk: int, tied_time: int). For more information see scikit-survival concordance_index_censored documentation.
        """
        ind, time = split_y(y)

        r = _estimate_concordance_index_antolini(
            event_indicator=ind,
            event_time=time,
            estimate=1.0 - estimator.predict_survival(X, time),  # use failure
            weights=np.full(len(y), 1.0),
        )
        return r if return_all else r[0]
    @dataclass
    class CIndexAntoliniScorer:
        implementation: Literal['vectorial', 'sksurv', 'pycox'] = 'vectorial'
        time_ties: Literal[''] = 'none'

        # FIXME this depends on the model calling convention
        def __call__(self, estimator, X, y, return_all=False):
            event_ind, event_time = split_y(y)
            est = predict_survival(estimator, X, event_time)
            est = model.predict('survival', X, y['time'][y['event']])
            #if self.implementation == 'vectorial':

            return c_index_antolini_vector(est, y, time_ties=self.time_ties, return_all=return_all)

