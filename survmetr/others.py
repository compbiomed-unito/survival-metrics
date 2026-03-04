# add harrel c-index
from sksurv.metrics import concordance_index_censored

__all__ = ['harrel_c_index_scorer']


def harrel_c_index_scorer(mod, X, y):
    # this works on sksurv Cox
    preds = mod.predict(X)
    return concordance_index_censored(
        event_indicator=y['event'], 
        event_time=y['time'],
        estimate=preds,
    )[0]

