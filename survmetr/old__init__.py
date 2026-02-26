from sksurv.util import check_y_survival
#event, time = check_array_survival(X, y)
from dataclasses import dataclass
import numpy as np


# # Antolini

#__all? = [
#    "concordance_index_antolini_scorer",
#]

# code from survhive.metrics

import sksurv.metrics



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
    
    event, time = y['event'], y['time']
    assert event.dtype == np.bool

    r = _estimate_concordance_index_antolini(
        event_indicator=event,
        event_time=time,
        estimate=1.0 - estimator.predict_survival(X, get_time(y)),  # use failure
        weights=np.full(len(y), 1.0),
    )
    return r if return_all else r[0]
    

# # Basic simulation

if False:
    def generate_features(size, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        X_sim = np.random.normal(size=(size, n))
        return X_sim


# # Other scores


from sksurv.metrics import concordance_index_censored

def harrel_c_index_scorer(mod, X, y):
    # this works on sksurv Cox
    preds = mod.predict(X)
    return concordance_index_censored(
        event_indicator=y['event'], 
        event_time=y['time'],
        estimate=preds,
    )[0]



from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
def get_time(y):
    return y['time']
def get_indicator(y):
    return y['event']

