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


def _estimate_concordance_index_antolini(
    event_indicator, event_time, estimate, weights, tied_tol=0.0,
):
    # taken from sksurv.metrics, generalized to a time-dependant estimate matrix
    # not meant to be called directly but through the concordance_index_td_scorer,
    # since it only works if estimate has the y times as its second index

    order = np.argsort(event_time)

    tied_time = None

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask, tied_time in sksurv.metrics._iter_comparable(
        event_indicator, event_time, order
    ):
        est_i = estimate[order[ind], order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask], order[ind]]

        assert (
            event_i
        ), f"got censored sample at index {order[ind]}, but expected uncensored"

        ties = np.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    if tied_time is None:
        raise sksurv.exceptions.NoComparablePairException(
            "Data has no comparable pairs, cannot estimate concordance index."
        )

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time, numerator, denominator


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
    
def vec_anto(est, y, time_ties='none', tie_weight=0.5, return_all=False):
    '''time_ties: include time ties in the risk sets'''

    event, time = check_y_survival(y)
    #event, time = y['event'], y['time']
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
    if not all(diag_events == np.diag(est[event])):
        print('@'*80)
        print('@'*80)
        print(f'{est.shape=}')
        print(f'{event.shape=}')
        print('@'*80)
        print(f'{diag_events.shape=}')
        print(f'{diag_events=}')
        print('@'*80)
        print(f'{np.diag(est[event]).shape=}')
        print(f'{np.diag(est[event])=}')
        print('@'*80)
        print('@'*80)
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
            'cindex': c_index,
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

def make_survival_scorer(
    score_func,
    needs="failure",
    classification=False,
    aggregate="mean",
    time_mode="events",
    time_values=None,
    **kwargs,
):
    """
    Create a time-dependent survival scoring function for survival analysis.

    Parameters:
    - score_func (callable, with signature (y_pred, y_true)): A function that
    computes a score based on predicted and true values.
    - needs (str, optional): The type of predictions needed. Either "failure"
    or "survival" probability predictions. Default is "failure".
    - classification (bool, optional): If True, treat score_func as a
    classification score and run it on positive/negative events computed
    separately for each time point. Default is False.
    - aggregate (str, optional): The method to aggregate scores over different
    time points. Options include 'mean', 'median', 'sum', or 'no' for no
    aggregation. Default is 'mean'.
    - time_mode (str, optional): The mode for specifying prediction times.
    Options are "events" (using event times), "quantiles" (using quantiles of
    event times), or "absolute" (using specified absolute time values).
    Default is "events".
    - time_values (array-like or float, optional): The time values depending on
    the chosen time_mode. If time_mode is "events", time_values should be
    None. If time_mode is "quantiles", time_values should be an array of
    quantiles between 0 and 1. If time_mode is "absolute", time_values should
    be an array-like object or a float representing absolute time values.
    - **kwargs: Additional keyword arguments to be passed to the underlying
    score_func.

    Returns:
    - scorer (callable with signature (estimator, X, y)): A time-dependent
    scoring function that computes score_func at different time points and
    aggregate the results.

    Notes:
    - The resulting scorer can be used as a standard scikit-learn scorer with
    survival outcomes and survhive models. See the example

    ```
    from survhive import CoxNet, load_test_data
    from sklearn.metrics import roc_auc_score, brier_score_loss
    roc_auc_at_quartiles = make_survival_scorer(roc_auc_score, classification=True,
                                                time_mode='quantiles',
                                                time_values=[0.25, 0.5, 0.75])
    brier_at_quartiles = make_survival_scorer(lambda *args: -brier_score_loss(*args),
                                            classification=True,
                                            time_mode='quantiles',
                                            time_values=[0.25, 0.5, 0.75]),

    X, y = load_test_data('veterans_lung_cancer')

    cross_val_score(CoxNet(), X, y, scoring=roc_auc_at_quartiles)
    ```
    """

    def scorer(estimator, X, y):
        event_times = get_time(y)[get_indicator(y)]

        # get evaluation times
        if time_mode == "events":
            pred_times = event_times
        elif time_mode == "quantiles":
            pred_times = np.quantile(event_times, time_values)
        elif time_mode == "absolute":
            pred_times = time_values
        else:  # keep as is, must be a scalar or sequence
            raise ValueError('needs must be either "events", "quantiles" or "absolute"')

        # compute predictions at pred_times
        if needs == "failure":
            y_pred = 1.0 - estimator.predict_survival(X, pred_times)
        elif needs == "survival":
            y_pred = estimator.predict_survival(X, pred_times)
        else:
            raise ValueError('needs must be either "failure" or "survival"')

        # run score_func at each time
        scores = []
        for p, t in zip(y_pred.T, pred_times):
            if classification:
                y_time = get_time(y)
                y_ind = get_indicator(y)

                informative = (y_time > t) | y_ind
                positive = (y_time <= t) & y_ind

                score = score_func(positive[informative], p[informative])
            else:
                score = score_func(y, p)
            if score != score:
                print(f"bad survival score at time {t} computed by {score_func}")
            scores.append(score)

        # aggregate scores for different times
        if aggregate == "no":
            return np.array(scores)
        else:
            if hasattr(np, aggregate):
                return getattr(np, aggregate)(scores)
            else:
                raise ValueError(f"unknonw aggregate value `{aggregate}`")

    scorer.__name__ = score_func.__name__ + "_td_scorer"

    return scorer


def _create_default_scorers():
    """
    Create scorers for common survival metrics.
    """

    quantiles = {
        "quartiles": np.linspace(0, 1, 4 + 1)[1:-1],
        "deciles": np.linspace(0, 1, 10 + 1)[1:-1],
        # "percentiles": np.linspace(0, 1, 100 + 1)[1:-1],
    }
    classification_metrics = {
        "roc-auc": roc_auc_score,
        "neg-brier": lambda *args: -brier_score_loss(*args),
        "neg-log": lambda *args: -log_loss(*args),
    }

    scorers = {
        "c-index-antolini": concordance_index_antolini_scorer,
        "c-index-antolini-vec": AntoliniCIndexVecScorer(),
    }

    if False:
        scorers.update(
            {
                f"c-index-{quantile_name}": make_survival_scorer(
                    concordance_index_score,
                    classification=False,
                    time_mode="quantiles",
                    time_values=quantile_breaks,
                )  # FIXME this is not a good score, maybe remove it from this list
                for quantile_name, quantile_breaks in quantiles.items()
            }
        )

    scorers.update(
        {
            f"{score_name}-{quantile_name}": make_survival_scorer(
                score_func,
                classification=True,
                time_mode="quantiles",
                time_values=quantile_breaks,
            )
            for score_name, score_func in classification_metrics.items()
            for quantile_name, quantile_breaks in quantiles.items()
        }
    )

    return scorers

scorers = _create_default_scorers()