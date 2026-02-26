# derive survival scores from classification scores and a time grid
import numpy
from .util import split_y


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
        indicator, times = split_y(y)
        event_times = times[indicator]

        # get evaluation times
        if time_mode == "events":
            pred_times = event_times
        elif time_mode == "quantiles":
            pred_times = numpy.quantile(event_times, time_values)
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
                y_ind, y_time = split_y(y)

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
            return numpy.array(scores)
        else:
            if hasattr(numpy, aggregate):
                return getattr(numpy, aggregate)(scores)
            else:
                raise ValueError(f"unknonw aggregate value `{aggregate}`")

    scorer.__name__ = score_func.__name__ + "_td_scorer"

    return scorer

def _create_default_scorers():
    """
    Create scorers for common survival metrics.
    """

    quantiles = {
        "quartiles": numpy.linspace(0, 1, 4 + 1)[1:-1],
        "deciles": numpy.linspace(0, 1, 10 + 1)[1:-1],
        # "percentiles": numpy.linspace(0, 1, 100 + 1)[1:-1],
    }
    classification_metrics = {
        "roc-auc": roc_auc_score,
        "neg-brier": lambda *args: -brier_score_loss(*args),
        "neg-log": lambda *args: -log_loss(*args),
    }

    # TODO move these to antolini.py or to __init__.py, keep only classification scores here
    scorers = {
        "c-index-antolini": concordance_index_antolini_scorer,
        "c-index-antolini-vec": AntoliniCIndexVecScorer(),
    }

    if False: # these should not be used, but maybe give option to have them
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