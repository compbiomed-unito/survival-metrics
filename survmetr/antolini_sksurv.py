import numpy as np
from sksurv.metrics import _iter_comparable
from sksurv.exceptions import NoComparablePairException

def _estimate_concordance_index_antolini(
    event_indicator, event_time, estimate, weights, tied_tol=1e-8
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
    for ind, mask, tied_time in _iter_comparable(
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
        raise NoComparablePairException(
            "Data has no comparable pairs, cannot estimate concordance index."
        )

    cindex = numerator / denominator
    return {
        'c-index': cindex,
        'concordant': concordant, 
        'discordant': discordant, 
        'tied_risk': tied_risk, 
        'tied_time': tied_time, 
        'numerator': numerator, 
        'comparable': denominator,
    }