import numpy as np
import pytest
from sksurv.util import Surv


@pytest.fixture
def synthetic_survival_data():
    """Generate synthetic survival data with structured array and prediction matrices."""
    rng = np.random.default_rng(42)
    n_samples = 50

    times = rng.exponential(scale=10, size=n_samples)
    events = rng.choice([True, False], size=n_samples, p=[0.6, 0.4])

    y = Surv.from_arrays(event=events, time=times)
    n_events = events.sum()

    # Full prediction matrix: (n_samples, n_samples) failure probs at all sample times.
    # Used by sksurv (needs all-times columns) and vector (auto-filters to event columns).
    failure_probs_full = rng.uniform(0, 1, size=(n_samples, n_samples))

    # Event-only prediction matrix: (n_samples, n_events) failure probs at event times.
    # Used directly by the vector implementation.
    failure_probs = failure_probs_full[:, events]

    # Pycox prediction matrix: (n_samples, n_samples) survival function.
    # The wrapper does estimate.T and uses arange(n_samples) as surv_idx.
    surv_func_pycox = np.sort(rng.uniform(0.1, 1.0, size=(n_samples, n_samples)), axis=1)[:, ::-1]

    return {
        'y': y,
        'events': events,
        'times': times,
        'n_samples': n_samples,
        'n_events': n_events,
        'failure_probs': failure_probs,
        'failure_probs_full': failure_probs_full,
        'surv_func_pycox': surv_func_pycox,
    }
