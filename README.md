# survmetr

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Advanced predictive metrics for survival analysis, focused on Antolini's time-dependent concordance index.

## Installation

```bash
pip install survmetr
```

For numba-accelerated computation:

```bash
pip install survmetr[numba]
```

## Quick Start

```python
import numpy as np
from sksurv.util import Surv
from survmetr import c_index_antolini

# Create survival data
y = Surv.from_arrays(
    event=[True, True, False, True, False],
    time=[1.0, 3.0, 4.0, 5.0, 6.0],
)
n_events = y['event'].sum()

# Prediction matrix: (n_samples, n_events) failure probabilities
estimate = np.random.rand(len(y), n_events)

# Compute Antolini's time-dependent C-index
c = c_index_antolini(estimate, y)
print(f"C-index: {c:.3f}")

# Get detailed statistics
result = c_index_antolini(estimate, y, return_all=True)
print(result)
# {'c_index': ..., 'concordant': ..., 'comparable': ...,
#  'tied_risk': ..., 'discordant': ..., 'numerator': ...}
```

## Available Metrics

| Function | Description |
|----------|-------------|
| `c_index_antolini` | Antolini's time-dependent C-index (alias for vector impl) |
| `c_index_antolini_vector` | Vectorized NumPy implementation |
| `c_index_antolini_sksurv` | Implementation using scikit-survival internals |
| `c_index_antolini_pycox` | Numba-JIT parallel implementation |
| `make_survival_scorer` | Factory for time-dependent classification scorers |
| `harrel_c_index_scorer` | Harrell's standard C-index scorer |
| `AntoliniCIndexVecScorer` | Dataclass scorer for the vector implementation |
| `split_y` | Extract event/time from structured arrays |

## Tutorial

See [Tutorial.ipynb](Tutorial.ipynb) for a detailed walkthrough.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## TODO

- Try to be flexible in how outcomes are passed: two arrays, one structured (with arbitrary names?)
- Assume a certain API for models in sklearn-style scorers? like predict('failure', X, times)?
- Write a tutorial notebook
- IPCW metrics
- Handle competing risks
