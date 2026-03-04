import numpy as np
import pytest
from survmetr import c_index_antolini_vector, c_index_antolini_sksurv


class TestConsistency:

    def test_vector_vs_sksurv_same_result(self, synthetic_survival_data):
        """Vector and sksurv implementations should produce the same c-index."""
        d = synthetic_survival_data

        # Both use the full (n_samples, n_samples) matrix.
        # Vector takes failure probs (auto-filters to event cols).
        vec_result = c_index_antolini_vector(d['failure_probs_full'], d['y'])

        # sksurv takes survival probabilities.
        surv_probs = 1.0 - d['failure_probs_full']
        sksurv_result = c_index_antolini_sksurv(surv_probs, d['y'])

        np.testing.assert_allclose(vec_result, sksurv_result, atol=1e-10,
                                   err_msg="Vector and sksurv implementations disagree")

    def test_vector_vs_sksurv_return_all_concordant(self, synthetic_survival_data):
        """Both implementations should agree on concordant pair count."""
        d = synthetic_survival_data

        vec_r = c_index_antolini_vector(d['failure_probs_full'], d['y'], return_all=True)
        surv_probs = 1.0 - d['failure_probs_full']
        sksurv_r = c_index_antolini_sksurv(surv_probs, d['y'], return_all=True)

        assert vec_r['concordant'] == sksurv_r['concordant'], \
            f"Concordant pairs differ: vector={vec_r['concordant']}, sksurv={sksurv_r['concordant']}"

    def test_vector_vs_sksurv_comparable_pairs_match(self, synthetic_survival_data):
        """Both implementations should agree on the number of comparable pairs."""
        d = synthetic_survival_data

        vec_r = c_index_antolini_vector(d['failure_probs_full'], d['y'], return_all=True)
        surv_probs = 1.0 - d['failure_probs_full']
        sksurv_r = c_index_antolini_sksurv(surv_probs, d['y'], return_all=True)

        assert vec_r['comparable'] == sksurv_r['comparable'], \
            f"Comparable pairs differ: vector={vec_r['comparable']}, sksurv={sksurv_r['comparable']}"
