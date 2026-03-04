import numpy as np
import pytest
from survmetr import c_index_antolini_sksurv


class TestCIndexAntoliniSksurv:

    def test_basic_returns_float_in_range(self, synthetic_survival_data):
        d = synthetic_survival_data
        # sksurv needs (n_samples, n_samples) survival probs
        surv_probs = 1.0 - d['failure_probs_full']
        result = c_index_antolini_sksurv(surv_probs, d['y'])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_return_all_dict_keys(self, synthetic_survival_data):
        d = synthetic_survival_data
        surv_probs = 1.0 - d['failure_probs_full']
        result = c_index_antolini_sksurv(surv_probs, d['y'], return_all=True)
        assert isinstance(result, dict)
        assert 'c_index' in result
        assert 'concordant' in result
        assert 'comparable' in result
        assert 'tied_risk' in result
        assert 'numerator' in result

    def test_return_dict_keys_match_vector(self, synthetic_survival_data):
        """Verify that sksurv return dict has a superset of the vector keys."""
        d = synthetic_survival_data
        surv_probs = 1.0 - d['failure_probs_full']
        result = c_index_antolini_sksurv(surv_probs, d['y'], return_all=True)
        vector_keys = {'c_index', 'concordant', 'comparable', 'tied_risk', 'discordant', 'numerator'}
        assert vector_keys.issubset(set(result.keys()))
