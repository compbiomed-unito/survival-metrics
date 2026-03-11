import numpy as np
import pytest
from survmetr import c_index_antolini_vector


class TestCIndexAntoliniVector:

    def test_basic_returns_float_in_range(self, synthetic_survival_data):
        d = synthetic_survival_data
        result = c_index_antolini_vector(d['failure_probs'], d['y'])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_return_all_dict_keys(self, synthetic_survival_data):
        d = synthetic_survival_data
        result = c_index_antolini_vector(d['failure_probs'], d['y'], return_all=True)
        assert isinstance(result, dict)
        expected_keys = {'c-index', 'concordant', 'comparable', 'tied_risk', 'discordant', 'numerator'}
        assert set(result.keys()) == expected_keys

    @pytest.mark.parametrize("time_ties", ['none', 'censored', 'all'])
    def test_time_ties_modes(self, synthetic_survival_data, time_ties):
        d = synthetic_survival_data
        result = c_index_antolini_vector(d['failure_probs'], d['y'], time_ties=time_ties)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_tie_weight_affects_result(self, synthetic_survival_data):
        d = synthetic_survival_data
        r1 = c_index_antolini_vector(d['failure_probs'], d['y'], tie_weight=0.0, return_all=True)
        r2 = c_index_antolini_vector(d['failure_probs'], d['y'], tie_weight=1.0, return_all=True)
        # With different tie weights, the numerator should differ (if there are ties)
        if r1['tied_risk'] > 0:
            assert r1['c-index'] != r2['c-index']

    def test_all_nan_raises(self, synthetic_survival_data):
        d = synthetic_survival_data
        nan_estimate = np.full_like(d['failure_probs'], np.nan)
        with pytest.raises(ValueError, match='all risk values are nan'):
            c_index_antolini_vector(nan_estimate, d['y'])

    def test_full_estimate_auto_filters(self, synthetic_survival_data):
        """Passing (n_samples, n_samples) estimate auto-filters to event columns."""
        d = synthetic_survival_data
        r_full = c_index_antolini_vector(d['failure_probs_full'], d['y'])
        r_event = c_index_antolini_vector(d['failure_probs'], d['y'])
        assert r_full == r_event

    def test_invalid_time_ties_raises(self, synthetic_survival_data):
        d = synthetic_survival_data
        with pytest.raises(ValueError, match="Invalid time_ties"):
            c_index_antolini_vector(d['failure_probs'], d['y'], time_ties='invalid')
