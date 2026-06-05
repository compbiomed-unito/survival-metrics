import numpy as np
import pytest
from survmetr.antolini import c_index_antolini_pycox


class TestCIndexAntoliniPycox:

    def test_basic_returns_float_in_range(self, synthetic_survival_data):
        d = synthetic_survival_data
        result = c_index_antolini_pycox(d['surv_func_pycox'], d['y'])
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_return_all_dict_keys(self, synthetic_survival_data):
        d = synthetic_survival_data
        result = c_index_antolini_pycox(d['surv_func_pycox'], d['y'], return_all=True)
        assert isinstance(result, dict)
        assert 'c-index' in result
        assert 'concordant' in result
        assert 'comparable' in result
        assert 'numerator' in result

    @pytest.mark.parametrize("method", ['antolini', 'adj_antolini'])
    def test_both_methods(self, synthetic_survival_data, method):
        d = synthetic_survival_data
        result = c_index_antolini_pycox(d['surv_func_pycox'], d['y'], method=method)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
