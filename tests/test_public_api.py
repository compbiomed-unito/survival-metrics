import pytest
import survmetr


class TestPublicAPI:

    EXPECTED_EXPORTS = [
        'c_index_antolini',
        'c_index_antolini_vector',
        'c_index_antolini_sksurv',
        'c_index_antolini_pycox',
        'AntoliniCIndexVecScorer',
        'harrel_c_index_scorer',
        'make_survival_scorer',
        'split_y',
    ]

    def test_all_symbols_importable(self):
        for name in self.EXPECTED_EXPORTS:
            assert hasattr(survmetr, name), f"{name} not importable from survmetr"

    def test_all_matches_exports(self):
        assert set(survmetr.__all__) == set(self.EXPECTED_EXPORTS)

    def test_antolini_is_vector(self):
        assert survmetr.c_index_antolini is survmetr.c_index_antolini_vector
