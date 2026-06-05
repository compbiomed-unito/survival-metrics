import pytest
import survmetr


class TestPublicAPI:

    EXPECTED_EXPORTS = [
        'AntoliniCIndexVecScorer',
        'harrel_c_index_scorer',
        'make_survival_scorer',
        'default_scorers',
        'split_y',
    ]

    def test_all_symbols_importable(self):
        for name in self.EXPECTED_EXPORTS:
            assert hasattr(survmetr, name), f"{name} not importable from survmetr"

    def test_all_matches_exports(self):
        assert set(survmetr.__all__) == set(self.EXPECTED_EXPORTS)
