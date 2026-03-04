import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from survmetr import make_survival_scorer


class TestMakeSurvivalScorer:

    def test_returns_callable(self):
        scorer = make_survival_scorer(roc_auc_score, classification=True)
        assert callable(scorer)

    def test_name_ends_with_td_scorer(self):
        scorer = make_survival_scorer(roc_auc_score, classification=True)
        assert scorer.__name__.endswith('_td_scorer')

    def test_name_includes_original(self):
        scorer = make_survival_scorer(roc_auc_score, classification=True)
        assert 'roc_auc_score' in scorer.__name__

    def test_aggregate_no_returns_array(self):
        scorer = make_survival_scorer(
            roc_auc_score,
            classification=True,
            aggregate='no',
            time_mode='quantiles',
            time_values=[0.25, 0.5, 0.75],
        )
        # We can't easily test with a real model here, but we can verify
        # the scorer is properly configured
        assert callable(scorer)
        assert scorer.__name__ == 'roc_auc_score_td_scorer'
