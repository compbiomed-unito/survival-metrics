import numpy as np
import pytest
from sksurv.util import Surv
from survmetr import split_y


class TestSplitY:

    def test_extracts_correct_fields(self):
        events = np.array([True, False, True, True])
        times = np.array([1.0, 2.5, 3.0, 4.5])
        y = Surv.from_arrays(event=events, time=times)

        extracted_events, extracted_times = split_y(y)
        np.testing.assert_array_equal(extracted_events, events)
        np.testing.assert_array_equal(extracted_times, times)

    def test_return_types(self):
        y = Surv.from_arrays(event=[True, False], time=[1.0, 2.0])
        ev, ti = split_y(y)
        assert ev.dtype == bool
        assert ti.dtype == np.float64
