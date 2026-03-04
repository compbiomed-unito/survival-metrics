__all__ = ['split_y']

# extract time and event from a structured array in scikit-survival style
def split_y(y):
    # this is what sksurv does
    event_field, time_field = y.dtype.names
    return y[event_field], y[time_field]
