"""

"""

# flake8: noqa

from pandas.tseries.index import DatetimeIndex, date_range, bdate_range
from pandas.tseries.frequencies import infer_freq
from pandas.tseries.tdi import Timedelta, TimedeltaIndex, timedelta_range
from pandas.tseries.period import Period, PeriodIndex, period_range
from pandas.tseries.resample import TimeGrouper
from pandas.tseries.timedeltas import to_timedelta
from pandas.lib import NaT
import pandas.tseries.offsets as offsets

# deprecation, xref #13790
def pnow(freq=None):
    import warnings

    warnings.warn("pd.pnow() is deprecated. Please use pandas.tseries.period.pnow()",
                  FutureWarning, stacklevel=2)
    from pandas.tseries.period import pnow
    return pnow(freq=freq)
