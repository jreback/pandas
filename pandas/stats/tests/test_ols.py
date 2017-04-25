"""
Unit test suite for OLS
"""

from __future__ import division

import pytest
from datetime import datetime
import numpy as np

from pandas import (date_range, compat,
                    DataFrame, Series, offsets)
from pandas.stats.ols import _filter_data, ols
import pandas.util.testing as tm
import statsmodels.api as sm


def _compare_ols_results(model1, model2):
    assert isinstance(model1, type(model2))

    if hasattr(model1, '_window_type'):
        _compare_moving_ols(model1, model2)
    else:
        _compare_fullsample_ols(model1, model2)


def _compare_fullsample_ols(model1, model2):
    tm.assert_series_equal(model1.beta, model2.beta)


def _compare_moving_ols(model1, model2):
    tm.assert_frame_equal(model1.beta, model2.beta)


class TestOLS(object):

    # TODO: Add tests for OLS y predict
    # TODO: Right now we just check for consistency between full-sample and
    # rolling/expanding results of the panel OLS.  We should also cross-check
    # with trusted implementations of panel OLS (e.g. R).
    # TODO: Add tests for non pooled OLS.

    def test_with_datasets_ccard(self):
        self.check_data_set(sm.datasets.ccard.load(), skip_moving=True)
        self.check_data_set(sm.datasets.cpunish.load(), skip_moving=True)
        self.check_data_set(sm.datasets.longley.load(), skip_moving=True)
        self.check_data_set(sm.datasets.stackloss.load(), skip_moving=True)

    def test_with_datasets_copper(self):
        self.check_data_set(sm.datasets.copper.load())

    def test_with_datasets_scotland(self):
        self.check_data_set(sm.datasets.scotland.load())

        # degenerate case fails on some platforms
        # self.check_data_set(datasets.ccard.load(), 39, 49) # one col in X all
        # 0s

    def test_WLS(self):
        # WLS centered SS changed (fixed) in 0.5.0
        X = DataFrame(np.random.randn(30, 4), columns=['A', 'B', 'C', 'D'])
        Y = Series(np.random.randn(30))
        weights = X.std(1)

        self._check_wls(X, Y, weights)

        weights.ix[[5, 15]] = np.nan
        Y[[2, 21]] = np.nan
        self._check_wls(X, Y, weights)

    def _check_wls(self, x, y, weights):
        result = ols(y=y, x=x, weights=1 / weights)

        combined = x.copy()
        combined['__y__'] = y
        combined['__weights__'] = weights
        combined = combined.dropna()

        endog = combined.pop('__y__').values
        aweights = combined.pop('__weights__').values
        exog = sm.add_constant(combined.values, prepend=False)

        sm_result = sm.WLS(endog, exog, weights=1 / aweights).fit()

        tm.assert_almost_equal(sm_result.params, result._beta_raw)
        tm.assert_almost_equal(sm_result.resid, result._resid_raw)

        self.check_moving_ols('rolling', x, y, weights=weights)
        self.check_moving_ols('expanding', x, y, weights=weights)

    def check_data_set(self, dataset, start=None, end=None, skip_moving=False):
        exog = dataset.exog[start: end]
        endog = dataset.endog[start: end]
        x = DataFrame(exog, index=np.arange(exog.shape[0]),
                      columns=np.arange(exog.shape[1]))
        y = Series(endog, index=np.arange(len(endog)))

        self.check_ols(exog, endog, x, y)

        if not skip_moving:
            self.check_moving_ols('rolling', x, y)
            self.check_moving_ols('rolling', x, y, nw_lags=0)
            self.check_moving_ols('expanding', x, y, nw_lags=0)
            self.check_moving_ols('rolling', x, y, nw_lags=1)
            self.check_moving_ols('expanding', x, y, nw_lags=1)
            self.check_moving_ols('expanding', x, y, nw_lags=1,
                                  nw_overlap=True)

    def check_ols(self, exog, endog, x, y):
        reference = sm.OLS(endog, sm.add_constant(exog, prepend=False)).fit()
        result = ols(y=y, x=x)

        # check that sparse version is the same
        sparse_result = ols(y=y.to_sparse(), x=x.to_sparse())
        _compare_ols_results(result, sparse_result)

        tm.assert_almost_equal(reference.params, result._beta_raw)
        tm.assert_almost_equal(reference.df_model, result._df_model_raw)
        tm.assert_almost_equal(reference.df_resid, result._df_resid_raw)
        tm.assert_almost_equal(reference.fvalue, result._f_stat_raw[0])
        tm.assert_almost_equal(reference.pvalues, result._p_value_raw)
        tm.assert_almost_equal(reference.rsquared, result._r2_raw)
        tm.assert_almost_equal(reference.rsquared_adj, result._r2_adj_raw)
        tm.assert_almost_equal(reference.resid, result._resid_raw)
        tm.assert_almost_equal(reference.bse, result._std_err_raw)
        tm.assert_almost_equal(reference.tvalues, result._t_stat_raw)
        tm.assert_almost_equal(reference.cov_params(), result._var_beta_raw)
        tm.assert_almost_equal(reference.fittedvalues, result._y_fitted_raw)

        _check_non_raw_results(result)

    def check_moving_ols(self, window_type, x, y, weights=None, **kwds):
        window = np.linalg.matrix_rank(x.values) * 2

        moving = ols(y=y, x=x, weights=weights, window_type=window_type,
                     window=window, **kwds)

        # check that sparse version is the same
        sparse_moving = ols(y=y.to_sparse(), x=x.to_sparse(),
                            weights=weights,
                            window_type=window_type,
                            window=window, **kwds)
        _compare_ols_results(moving, sparse_moving)

        index = moving._index

        for n, i in enumerate(moving._valid_indices):
            if window_type == 'rolling' and i >= window:
                prior_date = index[i - window + 1]
            else:
                prior_date = index[0]

            date = index[i]

            x_iter = {}
            for k, v in compat.iteritems(x):
                x_iter[k] = v.truncate(before=prior_date, after=date)
            y_iter = y.truncate(before=prior_date, after=date)

            static = ols(y=y_iter, x=x_iter, weights=weights, **kwds)

            self.compare(static, moving, event_index=i,
                         result_index=n)

        _check_non_raw_results(moving)

    FIELDS = ['beta', 'df', 'df_model', 'df_resid', 'f_stat', 'p_value',
              'r2', 'r2_adj', 'rmse', 'std_err', 't_stat',
              'var_beta']

    def compare(self, static, moving, event_index=None,
                result_index=None):

        index = moving._index

        # Check resid if we have a time index specified
        if event_index is not None:
            ref = static._resid_raw[-1]

            label = index[event_index]

            res = moving.resid[label]

            tm.assert_almost_equal(ref, res)

            ref = static._y_fitted_raw[-1]
            res = moving.y_fitted[label]

            tm.assert_almost_equal(ref, res)

        # Check y_fitted

        for field in self.FIELDS:
            attr = '_%s_raw' % field

            ref = getattr(static, attr)
            res = getattr(moving, attr)

            if result_index is not None:
                res = res[result_index]

            tm.assert_almost_equal(ref, res)

    def test_ols_object_dtype(self):
        df = DataFrame(np.random.randn(20, 2), dtype=object)
        model = ols(y=df[0], x=df[1])
        summary = repr(model)  # noqa


class TestOLSMisc(object):

    """
    For test coverage with faux data
    """

    def test_f_test(self):
        x = tm.makeTimeDataFrame()
        y = x.pop('A')

        model = ols(y=y, x=x)

        hyp = '1*B+1*C+1*D=0'
        result = model.f_test(hyp)

        hyp = ['1*B=0',
               '1*C=0',
               '1*D=0']
        result = model.f_test(hyp)
        tm.assert_almost_equal(result['f-stat'], model.f_stat['f-stat'])

        with pytest.raises(Exception):
            model.f_test('1*A=0')

    def test_r2_no_intercept(self):
        y = tm.makeTimeSeries()
        x = tm.makeTimeDataFrame()

        x_with = x.copy()
        x_with['intercept'] = 1.

        model1 = ols(y=y, x=x)
        model2 = ols(y=y, x=x_with, intercept=False)
        tm.assert_series_equal(model1.beta, model2.beta)

        # TODO: can we infer whether the intercept is there...
        assert model1.r2 != model2.r2

        # rolling

        model1 = ols(y=y, x=x, window=20)
        model2 = ols(y=y, x=x_with, window=20, intercept=False)
        tm.assert_frame_equal(model1.beta, model2.beta)
        assert (model1.r2 != model2.r2).all()

    def test_summary_many_terms(self):
        x = DataFrame(np.random.randn(100, 20))
        y = np.random.randn(100)
        model = ols(y=y, x=x)
        model.summary

    def test_y_predict(self):
        y = tm.makeTimeSeries()
        x = tm.makeTimeDataFrame()
        model1 = ols(y=y, x=x)
        tm.assert_series_equal(model1.y_predict, model1.y_fitted)
        tm.assert_almost_equal(model1._y_predict_raw, model1._y_fitted_raw)

    def test_predict(self):
        y = tm.makeTimeSeries()
        x = tm.makeTimeDataFrame()
        model1 = ols(y=y, x=x)
        tm.assert_series_equal(model1.predict(), model1.y_predict)
        tm.assert_series_equal(model1.predict(x=x), model1.y_predict)
        tm.assert_series_equal(model1.predict(beta=model1.beta),
                               model1.y_predict)

        exog = x.copy()
        exog['intercept'] = 1.
        rs = Series(np.dot(exog.values, model1.beta.values), x.index)
        tm.assert_series_equal(model1.y_predict, rs)

        x2 = x.reindex(columns=x.columns[::-1])
        tm.assert_series_equal(model1.predict(x=x2), model1.y_predict)

        x3 = x2 + 10
        pred3 = model1.predict(x=x3)
        x3['intercept'] = 1.
        x3 = x3.reindex(columns=model1.beta.index)
        expected = Series(np.dot(x3.values, model1.beta.values), x3.index)
        tm.assert_series_equal(expected, pred3)

        beta = Series(0., model1.beta.index)
        pred4 = model1.predict(beta=beta)
        tm.assert_series_equal(Series(0., pred4.index), pred4)

    def test_predict_longer_exog(self):
        exogenous = {"1998": "4760", "1999": "5904", "2000": "4504",
                     "2001": "9808", "2002": "4241", "2003": "4086",
                     "2004": "4687", "2005": "7686", "2006": "3740",
                     "2007": "3075", "2008": "3753", "2009": "4679",
                     "2010": "5468", "2011": "7154", "2012": "4292",
                     "2013": "4283", "2014": "4595", "2015": "9194",
                     "2016": "4221", "2017": "4520"}
        endogenous = {"1998": "691", "1999": "1580", "2000": "80",
                      "2001": "1450", "2002": "555", "2003": "956",
                      "2004": "877", "2005": "614", "2006": "468",
                      "2007": "191"}

        endog = Series(endogenous)
        exog = Series(exogenous)
        model = ols(y=endog, x=exog)

        pred = model.y_predict
        tm.assert_index_equal(pred.index, exog.index)

    def test_series_rhs(self):
        y = tm.makeTimeSeries()
        x = tm.makeTimeSeries()
        model = ols(y=y, x=x)
        expected = ols(y=y, x={'x': x})
        tm.assert_series_equal(model.beta, expected.beta)

        # GH 5233/5250
        tm.assert_series_equal(model.y_predict, model.predict(x=x))

    def test_various_attributes(self):
        # just make sure everything "works". test correctness elsewhere

        x = DataFrame(np.random.randn(100, 5))
        y = np.random.randn(100)
        model = ols(y=y, x=x, window=20)

        series_attrs = ['rank', 'df', 'forecast_mean', 'forecast_vol']

        for attr in series_attrs:
            value = getattr(model, attr)
            assert isinstance(value, Series)

        # works
        model._results

    def test_catch_regressor_overlap(self):
        df1 = tm.makeTimeDataFrame().ix[:, ['A', 'B']]
        df2 = tm.makeTimeDataFrame().ix[:, ['B', 'C', 'D']]
        y = tm.makeTimeSeries()

        data = {'foo': df1, 'bar': df2}

        with pytest.raises(Exception):
            ols(y=y, x=data)

    def test_columns_tuples_summary(self):
        # #1837
        X = DataFrame(np.random.randn(10, 2), columns=[('a', 'b'), ('c', 'd')])
        Y = Series(np.random.randn(10))

        # it works!
        model = ols(y=Y, x=X)
        model.summary


def _check_non_raw_results(model):

    def _check_repr(obj):
        repr(obj)
        str(obj)

    _check_repr(model)
    _check_repr(model.resid)
    _check_repr(model.summary_as_matrix)
    _check_repr(model.y_fitted)
    _check_repr(model.y_predict)


@pytest.fixture
def TS1():
    date_index = date_range(datetime(2009, 12, 11), periods=3,
                            freq=offsets.BDay())
    return Series([3, 1, 4], index=date_index)


@pytest.fixture
def TS2():
    date_index = date_range(datetime(2009, 12, 11), periods=5,
                            freq=offsets.BDay())
    return Series([1, 5, 9, 2, 6], index=date_index)


@pytest.fixture
def TS3():
    date_index = date_range(datetime(2009, 12, 11), periods=3,
                            freq=offsets.BDay())
    return Series([5, np.nan, 3], index=date_index)


@pytest.fixture
def TS4():
    date_index = date_range(datetime(2009, 12, 11), periods=5,
                            freq=offsets.BDay())
    return Series([np.nan, 5, 8, 9, 7], index=date_index)


@pytest.fixture
def DF1(TS2, TS4):
    data = {'x1': TS2, 'x2': TS4}
    return DataFrame(data=data)


@pytest.fixture
def DICT1(TS2, TS4):
    return {'x1': TS2, 'x2': TS4}


class TestOLSFilter(object):

    def test_filter_with_series_rhs(self, TS1, TS2):
        (lhs, rhs, weights, rhs_pre,
         index, valid) = _filter_data(TS1, {'x1': TS2}, None)
        tm.assert_series_equal(TS1.astype(np.float64), lhs, check_names=False)
        tm.assert_series_equal(TS2[:3].astype(np.float64), rhs['x1'],
                               check_names=False)
        tm.assert_series_equal(TS2.astype(np.float64), rhs_pre['x1'],
                               check_names=False)

    def test_filter_with_series_rhs2(self, TS1, TS2):
        (lhs, rhs, weights, rhs_pre,
         index, valid) = _filter_data(TS2, {'x1': TS1}, None)
        tm.assert_series_equal(TS2[:3].astype(np.float64), lhs,
                               check_names=False)
        tm.assert_series_equal(TS1.astype(np.float64), rhs['x1'],
                               check_names=False)
        tm.assert_series_equal(TS1.astype(np.float64), rhs_pre['x1'],
                               check_names=False)

    def test_filter_with_series_rhs3(self, TS3, TS4):
        (lhs, rhs, weights, rhs_pre,
         index, valid) = _filter_data(TS3, {'x1': TS4}, None)
        exp_lhs = TS3[2:3]
        exp_rhs = TS4[2:3]
        exp_rhs_pre = TS4[1:]
        tm.assert_series_equal(exp_lhs, lhs, check_names=False)
        tm.assert_series_equal(exp_rhs, rhs['x1'], check_names=False)
        tm.assert_series_equal(exp_rhs_pre, rhs_pre['x1'], check_names=False)

    def test_filter_with_DataFrame_rhs(self, TS1, TS2, TS4, DF1):
        (lhs, rhs, weights, rhs_pre,
         index, valid) = _filter_data(TS1, DF1, None)
        exp_lhs = TS1[1:].astype(np.float64)
        exp_rhs1 = TS2[1:3]
        exp_rhs2 = TS4[1:3].astype(np.float64)
        tm.assert_series_equal(exp_lhs, lhs, check_names=False)
        tm.assert_series_equal(exp_rhs1, rhs['x1'], check_names=False)
        tm.assert_series_equal(exp_rhs2, rhs['x2'], check_names=False)

    def test_filter_with_dict_rhs(self, TS1, TS2, TS4, DICT1):
        (lhs, rhs, weights, rhs_pre,
         index, valid) = _filter_data(TS1, DICT1, None)
        exp_lhs = TS1[1:].astype(np.float64)
        exp_rhs1 = TS2[1:3].astype(np.float64)
        exp_rhs2 = TS4[1:3].astype(np.float64)
        tm.assert_series_equal(exp_lhs, lhs, check_names=False)
        tm.assert_series_equal(exp_rhs1, rhs['x1'], check_names=False)
        tm.assert_series_equal(exp_rhs2, rhs['x2'], check_names=False)
