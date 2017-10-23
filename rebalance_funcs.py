import pandas as pd
import numpy as np
from rebalance_helpers import *
import datetime as dt
from helpers import *
import logging
logger = logging.getLogger(__name__)

def rebalance(universe=None,
              rebal_func=None,
              p_cap=1.,
              lb=365,
              rebal_cycle='BQ',
              mom_lb=6*30, tact_lb=200, c=8., bond=None, groups=None,
             ):

    prices = getFundsPrices(universe).dropna()
    dividends = getFundsDistributions(universe).dropna()

    groups = universe if groups is None else groups

    acc_dividends = pd.Series()
    shares = None
    m_val = pd.DataFrame(None, columns=prices.columns)

    tlb = max([lb, mom_lb, tact_lb])
    start_date = prices.index[tlb]
    end_date = prices.index[-1]

    dates = pd.date_range(start_date, end_date, None, 'B')
    rebal_dates = pd.date_range(start_date, end_date, None, rebal_cycle)

    for i in dates:
        psub = prices[prices.index <= i]
        j = psub.iloc[-1] # get first available price

        try: # try to get all dividends paid/share today
            acc_dividends = dividends.loc[[i]].set_index('fund_id')['dividend']
        except:
            acc_dividends = pd.Series(0.0, index=j.index) # else no dividends

        if (i in rebal_dates) or (shares is None):
            rebal = rebal_func(psub, lb=lb, mom_lb=mom_lb, tact_lb=tact_lb, c=c, bond=bond, groups=groups)
            shares = rebal*p_cap/j
            logger.debug('Rebalancing: %s Capital - %.2f' % (dt.datetime.strftime(i, "%d/%m/%Y"), p_cap))

        shares += ((shares*acc_dividends)/j).fillna(0.0) # new amount of shares purchased from dividends
        m_val.loc[i,:] = (shares*j)
        p_cap = m_val.loc[i].sum()

    return m_val.astype(float) # To prevent df from turning into object





def reb_ew(p, **k): # Equal Weight
    return pd.Series([1./len(p.columns)]*len(p.columns), index=p.columns)

def reb_ewvt(p, **k): #c=8.
    ret = (p / p.shift(1) - 1.)*100.
    ret = returnWindow(ret, k['lb'])
    covmat = clusterCovariance(ret, k['groups'])
    g = optim(covmat, typ='ew', c=k['c'])
    return pd.Series(g.x, index=covmat.index)

def reb_rcf(p, **k):
    ret = (p / p.shift(1) - 1.)*100.
    ret = returnWindow(ret, k['lb'])
    covmat = clusterCovariance(ret, k['groups'])
    g = optim(covmat, typ='rcf')
    return pd.Series(g.x, index=covmat.index)

def reb_mom(p, **k): #mom_lb=6*30
    ret = (p / p.shift(1) - 1.)*100.
    b_ret = returnWindow(ret, k['lb'])
    mom_ret = (p/p.shift(k['mom_lb'])).dropna().iloc[-1].order(ascending=False).index
    covmat = clusterCovariance(b_ret, k['groups']).loc[mom_ret,mom_ret]
    g = optim(covmat, typ='sort', c=k['c'])
    w = pd.Series(g.x, index=covmat.index)
    return w


def reb_tact(p, **k): #tact_lb=200, c=8., bond='28'
    ret = (p / p.shift(1) - 1.)*100.
    ret = returnWindow(ret, k['lb'])
    tact_shift = pd.rolling_mean(p, window=k['tact_lb']).dropna().iloc[-1]
    tact_shift = (p.iloc[-1] - tact_shift)
    ind = tact_shift.order(ascending=False).index
    nulle = tact_shift[tact_shift.index != k['bond']][tact_shift < 0].index
    covmat = clusterCovariance(ret, k['groups']).loc[ind,ind]
    g = optim(covmat, typ='sort', c=k['c'])
    w = pd.Series(g.x, index=covmat.index)
    to_bonds = sum(w[nulle])
    w[nulle] = 0.0
    w[k['bond']] += to_bonds #Short Term Bonds or Cash Equivalent
    return w

def reb_momtact(p, **k): #mom_lb=6*30, tact_lb=200, c=8., bond='28'
    w = reb_mom(p, **k)
    w_tact = reb_tact(p, **k)
    tact_shift = w_tact[w_tact == 0.0].index
    to_bonds = w[tact_shift].sum()
    w[tact_shift] = 0.0
    w[k['bond']] += to_bonds #Short Term Bonds or CE

    return w

def reb_benchmark(p, **k):
    return pd.Series([1.], index=p.columns)