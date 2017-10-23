
from flask import send_file
import pandas as pd
import numpy as np
import datetime as dt
import StringIO
import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')

def fixDate(df):
    df['Date'] = df['Date'].apply(pd.to_datetime)
    df = df.set_index('Date')
    return df
def getPortfolioPrices(inst, port):
    p = pd.read_pickle(APP_STATIC+'/folio/'+inst+'/'+port+'/p.pkl')
    return p
def getBenchmarkPrices(inst, as_series=True):
    p = getPortfolioPrices(inst, 'benchmark')
    if(as_series):
        return p.iloc[:,0]
    else:
        return p
def getRollingStatistics(inst, port):
    p = pd.read_pickle(APP_STATIC+'/folio/'+inst+'/'+port+'/rs.pkl')
    return p
def getFamaFrenchAttribution(inst, port):
    return pd.read_pickle(APP_STATIC+'/folio/'+inst+'/'+port+'/ff.pkl')
def getCountryFundsAllocation():
    return pd.read_csv(APP_STATIC+'/ducket/countrypct.csv')
def getCountryPortfolioMetrics(inst, port):
    return pd.read_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/cm.csv')
def getEfficientFrontier(inst, port):
    return pd.read_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/ef.csv').set_index('0')
def getCrisisBreakdownPerformance(inst, port):
    return pd.read_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/crises/summary.csv')
def getReturnContributions(inst, port):
    return pd.Series.from_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/rc.csv')
def getPortfolioPeriodicReturns(inst, port):
    return pd.Series.from_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/pr.csv')
def getReturnYield(inst, port):
    return pd.Series.from_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/yi.csv')
def getConditionalQuantileReturns(inst, port):
    return pd.read_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/condq.csv').set_index('quantile')
def getMutualFundReturns():
    return pd.read_csv(APP_STATIC+'/ducket/mutfundrets.csv')
def getMutualFundProportions(per=None, perf=None):
    o = pd.read_pickle(APP_STATIC+'/ducket/mut_fund_prop.pkl')
    o = o.loc[per] if per is not None else o
    o = o.loc[:,perf] if perf is not None else o
    return o
def getPortfolioStats(inst, port):
    return pd.Series.from_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/s.csv')
def prepareCSV(obj):
    strIO = StringIO.StringIO()
    strIO.write(obj.to_csv())
    strIO.seek(0)
    return send_file(strIO, attachment_filename="file.csv",as_attachment=True)
def getFunds():
    return pd.read_csv(APP_STATIC+'/ducket/funds.csv').set_index('value')
def get_by_universe(a, universe):
    if(universe):
        return a[a['fund_id'].isin(universe)].dropna()
    return a.dropna()    
def getFundsPrices(universe=None):
    r = get_by_universe(pd.read_pickle(APP_STATIC+'/ducket/fund.prices.pkl'), universe)
    return r.pivot(columns="fund_id", values="price")
def getFundsDistributions(universe=None):
    r = get_by_universe(pd.read_pickle(APP_STATIC+'/ducket/fund.divs.pkl'), universe)
    return r
def getCrisisSeries(inst, port, crisis):
    p = pd.read_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/crises/'+crisis+'.csv')
    p = fixDate(p)
    return p
def fundId2Name(l):
    funds = getFunds()
    if(type(l) == list):
        return funds.loc[l,'label'].tolist()
    else:
        return funds.loc[l,'label']