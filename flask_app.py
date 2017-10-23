from flask import Flask, url_for, render_template, send_file, jsonify, request
import StringIO
import pandas as pd
import numpy as np
import requests as wr
from scipy.optimize import minimize
import csv
import datetime as dt
import os

from statistics import *
from helpers import *
from constants import *
from calculate import *

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True #use in development only
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')



# INDEX
########
@app.route('/')
def hello_world():
    dis_year = dt.datetime.now().year
    return render_template('index.html', year=dis_year)
    
# ABOUT
########
@app.route('/about')
def about():
    dis_year = dt.datetime.now().year
    return render_template('about.html', year=dis_year)

# PORTFOLIOS
##############
@app.route('/portfolios')
def portfolios():
  dis_year = dt.datetime.now().year
  if ('folios' in request.args) and ('inst' in request.args):
    portfolio = request.args['folios']
    institution = request.args['inst']
    return render_template('portfolio.html', year=dis_year, folio=portfolio, inst=institution)
  return render_template('portfolios.html', year=dis_year)
  
# Retrieve Available portfolios
################################
@app.route('/getAvailableFolios')
def getAvailableFolios():
	folios = {f: FOLIOS[f]['name'] for f in FOLIOS}
	folios.pop('benchmark', None) # get rid of benchmark
	return jsonify(folios)
    
# Retrieve portfolio name and description
#########################################
@app.route('/getFolioDetails/<string:inst>/<string:port>')
def getFolioDetails(inst, port):
    if port in FOLIOS:
        return jsonify({
        	'inst': INST[inst],
            'name': FOLIOS[port]['name'],
            'desc': FOLIOS[port]['desc'],
            #'benchmark': FOLIOS[port]['benchmark']
        })
    else:
        return jsonify({})

# Retrieve Portfolio Statistics as JSON
#######################################
@app.route('/<string:inst>/<string:port>/getFolioStat')
def getFolioStat(inst, port):
    return getPortfolioStats(inst, port).to_json()
    
# Retrieve portfolio holdings information as json
##################################################
@app.route('/<string:inst>/<string:port>/getHoldingsInformation')
def getHoldingsInformation(inst, port):
    funds = getFunds()
    contrib = getReturnContributions(inst, port)
    yields = getReturnYield(inst, port)
    val = getPortfolioPrices(inst, port).iloc[-1]
    funds['contrib'] = contrib
    funds['yield'] = yields
    funds['yield'] = funds['yield'].fillna(0)
    funds['weight'] = val/val.sum()
    return funds.dropna().reset_index(drop=True).to_json(orient='records')

# Rolling Statistics
#####################
@app.route('/<string:inst>/<string:port>/graphRollingStatistics/<string:name>')
def graphRollingStatistics(inst, port, name):
    s = getRollingStatistics(inst, port).loc[:,[name]]
    return prepareCSV(s)
    
# get crisis performance - series and table
##########################
@app.route('/<string:inst>/<string:port>/graphCrisisPerformance/<string:name>')
def graphCrisisPerformance(inst, port, name):
  return prepareCSV(getCrisisSeries(inst, port, name))

@app.route('/<string:inst>/<string:port>/getCrisisBreakdown')
def getCrisisBreakdown(inst, port):
  o = getCrisisBreakdownPerformance(inst, port)
  o.columns = ['name', 'Benchmark', 'Fund']
  return o.reset_index(drop=True).to_json(orient='records')
  
# graph conditional quantile in event analysis
################################################
@app.route('/<string:inst>/<string:port>/graphConditionalQuantileReturns')
def graphConditionalQuantileReturns(inst, port):
  o = getConditionalQuantileReturns(inst, port)*100
  o.index = o.apply(lambda x: x.name+'<br/>'+str(x['anchor'].round(2))+'%', axis=1) # add SP 500 returns ontop of xaxis
  o.index.name = "category"
  return prepareCSV(o.loc[:,['benchmark','fund']])

# Graph allocation map - output is json
#############################
@app.route('/<string:inst>/<string:port>/graphCountryPortfolioMetrics')
def graphCountryPortfolioMetrics(inst, port):
  o = getCountryPortfolioMetrics(inst, port)
  o[['rc','mkt_pct']] *= 100
  o.columns = ['name', 'code', 'mktval', 'value', 'mktw'] # for highmaps
  o = o.set_index('name')
  return o.to_json(orient='records')

# TS-Allocation
################
@app.route('/<string:inst>/<string:port>/graphAllocationSeries/<string:output>/<string:isIndex>')
def graphAllocation(inst, port, output, isIndex):
    p = getPortfolioPrices(inst, port)
    p.columns = fundId2Name(p.columns)
    if(output == 'pct'):
        p = p.divide(p.sum(axis=1), axis=0)
    if(isIndex == '1'):
        p = p.groupby(pd.TimeGrouper(RESAMPLE_PERIOD)).first()
    return prepareCSV(p.round(3))

# TS-Performance
#################
@app.route('/<string:inst>/<string:port>/graphPerformanceSeries/<string:isIndex>')
def graphFolioPerformance(inst, port, isIndex, sDate=None, eDate=None):
  p = (getPortfolioPrices(inst, port)).sum(axis=1)
  b = (getBenchmarkPrices(inst))
  bench_name = fundId2Name(PREF[inst]['benchmark'])
  sDate = dt.datetime.strptime(request.args['sDate'], '%Y-%m-%d') if 'sDate' in request.args else p.index[0]
  eDate = dt.datetime.strptime(request.args['eDate'], '%Y-%m-%d') if 'eDate' in request.args else dt.datetime.now()
  out = pd.DataFrame([p,b], index=['Fund', bench_name]).T
  out = out[(out.index >= sDate) & (out.index <= eDate)].dropna()
  out = (out/out.shift(1)).cumprod()*INIT_CAP
  if(isIndex == '1'):
      out = out.groupby(pd.TimeGrouper(RESAMPLE_PERIOD)).first()
  return prepareCSV(out.round(3))


# Fama French Regression coeficients
#########################################
@app.route('/<string:inst>/<string:port>/graphFamaFrenchAttribution')
def graphFamaFrenchAttribution(inst, port):
  p = getFamaFrenchAttribution(inst, port)
  p[['corr', 'corr_b']] = p[['corr', 'corr_b']]*100
  alpha = p.loc['intercept', ['coef', 'coef_b']]*252 # get model alpha
  p = p.drop('intercept')
  b = p[['coef_b', 'corr_b']]
  p = p[['coef', 'corr']]
  b.columns = ['coef', 'y']
  p.columns = ['coef', 'y']
  o = {
    'alpha': alpha.to_dict(),
    'series_fund': {
      'name': 'fund',
      'data': p.to_dict(orient='records'),
      'pointPlacement': 'on'
    },
    'series_benchmark': {
      'name': 'benchmark',
      'data': b.to_dict(orient='records'),
      'pointPlacement': 'on'
    }
  }
  return jsonify(o)

# Get the Mutual fund return quantiles as json
#####################################################
@app.route('/graphMutualFundPerformances')
def graphMutualFundPerformances():
  a = calculateMutualFundQuantiles(getMutualFundReturns())
  a.index = ['poor', 'bad', 'good', 'excellent', 'na']
  series = []
  for i,x in enumerate(a.iterrows()):
      if i == 4: break #prevent last row from being operated
      
      df = a.iloc[i:i+2].T.reset_index(drop=True) #convert to low,high df
      df.columns = ['low', 'high']
      series.append({
              'name': x[0],
              'data': df.to_dict(orient='records')
          })
  return jsonify(series)

# get the mutual fund porportions for a performance tier as json
#################################################################
@app.route('/<string:per>/graphMutualFundProportions')
def graphMutualFundProportions(per):
  prop = getMutualFundProportions(per)
  o = {}
  names = ['poor', 'bad', 'good', 'excellent']
  for i in range(0,4):
      propsub = prop[i][prop[i] > 0].sort_values(ascending=False).reset_index()
      propsub.columns = ['name','y']
      o[names[i]] = propsub.to_dict(orient='records')
  return jsonify(o)

# get the portfolio periodical returns
#####################################################
@app.route('/<string:inst>/<string:port>/graphPortfolioPeriodicReturns')
def graphPortfolioPeriodicReturns(inst, port):
  o = getPortfolioPeriodicReturns(inst, port)*100
  return o.to_json()

# remove below if on pyanywhere
if __name__ == "__main__":
    app.run()
