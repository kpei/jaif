import pandas as pd
import numpy as np
import requests as wr
from pandas.tseries.offsets import BDay
import datetime as dt
from pandas.stats.api import ols
import json
from StringIO import StringIO
import zipfile
from statistics import *
from helpers import *
from constants import *
from rebalance_funcs import *
from rebalance_helpers import *
import logging
logger = logging.getLogger(__name__)

# /* PULL / CALCULATE */ #

def calculatePortfolioStats(p, bench_p):
    bench_name = bench_p.columns[0]
    bench_p = bench_p.iloc[:,0].dropna()
    w = p.iloc[-1]/p.iloc[-1].sum()
    w.index = w.index
    mers = getFunds().loc[:, 'mer']
    bmer = mers[bench_name]
    port_mer = w.dot(mers[w.index])
    p = p.sum(axis=1)

    if(bench_p.index[0] > p.index[0]): #Check to see if benchmark starts after portfolio start date
        p = p[p.index >= bench_p.index[0]]

    p_ret = (p/p.shift(1)-1.).dropna()
    bench_ret = (bench_p/bench_p.shift(1)-1.).dropna()

    te = st_trackingerror(p_ret, bench_ret)
    ret = st_ret(p_ret)
    cagr = st_cagr(p_ret)
    avol = st_vol(p_ret)
    ret = st_ret(p_ret)
    var = st_var(p_ret)
    ir = st_ir(p_ret, bench_ret)
    tr = st_tr(p_ret)
    skew = p_ret.skew()
    kurt = p_ret.kurtosis()
    sharpe = st_sharpe(p_ret, bench_ret)
    beta = st_beta(p_ret, bench_ret)
    alpha = st_alpha(p_ret, bench_ret)
    return pd.Series([ret, cagr, avol, sharpe, beta, alpha,
                      te, var, ir, tr, skew, kurt, port_mer, bmer],
                     index=['return', 'cagr', 'vol', 'sharpe',
                            'beta', 'alpha', 'te', 'var', 'ir', 'tr',
                            'skew', 'kurt', 'mer', 'bmer'])

def calculateWeightedReturnContribution(val):
    rebal_numerator = list(pd.date_range(val.index[0], val.index[-1], freq='BQ')-BDay(1))
    rebal_numerator.append(val.index[-1])
    num = val.loc[rebal_numerator].reset_index(drop=True).astype('float64')
    rebal_denominator = list(pd.date_range(val.index[0], val.index[-1], freq='BQ'))
    rebal_denominator.insert(0, val.index[0])
    denom = val.loc[rebal_denominator].reset_index(drop=True).astype('float64')
    a = (num/denom-1.).fillna(0) #incase of 0% allocation
    b = (denom.divide(denom.sum(axis=1), axis=0))
    return (a*b).sum()

def calculateWeightedYield(val, inst):
    p = pd.melt(getFundsPrices().reset_index(), id_vars=['Date'],
                value_vars=PREF[inst]["universe"], var_name='fund_id', value_name='price').dropna()
    p_val = val.sum(axis=1)
    distr = getFundsDistributions()
    distr = distr[distr['fund_id'].isin(PREF[inst]["universe"])]
    w = val.divide(p_val, axis=0)
    w = pd.melt(w.reset_index(), id_vars=['Date'], value_vars=PREF[inst]["universe"],
                var_name='fund_id', value_name='pct') #unpivot
    distr = distr.reset_index().merge(w, on=['Date', 'fund_id']).merge(p, on=['Date', 'fund_id']).set_index('Date')
    distr['yield'] = (distr['dividend']/distr['price'])*distr['pct']
    return distr.groupby('fund_id')['yield'].sum()

def calculateRollingStatistics(p, bench_p, period, start_date=None):
    bench_ret = (bench_p/bench_p.shift(1)-1.).dropna()
    p_ret = (p/p.shift(1)-1.).dropna()

    if(bench_p.index[0] > p.index[0]): #Check to see if benchmark starts after portfolio start date
        p = p[p.index >= bench_p.index[0]]

    df = pd.DataFrame(p_ret).join(bench_ret).dropna()

    if start_date is not None:
        offset = sum(df.index >= start_date)+period
        df = df.iloc[-offset:,:]

    a = pd.DataFrame(index = df.index, columns = ['vol', 'sharpe', 'beta', 'alpha', 'te', 'var', 'ir', 'tr', 'skew', 'kurt'])
    def calc_stat(p_ret, bench_ret):
        te = st_trackingerror(p_ret, bench_ret)
        avol = st_vol(p_ret)
        var = st_var(p_ret)
        ir = st_ir(p_ret, bench_ret)
        tr = st_tr(p_ret)
        skew = p_ret.skew()
        kurt = p_ret.kurtosis()
        sharpe = st_sharpe(p_ret, bench_ret)
        beta = st_beta(p_ret, bench_ret)
        alpha = st_alpha(p_ret, bench_ret)
        return pd.Series([avol, sharpe, beta, alpha,
                          te, var, ir, tr, skew, kurt],
                         index=['vol', 'sharpe', 'beta', 'alpha', 'te', 'var', 'ir', 'tr', 'skew', 'kurt']).round(5)
    for i in range(period, len(df)+1):
        sub_df = df.iloc[max(i-period, 0):i,:] #I edited here
        ret = sub_df.iloc[:,0]
        b_ret = sub_df.iloc[:,1]
        a.loc[sub_df.index[-1], :] = calc_stat(ret, b_ret)

    return a.dropna().astype(float)

def calculateConditionalQuantileReturn(port_p, bench_p, anchor):
    df = getFundsPrices([anchor])
    df['fund'] = port_p
    df['benchmark'] = bench_p
    df = df.dropna().resample('W')
    df_ret = (df/df.shift(1)-1.).dropna()

    anchs = df_ret[anchor].quantile([0, 0.15, 0.30, 0.7, 0.85, 1.]) # calculate anchor quantiles
    cond_q = pd.DataFrame(0, index=['worst','bad','average','good','best'], columns=['benchmark', 'fund','anchor'])
    for i in range(0,len(anchs)-1):
        q_date = df_ret[(df_ret[anchor] >= anchs.iloc[i]) &
                        (df_ret[anchor] < anchs.iloc[i+1])].index # get the dates of each anchor quantile
        cond_q.ix[i,['benchmark','fund']] = df_ret.loc[q_date, ['benchmark', 'fund']].mean().values
        cond_q.ix[i,'anchor'] = 0.5*(anchs.iloc[i]+anchs.iloc[i+1]) # avg anchor return between two quantile
    cond_q.index.name = 'quantile'
    return cond_q

def calculatePortfolioPeriodicReturns(p):
    returns = {}
    val_today = p.iloc[-1]
    today = p.index[-1]
    returns['M'] = val_today/p[p.index <= today-pd.DateOffset(months=1)].iloc[-1]-1.
    returns['3M'] = val_today/p[p.index <= today-pd.DateOffset(months=3)].iloc[-1]-1.
    returns['6M'] = val_today/p[p.index <= today-pd.DateOffset(months=6)].iloc[-1]-1.
    returns['YTD'] = val_today/p[p.index <= dt.datetime(today.year,1,1)].iloc[-1]-1.
    returns['Y'] = val_today/p[p.index <= today-pd.DateOffset(months=12)].iloc[-1]-1.
    returns['2Y'] = val_today/p[p.index <= today-pd.DateOffset(months=24)].iloc[-1]-1.
    return pd.Series(returns)

def getSaveCrisisPerformance(inst, port): # Does not need to be calculated periodically unless new crises emerge
    p = (getPortfolioPrices(inst, port)).sum(axis=1)
    b = (getBenchmarkPrices(inst))
    out = pd.DataFrame([p,b], index=['Fund', 'benchmark']).T.dropna()
    ret = pd.DataFrame(0,index=CRISES.keys()+['crisis alpha','crisis beta', 'non-crisis alpha', 'non-crisis beta'],
                           columns=['benchmark', 'Fund'])
    tot_crisis = pd.DataFrame()
    for name in CRISES:
        crisis = out[(out.index >= CRISES[name]['sDate']) & (out.index <= CRISES[name]['eDate'])]
        crisis_ret = crisis/crisis.shift(1)-1.
        tot_crisis = pd.concat([tot_crisis, crisis_ret])
        crisis = (crisis_ret+1.).cumprod().dropna()-1.
        ret.loc[name,:] = crisis.iloc[-1]
        crisis.to_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/crises/'+name+'.csv', float_format='%.5f')
    # now calculate crisis vs non crisis alpha
    out_ret = out/out.shift(1)-1.
    tot_noncrisis = out_ret[~out_ret.index.isin(tot_crisis.index)]
    crisisbeta = st_beta(tot_crisis['Fund'], tot_crisis['benchmark'])
    crisisalpha = st_alpha(tot_crisis['Fund'], tot_crisis['benchmark'])
    noncrisisbeta = st_beta(tot_noncrisis['Fund'], tot_noncrisis['benchmark'])
    noncrisisalpha = st_alpha(tot_noncrisis['Fund'], tot_noncrisis['benchmark'])
    ret.loc['crisis alpha',:] = [0,crisisalpha]
    ret.loc['crisis beta', :] = [1,crisisbeta]
    ret.loc['non-crisis alpha',:] = [0,noncrisisalpha]
    ret.loc['non-crisis beta', :] = [1,noncrisisbeta]
    # Save summary returns
    ret.index.name = 'name'
    ret.to_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/crises/summary.csv', float_format='%.5f')
    return ret

def calculateCountryMarketValue(port):
    a = port.iloc[-1]*INIT_CAP
    b = getCountryFundsAllocation()
    b = b.set_index('value')
    b['mkt'] = a
    b['mkt_pct'] = b['mkt']*b['pct']
    return b.groupby(['country', 'code']).sum()['mkt_pct']

def calculateCountryRiskContribution(port):
    p = getFundsPrices()
    p = p[p.index >= dt.datetime.now()-dt.timedelta(90)]
    ret = p/p.shift(1)-1.
    covmat = ret.cov()
    w = port.iloc[-1]
    w = w/w.sum()
    covmat = covmat.loc[w.index,w.index]
    rc = (covmat.dot(w)*w)/(np.sqrt(covmat.dot(w).dot(w)))*np.sqrt(252)
    rc.index = rc.index
    b = getCountryFundsAllocation()
    b = b.set_index('value')
    b['rc'] = rc
    b['rc_pct'] = b['rc']*b['pct']
    return b.groupby(['country', 'code']).sum()['rc_pct']

def calculateCountryMetrics(port):
    a = calculateCountryMarketValue(port)
    b = calculateCountryRiskContribution(port)
    c = pd.DataFrame()
    c['mkt'] = a
    c['rc'] = b
    c['mkt_pct'] = c['mkt']/c['mkt'].sum()
    return c

def calculateMutualFundQuantiles(df):
    return df.quantile([0.05, 0.25, 0.5, 0.75, 0.95])

def getSaveMutualFundReturns():
    df = []
    cats = []
    for pg in range(1,24):
        print pg
        # Get Return Performance
        link = 'http://www.fundlibrary.com/funds/listing.asp?filtertype=custom&filterid=10039224&id=&t=3&l=all&lc=100&ps=100&p='+str(pg)+'&s=F.EnglishName50&sd=asc&rating='
        req = wr.get(link)
        html_cells = req.content.replace('  ','').replace('\n','')
        flrow = html_cells.split('<td align="right" valign="top" class="FLROW" nowrap><span class="fNumber">')[1:]
        flrow2 = html_cells.split('<td align="right" valign="top" class="FLROW1" nowrap><span class="fNumber">')[1:]
        data = flrow+flrow2
        arr = []
        i = 1
        for num in data:
            d = num.split('</span></td>')[0]
            d = '0' if d == '--' else d
            arr.append(float(d))
            i+=1
            if i%8 == 0:
                df.append(arr)
                arr = []
                i=1
        # Get Mutual Fund category
        link = 'http://www.fundlibrary.com/funds/listing.asp?filtertype=custom&filterid=10039224&id=&t=1&l=all&lc=100&ps=100&p='+str(pg)+'&s=F.EnglishName50&sd=asc&rating='
        req = wr.get(link)
        html_cells = req.content.replace('  ','').replace('\n','')
        flrow = html_cells.split('<td align="left" valign="top" class="FLROW" ><span class="fText">')[1:]
        flrow2 = html_cells.split('<td align="left" valign="top" class="FLROW1" ><span class="fText">')[1:]
        data = flrow+flrow2
        for i in data:
            cat = i.split('</span>')[0].replace('\x96','-')
            if len(cat) > 7:
                cats.append(cat)

    df = pd.DataFrame(df,columns = ['NAVPS','m','3m','6m','ytd','y','yy'])
    df['category'] = cats
    df = df[df['m'] != 0].iloc[:,1:] # elimnate non data mutual funds and NAVPS column
    # save it
    df.to_csv(APP_STATIC+'/ducket/mutfundrets.csv', float_format='%.5f', index=False)
    return df[df['m'] != 0].iloc[:,1:]

def getSaveMorningStarAllocation(subset=None):
    a = getFunds()
    a = a if subset is None else a.loc[subset,:]
    b = pd.DataFrame()
    for i,j in a.iterrows():
        print i+": "+j['label']
        c = requestCountryAllocation(i)
        df = pd.DataFrame({'value':[i]*len(c),
                           'pct': list(c.values),
                           'country': list(c.index)})
        b = pd.concat([b, df], ignore_index=True)
    c = pd.read_csv(APP_STATIC+'/ducket/countrycodes.csv', encoding='utf-8-sig')
    b = b.merge(c, on='country')
    b.to_csv(APP_STATIC+'/ducket/countrypct.csv', index=False, float_format='%.5f')
    return b

def requestCountryAllocation(code):
    req = wr.get('http://quote.morningstar.ca/QuickTakes/fund/PortfolioOverviewNew.aspx?t='+code+'&region=CAN&culture=en-CA')
    if(req.content.find('Top 10 Country Breakdown') > 0):
        identifier = 'Top 10 Country Breakdown' # fixed income fund
        del_country = '<tr><td>'
        single_split = True
        del_ctext = '</td>'
    else:
        identifier = 'id="world_regions_tab"'  # equity fund
        del_country = '<td>&nbsp;</td>'
        single_split = False
        del_ctext_start = '<td>'; del_ctext_end = '</td>';

    html_table = req.content.split(identifier)[1].split('</table>')[0]
    html_countries = html_table.replace('  ','').replace('\n','').split(del_country)[1:]
    out = pd.Series()
    for text in html_countries:
        country = text.split(del_ctext)[0] if single_split else text.split(del_ctext_start)[1].split(del_ctext_end)[0]
        pct = text.split('<td align="right">')[1].split('</td>')[0]
        if(pct == "&mdash;"): # no data, skip
            print "no data detected, skipping"
            continue
        out[country] = float(pct)/100.
    return out[out > 0]

def calculateMutualFundProportions():
    mut_ret = getMutualFundReturns()
    q = calculateMutualFundQuantiles(mut_ret)
    unq_cats = mut_ret['category'].unique()
    pnl = pd.Panel(None, major_axis=unq_cats, minor_axis=[0,1,2,3])
    for period,rets in q.T.iterrows():
        df = pd.DataFrame(None,index=unq_cats)
        for i in range(0,4):
            mut_ret_sub = mut_ret.loc[(mut_ret[period] >= rets.iloc[i]) & (mut_ret[period]< rets.iloc[i+1])]
            df[i] = mut_ret_sub['category'].value_counts()
        df.fillna(0, inplace=True)
        pnl[period] = df
    return pnl

def calculateSaveFamaFrenchAttribution(inst, port):
    ff = pd.read_pickle(APP_STATIC+'/ducket/ff6.pkl')
    p = getPortfolioPrices(inst, port).sum(axis=1)
    bm = getBenchmarkPrices(inst).dropna()
    bm_ret = bm/bm.shift(1)-1.
    ret = p/p.shift(1)-1.
    #recent performance based on FAMA-FRENCH !!
    bm_ret = bm_ret[bm_ret.index >= ff.index[-1]-dt.timedelta(days=ROLLING_WINDOW)]
    ret = ret[ret.index >= ff.index[-1]-dt.timedelta(days=ROLLING_WINDOW)]
    ff['fund'] = ret*100-ff['RF']
    ff['bench'] = bm_ret*100-ff['RF']
    ff.dropna(inplace=True)
    res = ols(y=ff['fund'], x=ff[['Mkt-RF','SMB','HML','RMW','CMA','WML']])
    res_b = ols(y=ff['bench'], x=ff[['Mkt-RF','SMB','HML','RMW','CMA','WML']])
    corrs = ff[['fund', 'Mkt-RF','SMB','HML','RMW','CMA','WML']].corr().iloc[1:,0]
    corrs_b = ff[['bench', 'Mkt-RF','SMB','HML','RMW','CMA','WML']].corr().iloc[1:,0]
    output = pd.DataFrame({'coef':res.beta, 'coef_b':res_b.beta, 'corr':corrs, 'corr_b':corrs_b})
    output.to_pickle(APP_STATIC+'/folio/'+inst+'/'+port+'/ff.pkl')
    return output

def requestFFData():
    def download_file(url, name):
        # NOTE the stream=True parameter
        r = wr.get(url, stream=True)
        name = name+'.'+r.url.split('.')[-1]
        with open(name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    #f.flush() commented by recommendation from J.F.Sebastian
        return name

    # Download Fama French global 5 Factor
    link = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Global_5_Factors_Daily_CSV.zip"
    download_file(link, 'ff5')
    link = "http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Global_Mom_Factor_Daily_CSV.zip"
    download_file(link, 'ffmom')
    # unzip
    zip = zipfile.ZipFile('ff5.zip')
    zip.extractall(APP_STATIC+'/ducket/')
    zip.close()
    zip = zipfile.ZipFile('ffmom.zip')
    zip.extractall(APP_STATIC+'/ducket/')
    zip.close()

    #convert to pandas
    ff = pd.read_csv(APP_STATIC + '/ducket/Global_5_Factors_Daily.csv', skiprows=6, index_col=0)
    ff.index = pd.to_datetime(ff.index, format='%Y%m%d')
    ffmom = pd.read_csv(APP_STATIC + '/ducket/Global_MOM_Factor_Daily.csv', skiprows=6, index_col=0)
    ffmom.index = pd.to_datetime(ffmom.index, format='%Y%m%d')
    ff = ff.join(ffmom).dropna()
    ff.to_pickle(APP_STATIC+'/ducket/ff6.pkl')
    return ff


# /* PULL / CALCULATE */ #

# /* PULL MSTAR */ #

def mstarStampToDate(t):
    return dt.datetime(1900,1,1)+dt.timedelta(days=t)

def request_mstar_info(fundId):
    r = wr.get("http://globalquote.morningstar.com/globalcomponent/RealtimeHistoricalStockData.ashx?ticker="+fundId+"&showVol=false&dtype=his&f=d&curry=CAD&range=1900-1-1|" +
            dt.date.today().strftime('%Y-%m-%d')+"&isD=true&isS=true&hasF=true&ProdCode=DIRECT")
    io = StringIO.StringIO(r.content)
    return json.load(io)

def get_mstar_price_series(r_json):
    price_series = map(lambda x: x[0], r_json['PriceDataList'][0]['Datapoints']) # take out of the per-dp list
    date_series = map(mstarStampToDate, r_json['PriceDataList'][0]['DateIndexs']) # format date
    return pd.Series(price_series, index = date_series)

def get_mstar_div(card):
    date = dt.datetime.strptime(card['Date'], '%Y-%m-%d')
    dividend = float(card['Desc'].split(":")[1].split("<br>")[0])
    return [date, dividend]

def get_mstar_div_series(r_json):
    a = map(get_mstar_div, r_json['DividendData'])
    if(len(a) > 0 ):
        dates, divs = zip(*a)
        div_series = pd.Series(divs, index=dates)
        return div_series
    else:
        return []

def savePrices(p):
    p.index.name = 'Date'
    pd.to_pickle(p, APP_STATIC+'/ducket/fund.prices.pkl')

def saveDivs(d):
    d.index.name = 'Date'
    pd.to_pickle(d, APP_STATIC+'/ducket/fund.divs.pkl')



def pull():
    logger.info("retrieving list of funds")
    funds = getFunds()
    p = pd.DataFrame()
    d = pd.DataFrame()
    for ind,row in funds.iterrows():
        fund_id = ind
        fund_name = row['label']
        logger.info(fund_id+": "+fund_name)
        request = request_mstar_info(fund_id)
        divs = get_mstar_div_series(request)
        df_divs = pd.DataFrame(divs, columns = ["dividend"])
        df_divs['fund_id'] = fund_id
        logger.debug("Retrieved Dividends Preview: %s" % df_divs.tail())
        prices = get_mstar_price_series(request)
        df_prices = pd.DataFrame(prices, columns = ["price"])
        df_prices['fund_id'] = fund_id
        logger.debug("Retrieved Prices Preview: %s" % df_prices.tail())
        p = pd.concat([p, df_prices])
        d = pd.concat([d, df_divs])

    logger.info("Saving all prices and divs")
    savePrices(p)
    saveDivs(d)
    logger.info("Done Pull()")
    return True


# /* PULL MSTAR */ #


# /* CONCATENATED */ #

def calculateBenchmarkPrices():
    logger.info("Calculating Benchmark Prices")
    for inst in PREF:
        logger.info("Calculating price on "+inst+" benchmark...")
        port_price = rebalance(universe=[PREF[inst]['benchmark']],
              rebal_func=reb_benchmark,
              p_cap=1.,
              lb=0,
              rebal_cycle='90000D', #some long time bick boi
              mom_lb=0, tact_lb=0, c=0, bond=None, groups=None,
             )
        port_price.index.name = 'Date'
        logger.debug("Calculated Benchmark Prices Preview: %s" % port_price.tail())
        logger.info("Saving benchmark prices")
        port_price.to_pickle(APP_STATIC+'/folio/'+inst+'/benchmark/p.pkl')
    logger.info("Done calculateBenchmarkPrices()")
    return True

def calculatePortfolios():
    logger.info("Calculating Portfolios ")
    for inst in PREF:
        logger.info("Calculating on %s portfolios..." % inst)
        for port in FOLIOS:
            if(port == 'benchmark'): continue # skip all benchmarks, calculate them beforehand
            logger.debug("Calculating Prices/Stats for %s" % port)
            opts = FOLIOS[port]['rebal_params'].copy()
            opts.update(PREF[inst]['params'])
            port_price = rebalance(universe = PREF[inst]["universe"],
                                   rebal_func = FOLIOS[port]['rebal_func'],
                                   groups = PREF[inst]['cluster'],
                                  **opts)
            port_price.index.name = 'Date'
            total_port_price = port_price.sum(axis=1)
            logger.debug("Calculated Portfolio Prices Preview: %s" % port_price.tail())
            logger.debug("Saving %s portfolio prices" % port)
            port_price.to_pickle(APP_STATIC+'/folio/'+inst+'/'+port+'/p.pkl')
            logger.info("Price calculation complete, calculating statistics...")

            bench_price = getBenchmarkPrices(inst, as_series=False)
            logger.info("   Statistics %s:" % port)
            new_stats = calculatePortfolioStats(port_price, bench_price)
            logger.debug("      %s" % new_stats)
            logger.info("   Weighted Return Contribution %s:" % port)
            new_rc = calculateWeightedReturnContribution(port_price)
            logger.debug("      %s" % new_rc)
            logger.info("   Weighted Yield %s:" % port)
            new_yield = calculateWeightedYield(port_price, inst)
            logger.debug("      %s" % new_yield)
            logger.info("   Rolling Statistics %s:" % port)
            new_rolling_stats = calculateRollingStatistics(total_port_price, bench_price.iloc[:,0], ROLLING_WINDOW)
            logger.debug("      %s" % new_rolling_stats.tail())
            logger.info("   Conditional Quantile %s:" % port)
            new_condquantile = calculateConditionalQuantileReturn(total_port_price, bench_price.iloc[:,0], PREF[inst]["anchor"])
            logger.debug("      %s" % new_condquantile)
            logger.info("   Country Metrics %s:" % port)
            new_countrymetrics = calculateCountryMetrics(port_price)
            logger.debug("      %s" % new_countrymetrics)
            logger.info("   Periodic Returns %s:" % port)
            new_periodrets = calculatePortfolioPeriodicReturns(total_port_price)
            logger.debug("      %s" % new_periodrets)

            logger.info("")
            logger.info("Saving all Calculated statistics")
            new_countrymetrics.to_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/cm.csv', float_format='%.5f')
            new_periodrets.to_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/pr.csv', float_format='%.5f')
            new_rc.to_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/rc.csv', float_format='%.5f')
            new_yield.to_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/yi.csv', float_format='%.5f')
            new_stats.to_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/s.csv', float_format='%.5f')
            new_rolling_stats.to_pickle(APP_STATIC+'/folio/'+inst+'/'+port+'/rs.pkl')
            new_condquantile.to_csv(APP_STATIC+'/folio/'+inst+'/'+port+'/condq.csv', float_format='%.5f')
            logger.info("------------------------------------------------------")
    logger.info("Done calculatePortfolios()")
    return True

def calculateOccasionals():
    logger.info("Requesting Fama French Attributions data")
    requestFFData()
    for inst in PREF:
        logger.info("Calculating Occasionals on %s portfolios..." % inst)
        for port in FOLIOS:
            if(port == 'benchmark'): continue # skip all benchmarks, calculate them beforehand
            print port
            logger.info("Calculating/Saving Crisis Performances")
            getSaveCrisisPerformance(inst, port)
            logger.info("Calculating/Saving Fama French Attribution Performances")
            calculateSaveFamaFrenchAttribution(inst, port)
    logger.info("Requesting and Calculating mutual fund Proportions")
    getSaveMutualFundReturns()
    pnl = calculateMutualFundProportions()
    logger.info("Saving mutual fund Proportions")
    pnl.to_pickle(APP_STATIC+'/ducket/mut_fund_prop.pkl')
    logger.info("Requesting Morningstar Allocations data")
    getSaveMorningStarAllocation()
    logger.info("Done calculateOccasionals()")
    return True

# /* CONCATENATED */ #