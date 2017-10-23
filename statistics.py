import pandas as pd
import numpy as np
import datetime as dt

def st_trackingerror(a,b):
    return (a-b).std()*np.sqrt(252)
def st_ret(a):
    return (a+1.).prod()-1.
def st_cagr(a):
    length = (a.index[-1]-a.index[0]).days
    decay_factor = 1/(length/365.)
    return ((st_ret(a)+1.)**(decay_factor)-1.)
def st_vol(a):
    return a.std()*np.sqrt(252)
def st_var(a):
    return a.mean()*252-1.96*st_vol(a)
def st_ir(a,b):
    return (a.mean()-b.mean()/st_trackingerror(a,b))*252
def st_tr(a):
    return np.percentile(a, 95)/abs(np.percentile(a, 5))
def st_sharpe(a,b):
    return (a.mean()*252)/st_vol(a) - (b.mean()*252)/(st_vol(b))
def st_beta(a,b):
    return a.cov(b)/b.var()
def st_alpha(a,b):
    return (a.mean()-b.mean()*st_beta(a,b))*252