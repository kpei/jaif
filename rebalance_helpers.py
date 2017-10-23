import pandas as pd
import numpy as np
from scipy.optimize import minimize
import datetime as dt
from scipy.stats import norm

def returnWindow(ret, lb):
    return ret[(ret.index >= ret.index[-lb])]

def frontierOptim(c, e_ret, covmat):
    def optim_ret(x, ret):
        return -x.dot(ret)
    cons = (
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'eq', 'fun': lambda x: 1.-np.sum(x)},
        {'type': 'eq', 'fun': lambda x,cov: np.sqrt(cov.dot(x).dot(x))-(c), 'args': (covmat,)},
    )
    g = minimize(optim_ret,
        x0 = [1./float(len(covmat))]*len(covmat),
        args=(e_ret), jac=False, options=({'maxiter': 1e4}), method='SLSQP',
        constraints=cons)
    if(g['success'] == False): print 'Error in optimizing'
    return -g.fun

def clusterCorrelation(ret, g):
    c = pd.ewmcorr(ret,span=32.33,adjust=False).iloc[-1,:,:]
    o = pd.DataFrame(0, index=range(0,len(g)), columns=range(0,len(g)))
    for i in range(0,len(g)):
        for j in range(i, len(g)):
            if(i==j): # will be a corr square
                corr_m = np.triu(c.loc[g[i], g[j]], k=1).flatten()
                c_corr = corr_m[corr_m != 0].mean()
            else:
                c_corr = (c.loc[g[i], g[j]]).stack().mean()
            o.loc[i,j] = c_corr
            o.loc[j,i] = c_corr
        inGroup = c.loc[g[i],g[i]]
        inGroup.values[[np.arange(len(inGroup))]*2] = None
        o.loc[i,i] = inGroup.stack().dropna().mean()
    o.fillna(1., inplace=True)
    return o

def cluster2asset(g,cl):
    l = sum(g,[])
    o = pd.DataFrame(0, index=l, columns=l) #unlists
    for ind, gr in enumerate(g): # group x
        for jnd, gro in enumerate(g): # group x + 1
            for i in gr:
                for j in gro:
                    o.loc[i,j] = cl.loc[ind, jnd]
    o.values[[np.arange(len(o))]*2] = 1.
    return o

def cor2cov(corr, std):
    return corr*pd.DataFrame(std).dot(pd.DataFrame(std).T)

def clusterCovariance(ret, groups):
    cl = clusterCorrelation(ret, groups)
    corrmat = cluster2asset(groups, cl)
    std = pd.ewmstd(ret,span=32.33,adjust=False).iloc[-1,:]
    return cor2cov(corrmat, std)

def rcf(x,cov):
    o = 0.0
    mrc = cov.dot(x)
    for l,i in enumerate(mrc):
        for k,j in enumerate(mrc):
            o += (x[l]*i-x[k]*j)**2
    return o

def vol(x,cov):
    return float(cov.dot(x).dot(x))

def ew(x):
    o = 0.0
    for l in x:
        for k in x:
            o += (l-k)**2
    return o

def optim(cov, typ='rcf', c=5):
    cons = ({'type': 'ineq', 'fun': lambda x: x}, {'type': 'eq', 'fun': lambda x: 1.-np.sum(x)},)
    if(typ=='rcf'):
        arg = (cov,)
        func = rcf
    elif(typ=='vol'):
        arg = (cov,)
        func = vol
    elif(typ=='ew'):
        func = ew
        arg = ()
        cons += ({'type': 'eq', 'fun': lambda x,cov: np.sqrt(cov.dot(x).dot(x)*252)-(c), 'args': (cov,)},)
    elif(typ=="sort"):
        n= len(cov)
        alpha = 0.4424-0.1185*n**(-0.21)
        centroid = norm.ppf((n+1-np.array(range(1,n+1))-alpha)/(n-2*alpha+1))
        func = lambda x,c: -1*x.dot(c)
        arg = (centroid,)
        cons += ({'type': 'eq', 'fun': lambda x,cov: np.sqrt(cov.dot(x).dot(x)*252)-(c), 'args': (cov,)},)

    return minimize(func, x0 = [1./float(len(cov))]*len(cov), args=arg, jac=False, options=({'maxiter': 1e4}), method='SLSQP', constraints=cons)