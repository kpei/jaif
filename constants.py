from rebalance_funcs import *

INIT_CAP = 1000.

RESAMPLE_PERIOD = '3M'

ROLLING_WINDOW = 6*30

COND_QUANTILE_ANCHOR = 3271

CRISES = {
    "subprime":{
        "sDate": dt.datetime(2008,9,1),
        "eDate": dt.datetime(2009,2,1)
    },
    "euro":{
        "sDate": dt.datetime(2010,1,1),
        "eDate": dt.datetime(2010,12,1)
    },
    "aug": {
        "sDate": dt.datetime(2011,7,1),
        "eDate": dt.datetime(2012,2,1)
    },
    "2015":{
        "sDate": dt.datetime(2015,8,15),
        "eDate": dt.datetime(2015,12,1)
    },
    "brexit":{
        "sDate": dt.datetime(2016,6,1),
        "eDate": dt.datetime(2016,7,1)
    },
}

INST = {
	'td': {
		'name': 'TD',
		'color':'#3c763d'
	},
	'scotia': {
		'name': 'Scotiabank',
		'color':'#a94442'
	},
    'cibc': {
        'name': 'CIBC',
        'color':'#a94442'
    },
	'rbc': {
		'name': 'RBC',
		'color':'#072C67'
	},
}

PREF = {
    'td': {
        'universe': ['F0CAN05MYF','F0CAN05NJO','F0CAN05MYB','0P000071W8','0P000071HA','0P000070GR','F0CAN05MF4'],
        'cluster': [['F0CAN05MYF'],['0P000071W8','F0CAN05NJO'],['F0CAN05MYB'],['F0CAN05MF4'],['0P000070GR'],['0P000071HA'],],
		'params': { 'bond': 'F0CAN05MYF'},
		'benchmark':  '0P000070LK', 
		'anchor': '0P000071W8' 
    },
    'scotia':{
        'universe': ['F0CAN05MS0','F0CAN05M16','F0CAN05M17','F0CAN05MRZ','F0CAN05LYE', 'F0CAN05MZ0'],
        'cluster': [['F0CAN05MS0', 'F0CAN05MZ0'],['F0CAN05M16','F0CAN05M17'],['F0CAN05MRZ'],['F0CAN05LYE'],],
		'params': { 'bond': 'F0CAN05MZ0',},
		'benchmark':  'F0CAN05NSU', 
		'anchor': 'F0CAN05M17' 
    },
    'rbc': {
        'universe': ['F0CAN05NCN','F0CAN05MVZ','F0CAN05P92','F0CAN05LYG','F0CAN05OYJ','F0CAN05NGC'],
        'cluster': [['F0CAN05LYG','F0CAN05P92'],['F0CAN05MVZ','F0CAN05NCN'],['F0CAN05OYJ'],['F0CAN05NGC'],],
		'params': { 'bond': 'F0CAN05NCN',},
		'benchmark':  'F0CAN05LUC', 
		'anchor': 'F0CAN05LYG' 
    },
    'cibc': {
        'universe': ['0P0000704P','0P000070TE','0P000073I7','0P0000704N','0P0000707T','0P000071B6','0P000071GP','0P0000704T','0P000070TG'],
        'cluster': [['0P0000704P','0P0000704N'],['0P0000707T','0P000071GP','0P000071B6'],['0P000070TE'],['0P000073I7'],['0P0000704T'],['0P000070TG']],
        'params': { 'bond': '0P000070TE',},
        'benchmark':  'F0CAN05OEZ', 
        'anchor': '0P0000704P' 
    },
}

FOLIOS = {
    'ew': {
        'name': 'equal weight portfolio',
        'desc': 'the naive approach to a global allocation',
        'rebal_func': reb_ew,
        'rebal_params': {},
    },
    '8vt': {
        'name': '8% risk-target portfolio',
        'desc': 'an annual 8% risk-targeted allocation with focus on equal weights',
        'rebal_func': reb_ewvt,
        'rebal_params': {'lb': 60, 'c': 8.},
    },
    'rp': {
        'name': 'structured risk parity portfolio',
        'desc': 'prioritizing safety first',
        'rebal_func': reb_rcf,
        'rebal_params': {'lb': 60},
    },
    'mom': {
        'name': 'momentum portfolio',
        'desc': 'a 8% risk-targeted portfolio selected toward recent winners',
        'rebal_func': reb_mom,
        'rebal_params': {'lb': 60, 'c': 10., 'mom_lb': 180},
    },
    'tact': {
        'name': 'tactical portfolio',
        'desc': 'an 8% risk-targeted portfolio with technical signals overlay',
        'rebal_func': reb_tact,
        'rebal_params': {'lb': 60, 'tact_lb': 200, 'c': 8.},
    },
    'momtact': {
        'name': 'tactical momentum portfolio',
        'desc': 'a combination of both momentum allocation and tactical overlay',
        'rebal_func': reb_momtact,
        'rebal_params': {'lb': 60, 'mom_lb': 180, 'tact_lb': 200, 'c': 8.},
    },
    'benchmark': {
        'name': 'the benchmark',
        'desc': 'buy and hold one security',
        'rebal_func': reb_benchmark,
        'rebal_params': {'lb': 60, 'mom_lb': 180, 'tact_lb': 200, 'c': 8.},
    },
}
