import pandas as pd

def calculateMutualFundQuantiles(df):
    return df.quantile([0.05, 0.25, 0.5, 0.75, 0.95])