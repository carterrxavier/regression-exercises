import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def get_risiduals(df ,act, pred):
    df['risiduals'] = act - pred
    df['baseline_risiduals'] = act - act.mean()
    return df

def plot_residuals(act, pred, res, baseline):
    plt.title('Residuals')
    res.hist()
    plt.show()
    plt.title('Baseline Residuals')
    baseline.hist()
    
    fig , ax = plt.subplots(figsize=(10,5))
    ax.scatter(act, pred)
    ax.set(xlabel='actual', ylabel='prediction')
    ax.plot(act, act,  ls=":", color='black')
    
    
    fig , ax = plt.subplots(figsize=(10,5))
    ax.scatter(act, res)
    ax.set(xlabel='actual', ylabel='residual')
    ax.hlines(0, *ax.get_xlim(), ls=":",color='black')
    
def regression_errors(y, yhat):
    sse = ((y-yhat) ** 2).sum()
    mse = sse / y.shape[0]
    rmse = math.sqrt(mse)
    ess = ((yhat - y.mean())**2).sum()
    tss = ((y - y.mean())**2).sum()
    r_2 = ess/tss
    
    return sse, mse, rmse, ess, tss, r_2

def baseline_mean_errors(y):
    sse_baseline = ((y-y.mean()) ** 2).sum()
    mse_baseline = sse_baseline / y.shape[0]
    rmse_baseline = math.sqrt(mse_baseline)
    
    return sse_baseline, mse_baseline, rmse_baseline

def better_than_baseline(y, yhat):
    return regression_errors(y,yhat)[2] < baseline_mean_errors(y)[2]
    

    



