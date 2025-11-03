url = lambda ticker: f'https://api.kraken.com/0/public/OHLC?pair={ticker}&interval=1'

import numpy as np
import ctypes
import matplotlib.pyplot as plt
import requests
import json

data = requests.get(url('BTCUSD')).json()
prices = np.array(data['result']['XXBTZUSD'], dtype=float)[:, 4]

def AutoCorr(xdata):
    def LoadData(ticker, vin=30):
        ror = xdata[1:]/xdata[:-1] - 1.0
        volt = np.array([np.std(ror[i-vin:i]) for i in range(vin, len(ror))])

        prices = xdata[vin+1:]
        ror = ror[vin:]

        return prices, ror, volt

    def Signal(vol):
        v0, v1 = vol[:-5], vol[5:]
        E = np.cov(v0, v1)
        beta = E[0, 1]/E[0, 0]
        alpha = np.mean(v1) - beta*np.mean(v0)
        spread = v1 - (alpha + beta*v0)
        return (spread[-1] - np.mean(spread))/np.std(spread)

    def TradingStrategy(price, returns, volatility, z1=-2, z2=2, vin=50):
        side = 'neutral'
        price_in = 0
        price_out = 0
        pnl = 1.0
        tx = 0.004

        for t in range(vin, len(price)):
            hold_vol = volatility[t-vin:t]
            tstat = Signal(hold_vol)
            
            if tstat > z2 and side == 'long':
                print("Exit Long Position")
                side = 'neutral'
                price_out = price[t]

                pnl *= (price_out*(1-tx))/(price_in*(1+tx))

            if tstat < z1 and side == 'neutral':
                print("Enter Long Position")
                side = 'long'
                price_in = price[t]

        return pnl - 1.0

    p, r, v = LoadData('BTCUSD')
    n = 20

    az1, az2 = np.meshgrid(np.linspace(-5, -1, n), np.linspace(1, 5, n))
    pnl = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            print(i, j)
            pnl[i, j] = TradingStrategy(p, r, v, z1=az1[i, j], z2=az2[i, j])
    
    return az1, az2, pnl

def MeanReversion(prices):
    def Stochastic(price):
        quant = ctypes.CDLL("./gbm.so")
        quant.geometric.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
        quant.geometric.restype = ctypes.c_double

        window = 30

        for i in range(window, len(price)):
            hold = price[i-window:i]
            ror = hold[1:]/hold[:-1] - 1.0
            drift = np.mean(ror)
            volt = np.std(ror)**2
            s = price[i]

            gbm_s = quant.geometric(s, drift, volt)
            tstat = (gbm_s - np.mean(hold))/(np.std(hold))

            yield tstat, s

    N = 20
    x, y = np.meshgrid(np.linspace(-5, -1, N), np.linspace(1, 5, N))
    pnl = np.zeros((N, N))

    tx_fee = 0.004

    for i in range(N):
        for j in range(N):

            side = 'neutral'
            entry_price = 0
            exit_price = 0

            profit = 1.0

            for tstat, price in Stochastic(prices):

                if tstat > y[i, j] and side == 'long':
                    print("Exit Long Position: ", i, j)
                    side = 'neutral'
                    exit_price = price

                    profit *= (exit_price*(1-tx_fee)) / (entry_price*(1 + tx_fee))
                
                if tstat < x[i, j] and side == 'neutral':
                    print("Enter Long Position: ", i, j)
                    side = 'long'
                    entry_price = price

            pnl[i, j] = profit - 1.0
    
    return x, y, pnl


meanx, meany, meanz = MeanReversion(prices)
autox, autoy, autoz = AutoCorr(prices)

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(121)
ay = fig.add_subplot(122)

ax.set_title('Mean Reversion Backtest w/ GBM')
ay.set_title('Volatility Autocorrelation Backtest')

ax.set_xlabel('Entry Test Statistic')
ay.set_xlabel('Entry Test Statistic')

ax.set_ylabel('Exit Test Statistic')
ay.set_ylabel('Exit Test Statistic')

p1 = ax.contourf(meanx, meany, meanz, cmap='jet_r')
p2 = ay.contourf(autox, autoy, autoz, cmap='jet_r')

fig.colorbar(p1, ax=ax)
fig.colorbar(p2, ax=ay)

plt.show()
