url = lambda ticker: f'https://api.kraken.com/0/public/OHLC?pair={ticker}&interval=1'

import numpy as np
import ctypes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
import json

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

data = requests.get(url('BTCUSD')).json()
prices = np.array(data['result']['XXBTZUSD'], dtype=float)[:, 4]

N = 20
x, y = np.meshgrid(np.linspace(-5, -1, N), np.linspace(1, 5, N))
pnl = np.zeros((N, N))

tx_fee = 0.004

for i in range(N):
    for j in range(N):

        side = 'neutral'
        entry_price = 0
        exit_price = 0

        for tstat, price in Stochastic(prices):

            if tstat > y[i, j] and side == 'long':
                print("Exit Long Position: ", i, j)
                side = 'neutral'
                exit_price = price

                pnl[i, j] = (exit_price*(1-tx_fee)) / (entry_price*(1 + tx_fee)) - 1.0
            
            if tstat < x[i, j] and side == 'neutral':
                print("Enter Long Position: ", i, j)
                side = 'long'
                entry_price = price


fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(121)
ay = fig.add_subplot(122, projection='3d')

ax.set_title('GBM Mean Reversion')
ax.set_xlabel('Entry TStat')
ax.set_ylabel('Exit TStat')
myplot = ax.contourf(x, y, pnl, cmap='jet_r')
ay.plot_surface(x, y, pnl, cmap='hsv')

fig.colorbar(myplot, ax=ax)
plt.show()

