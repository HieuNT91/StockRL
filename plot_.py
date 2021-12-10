import os
import numpy as np
import csv
from matplotlib import pyplot as plt
path_ddpg = "result/VN/_ddpg_experiment_2"
returns_ddpg = []
for root, dir, files in os.walk(path_ddpg, topdown=False):
    for name in files:
        with open(os.path.join(root, name), newline="") as file:
            reader = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
            returns_ddpg.append(next(reader))

returns_ddpg = np.array(returns_ddpg).astype(np.float)
avg_ddpg = np.mean(returns_ddpg, axis=0)
min = returns_ddpg.min(axis=0)
max = returns_ddpg.max(axis=0)
x = np.arange(0, len(avg_ddpg))
fig, ax = plt.subplots()
#ax.plot(avg_ddpg, color="#CF424E", label="DDPG")
ax.plot(avg_ddpg, color="black", label="DDPG")
#ax.fill_between(x,min, max, alpha=0.2, color="r")
print(f'ddpg: {avg_ddpg[-1]}')

path_td3 = "result/VN/_td3_experiment_2"
returns_td3 = []
for root, dir, files in os.walk(path_td3, topdown=False):
    for name in files:
        with open(os.path.join(root, name), newline="") as file:
            reader = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
            returns_td3.append(next(reader))

returns_td3 = np.array(returns_td3).astype(np.float)
avg_td3 = np.mean(returns_td3, axis=0)
min = returns_td3.min(axis=0)
max = returns_td3.max(axis=0)
x = np.arange(0, len(avg_td3))
ax.plot(avg_td3, color="green", label="TD3")
#ax.plot(avg_td3, color="#553E3D", label="TD3")
#ax.fill_between(x,min, max, color="g" , alpha=0.2)
print(f'td3: {avg_td3[-1]}')

path_sac = "result/VN/_sac_experiment_1"
returns_sac = []
for root, dir, files in os.walk(path_sac, topdown=False):
    for name in files:
        with open(os.path.join(root, name), newline="") as file:
            reader = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
            returns_sac.append(next(reader))

returns_sac = np.array(returns_sac).astype(np.float)
avg_sac = np.mean(returns_sac, axis=0)
min = returns_sac.min(axis=0)
max = returns_sac.max(axis=0)
x = np.arange(0, len(avg_sac))
ax.plot(avg_sac, color="red", label="SAC")
#ax.plot(avg_sac, color="#60B3A3", label="SAC")
#ax.fill_between(x,min, max, color="y", alpha=0.2)
print(f'sac: {avg_sac[-1]}')
path_sac = "result/VN/_rand_experiment_4"
returns_rand = []
for root, dir, files in os.walk(path_sac, topdown=False):
    for name in files:
        with open(os.path.join(root, name), newline="") as file:
            reader = csv.reader(file,quoting=csv.QUOTE_NONNUMERIC)
            returns_rand.append(next(reader))

returns_rand = np.array(returns_rand).astype(np.float)
avg_rand = np.mean(returns_rand, axis=0)
min = returns_rand.min(axis=0)
max = returns_rand.max(axis=0)
x = np.arange(0, len(avg_rand))
ax.plot(avg_rand, color="blue", linestyle=':', label="Random")
#ax.fill_between(x,min, max, color="y", alpha=0.2)
fig.legend()
plt.show()
print(f'rand: {avg_rand[-1]}')
from scipy import stats




from data_preprocess import countVNTradingDays, countTradingDays
start_days = ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01"]
end_days = ["2013-12-31", "2014-12-31","2015-12-31","2016-12-31","2017-12-31","2018-12-31","2019-12-31"]

trading_days = []
for i in range(len(start_days)):
    trading_days.append(countVNTradingDays('ree', start_days[i], end_days[i], 'data/VN30/'))

def get_end_of_year(returns, trading_days):
    asset = []
    tmp = 0
    for day in trading_days:
        day = day + tmp - 1
        #if day >= len(returns[0]):
        #    day -= 1
        asset.append([x[day] for x in returns])
        tmp = day
    return asset

def get_mean_end(returns, trading_days):
    asset = []
    tmp = 0
    for day in trading_days:
        day = day + tmp
        if day >= len(returns):
            day -= 1
        asset.append(returns[day])
        tmp = day
    return asset

import pandas as pd
def get_anul(asset):
    return (pd.DataFrame(asset).pct_change(1).sum())/len(asset)

def get_anul_by_mean(avg):
    return (pd.Series(avg).pct_change(1).sum())/len(avg)

print("------ MEAN -------")
print(get_anul_by_mean(get_mean_end(avg_ddpg, trading_days)))
print(round(get_anul_by_mean(get_mean_end(avg_td3, trading_days)),3))
print(get_anul_by_mean(get_mean_end(avg_sac, trading_days)))
print(get_anul_by_mean(get_mean_end(avg_rand, trading_days)))
print("------ 30 runs -------")

print(sum(get_anul(get_end_of_year(returns_ddpg, trading_days)))/len(get_anul(get_end_of_year(returns_ddpg, trading_days))))
print(sum(get_anul(get_end_of_year(returns_td3, trading_days)))/len(get_anul(get_end_of_year(returns_td3, trading_days))))
print(sum(get_anul(get_end_of_year(returns_sac, trading_days)))/len(get_anul(get_end_of_year(returns_sac, trading_days))))
print(sum(get_anul(get_end_of_year(returns_rand, trading_days)))/len(get_anul(get_end_of_year(returns_rand, trading_days))))

print(stats.ttest_ind(get_anul(get_end_of_year(returns_rand, trading_days)), get_anul(get_end_of_year(returns_ddpg, trading_days))))