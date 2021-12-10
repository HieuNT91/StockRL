import gym
import numpy as np
import time
from stable_baselines import DDPG, TD3, SAC
from stable_baselines.td3.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlp
from stable_baselines.ddpg.policies import LnMlpPolicy as ddpgLnMlp
from stable_baselines.sac.policies import MlpPolicy as sacMlp
from stable_baselines.sac.policies import LnMlpPolicy as sacLnMlp
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt
from env.megaEnv import megaStockEnv
from env.slidingTestEnv import slidingMegaStockTestEnv
from data_preprocess import countTradingDays

SEED = 10

start_train_day = "2011-01-01"
end_train_day = "2012-12-31"
train_trading_days = 502

start_days = ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01"]
end_days = ["2013-12-31", "2014-12-31","2015-12-31","2016-12-31","2017-12-31","2018-12-31","2019-12-31"]

trading_days = []
for i in range(len(start_days)):
    trading_days.append(countTradingDays('BBH', start_days[i], end_days[i], 'data/HUGE_STOCK/'))

ticker_list = open("data/tickers.txt", 'r').readlines()
for i in range(len(ticker_list)):
    ticker_list[i] = ticker_list[i].replace('\n', '')

def run_TD3(times, money=10000, random_seed=1):
    final_asset = []
    returns = []
    running_time = time.time()
    for i in range(times):
        env = megaStockEnv(start_train_day, end_train_day, train_trading_days, ticker_list, algoMode="TD3")
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = TD3(MlpPolicy, env, action_noise=action_noise, seed=random_seed, n_cpu_tf_sess=1)
        model.learn(total_timesteps=int(5e3))
        env.close()

        for i in range(len(start_days)):
            env = slidingMegaStockTestEnv(start_days[i], end_days[i], trading_days[i], ticker_list, algoMode="TD3", money=money)
            obs = env.reset()
            info = {}
            for _ in range(trading_days[i]):
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                print(env.render())
            returns.append(info['memory'])
            money = info['money']
            final_asset.append(money)

            env.close()
            if i != len(start_days)-1:
                days = countTradingDays("BBH", start_train_day, end_days[i],'data/HUGE_STOCK/')
                print(f'train days: {start_train_day} -- {end_days[i]} {days}')
                env = megaStockEnv(start_train_day, end_days[i], days, ticker_list, algoMode="TD3", money=10000)
                model = TD3(MlpPolicy, env, action_noise=action_noise, seed=random_seed, n_cpu_tf_sess=1)
                model.learn(total_timesteps=int(1e4))
                env.close()
        running_time = round(time.time()-running_time, 3)
    return final_asset, sum(final_asset), returns, running_time


def run_SAC(times, money=10000, random_seed=1):
    final_asset = []
    returns = []
    running_time = time.time()
    for i in range(times):
        env = megaStockEnv(start_train_day, end_train_day, train_trading_days, ticker_list, algoMode="SAC")
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = SAC(sacMlp, env, action_noise=action_noise, seed=random_seed, n_cpu_tf_sess=1)
        model.learn(total_timesteps=int(5e3))
        env.close()

        for i in range(len(start_days)):
            env = slidingMegaStockTestEnv(start_days[i], end_days[i], trading_days[i], ticker_list, algoMode="SAC", money=money)
            obs = env.reset()
            info = {}
            for _ in range(trading_days[i]):
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                print(env.render())
            returns.append(info['memory'])
            money = info['money']
            final_asset.append(money)
            env.close()
            if i != len(start_days)-1:
                days = countTradingDays("BBH", start_train_day, end_days[i],'data/HUGE_STOCK/')
                print(f'train days: {start_train_day} -- {end_days[i]} {days}')
                env = megaStockEnv(start_train_day, end_days[i], days, ticker_list, algoMode="DDPG", money=10000)
                model = SAC(sacMlp, env, action_noise=action_noise, seed=random_seed, n_cpu_tf_sess=1)
                model.learn(total_timesteps=int(1e4))
                env.close()
        running_time = round(time.time() - running_time, 3)
    return final_asset, sum(final_asset), returns, running_time

def run_DDPG(times, money=10000, random_seed=1):
    final_asset = []
    returns = []
    running_time = 0
    for i in range(times):
        running_time = time.time()
        env = megaStockEnv(start_train_day, end_train_day, train_trading_days, ticker_list, algoMode="DDPG")
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG(ddpgMlp, env, action_noise=action_noise, seed=random_seed, n_cpu_tf_sess=1)
        model.learn(total_timesteps=int(5e3))
        env.close()

        for i in range(len(start_days)):
            env = slidingMegaStockTestEnv(start_days[i], end_days[i], trading_days[i], ticker_list, algoMode="DDPG", money=money)
            obs = env.reset()
            info = {}
            for _ in range(trading_days[i]):
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                print(env.render())
            returns.append(info['memory'])
            money = info['money']
            final_asset.append(money)
            env.close()
            if i != len(start_days)-1:
                days = countTradingDays("BBH", start_train_day, end_days[i],'data/HUGE_STOCK/')
                print(f'train days: {start_train_day} -- {end_days[i]} {days}')
                env = megaStockEnv(start_train_day, end_days[i], days, ticker_list, algoMode="DDPG", money=10000)
                model = DDPG(ddpgMlp, env, action_noise=action_noise, seed=random_seed, n_cpu_tf_sess=1)
                model.learn(total_timesteps=int(1e4))
                env.close()
        running_time = round(time.time() - running_time, 3)
    return final_asset, sum(final_asset), returns, running_time


def run_random(money, random_seed=1):
    np.random.seed(random_seed)
    final_asset = []
    returns = []
    days = countTradingDays("BBH", '2013-01-01', '2019-12-31','data/HUGE_STOCK/')
    env = slidingMegaStockTestEnv('2013-01-01', '2019-12-31', days, ticker_list, algoMode="RANDOM",
                                          money=money)
    env.reset()
    info = {}
    for _ in range(days):
        action = np.random.randint(-5, 5, size=env.action_space.shape[-1])
        obs, rewards, done, info = env.step(action)
        print(env.render())
    print(info)
    returns.append(info['memory'])
    money = info['money']
    final_asset.append(money)
    env.close()
    return final_asset, sum(final_asset), returns

import csv
for SEED in range(0, 310, 10):
    a, result, returns, running_times = run_TD3(1, random_seed=SEED)
    returns_random = []
    for retur in returns:
        returns_random += retur


    fig, ax = plt.subplots()
    [plt.axvline(x=x_, linewidth=2, color='k', linestyle='--') for x_ in np.array(trading_days).cumsum()]
    ax.plot(returns_random, color="y", label="TD3 Returns")
    ax.legend()
    fig.savefig(f'plot/us_td3_new/{SEED}_{running_times}.png')
    with open(f"result/us_td3_new/{SEED}_{running_times}.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(returns_random)

for SEED in range(0, 310, 10):
    a, result, returns, running_times = run_DDPG(1, random_seed=SEED)
    returns_random = []
    for retur in returns:
        returns_random += retur


    fig, ax = plt.subplots()
    [plt.axvline(x=x_, linewidth=2, color='k', linestyle='--') for x_ in np.array(trading_days).cumsum()]
    #ax.plot(returns_DDPG, color="r", label="DDPG Returns")
    #ax.plot(returns_DDPG, color="b", label="DDPG Returns")
    #ax.plot(returns_random, color="g", label="SAC Returns")
    ax.plot(returns_random, color="y", label="DDPG Returns")
    ax.legend()
    fig.savefig(f'plot/us_ddpg_new/{SEED}_{running_times}.png')
    with open(f"result/us_ddpg_new/{SEED}_{running_times}.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(returns_random)

for SEED in range(0, 310, 10):
    a, result, returns, running_times = run_SAC(1, random_seed=SEED)
    returns_random = []
    for retur in returns:
        returns_random += retur


    fig, ax = plt.subplots()
    [plt.axvline(x=x_, linewidth=2, color='k', linestyle='--') for x_ in np.array(trading_days).cumsum()]
    ax.plot(returns_random, color="y", label="SAC Returns")
    ax.legend()
    fig.savefig(f'plot/us_sac_new/{SEED}_{running_times}.png')
    with open(f"result/us_sac_new/{SEED}_{running_times}.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(returns_random)