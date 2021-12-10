import gym
import numpy as np

from stable_baselines import DDPG, TD3, SAC
from stable_baselines.td3.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpgMlp
from stable_baselines.ddpg.policies import LnMlpPolicy as ddpgLnMlp
from stable_baselines.sac.policies import MlpPolicy as sacMlp
from stable_baselines.sac.policies import LnMlpPolicy as sacLnMlp
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt
from env.vnStockMarket import VNStockEnv
from env.vnSlidingTestEnv import slidingMegaStockTestEnv as random_env
#from env.vnSlidingTestEnv_withStopLoss import slidingMegaStockTestEnv
from env.vnSlidingTestEnv_withStopLossForEachStock import slidingMegaStockTestEnv
from data_preprocess import countVNTradingDays
import time
from pathlib import Path

start_train_day = "20110101"
end_train_day = "20121231"
train_trading_days = 498

start_days = ["2013-01-01", "2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01"]
end_days = ["2013-12-31", "2014-12-31","2015-12-31","2016-12-31","2017-12-31","2018-12-31","2019-12-31"]

SEED=1
trading_days = []
for i in range(len(start_days)):
    trading_days.append(countVNTradingDays('ree', start_days[i], end_days[i], 'data/VN30/'))

ticker_list = open("data/vn_tickers.txt", 'r').readlines()
for i in range(len(ticker_list)):
    ticker_list[i] = ticker_list[i].replace('\n', '')

def run_TD3(times, money=10000, random_seed=1):
    final_asset = []
    returns = []
    running_time = 0
    for i in range(times):
        env = VNStockEnv(start_train_day, end_train_day, train_trading_days, ticker_list, algoMode="TD3")
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        running_time = time.time()
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
                days = countVNTradingDays("ree", start_train_day, end_days[i],'data/VN30/')
                print(f'train days: {start_train_day} -- {end_days[i]} {days}')
                env = VNStockEnv(start_train_day, end_days[i], days, ticker_list, algoMode="TD3", money=money)
                model = TD3(MlpPolicy, env, action_noise=action_noise, seed=random_seed, n_cpu_tf_sess=1)
                model.learn(total_timesteps=int(1e4))
                env.close()
        running_time = round(time.time() - running_time, 2)
    return final_asset, sum(final_asset), returns, running_time

def run_DDPG(times, money=10000, random_seed=1):
    final_asset = []
    returns = []
    running_time= 0
    for i in range(times):
        env = VNStockEnv(start_train_day, end_train_day, train_trading_days, ticker_list, algoMode="DDPG")
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        running_time = time.time()
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
                days = countVNTradingDays("ree", start_train_day, end_days[i],'data/VN30/')
                print(f'train days: {start_train_day} -- {end_days[i]} {days}')
                env = VNStockEnv(start_train_day, end_days[i], days, ticker_list, algoMode="DDPG", money=money)
                model = DDPG(ddpgMlp, env, action_noise=action_noise, seed=random_seed, n_cpu_tf_sess=1)
                model.learn(total_timesteps=int(1e4))
                env.close()
        running_time = round(time.time() - running_time, 2)
    return final_asset, sum(final_asset), returns, running_time


def run_SAC(times, money=10000, random_seed=1):
    final_asset = []
    returns = []
    running_time = 0
    for i in range(times):
        env = VNStockEnv(start_train_day, end_train_day, train_trading_days, ticker_list, algoMode="SAC")
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        running_time = time.time()
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
                days = countVNTradingDays("ree", start_train_day, end_days[i],'data/VN30/')
                print(f'train days: {start_train_day} -- {end_days[i]} {days}')
                env = VNStockEnv(start_train_day, end_days[i], days, ticker_list, algoMode="SAC", money=money)
                model = SAC(sacMlp, env, action_noise=action_noise, seed=random_seed, n_cpu_tf_sess=1)
                model.learn(total_timesteps=int(1e4))
                env.close()
        running_time = round(time.time() - running_time, 2)
    return final_asset, sum(final_asset), returns, running_time

def run_random(money=10000, random_seed=1):
    np.random.seed(random_seed)
    final_asset = []
    returns = []
    days = countVNTradingDays("ree", '20130101', '20191231','data/VN30/')
    env = random_env('20130101', '20191231', days, ticker_list, algoMode="RANDOM",
                                          money=money)
    env.reset()
    info = {}
    for _ in range(days):
        action = np.random.uniform(-1, 1, size=env.action_space.shape[-1])
        obs, rewards, done, info = env.step(action)
        print(env.render())
    print(info)
    returns.append(info['memory'])
    money = info['money']
    final_asset.append(money)
    env.close()
    return final_asset, sum(final_asset), returns

def run_random_with_stop_loss(times, money=10000, random_seed=1):
    np.random.seed(random_seed)
    final_asset = []
    returns = []
    for i in range(times):
        for i in range(len(start_days)):
            env = slidingMegaStockTestEnv(start_days[i], end_days[i], trading_days[i], ticker_list, money=money)
            info = {}
            for _ in range(trading_days[i]):
                action = np.random.uniform(-1, 1, size=env.action_space.shape[-1])
                obs, rewards, done, info = env.step(action)
                print(env.render())
            returns.append(info['memory'])
            money = info['money']
            final_asset.append(money)
            env.close()
    return final_asset, sum(final_asset), returns,0

import csv
Path("plot\VN\_td3_experiment_4").mkdir(parents=True, exist_ok=True)
Path("plot\VN\_ddpg_experiment_4").mkdir(parents=True, exist_ok=True)
Path("plot\VN\_sac_experiment_4").mkdir(parents=True, exist_ok=True)
Path("plot\VN\_rand_experiment_5").mkdir(parents=True, exist_ok=True)
Path("result\VN\_td3_experiment_4").mkdir(parents=True, exist_ok=True)
Path("result\VN\_ddpg_experiment_4").mkdir(parents=True, exist_ok=True)
Path("result\VN\_sac_experiment_4").mkdir(parents=True, exist_ok=True)
Path("result\VN\_rand_experiment_5").mkdir(parents=True, exist_ok=True)

for SEED in range(0, 311, 10):
    a, result, returns,run_times = run_random_with_stop_loss(1,money=200000, random_seed=SEED)
    returns_rand = []
    for retur in returns:
        returns_rand += retur

    fig, ax = plt.subplots()
    [plt.axvline(x=x_, linewidth=2, color='k', linestyle='--') for x_ in np.array(trading_days).cumsum()]
    #ax.plot(returns_TD3, color="r", label="TD3 Returns")
    ax.plot(returns_rand, color="b", label="RANDOM Returns")
    #ax.plot(returns_DDPG, color="g", label="SAC Returns")
    #ax.plot(returns_random, color="y", label="random Returns")
    ax.legend()
    fig.savefig(f'plot\VN\_rand_experiment_5\{SEED}_{run_times}.png')

    with open(f"result\VN\_rand_experiment_5\{SEED}_{run_times}.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(returns_rand)


# for SEED in range(61, 311, 10):
#     a, result, returns, run_times = run_DDPG(1,money=200000, random_seed=SEED)
#     returns_rand = []
#     for retur in returns:
#         returns_rand += retur
#
#     fig, ax = plt.subplots()
#     [plt.axvline(x=x_, linewidth=2, color='k', linestyle='--') for x_ in np.array(trading_days).cumsum()]
#     #ax.plot(returns_TD3, color="r", label="TD3 Returns")
#     ax.plot(returns_rand, color="b", label="RANDOM Returns")
#     #ax.plot(returns_DDPG, color="g", label="SAC Returns")
#     #ax.plot(returns_random, color="y", label="random Returns")
#     ax.legend()
#     fig.savefig(f'plot\VN\_ddpg_experiment_4\{SEED}_{run_times}.png')
#
#     with open(f"result\VN\_ddpg_experiment_4\{SEED}_{run_times}.csv", 'w', newline='') as myfile:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         wr.writerow(returns_rand)
#
#
# for SEED in range(1, 311, 10):
#     a, result, returns, run_times = run_TD3(1,money=200000, random_seed=SEED)
#     returns_rand = []
#     for retur in returns:
#         returns_rand += retur
#
#     fig, ax = plt.subplots()
#     [plt.axvline(x=x_, linewidth=2, color='k', linestyle='--') for x_ in np.array(trading_days).cumsum()]
#     #ax.plot(returns_TD3, color="r", label="TD3 Returns")
#     ax.plot(returns_rand, color="b", label="RANDOM Returns")
#     #ax.plot(returns_DDPG, color="g", label="SAC Returns")
#     #ax.plot(returns_random, color="y", label="random Returns")
#     ax.legend()
#     fig.savefig(f'plot\VN\_td3_experiment_4\{SEED}_{run_times}.png')
#
#     with open(f"result\VN\_td3_experiment_4\{SEED}_{run_times}.csv", 'w', newline='') as myfile:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         wr.writerow(returns_rand)
#
# for SEED in range(1, 311, 10):
#     a, result, returns, run_times = run_SAC(1,money=200000, random_seed=SEED)
#     returns_rand = []
#     for retur in returns:
#         returns_rand += retur
#
#     fig, ax = plt.subplots()
#     [plt.axvline(x=x_, linewidth=2, color='k', linestyle='--') for x_ in np.array(trading_days).cumsum()]
#     #ax.plot(returns_TD3, color="r", label="TD3 Returns")
#     ax.plot(returns_rand, color="b", label="RANDOM Returns")
#     #ax.plot(returns_DDPG, color="g", label="SAC Returns")
#     #ax.plot(returns_random, color="y", label="random Returns")
#     ax.legend()
#     fig.savefig(f'plot\VN\_sac_experiment_4\{SEED}_{run_times}.png')
#
#     with open(f"result\VN\_sac_experiment_4\{SEED}_{run_times}.csv", 'w', newline='') as myfile:
#         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#         wr.writerow(returns_rand)