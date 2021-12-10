import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
from datetime import datetime
import matplotlib.pyplot as plt
from data_preprocess import getTrainData

#trading_days = 2264
#start_day = "2011-01-01"
#end_day = "2019-12-31"

#ticker_list = open("data/tickers.txt", 'r').readlines()
#for i in range(len(ticker_list)):
#    ticker_list[i] = ticker_list[i].replace('\n', '')


class megaStockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, start_day, end_day, trading_days, ticker_list, day=0, money=10000, algoMode=None):
        self.money = money
        self.day = day
        self.start_day = start_day
        self.end_day = end_day
        self.trading_days = trading_days
        self.ticker_list = ticker_list

        self.train_daily_data = getTrainData(ticker_list, self.start_day, self.end_day, self.trading_days)
        self.data = self.train_daily_data[self.day]

        self.algoMode = algoMode
        self.iteration = 0

        self.action_space = spaces.Box(low=-5, high=5, shape=(len(ticker_list),), dtype=np.int8)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2*len(ticker_list)+1,))

        self.state = [self.money] + self.data[ticker_list].iloc[0].tolist() + [0 for _ in range(len(ticker_list))]
        self.reward = 0
        self.asset_memory = [self.money]
        self.terminal = False
        self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        if self.state[index + len(self.ticker_list) + 1] > 0:
            self.state[0] += self.state[index + 1] * min(abs(action), self.state[index + len(self.ticker_list) + 1])
            self.state[index + len(self.ticker_list) + 1] -= min(abs(action), self.state[index + len(self.ticker_list) + 1])
        else:
            pass

    def _buy_stock(self, index, action):
        available_amount = self.state[0] // self.state[index + 1]
        # print('available_amount:{}'.format(available_amount))
        self.state[0] -= self.state[index + 1] * min(available_amount, action)
        self.state[index + len(self.ticker_list) + 1] += min(available_amount, action)

    def step(self, actions):
        #print(f'day: {self.day}')
        self.terminal = self.day >= self.trading_days - 1
        #print(actions)

        if self.terminal:
            #plt.plot(self.asset_memory, 'y')
            #if self.algoMode == "TD3":
            #    plt.savefig("C:\BackupD\JunHill\Projects\stock_first_try\plot\_td3\iteration{}.png".format(self.iteration))
            #elif self.algoMode == "DDPG":
            #    plt.savefig('C:\BackupD\JunHill\Projects\stock_first_try\plot\ddpg\iteration{}.png'.format(self.iteration))
            #elif self.algoMode == "SAC":
            #    plt.savefig('C:\BackupD\JunHill\Projects\stock_first_try\plot\sac\iteration{}.png'.format(self.iteration))
            #else:
            #    raise Exception("Invalid algorithm!")
            #plt.close()
            print(f"Iteration {self.iteration}: ")
            print("total_reward:{}".format(
                self.state[0] + sum(np.array(self.state[1:len(self.ticker_list) + 1]) * np.array(self.state[len(self.ticker_list) + 1:])) - self.money))

            print('total asset: {}'.format(self.state[0] + sum(np.array(self.state[1:len(self.ticker_list) + 1]) * np.array(self.state[len(self.ticker_list) + 1:]))))
            return self.state, self.reward, self.terminal, {}
        else:
            # print(np.array(self.state[1:29]))

            begin_total_asset = self.state[0] + sum(np.array(self.state[1:len(self.ticker_list) + 1]) * np.array(self.state[len(self.ticker_list) + 1:]))
            # print("begin_total_asset:{}".format(begin_total_asset))
            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.train_daily_data[self.day]

            # print("stock_shares:{}".format(self.state[29:]))
            self.state = [self.state[0]] + self.data[self.ticker_list].iloc[0].tolist() + list(self.state[len(self.ticker_list) + 1:])
            end_total_asset = self.state[0] + sum(np.array(self.state[1:len(self.ticker_list) + 1]) * np.array(self.state[len(self.ticker_list) + 1:]))
            #print("end_total_asset:{}".format(end_total_asset))

            self.reward = end_total_asset - begin_total_asset
            #print("step_reward:{}".format(self.reward))

            self.asset_memory.append(end_total_asset)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.money]
        self.day = 0
        self.data = self.train_daily_data[self.day]
        self.state = [self.money] + self.data[self.ticker_list].iloc[0].tolist() + [0 for _ in range(len(self.ticker_list))]

        self.iteration += 1
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
