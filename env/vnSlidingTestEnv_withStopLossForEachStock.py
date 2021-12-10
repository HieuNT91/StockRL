import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib.pyplot as plt
from data_preprocess import getTrainData

MAX_STOCK_PER_TRADE = 100

class slidingMegaStockTestEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, start_day, end_day, trading_days, ticker_list, day=0, money=200000, algoMode=None):
        self.money = money
        self.init_budget = money

        self.day = day
        self.start_day = start_day
        self.end_day = end_day
        self.trading_days = trading_days
        self.ticker_list = ticker_list
        self.test_daily_data = getTrainData(ticker_list, self.start_day, self.end_day, self.trading_days, mode="VN")
        self.data = self.test_daily_data[self.day]
        self.algoMode = algoMode
        self.iteration = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ticker_list),))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(2 * len(ticker_list) + 1,))
        self.price_last_bought = np.zeros((1,len(ticker_list)))
        self.stop_loss = np.zeros((1, len(ticker_list)))
        self.state = [self.money] + self.data[ticker_list].iloc[0].tolist() + [0 for _ in range(len(ticker_list))]
        self.reward = 0
        self.asset_memory = [self.money]
        self.terminal = False
        self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        if self.state[index + len(self.ticker_list) + 1] > 0:
            self.state[0] += self.state[index + 1] * min(abs(action), self.state[index + len(self.ticker_list) + 1])
            self.state[index + len(self.ticker_list) + 1] -= min(abs(action),
                                                                 self.state[index + len(self.ticker_list) + 1])
        else:
            pass

    def _buy_stock(self, index, action):
        available_amount = self.state[0] // self.state[index + 1]
        self.state[0] -= self.state[index + 1] * min(available_amount, action)
        self.state[index + len(self.ticker_list) + 1] += min(available_amount, action)
        self.price_last_bought[0][index] = max(self.state[index+1], self.price_last_bought[0][index])

    def step(self, actions):
        print(f'day: {self.day}')
        self.terminal = self.day >= self.trading_days - 1
        if self.terminal:
            x = self.state[0] + sum(np.array(self.state[1:len(self.ticker_list) + 1]) * np.array(self.state[len(self.ticker_list) + 1:]))
            print("total_reward:{}".format(self.state[0] + sum(np.array(self.state[1:len(self.ticker_list) + 1]) * np.array(self.state[len(self.ticker_list) + 1:])) - self.money))

            return self.state, self.reward, self.terminal, {'money': x, 'memory': self.asset_memory}

        else:
            for i, x in enumerate(self.stop_loss[0]):
                if x == 1:
                    actions[i] = -(self.state[i+1+len(self.ticker_list)] / MAX_STOCK_PER_TRADE)


            actions = actions * MAX_STOCK_PER_TRADE
            #print(f'action: {actions}')
            #print(f'stop_loss: {self.stop_loss}')
            #print(f'holdings: {self.state[len(self.ticker_list)+1:]}')
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1:len(self.ticker_list) + 1]) * np.array(self.state[len(self.ticker_list) + 1:]))
            if begin_total_asset > self.init_budget:
                self.init_budget = begin_total_asset

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.test_daily_data[self.day]
            self.state = [self.state[0]] + self.data[self.ticker_list].iloc[0].tolist() + list(
                self.state[len(self.ticker_list) + 1:])
            end_total_asset = self.state[0] + sum(np.array(self.state[1:len(self.ticker_list) + 1]) * np.array(
                self.state[len(self.ticker_list) + 1:]))
            self.reward = end_total_asset - begin_total_asset
            self.asset_memory.append(end_total_asset)

            for i in range(len(self.stop_loss[0])):
                if self.state[i+1] < self.price_last_bought[0][i] * 0.9:
                    self.stop_loss[0][i] = 1

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.price_last_bought = np.zeros((1, len(self.ticker_list)))
        self.stop_loss = np.zeros((1, len(self.ticker_list)))
        self.asset_memory = [self.money]
        self.day = 0
        self.data = self.test_daily_data[self.day]
        self.state = [self.money] + self.data[self.ticker_list].iloc[0].tolist() + [0 for _ in range(len(self.ticker_list))]

        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]