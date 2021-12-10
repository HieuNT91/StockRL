import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces

import matplotlib.pyplot as plt

dji = pd.read_csv("C:\BackupD\JunHill\Projects\stock_first_try\plot\^DJI.csv")
test_dji = dji[dji['Date'] > '2016-01-01']
dji_price = test_dji['Adj Close']
dji_date = test_dji['Date']
daily_return = dji_price.pct_change(1)
daily_return = daily_return[1:]
daily_return.reset_index()
initial_amount = 10000

total_amount = initial_amount
account_growth = list()
account_growth.append(initial_amount)
for i in range(len(daily_return)):
    total_amount = total_amount * daily_return.iloc[i] + total_amount
    account_growth.append(total_amount)

data_1 = pd.read_csv('C:\BackupD\JunHill\Projects\stock_first_try\plot\dow_jones_30_daily_price.csv')

equal_4711_list = list(data_1.tic.value_counts() == 4711)
names = data_1.tic.value_counts().index

# select_stocks_list = ['NKE','KO']
select_stocks_list = list(names[equal_4711_list]) + ['NKE', 'KO']

data_2 = data_1[data_1.tic.isin(select_stocks_list)][~data_1.datadate.isin(['20010912', '20010913'])]

data_3 = data_2[['iid', 'datadate', 'tic', 'prccd', 'ajexdi']]

data_3['adjcp'] = data_3['prccd'] / data_3['ajexdi']

# train_data = data_3[(data_3.datadate > 20090000) & (data_3.datadate < 20160000)]
test_data = data_3[data_3.datadate > 20160000]

# train_daily_data = []

# for date in np.unique(train_data.datadate):
#     train_daily_data.append(train_data[train_data.datadate == date])

# print(len(train_daily_data))
test_daily_data = []

for date in np.unique(test_data.datadate):
    test_daily_data.append(test_data[test_data.datadate == date])

# whole_data = train_daily_data+test_daily_data

iteration = 0


class StockTestEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, day=0, money=10000, algoMode=None):
        self.algoMode = algoMode
        self.money = money
        self.day = day
        self.action_space = spaces.Box(low=-5, high=5, shape=(28,), dtype=np.int8)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(57,))
        self.data = test_daily_data[self.day]
        self.terminal = False
        self.state = [self.money] + self.data.adjcp.values.tolist() + [0 for i in range(28)]
        self.reward = 0
        self.asset_memory = [self.money]
        self.reset()
        self._seed()
        print("------Enter playing mode------")
    def _sell_stock(self, index, action):
        if self.state[index + 29] > 0:
            self.state[0] += self.state[index + 1] * min(abs(action), self.state[index + 29])
            self.state[index + 29] -= min(abs(action), self.state[index + 29])
        else:
            pass

    def _buy_stock(self, index, action):
        available_amount = self.state[0] // self.state[index + 1]
        # print('available_amount:{}'.format(available_amount))
        self.state[0] -= self.state[index + 1] * min(available_amount, action)
        self.state[index + 29] += min(available_amount, action)

    def step(self, actions):
        actions = np.array([round(i) for i in actions])
        print(f'day: {self.day}')
        self.terminal = self.day >= 685
        print(actions)

        if self.terminal:
            plt.plot(self.asset_memory, 'g')
            plt.plot(account_growth)
            if self.algoMode == "TD3":
                plt.savefig('D:\JunHill\Projects\stock_first_try\plot\_td3.png')
            elif self.algoMode == "DDPG":
                plt.savefig('D:\JunHill\Projects\stock_first_try\plot\ddpg.png')
            elif self.algoMode == "SAC":
                plt.savefig('D:\JunHill\Projects\stock_first_try\plot\sac.png')
            else:
                raise Exception("Invalid Algorithm!")
            plt.close()
            x = self.state[0] + sum(np.array(self.state[1:29]) * np.array(self.state[29:])) - self.money
            print("total_reward:{}".format(self.state[0] + sum(np.array(self.state[1:29]) * np.array(self.state[29:])) - self.money))

            return self.state, self.reward, self.terminal, {x}

        else:
            # print(np.array(self.state[1:29]))

            begin_total_asset = self.state[0] + sum(np.array(self.state[1:29]) * np.array(self.state[29:]))
            #print("begin_total_asset:{}".format(begin_total_asset))
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
            self.data = test_daily_data[self.day]

            self.state = [self.state[0]] + self.data.adjcp.values.tolist() + list(self.state[29:])
            end_total_asset = self.state[0] + sum(np.array(self.state[1:29]) * np.array(self.state[29:]))
            print("end_total_asset:{}".format(end_total_asset))

            self.reward = end_total_asset - begin_total_asset
            print("step_reward:{}".format(self.reward))

            self.asset_memory.append(end_total_asset)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.money]
        self.day = 0
        self.data = test_daily_data[self.day]
        self.state = [self.money] + self.data.adjcp.values.tolist() + [0 for i in range(28)]

        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]