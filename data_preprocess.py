import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def getVNTickDataFromFile(ticker_names, start_day='20110101', end_day='20191231', trading_days=2242, dir="data/VN30/"):
    cols = ["<DTYYYYMMDD>", "<CloseFixed>"]
    file_names = ticker_names + '.csv'
    data = pd.read_csv(dir+file_names, usecols=cols)
    dates = []
    for i in range(len(data)):
        dates.append(datetime.strptime(str(data["<DTYYYYMMDD>"][i]), '%Y%m%d'))
    del data["<DTYYYYMMDD>"]
    data["date"] = dates
    data.rename(columns={"<CloseFixed>": "adj"}, inplace=True)
    mask = (data['date'] >= start_day) & (data['date'] <= end_day)
    return data[mask].reset_index().drop(["index"], axis=1)

def getTickDataFromFile(ticker_names, start_day, end_day, trading_days, dir="data/HUGE_STOCK/", ):
    cols = ["Date", "Adj Close"]
    file_names = ticker_names + '.csv'
    data = pd.read_csv(dir+file_names, usecols=cols)
    dates = []
    for i in range(len(data)):
        dates.append(datetime.strptime(data["Date"][i], '%Y-%m-%d'))
    del data["Date"]
    data["date"] = dates
    data.rename(columns={"Adj Close": "adj"}, inplace=True)
    mask = (data['date'] >= start_day) & (data['date'] <= end_day)
    return data[mask].reset_index().drop(["index"], axis=1)


def getTrainData(tickers,start_day, end_day, trading_days, mode="Huge Stock"):
    train_data = []
    if mode == "Huge Stock":
        data = getTickDataFromFile(tickers[0], start_day, end_day, trading_days)
    else:
        data = getVNTickDataFromFile(tickers[0], start_day, end_day, trading_days)
    data.rename(columns={"adj": tickers[0]}, inplace=True)
    for i in range(1, len(tickers)):
        if mode == "Huge Stock":
            data[tickers[i]] = getTickDataFromFile(tickers[i], start_day, end_day, trading_days)['adj']
        elif mode == "VN":
            data[tickers[i]] = getVNTickDataFromFile(tickers[i], start_day, end_day, trading_days)['adj']
    cols = ["date"] + tickers
    data = data[cols]
    for date in data["date"]:
        train_data.append(data[data["date"] == date])
    return train_data


def getTickers(file):
    ticker_list = open(file, 'r').readlines()
    for i in range(len(ticker_list)):
        ticker_list[i] = ticker_list[i].replace('\n', '')
    return ticker_list

def graphClosePrice(tickers, saveDir, start_day, end_day, trading_days, mode="render"):
    for i in range(len(tickers)):
        ax = plt.gca()
        ax.set_title(tickers[i] + " ADJ CLOSE PRICE")
        data = getVNTickDataFromFile(tickers[i], start_day, end_day, trading_days)
        data.plot(kind='line', x='date', y='adj', ax=ax)
        if mode == "save":
            plt.savefig(saveDir+tickers[i]+".png")
        elif mode == "render":
            plt.show()
        plt.close()

def countTradingDays(ticker_name, start_day, end_day, dir):
    cols = ["Date"]
    file_names = ticker_name + '.csv'
    data = pd.read_csv(dir + file_names, usecols=cols)
    dates = []
    for i in range(len(data)):
        dates.append(datetime.strptime(data["Date"][i], '%Y-%m-%d'))
    del data["Date"]
    data["date"] = dates
    mask = (data['date'] >= start_day) & (data['date'] <= end_day)
    return len(data[mask])

def countVNTradingDays(ticker_name, start_day, end_day, dir):
    cols = ["<DTYYYYMMDD>", "<CloseFixed>"]
    file_names = ticker_name + '.csv'
    data = pd.read_csv(dir + file_names, usecols=cols)
    dates = []
    for i in range(len(data)):
        dates.append(datetime.strptime(str(data["<DTYYYYMMDD>"][i]), '%Y%m%d'))
    del data["<DTYYYYMMDD>"]
    data["date"] = dates
    data.rename(columns={"<CloseFixed>": "adj"}, inplace=True)
    mask = (data['date'] >= start_day) & (data['date'] <= end_day)
    return len(data[mask])