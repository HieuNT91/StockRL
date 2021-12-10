import matplotlib.pyplot as plt
from data_preprocess import getVNTickDataFromFile, getTickers

def graphClosePrice(tickers, saveDir, start_day, end_day, trading_days, mode="save"):
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

ticks = getTickers("data/vn_tickers.txt")
graphClosePrice(ticks, 'plot/vn30_graph/', '20110101', '20191231')