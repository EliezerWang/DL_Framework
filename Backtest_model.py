import pandas as pd
from data_gen.var_lib import *
from tqdm import tqdm
import numpy as np


class Back_test(object):

    # def __init__(self, pred_data):
    #     self.pred_data = pred_data
    #     self.temp_data = pd.DataFrame()
    #     self.temp_weight = 1
    #     self.stock_set = pd.DataFrame()
    #     self.change = []
    #     self.profit = []
    #     self.date = []
    #     self.adj_profit = []
    #     self.sharpe_ratio = 0
    #     self.year_profit = 0
    #
    # def buy_set(self):
    #     def make_weight(data):
    #         data = data[~data.high == data.low]
    #         data = data.pred
    #         data[data < 0] = -0.001
    #         data[(data >= 0) & (data < 0.0012)] = 0
    #         data = data / data.sum()
    #         data[data > single_max] = single_max  # 确保最大单票持仓为10%
    #         return data
    #
    #     self.pred_data['weight'] = self.pred_data.groupby(by='tradeDate').apply(make_weight)
    #     self.temp_data = list(self.pred_data.groupby(by='tradeDate'))[0][1]
    #     self.stock_set = self.temp_data[self.temp_data.weight > 0][['weight', 'code']]
    #     self.stock_set.rename(columns={'weight': 'weight_last'}, inplace=True)

    def __init__(self, pred_data):
        self.pred_data = pred_data.set_index('tradeDate')
        self.temp_data = pd.DataFrame()
        self.temp_weight = 1
        self.stock_set = pd.DataFrame()
        self.change = []
        self.profit = []
        self.date = []
        self.adj_profit = []
        self.sharpe_ratio = 0
        self.year_profit = 0
        self.temp = pd.DataFrame

    def buy_set(self):
        def make_weight(data):
            data[data < 0] = -0.001
            data[(data >= 0) & (data < 0.0012)] = 0
            data = data / data.sum()
            data[data > single_max] = single_max  # 确保最大单票持仓为10%
            # data = pd.DataFrame(data)
            # data.reset_index(inplace=True)
            # data.drop(['tradeDate'], axis=1, inplace=True)
            # data = data.set_index(['code'])
            return data

        temp = pd.DataFrame(self.pred_data.pred.groupby(level='tradeDate').apply(make_weight))
        temp.rename(columns={'pred': 'weight'}, inplace=True)
        self.pred_data = pd.concat([temp.reset_index(drop=True), self.pred_data.reset_index()], axis=1)
        self.pred_data = self.pred_data.set_index('tradeDate')
        self.temp_data = list(self.pred_data.groupby(level='tradeDate'))[0][1]
        self.stock_set = self.temp_data[self.temp_data.weight > 0][['weight', 'code']]
        self.stock_set.rename(columns={'weight': 'weight_last'}, inplace=True)

    def position_set(self):
        for all_data in tqdm(list(self.pred_data.groupby(by='tradeDate'))):
            date = all_data[0]
            data = all_data[1]
            data = pd.merge(data, self.stock_set, how='left', on='code')
            data.weight_last.fillna(0, inplace=True)
            data_1 = data[(data.weight <= 0) & (data.weight < data.weight_last)]
            data_1 = data_1[data_1.weight_last > 0]

            weight = 1 - (data_1.weight_last.sum()-data_1.weight.sum())
            data.weight = data.weight * weight
            data.weight[(data.weight <= 0) & (data.weight < data.weight_last)] = data.weight_last[
                (data.weight <= 0) & (data.weight < data.weight_last)]

            change = abs(data.weight - data.weight_last).sum()

            # data.test[data.y_close == data.low] = data.test[data.close == data.low] - 0.001   等有了完整数据再放上来
            profit = (data.weight * data.test).sum()/data.weight.sum()
            adj_profit = profit * (1 - change * commission_rate)

            self.change.append(change)
            self.profit.append(profit)
            self.adj_profit.append(adj_profit)

            self.date.append(date)

            self.stock_set = data[data.weight > 0][['weight', 'code']]
            self.stock_set.rename(columns={'weight': 'weight_last'}, inplace=True)

    def build_index(self):
        self.change = np.array(self.change)

        self.profit = np.array(self.profit).cumprod()[-1]
        self.adj_profit = np.array(self.adj_profit)
        self.year_profit = self.adj_profit.cumprod()[-1] ** (250 / len(self.date))
        self.sharpe_ratio = (self.adj_profit.cumprod()[-1]-1) / (self.adj_profit.std())
        print('%d 个交易日，收益率为:百分之 %f，年化收益率为百分之： %f, 夏普比率为： %f' % (len(self.date), (self.adj_profit.cumprod()[-1]-1)*100, (self.year_profit-1)*100, self.sharpe_ratio))

    def main(self):
        self.buy_set()
        self.position_set()
        self.build_index()