from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from data_gen.var_lib import *
from CNN_train import CNN
from ML_train import machine_learning
from tqdm import tqdm
import numpy as np
import datetime


class model_gen(object):
    def __init__(self, data):
        self.data = data.reset_index()
        self.train_list = []
        self.test_list = []
        self.score = []
        self.year_profit = []
        self.sharpe_ratio = []
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.profit_list = np.array([1])
        self.total_return = 0
        self.total_year_return = 0
        self.total_sharpe = 0

        col_name = list(data.columns)
        col_pos = list(range(len(col_name)))
        self.dict_name = dict(zip(col_name, col_pos))

    def get_time_split(self):  # vT0.0.3   下个版本可以改成基于日期进行划分(现在是基于资料数)
        if how_split == 0:
            tscv = TimeSeriesSplit(n_splits=time_series_split)
            for train, test in tscv.split(self.data):
                self.train_list.append(train)
                self.test_list.append(test)

        elif how_split == 1:  # vT0.0.3

            unit_time_start = int((train_split_ratio*self.data.shape[0])/(time_series_split+train_split_ratio))
            unit_time_long = int(self.data.shape[0]/(time_series_split+train_split_ratio))

            temp_train = self.data.iloc[:unit_time_start]
            temp_test = self.data.iloc[unit_time_start:unit_time_start + unit_time_long]
            self.train_list.append(temp_train)
            self.test_list.append(temp_test)
            last_site = unit_time_start + unit_time_long

            for i in range(time_series_split-2):
                temp_train = self.data.iloc[(last_site-unit_time_start):last_site]
                temp_test = self.data.iloc[last_site: (last_site+unit_time_long)]
                last_site += unit_time_long
                self.train_list.append(temp_train)
                self.test_list.append(temp_test)

            temp_train = self.data.iloc[(last_site-unit_time_start):last_site]
            temp_test = self.data.iloc[last_site:]
            self.train_list.append(temp_train)
            self.test_list.append(temp_test)

    def train_model(self):
        def filter_stdlize(data):
            train = data[data.index.isin(self.train_list[i])].set_index(['tradeDate', 'code'])
            test = data[data.index.isin(self.test_list[i])].set_index(['tradeDate', 'code'])

            train = train.replace([np.inf, -np.inf], 1)
            test = test.replace([np.inf, -np.inf], 1)

            train_label = train.iloc[:, self.dict_name['label']:]
            test_label = test.iloc[:, self.dict_name['label']:]

            train = train.iloc[:, :self.dict_name['label']]
            test = test.iloc[:, :self.dict_name['label']]

            temp_max = train.max()
            temp_min = train.min()

            train = (train - temp_min) / (temp_max-temp_min)
            train = train.replace([np.inf, -np.inf], np.nan)
            train.fillna(train.mean(), inplace=True)
            train.fillna(0, inplace=True)

            test = (test - temp_min) / (temp_max-temp_min)
            test = test.replace([np.inf, -np.inf], np.nan)
            test.fillna(train.mean(), inplace=True)
            test.fillna(0, inplace=True)

            train = pd.concat([train, train_label], axis=1)
            test = pd.concat([test, test_label], axis=1)

            return train, test
        # ------------------------------------------------------------

        for i in range(time_series_split):

            # train_data, test_data = self.data.groupby(by='code').apply(filter_stdlize)
            self.train_data = pd.DataFrame()
            self.test_data = pd.DataFrame()

            for code_name, temp_data in tqdm(list(self.data.groupby(by='code')), desc='data_gen'):
                temp_train, temp_test = filter_stdlize(temp_data)
                self.train_data = pd.concat([self.train_data, temp_train])
                self.test_data = pd.concat([self.test_data, temp_test])

            # long = int(self.test_data.shape[0]/4)
            # self.test_data = self.test_data.iloc[:long]
            self.train_data.to_pickle('train_data_pre_CNN.pkl')
            self.test_data.to_pickle('test_data_pre_CNN.pkl')

            print('std_finish')
            print(self.train_data.columns)

            if which_model == 0:
                temp = CNN.Model_training(self.train_data, self.test_data)

            elif which_model == 2:
                temp = machine_learning.Model_training(self.data[self.train_list[i]], self.data[self.test_list[i]])

            temp.main()
            self.score.append(temp.score)
            self.year_profit.append(temp.year_profit)
            self.sharpe_ratio.append(temp.sharpe_ratio)
            self.profit_list = np.append(self.profit_list, temp.profit_list)

    def return_and_to_db(self):
        self.total_return = self.profit_list.cumprod()[-1]
        self.total_sharpe = (self.profit_list.cumprod()[-1]-1)/(self.profit_list-1).std()
        data = {'name': name, 'time': datetime.datetime.now(), 'total_performance': self.total_return,
                'total_sharpe': self.total_sharpe}
        x = save_db[change_part].insert(data)

    def main(self):
        self.get_time_split()
        self.train_model()
        self.return_and_to_db()
