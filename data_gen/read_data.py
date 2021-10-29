import pandas as pd
import os
from tqdm import tqdm
from var_lib import *


class data_sorting(object):

    def __init__(self, name, shift_num):
        self.file_name = name
        self.data = pd.DataFrame()
        self.shift_num = shift_num

    def read_raw(self):
        file = self.file_name
        store = pd.HDFStore(file, mode='r')
        self.data = store.select('data')
        self.data.reset_index(drop=True, inplace=True)
        self.data.ticker = self.data.ticker.astype('int')
        store.close()

    def shift_cell(self):
        self.data.reset_index(drop=True, inplace=True)
        test_data = self.data.copy()
        for i in tqdm(range(self.shift_num)):
            i += 1
            temp_dataframe = test_data.copy().shift(-i)
            temp_dataframe.ticker = temp_dataframe.ticker + i
            self.data = pd.concat([self.data, temp_dataframe], axis=0, join='outer', ignore_index=True).dropna()

    def clean_data(self):
        if self.data.isnull().sum().sum() != 0:
            self.data.fillna(method='ffill', inplace=True)
            self.data.dropna(inplace=True)
        self.data = self.data[['closePrice', 'highPrice', 'lowPrice', 'openPrice', 'totalValue', 'totalVolume']]
        self.data.rename(columns={'closePrice': 'close', 'highPrice': 'high', 'lowPrice': 'low', 'openPrice': 'open', 'totalValue': 'amount', 'totalVolume': 'vol'}, inplace=True)

    def sort_data(self):
        self.data['tradeTime'] = pd.to_datetime(self.data.dataDate + ' ' + self.data.barTime)
        self.data.rename(columns={'tradeTime': 'tradeDate', 'ticker': 'code'}, inplace=True)
        self.data['tick'] = self.data.barTime.copy()
        self.data['asset'] = self.data['code'].astype('str')
        self.data.set_index(['tradeDate', 'code', ], inplace=True)
        self.data.sort_values(by=['tradeDate', 'code'], inplace=True)

    def run(self):
        self.read_raw()
        # self.shift_cell()
        self.sort_data()
        self.clean_data()
        return self.data


def read_all(file_path):
    final_data = pd.DataFrame()
    for file in tqdm(os.listdir(file_path)):
        read_data = data_sorting((f'{file_path}/' + file), 20).run()
        final_data = pd.concat([final_data, read_data])
    return final_data