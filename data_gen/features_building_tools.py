import pandas as pd
from tqdm import tqdm


class Tools(object):

    def __init__(self, data, func, window, features):

        self.func = func
        self.window = window
        self.data = data
        self.temp_data = pd.DataFrame(data[features])

    def judge_method(self, data_temp):

        data_temp = pd.DataFrame(data_temp)

        if data_temp.shape[0] != 1 and data_temp.shape[1] != 1:
            raise ValueError('传入的数据只允许一行输出')

        if data_temp.shape[0] == 1 and data_temp.shape[1] != 1:
            data_temp = data_temp.T

        return data_temp

    def make_multi(self):  # 传入的函数输出需为np.array 并且不包含原数据

        data_final = self.func(self.temp_data[0:self.window])
        data_final = self.judge_method(data_final)

        for i in tqdm(range(1, self.data.shape[0] - self.window)):
            data_temp = self.func(self.temp_data[i:i + self.window])
            data_temp = self.judge_method(data_temp)
            data_final = pd.concat(objs=[data_final, data_temp], axis=0).reset_index(drop=True)

        self.data = pd.concat(objs=[self.data.iloc[self.window:].reset_index(drop=True), data_final], axis=1)

    def make_single(self):
        # self.data = self.data.rolling(self.window).apply(self.func)
        self.temp_data = self.temp_data.groupby(['code']).rolling(self.window).apply(self.func).sort_values(
            by=['tradeDate', 'code'])
        self.temp_data.index = self.temp_data.index.droplevel()
        self.data = pd.concat(objs=[self.temp_data, self.data], axis=1)

    def building(self):
        try:
            if self.temp_data.shape[1] > 1 and self.temp_data.shape[0] > 1:
                self.make_multi()
        except:
            if self.temp_data.shape[0] > 1:
                self.make_single()
            else:
                raise ValueError('just use your way to calculate')
        return self.data
