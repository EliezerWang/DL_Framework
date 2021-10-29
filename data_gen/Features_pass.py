import pandas as pd
from tqdm import tqdm


class Pass_features(object):

    def __init__(self, data, days, features):
        self.data = data
        self.days = days
        self.features = features
        self.name = pd.DataFrame(self.data[features].columns)

    def cre_past(self):

        def data_cut(data):
            data = data.reset_index().drop(['code'], axis=1).set_index('tradeDate')
            data = data.iloc[self.days-1:]
            return data

        for i in tqdm(range(1, self.days)):
            temp_name = self.name[0].apply(lambda x: x + '_' + str(i)).values.tolist()
            self.data[temp_name] = self.data[self.name[0].tolist()].groupby(['code']).shift(i)
            # self.data.groupby(['code']).fillna(method='bfill', inplace=True)
            self.data.groupby(['code']).fillna(0, inplace=True)
            # self.data.dropna(axis=0, inplace=True)
            # print(self.data.shape)

        self.data = self.data.groupby(['code']).apply(data_cut).reset_index().set_index(['tradeDate', 'code']).sort_values(by='tradeDate')

        return self.data
