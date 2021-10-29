import pandas as pd
from CNN_train.CNN_Construction import *
import Backtest_model_1
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.utils import to_categorical


class Model_training(object):

    def __init__(self, train_data, test_data):
        print(train_data.shape, test_data.shape)
        # print('是否含有nan值:', train_data.isna().any().any(), test_data.isna().any().any())
        # train_data.
        train_data, test_data = train_data.reset_index(), test_data.reset_index()
        split_spot = int(test_data.shape[0] * (1 - via_rate))
        via_data = test_data[:split_spot]
        test_data = test_data[split_spot:]

        col_name = list(train_data.columns)
        col_pos = list(range(len(col_name)))
        dict_name = dict(zip(col_name, col_pos))

        self.train_X, self.via_X, self.test_X = train_data.iloc[:, (dict_name['code'] + 1):dict_name['label']].values, \
                                                via_data.iloc[:, (dict_name['code'] + 1):dict_name['label']].values, test_data.iloc[
                                                :, (dict_name['code'] + 1):dict_name['label']].values
        self.train_Y, self.via_Y, self.test_Y = train_data.iloc[:, dict_name['label']].values, via_data.iloc[:,
                                                dict_name['label']].values, test_data.iloc[:,dict_name['label']].values

        self.test_date = test_data.iloc[:, dict_name['tradeDate']]
        # self.index_return = test_data.iloc[:, dict_name['index_daily']]
        self.shape_trans = shape_trans
        self.cl_or_reg = cl_or_reg
        self.mat = []
        self.trials = Trials()  # 通过直接传递一个Trials对象，我们可以检查实验期间计算的所有返回值
        self.test_code = test_data.iloc[:, dict_name['code']]
        self.test_high = test_data.iloc[:, dict_name['high_test']]
        self.test_open = test_data.iloc[:, dict_name['open_test']]
        self.test_low = test_data.iloc[:, dict_name['low_test']]
        self.test_close = test_data.iloc[:, dict_name['close_test']]
        self.test = 1 + test_data.iloc[:, dict_name['label']].values
        self.index_daily = test_data.iloc[:, dict_name['index_daily']]
        self.amount = test_data.iloc[:, dict_name['amount_test']]

        self.year_profit = 0
        self.sharpe_ratio = 0
        self.pred_Y = pd.DataFrame()
        self.pred_Y_com = pd.DataFrame()
        self.best = dict()
        self.profit_list = np.array([1])
        self.last_Y = pd.DataFrame()

    def create_poly(self):
        pl = PolynomialFeatures(degree=dg, interaction_only=io, include_bias=ib)
        self.train_X = pl.fit_transform(self.train_X)
        self.via_X = pl.fit_transform(self.via_X)
        self.test_X = pl.fit_transform(self.test_X)

    def make_category(self):

        if cl_or_reg > 2:
            self.train_Y[(self.train_Y > 0) & (self.train_Y <= 0.0012)], self.via_Y[
                (self.via_Y > 0) & (self.via_Y <= 0.0012)], self.test_Y[
                (self.test_Y > 0) & (self.test_Y <= 0.0012)] = 1, 1, 1
            self.train_Y[self.train_Y <= 0], self.via_Y[self.via_Y <= 0], self.test_Y[self.test_Y <= 0] = 0, 0, 0
            split_last = 0.0012
            self.mat = [-1, 0]
            for i in range(1, cl_or_reg - 2):
                split_num = np.percentile(self.train_Y[(self.train_Y > split_last) & (self.train_Y < 1)],
                                          i / (self.cl_or_reg - 2))
                self.train_Y[(self.train_Y > split_last) & (self.train_Y <= split_num)], self.via_Y[
                    (self.via_Y > split_last) & (self.via_Y <= split_num)], \
                self.test_Y[(self.test_Y > split_last) & (self.test_Y <= split_num)] = i + 1, i + 1, i + 1
                split_last = split_num
                self.mat.append(i)

            self.train_Y[(self.train_Y > split_last) & (self.train_Y < 1)], self.via_Y[
                (self.via_Y > split_last) & (self.via_Y < 1)], self.test_Y[
                (self.test_Y > split_last) & (self.test_Y < 1)] = cl_or_reg - 1, cl_or_reg - 1, cl_or_reg - 1
            self.mat.append(self.mat[-1] + 1)
            self.train_Y, self.via_Y, self.test_Y = to_categorical(self.train_Y, cl_or_reg), to_categorical(self.via_Y,
                                                                                                            cl_or_reg), to_categorical(
                self.test_Y, cl_or_reg)

        self.train_X, self.via_X, self.test_X = self.train_X.reshape(
            tuple([self.train_X.shape[0]]) + self.shape_trans), self.via_X.reshape(
            tuple([self.via_X.shape[0]]) + self.shape_trans), self.test_X.reshape(
            tuple([self.test_X.shape[0]]) + self.shape_trans)

    def load_last(self):
        last_model = keras.models.load_model('model.h5')
        self.last_Y = last_model.predict(self.test_X)

    def hyper_train(self):

        self.best, self.trials = CNN_2_FNN(self.train_X, self.train_Y, self.via_X, self.via_Y, self.test_X, self.test_Y,
                                           self.mat, self.test, self.test_date, self.test_code, self.test_high,
                                           self.test_open, self.test_low, self.test_close, self.amount, self.index_daily)
        loss_list = np.array(self.trials.losses())
        key = np.where(loss_list == loss_list.min())[0][0]
        self.pred_Y = self.trials.results[key]['test_pred']
        self.score = self.trials.results[key]['test_score']
        self.params = self.trials.results[key]['params']
        model = self.trials.results[key]['model']
        model.save('model.h5')

    def aft_train(self):
        # 记得要承上面的self.mat得到最终的预期值
        self.pred_Y = (self.pred_Y * self.mat).sum(axis=1)
        self.test_Y = (self.test_Y * self.mat).sum(axis=1)
        # self.pred_Y_com = pd.DataFrame([self.pred_Y, self.test, self.test_date.values, self.test_code.values])  # self.index_return, self.test_date, self.test_code])
        self.pred_Y_com = pd.DataFrame([self.pred_Y, self.test, self.test_date.values,
                                        self.test_code.values, self.test_high, self.test_open, self.test_low,
                                        self.test_close, self.amount,self.index_daily])  # vT0.0.4
        self.pred_Y_com = self.pred_Y_com.T
        self.pred_Y_com.columns = ['pred', 'test', 'tradeDate', 'code', 'high', 'open', 'low',
                                   'close', 'amount', 'index_daily']  # vT0.0.4
        self.pred_Y_com.to_pickle('pred_Y.pkl')

    def back_test(self):
        temp = Backtest_model_1.Back_test(self.pred_Y_com, self.params, True)  # 0.0.2
        temp.main()
        self.year_profit = temp.year_profit
        self.sharpe_ratio = temp.sharpe_ratio
        self.profit_list = temp.adj_profit

    def main(self):
        self.make_category()
        self.hyper_train()
        self.aft_train()
        self.back_test()
