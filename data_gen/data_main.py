from data_gen.Features_building import CleanData
from data_gen.Features_Selection import Features_optimizer_split
from data_gen.var_lib import *
import pandas as pd
from data_gen import linear_packages
import time
from tqdm import tqdm
import tushare as ts


class final_program(object):
    def __init__(self):
        self.data_gen = CleanData()  # 加载features_building模块
        self.factor_data = self.data_gen.main()  # 使用features_building来导出数据，里面包含了因子和label用于后续的处理
        self.starttime = time.time()
        self.index_daily = pd.DataFrame()
    '''
    这里主要是在创建完成数据后进行滤波和简单的筛选（筛选建议不要）
    '''

    def get_index(self):
        enddate = self.data_gen.enddate
        pro = ts.pro_api('e0d39dfc7e1222262ebafddb38584f894b17c6d4a733062913d069f6')
        df = pro.index_daily(ts_code=code[0], strat_date=startdate, end_date=enddate)
        df.trade_date = df.trade_date.apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
        df = df.rename(columns={"ts_code": "code", 'trade_date': 'tradeDate', 'pct_chg': 'index_daily'})
        df = df[['tradeDate', 'index_daily']]
        df.index_daily = 1 + (df.index_daily/100)   # vT0.0.4
        self.index_daily = df    # vT0.0.4

    def get_data(self):

        self.factor_data.to_pickle(all_data_path)
        print('save_all_success')

    def data_filter(self):
        def filter_func(fil_data):  # 滤波器 传入df（单独因子）
            fil_data = pd.DataFrame(fil_data.values.reshape(past_days, int(len(fil_data) / past_days)))  # 因为之前加入了过去的past_days的数据所以将其折叠为天数和笔数的数据
            fil_data = fil_data.apply(linear_packages.EMD_filter, axis=0)  # 传入后使用EMD进行去噪
            fil_data = fil_data.apply(linear_packages.ewm_process, axis=0)  # 传入后使用ewm进行滤波
            fil_data = fil_data.values.flatten()  # 展平，恢复之前没折叠的样子
            fil_data = pd.Series(fil_data)  # 变为df方便后面处理
            return fil_data

        temp_split = int(self.factor_data.shape[0] / times)
        for i in tqdm(range(times)):
            self.factor_data.iloc[:temp_split, :-1] = self.factor_data.iloc[:temp_split, :-1].apply(filter_func, axis=1)
            temp_split = int(i * self.factor_data.shape[0] / times)

    def merge_index(self):
        self.factor_data = self.factor_data.rename(columns={'high': 'h', 'open': 'o', 'low': 'l', 'close': 'c'})
        temp_data = pd.merge(self.factor_data, self.data_gen.holc, how='left', right_index=True, left_index=True)
        temp_data = temp_data.reset_index()
        temp_data.tradeDate = temp_data.tradeDate.apply(lambda x: x.strftime('%Y-%m-%d'))
        temp_data = pd.merge(temp_data, self.index_daily, how='left', on='tradeDate').set_index(['tradeDate', 'code'])
        temp_data.to_pickle(pre_split_path)
        print('save_pre_success')
        # train_data, test_data = Features_optimizer_split(self.factor_data, self.split_spot).main()

        # train_data = train_data.reset_index()
        # test_data = test_data.reset_index()
        # train_data = pd.merge(train_data, self.data_gen.index_daily, how='left', on='tradeDate').set_index(
        #     ['tradeDate', 'code'])
        # train_data = pd.merge(train_data, self.data_gen.index_daily, how='left', on='tradeDate').set_index(
        #     ['tradeDate', 'code'])

        # train_data = pd.merge(train_data, self.holc, left_index=True, right_index=True, how='left')
        # test_data = pd.merge(test_data, self.holc, left_index=True, right_index=True, how='left')
        # endtime = time.time() - self.starttime
        # print("主程序运行时间：%.8s s" % endtime)  # 显示到微秒
        # train_data.to_pickle(train_path)
        # test_data.to_pickle(test_path)

    def main(self):
        self.get_data()
        self.get_index()
        self.data_filter()
        self.merge_index()

