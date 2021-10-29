import pandas as pd
import time
from data_gen.var_lib import *
from data_gen.Features_pass import Pass_features
import tushare as ts
import psutil
from data_gen import Features_call_create
from read_data import *


class CleanData(object):

    def __init__(self):
        self.startdate = startdate  # 如果是日频的这里 是否只需要到1年即可 这是我考虑的 后面的因子大部分都是基于此添加的
        self.mydb = db
        self.code = code
        self.label_str = label_str
        self.index_daily = pd.DataFrame()
        self.holc = pd.DataFrame()
        self.factortable = read_all(file_path)

    # def get_code(self):
    #     indexpricetemp = YoaFetchplusfactor.get_IndexPriceHistory(code=self.code, startdate=self.startdate,
    #                                                               cols=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME',
    #                                                                     'AMT']).sort_values(by=['tradeDate'])
    #     if end_date is None:
    #         enddate = list(indexpricetemp['tradeDate'])[-1]  # 需要提前几天 这里注意 的 要
    #         self.enddate = enddate.strftime("%Y%m%d")  # 这个enddate需要注意 时间一定要是我们有的因子的当天 相同
    #     else:
    #         self.enddate = end_date
    #     stockweigh = YoaFetchplusfactor.get_StockWeight(index=self.code, startdate=self.startdate,
    #                                                     enddate=self.enddate).reset_index().sort_values(
    #         by=['tradeDate', 'code'])
    #     date_len = len(list(indexpricetemp['tradeDate'])[:-1])  # 得到天数的长度
    #     self.split_spot = int(date_len*train_rate)
    #     self.stockcode = list(set(stockweigh['code']))
    #     print(self.startdate, self.enddate)
    #     self.split_spot = list(indexpricetemp['tradeDate'])[self.split_spot]  # 拿到训练集验证集区分的点
    #
    # def get_data(self):
    #     starttime = time.time()  # 在这里先不指定index 全部按照tradedate code
    #     stockprice = YoaFetchplusfactor.get_StockPriceHistory(code=self.stockcode, startdate=self.startdate,
    #                                                           enddate=self.enddate,
    #                                                           cols=['open', 'high', 'low', 'close', 'vol', 'amount'])
    #     # 录入因子用的是stockprice的时间唯度  stockprice是最重要的维度 所有的匹配和计算因子目前都是按照stockprice的维度计算的
    #     stockadj = YoaFetchplusfactor.get_StockAdjFactorHistory(code=self.stockcode, startdate=self.startdate,
    #                                                             enddate=self.enddate,
    #                                                             cols=['adjFactor'])
    #     stockbasic = YoaFetchplusfactor.get_StockBasicHistory(code=self.stockcode, startdate=self.startdate,
    #                                                           enddate=self.enddate,
    #                                                           cols=['capitalization', 'circulating_cap'])
    #     stockprice = stockprice.sort_values(by=['tradeDate', 'code'])
    #     stockadj = stockadj.sort_values(by=['tradeDate', 'code'])
    #     stockbasic = stockbasic.sort_values(by=['tradeDate', 'code'])
    #     prices = stockprice.merge(stockadj, on=['tradeDate', 'code'], how='left')
    #     #  merge concat join
    #     prices = prices.sort_values(by=['code', 'tradeDate'])
    #     prices['adjFactor'].fillna(method='backfill', inplace=True)
    #     #  merge concat join
    #     assert prices['adjFactor'].isna().sum() == 0
    #     prices = priceDfResample.getAdjPrice(prices)  # 获取复权价格
    #     prices.fillna(0, inplace=True)
    #     prices = prices.sort_values(by=['tradeDate', 'code'])
    #     # 如果针对自己的股票集群这里的code得换
    #     indexprice = YoaFetchplusfactor.get_IndexPriceHistory(code=self.code, startdate=self.startdate,
    #                                                           enddate=self.enddate,
    #                                                           cols=['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMT'])
    #     indexprice = indexprice.sort_values(by=['tradeDate']).rename(columns={'CLOSE': 'close'})
    #     self.stockindustry = YoaFetchplusfactor.get_StockIndustry(code=self.stockcode, startdate=self.startdate,
    #                                                               enddate=self.enddate).reset_index().sort_values(
    #         by=['tradeDate', 'code']) \
    #         .drop_duplicates(subset=['tradeDate', 'code'], keep='first', inplace=False)
    #     # 如果针对自己的股票集群这里的index得换
    #     stockweigh = YoaFetchplusfactor.get_StockWeight(index=self.code, startdate=self.startdate,
    #                                                     enddate=self.enddate).reset_index().sort_values(
    #         by=['tradeDate', 'code']).drop_duplicates(subset=['tradeDate', 'code'], keep='first', inplace=False)
    #     stockweigh['weight'].fillna(0.018716000000000008, inplace=True)
    #     endtime = time.time() - starttime
    #     prices['vwapclose'] = prices['amount'] / prices['vol']
    #     print("程序运行时间：%.8s s" % endtime)  # 显示到微秒
    #     prices['vwapclose'][prices['vwapclose'].isna()] = prices['close'][prices['vwapclose'].isna()]
    #     print('取数据工作完成')
    #
    #     self.stockweigh = stockweigh
    #     self.prices = prices
    #
    # def clean_data(self):
    #     stockweigh = self.stockweigh.drop_duplicates(subset=['tradeDate', 'code'], keep='first',
    #                                                  inplace=False).set_index(['tradeDate', 'code'])
    #     prices = self.prices.set_index(['tradeDate', 'code'])
    #     newprice = stockweigh.join(prices, how='left').join(
    #         self.stockindustry.set_index(['tradeDate', 'code'])[['sec_name1']],
    #         how='left').reset_index()  # 在相同的索引上完全合并 vol是成交量合并进来后 开始进行整体
    #     newprice.loc[newprice['sec_name1'].isna(), ['sec_name1']] = '数据库无记录'
    #     newprice = newprice[newprice['vol'] > 1000][
    #         newprice['sec_name1'].map(lambda x: 'ST' not in x)]  # 前一天可以买到的情况下的统计  剔除了价格上停牌 成交量不足的情况  整体切片的方法是最好的 速度最快
    #     newprice.drop(['sec_name1', 'index'], axis=1, inplace=True)
    #     newpricetable = pd.pivot_table(newprice, values='open', index='tradeDate', columns='code')
    #     newprice = newprice.set_index(['tradeDate', 'code'])  # 这步的目的依然是为了匹配因子索引
    #     self.newpricetable = newpricetable
    #     self.newprice = newprice
    #     self.holc = newprice[['high', 'open', 'low', 'close', 'amount']]
    #     self.holc.columns = ['high_test', 'open_test', 'low_test', 'close_test', 'amount_test']
    #     del self.prices
    #     del self.stockindustry
    #     del self.stockweigh
    #     print('data_cleaning_done')

    def get_index(self):
        pro = ts.pro_api()
        df = pro.index_daily(ts_code=self.code, strat_date=self.startdate, end_date=self.enddate)
        df.trade_date = df.trade_date.apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
        df = df.rename(columns={"ts_code": "code", 'tradeDate': 'tradeDate', 'pct_chg': 'index_daily'})
        df = df[['tradeDate', 'index_daily']]
        self.index_daily = df

    def call_features(self):

        # cursor = self.mydb['UNIQUEALPHASTOCKFACTOR'].find(
        #     {"code": {'$in': self.stockcode}, "tradeDate": {"$gte": pd.Timestamp(self.startdate), "$lte": pd.Timestamp(
        #         self.enddate)}},
        #     {"_id": 0}, batch_size=10000)
        # self.stockUNIQUEfactor = (pd.DataFrame(list(cursor)).drop_duplicates(subset=['tradeDate', 'code'], keep='first',
        #                                                                      inplace=False)).copy()
        #
        # self.stockUNIQUEfactor = (self.stockUNIQUEfactor.set_index(['tradeDate', 'code'])).copy()
        # self.factortable = pd.merge(self.newprice.reset_index(), self.stockUNIQUEfactor, how='left',
        #                             on=['tradeDate', 'code']).sort_values(by=['tradeDate', 'code']).set_index(
        #     ['tradeDate', 'code'])
        # del self.stockUNIQUEfactor
        # del self.newprice
        # self.factortable = self.factortable.astype('float32')
        # print('features_calling_done')

        self.factortable = Features_call_create.call_features(self.mydb, self.stockcode, self.startdate, self.enddate, self.newprice)
        del self.newprice

    def build_features(self):

        # 不允许使用 shift(-1), 不允许用到未来数据， pct_change（）这些只能和shift(1)搭配使用，数据排列是从早到晚
        # 构建你的特征 如果是需要输入多行数据做移动窗口的请使用这个 并自己构建函数 返回的data记得自己构建列名

        # def func(Series):
        #     array = Series.mean()
        #     return array
        #
        # features = ['open']
        # window = 10
        # self.factortable = Tools(self.factortable, func, window, features).building()
        # print('features_building_done')

        self.factortable = Features_call_create.build_features(self.factortable)

    def label_select(self):
        if self.label_str == 1:
            self.factortable['label'] = self.factortable.groupby(level='code').close.pct_change().groupby(level='code').shift(-1)
        elif self.label_str == 0:
            self.factortable['label'] = self.factortable.groupby(level='code').open.pct_change().groupby(level='code').shift(-2)
        else:
            self.factortable['label'] = (self.factortable.groupby(level='code').close.shift(-2) - self.factortable.groupby(level='code').open.shift(
                -1)) / self.factortable.groupby(level='code').open.shift(-1)
        print('label_selecting_done')

    def creat_pass(self):  # 输入的资料需要从早到晚的时间顺序
        features = self.factortable.columns
        # features = self.factortable.columns.difference(['','',''])
        self.factortable = Pass_features(self.factortable, past_days, features).cre_past()
        print('creat_pass_done')

    def del_nan(self):
        # self.factortable.fillna(method='ffill', inplace=True)
        self.factortable.groupby(level='code').fillna(method='bfill', inplace=True)
        self.factortable.dropna(inplace=True)
        print('nan_deleting_done')

    def stdlize_data(self):
        def _factorFiterAndNormize(factordf):
            zscore = (factordf - factordf.median()) / (factordf - factordf.median()).abs().median()
            zscore[zscore > 3] = 3
            zscore[zscore < -3] = -3
            factordf = zscore * (factordf - factordf.median()).abs().median() + factordf.median()
            factordf = (factordf - factordf.mean()) / factordf.std()
            return factordf

        def _factorFiterAndNormize_(data):
            std = data.std()
            mean = data.mean()
            std = std.replace(0, 1)
            data.iloc[:split_test] = (data.iloc[:split_test] - mean) / std
            data.iloc[split_test:] = (data.iloc[split_test:] - mean) / std
            return data

        def get_mean_std(data):
            new_data = pd.DataFrame()
            data['standard'] = data.std().values
            data['mean'] = data.mean().values
            return pd.Series(data)

        info = psutil.virtual_memory()
        if info.percent > 0.5:
            times = int(info.percent / (1 - info.percent))
        else:
            times = 1

        split_test = int(self.factortable.shape[0] * train_rate)

        # 计算训练集的分布数据
        for i in range(times):
            temp_split_pre = split_test * (i/times)
            temp_split_aft = split_test * ((i+1)/times)

        self.factortable.iloc[:split_test, :-1] = self.factortable.iloc[:split_test, :-1].groupby(level='code').apply(_factorFiterAndNormize_)
        self.factortable.iloc[split_test:, :-1] = self.factortable.iloc[split_test:, :-1].groupby(level='code').apply(_factorFiterAndNormize_)
        self.factortable.iloc[:split_test, :-1] = self.factortable.iloc[:split_test, :-1].groupby(level='tradeDate').apply(_factorFiterAndNormize_)
        self.factortable.iloc[split_test:, :-1] = self.factortable.iloc[split_test:, :-1].groupby(level='tradeDate').apply(_factorFiterAndNormize_)
        print('data_standardizing_done')

    def main(self):
        self.get_code()
        self.get_data()
        self.clean_data()
        self.call_features()
        self.build_features()
        self.creat_pass()
        print(self.factortable.shape)
        self.label_select()
        #self.stdlize_data()
        self.del_nan()
        print('All_tasks_in_Features_building_done')
        return self.factortable.astype('float32')
