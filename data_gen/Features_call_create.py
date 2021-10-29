from data_gen.features_building_tools import Tools
import pandas as pd


def call_features(mydb, stockcode, startdate, enddate, newprice):
    """
    自己从数据库找到想要的因子在这里输出
    """

    cursor = mydb['UNIQUEALPHASTOCKFACTOR'].find(
        {"code": {'$in': stockcode}, "tradeDate": {"$gte": pd.Timestamp(startdate), "$lte": pd.Timestamp(
            enddate)}},
        {"_id": 0}, batch_size=10000)
    stockUNIQUEfactor = (pd.DataFrame(list(cursor)).drop_duplicates(subset=['tradeDate', 'code'], keep='first',
                                                                    inplace=False)).copy()

    stockUNIQUEfactor = (stockUNIQUEfactor.set_index(['tradeDate', 'code'])).copy()
    factortable = pd.merge(newprice.reset_index(), stockUNIQUEfactor, how='left',
                           on=['tradeDate', 'code']).sort_values(by=['tradeDate', 'code']).set_index(
        ['tradeDate', 'code'])

    factortable = factortable.astype('float32')  # 降低内存
    print('features_calling_done')
    return factortable


def build_features(factortable):
    # 不允许使用 shift(-1), 不允许用到未来数据， pct_change（）这些只能和shift(1)搭配使用，数据排列是从早到晚
    # 构建你的特征 如果是需要输入多行数据做移动窗口的请使用这个 并自己构建函数 返回的data记得自己构建列名
    def func(Series):
        array = Series.mean()
        return array

    features = ['open']
    window = 10
    factortable = Tools(factortable, func, window, features).building()
    print('features_building_done')

    return factortable