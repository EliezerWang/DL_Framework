import os
import sys

import pymongo
import pynvml
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor

# 检查是否有gpu
try:
    pynvml.nvmlInit()
    gpu_num = pynvml.nvmlDeviceGetCount()
except:
    gpu_num = 0

name = '王逸洋'  # 请输入自己的名字   # T 0.0.2
save_client = pymongo.MongoClient('mongodb://localhost:27017/')  # 选择自己的本地数据库 用来上传日志 # T 0.0.2
change_part = 'CNN_Construction'  # 改动的部分 # T 0.0.2
save_db = 'Ori_Wang'  # 填入使用的模板的贡献者的名字作为表名 # T 0.0.2
change_part_path = r'C:\Users\Admin\PycharmProjects\pythonProject1\CNN_train\CNN_Construction.py'  # T 0.0.2
Path = r'C:\\Users\\Admin\\Desktop\\model'  # 训练中模型存储的位置 # T 0.0.2
Best_model_path = sys.path[0] + os.sep + 'model.h5'  # 最后参数测试结束后表现最好的模型的存储位置 # T 0.0.2
file_path = r'index_min\\data'

startdate = '20180601'  # 选择起始日期 结束日期默认是现在
end_date = None  # 如果选择None则为最近的一天为enddate
code = ['000300.SH']  # 选择你要的股票池（这里是沪深300）

myclient = pymongo.MongoClient("mongodb://192.168.17.19:27017/")  # 选择资料库
myclient.admin.authenticate('NXADMIN2', 'wY4SOOnSJYhUCmaH')  # 授权码
db = myclient["NxData"]

# 选择标签类型
label_str = 1  # 1:close to close 2:open to open 3:open to close

# 造因子
past_days = 5  # 1、能够加入前面的天数，弥补某些网络不会对过去的数据敏感的问题 2、可以在后面的Features_optimizer里充当滤波或者优化的窗口（不会使用到未来数据）
dg = 1  # 变量相乘的维度 2是 (a,b) (1,a,b,ab,a**2,b**2) 以此类推
io = False  # 允许变量自己平方 True为不允许
ib = False  # 不允许出现 1 这一项  True为允许


train_rate = 0.9  # 训练集划分
via_rate = 0.9  # 训练集中用作训练集的数量 建议大于0.7

# 统计模型筛选
aft_sel_num = 200  # 模型筛选后想留下来的因子数量（现在的版本默认为不使用，如果想要使用的话，请去模型里）
models = [LinearRegression(), Ridge(), Lasso(alpha=0.01, max_iter=10000), RandomForestRegressor(),
          GradientBoostingRegressor(), SVR(), LinearSVR(),
          ElasticNet(alpha=0.001, max_iter=10000), SGDRegressor(max_iter=1000, tol=1e-3), BayesianRidge(),
          KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
          ExtraTreesRegressor(), XGBRegressor(), CatBoostRegressor(), LGBMRegressor()]  # 筛选

model = models[1]  # 自行选择

times = 6  # 分解成几分运行 降低内存占用
# train_path = 'train_data_2.pkl'  # 最终训练集存储路径
# test_path = 'test_data_2.pkl'  # 最终测试集存储路径
pre_split_path = r'C:\Users\Admin\PycharmProjects\pythonProject1\data_gen\pre_split_data_1.pkl'
all_data_path = r'C:\Users\Admin\PycharmProjects\pythonProject1\data_gen\all_data_1.pkl'

shape_trans = (36, past_days, 1)  # 自己构造只要这3个乘起来是因子的总数就好 比如aft_sel_num是200 就可以选择(10,10,2)

commission_rate = 0.0006
cl_or_reg = 5  # 想做分类问题请选择3或3以上的数字（想分为几类就选择几），回归问题则选择1
single_max = 0.1  # 单票最大持仓

which_model = 0  # 0代表CNN 1代表LSTM 2代表统计模型
which_loss_func = 0  # 0代表基于准确率的损失函数 1代表基于基于回测的损失函数  vT0.0.3
time_series_split = 12  # 将数据分成几个进行交叉验证(总共训练几次产出最后结果) 两种分类方法都需要这个参数来确定  vT0.0.3
how_split = 0  # 0代表按照基于时间序列的交叉验证切分 1代表基于固定比例进行切分  vT0.0.3
train_split_ratio = 2  # 代表保持不动的对于数据的滚动切分是 x单位训练集：1单位测试集  vT0.0.3





