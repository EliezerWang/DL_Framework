# DL_Framework
Self-build Deep learnning Framework for stock market  
Environment: python 3.9  
requirements packages are listed at requirements.md  

## 训练资料的获取(data_gen)：  
### 一、总操作台：var_lib.py  
(1)startdate: 数据取得开始的日期 例子:’20180618’。 end_date:如果是None则设当日的前一天为结束日期, 可以指定结束日期 例子:’20200618’。  
(2)code:选择需要的股票池 例子:[‘000300.SH’]  (记得带上[]，进入格式需为list)  
(3)myclient选择数据库 例子：pymongo.MongoClient("mongodb://192.168.17.19:27017/"), myclient.admin.authenticate 选择授权码 例子:'XXXXXX', 'wXXXXXXXXXX'  db:数据库下的文件夹 例子:’Data’  (这里的数据库为想要提取出的因子的数据库)  
(4)label_str 选择label的形式，选之前请认真比对自己的因子和数据库得到的因子的时间，切忌用到未来数据。1代表当天的收盘到隔日的收盘的收益率(t close ~ t+1 close)，2代表隔日的开盘到隔日的下一日的开盘的收益率(t+1 open ~ t+1 open)，3 代表隔日的开盘到隔日的下一日收盘的收益率(t+1 open ~ t+1 close) 也支持自定义，请到data_gen  例子: 1  
(5)past_days 选择过去多少天(包含今天)的数据作为因子,假设原来的数据是(3600, 2000,1,1)， 加了200天作为过去的因子之后，就是(3401,200,199,1)  例子:100  
(6)dg 增加变量的维度 比如原来有 a b两个因子， dg 为2 就为 a2， b2, ab, a, b, 1 以此类推， 例子: 1   io 不允许变量生成自己的次方(a2,b2,a3,b3之类的) True为不允许  例子: False  ib 允许出现1这一项 True为允许  例子: False  
(7)train_rate 训练集的比例，剩下的为测试集 例子 0.9   via_rate 中间集的比例，中间集是训练集中间划分出来的作为调参依据的  例子 0.9   
(8)aft_sel_num 使用统计模型筛选有用的因子后剩下的数量 例子：200  models统计模型的集合，可以自己导入   model 在模型中选择一个作为筛选的基准模型 例子: models[1]  
(9)times 将一份数据分解成几分运行，降低内存占用 例子：6  
(10)path 这里主要定义了存储位置  pre_split_path 最终资料的输出位置  all_data_path 没有进行index合并前的资料存储位置  
(11)shape_trans （因子数量如果不同需要更改）想要转化的资料形态 只要这三个数字乘起来等于 因子数乘上过去天数就可以了  比如：框架中有36个因子 而过去天数是200 在不筛选的情况下 原本是(36,200,1)，转换后可以是(18,100,4) 相乘都是7200  例子: (36, past_days, 1)  
(12)time_series_split  将数据分解成几块进行时间序列的交叉验证(训练几次) 例子：12  
(13)commission_rate  交易的手续费  例子0.0012  single_max 单票最大持仓 例子 0.1  
(14)cl_or_reg  分类或者回归 回归选择1 分类可以选择分几类（必须大于等于3） 例子：5   which_model  选择模型 0代表CNN  1代表LSTM  2代表统计模型  例子：0  
(15)name  填入自己的名字  例子：’张三’  
(16) save_client  填入自己想要保存参数以及结果的数据库   
例子：'mongodb://localhost:27017/'  
(17)	change_part  填入自己在这次修改时所修改的部分  例子：’CNN_Construction’  
(18)	save_db  填入数据要存入的具体的表名 表名要写入参考的人的名字(Ori_+人的名		字) 		例子：’Ori_Wang’   
(19)	change_part_path  填入自己改动的这一部分的路径名  
例子：r'C:\Users\Admin\PycharmProjects\pythonProject1\CNN_train\CNN_Construction.py'  
(20)	Path 训练中模型的存储路径(不是最后的最优模型)    
例子：r'C:\\Users\\Admin\\Desktop\\model'  
(21)	Best_model_path  调参结束后，里面表现最好的模型的存储路径  
例子：sys.path[0] + os.sep + 'model.h5'  
(22)	Which_loss_func  填入想要调参基于的损失函数 0代表基于准去率的函数			((via_acc+train_acc)/(2*(1+abs(via_acc - train_acc)), 主要是为了提升准确率的同时降低		中间集和训练集的差距,来防止过拟合或者欠拟合)  1 代表使用基于回测收益和夏普		比率的函数(np.sign(year_profit)*year_profit*sharpe_ratio)  例子：0  
(23)	How_split 选择如何进行时间序列上的交叉验证的划分  0 代表用过即增加的方		式进行划分  1 代表固定的训练集和测试集的比率进行划分  例子：0  
(24)	如果在(23)中选了1 这个才会影响到模型 代表固定比率划分的切分比率是： x单位的训练集 ：1单位的测试集  例子：2  

### 二、因子建立，以及因子取出：Features_call_create.py  
(1)	因子的调出 call_features 可以在Navicat 里面查看因子的位置，若要用到新的数据库，可以在里面自定义mydb2等 详细原理和var_lib里的一样，需要有授权码和数据库的名称  写法可以参照上面，如果不熟悉可以在notebook上先尝试  
(2)	因子的建立 build_features 如果是生成因子只用到当日的数据，就直接写就好了，不需要用到Tools window features等。 但如果你的数据是需要用到一段时间产生的，比如当日和过去的十日产生的数据，就需要用到Tools。只需要写入你需要用到的features的名称, 并写好对其处理的func，再加入你需要的rolling的长度window 就可以运行。  
  
### 三、对于滤波器的参数调整：filter_var.py  滤波器在的位置：linear_packages.py  
(1)	滤波器的选择在data_main.py的data_filter里的filter_func里  
(2)	滤波器包含了：上下包络线(可以通过filter_var.py里的which来选择输出的是上包络线和下包络线)、sav_filter输出单行数据、inter_plot插值法会增大数据 输出单行数据，增大的倍数可以在  filter_lib.py里面调整、moving_avg 移动平均输出单行，参数lib可调、low_pass_filter低通滤波器 参数lib可调 输出单行数据、high_pass_filter高通滤波器 参数lib可调 输出单行数据、band_pass_filter通带滤波器 参数lib可调 输出单行数据、band_stop_filter阻带滤波器 参数lib可调 输出单行数据、gaussian_filter高斯滤波器 参数lib可调，支持单维以及二维输入输出，不支持多维，box_filter盒型滤波器 参数lib可调 支持单维以及二维输入输出，不支持多维、mean_filter平均滤波器 参数lib可调 支持单维以及二维输入输出，不支持多维、median_filter中位数滤波器 参数lib可调 支持任意维度输入以及输出、ewm_process半衰期加权滤波 参数不可调 支持任意维度输入输出  
(3)	特殊滤波：EMD_filter EMD分解将一个波分解为多个，然后去掉前几个噪音项，得到更加纯净的真实波 参数lib可调整 仅支持单维输入以及输出、trans_EMD 使用EMD将波分解每一层波都会返回，参数不可调，单维输入多维输出（波不同维度不同）  
  
  
## 模型训练(CNN_train、LSTM_train、ML_train):  
  
### 一、CNN 总操作台 CNN_construction.py （需要模型架构和参数空间寻找能力）  
(1)	space 为参数空间（更多选项可以参考 中的Parameter Expressions），hp.uniform为包含两边的正态分布 hp.uniformint为包含两边的正态分布，但只取整数 hp.choice为 在一个list里做选择。需要为字典的形式{名字:hp.uniform(名字,min,max)}   max_eval为尝试不同参数组合的最大的次数  Path为存储训练中模型的地方，可以通过修改filepath为随着不同参数就不同的名称，来保存每一次参数组合尝试的模型。  
(2)	模型架构 只需改动f_NN1中的架构就好，需要调用space中的参数，就是用字典一般取出来就好 变量=params[自己取的名字]，需要传出一个损失函数作为调参基准，原框架用的是accuracy但是调参需要越小越好，所以加了负号  
(3)	模型架构注意事项：此模型输入的有训练集、中间集、测试集。测试集的预测作为输出，训练集调整模型的权重，中间集调参数  
  
### 二、ML 总操作台 ML_Construction.py （普通模型仅需要参数空间寻找能力、浅层神经网络模型则还需要模型架构能力）  
(1) Space为参数空间，space 为参数空间，hp.uniform为包含两边的正态分布 hp.uniformint为包含两边的正态分布，但只取整数 hp.choice为 在一个list里做选择。需要为字典的形式{名字:hp.uniform(名字,min,max)}   max_eval为尝试不同参数组合的最大的次数。  
(2) 包含了不同的模型，cl_model为分类模型的集合、reg_model为回归模型的集合。其中voting模型的分类和回归的是为了给模型加权的，可以尝试模型的集成  
(3) 有两种训练方式可以自己选择，一种是针对于普通的模型，一种是针对于像xgboost、lightgbm和catboost这几类浅层神经网络的模型的，训练方式不同，架构也不同  
  
## 日志撰写：  
  
## 请在训练的时候在var_lib下面输入自己的名字、改动的地方(文件名：比如:var_lib.py)  
  
## 每一次训练完都会自动上传，请不要重复训练相同的东西，除非报错  
## 训练出来的表现可以在mongodb中查看  （文字在collection,文件在GridFS）  


