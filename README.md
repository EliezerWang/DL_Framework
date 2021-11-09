# DL_Framework
Self-build Deep learnning Framework for stock market  
Environment: python 3.9  
requirements packages are listed at requirements.txt and please ensure that mongodb has been installed


## Acquisition of Training Data(data_gen):
### 1. Master Configure: var_lib.py
(1)startdate: date when data acquisition started example:’20180618’. end_date:If it is None, set the day before the current day as the end date, and you can specify the end date example:’20200618’.
(2)code:select the required stock pool Example:[‘000300.SH’] (remember to bring [], the format must be list)
(3) Myclient selects the database example: pymongo.MongoClient("mongodb://192.168.17.19:27017/"), myclient.admin.authenticate selects the authorization code example:'XXXXXX','wXXXXXXXXXX' db: folder under the database Example:'Data' (the database here is the database of the factors you want to extract)
(4) label_str Choose the form of label. Please carefully compare your own factor with the time of the factor obtained in the database before choosing, and do not use future data. 1 represents the rate of return from the close of the day to the close of the next day (t close ~ t+1 close), 2 represents the rate of return from the opening of the next day to the opening of the next day of the next day (t+1 open ~ t+1 open), 3 Represents the rate of return from the opening of the next day to the closing of the next day of the next day (t+1 open ~ t+1 close). Customization is also supported. Please go to data_gen Example: 1
(5) Past_days selects the data of the past days (including today) as the factor. Assuming the original data is (3600, 2000,1,1), after adding 200 days as the past factor, it is (3401,200,199,1) Example: 100
(6) dg increases the dimension of the variable. For example, there are two factors of ab, dg is 2, then a2, b2, ab, a, b, 1 and so on. Example: 1 io does not allow variables to generate their own powers (a2 , b2, a3, b3 and the like) True is not allowed example: False ib is allowed to appear 1 This item is allowed, and True is allowed example: False
(7) Train_rate is the proportion of the training set, and the rest is the test set. Example 0.9 via_rate The proportion of the intermediate set. The intermediate set is divided between the training sets as the basis for tuning. Example 0.9
(8)aft_sel_num Use the statistical model to filter the remaining number of useful factors. Example: 200 models statistical model collection, you can import model by yourself. Choose one of the models as the benchmark model for filtering. Example: models[1]
(9)times divide a piece of data into several points to run, reducing memory usage Example: 6
(10) path This mainly defines the storage location pre_split_path The output location of the final data all_data_path The data storage location before index merging is not performed
(11)shape_trans (If the number of factors is different, you need to change) The data shape you want to transform is as long as these three numbers are multiplied by the factor number multiplied by the number of days in the past. For example: there are 36 factors in the frame and the number of days in the past is 200. In the case of filtering, it was originally (36,200,1), after conversion, it can be (18,100,4) and multiplication is 7200 Example: (36, past_days, 1)
(12) time_series_split splits the data into several pieces for cross-validation of the time series (training several times) Example: 12
(13) Commission_rate transaction fee example 0.0012 single_max single ticket maximum position example 0.1
(14) cl_or_reg classification or regression Regression selection 1 Classification can be divided into several categories (must be greater than or equal to 3) Example: 5 which_model selection model 0 represents CNN 1 represents LSTM 2 represents statistical model Example: 0
(15)name Fill in your own name Example:'Zhang San'
(16) save_client fill in the database where you want to save the parameters and results
Example:'mongodb://localhost:27017/'
(17) change_part Fill in the part that you modified during this modification. Example: ‘CNN_Construction’
(18) Save_db fill in the specific table name where the data will be saved. The table name must be written in the name of the reference person (Ori_+person’s name) Example:’Ori_Wang’
(19) change_part_path fill in the path name of the part that you changed
Example: r'C:\Users\Admin\PycharmProjects\pythonProject1\CNN_train\CNN_Construction.py'
(20) Path The storage path of the model in training (not the final optimal model)
Example: r'C:\\Users\\Admin\\Desktop\\model'
(21) Best_model_path The storage path of the best-performing model after the tuning is over
Example: sys.path[0] + os.sep +'model.h5'
(22) Which_loss_func fill in the loss function you want to adjust based on 0 represents the function based on the quasi-drop rate ((via_acc+train_acc)/(2*(1+abs(via_acc-train_acc)), mainly to improve accuracy At the same time reduce the gap between the intermediate set and the training set to prevent over-fitting or under-fitting) 1 means to use the function based on backtesting profit and Sharpe ratio (np.sign(year_profit)*year_profit*sharpe_ratio) Example: 0
(23) How_split chooses how to divide the time series for cross-validation. 0 represents the division after use and increase. 1 represents the fixed ratio of the training set and the test set. Example: 0
(24) If you choose 1 in (23), this will affect the model. The segmentation ratio representing the fixed ratio division is: x unit training set: 1 unit test set Example: 2

### 2. Factor creation and factor extraction: Features_call_create.py
(1) The call_features of the factor can be viewed in Navicat. If you want to use a new database, you can customize the detailed principles of mydb2 in it. It is the same as in var_lib. You need to have the authorization code and the name of the database. Refer to the above, if you are not familiar with it, you can try it on the notebook first
(2) The establishment of factors build_features If it is to generate factors that only use the data of the day, just write them directly, without using Tools window features and so on. But if your data needs to be generated over a period of time, such as data generated on the current day and the past ten days, you need to use Tools. You only need to write the name of the features you need, and write the func to be processed, and then add the length of the rolling window you need to run.
  
### Three. Adjust the parameters of the filter: filter_var.py The location of the filter: linear_packages.py
(1) The selection of the filter is in filter_func in data_filter of data_main.py
(2) The filter contains: upper and lower envelopes (you can select the output of the upper and lower envelopes through which in filter_var.py), sav_filter outputs single-line data, and inter_plot interpolation will increase the data output Single line data, the multiplier of increase can be adjusted in filter_lib.py, moving_avg moving average output single line, parameter lib adjustable, low_pass_filter low pass filter parameter lib adjustable output single line data, high_pass_filter high pass filter parameter lib adjustable output single line data , Band_pass_filter passband filter parameter lib adjustable output single line data, band_stop_filter stopband filter parameter lib adjustable output single line data, gaussian_filter Gaussian filter parameter lib adjustable, support single-dimensional and two-dimensional input and output, does not support multi-dimensional, box_filter Box filter parameter lib is adjustable Support single-dimensional and two-dimensional input and output, does not support multi-dimensional, mean_filter average filter parameter lib is adjustable Support single-dimensional and two-dimensional input and output, does not support multi-dimensional, median_filter median filter parameter lib Adjustable support arbitrary dimension input and output, ewm_process half-life weighted filter parameters are not adjustable, support arbitrary dimension input and output
(3) Special filtering: EMD_filter EMD decomposition decomposes a wave into multiple, and then removes the first few noise items to obtain a purer real wave. The parameters lib can be adjusted. Only single-dimensional input and output are supported. Trans_EMD uses EMD to decompose each wave. A layer of waves will return, the parameters are not adjustable, single-dimensional input and multi-dimensional output (waves have different dimensions)
  
  
## Model training (CNN_train, LSTM_train, ML_train):
  
### 1. CNN main console CNN_construction.py (requires model architecture and parameter space search capabilities)
(1) Space is the parameter space (for more options, please refer to Parameter Expressions in), hp.uniform is the normal distribution that includes both sides, hp.uniformint is the normal distribution that includes both sides, but only the integer hp.choice is in one Make a selection in the list. It needs to be in the form of a dictionary {name:hp.uniform(name,min,max)} max_eval is the maximum number of attempts of different parameter combinations. Path is the place where the model in training is stored. You can modify the filepath to be different with different parameters Name to save the model for each parameter combination attempt.
(2) The model architecture only needs to change the architecture in f_NN1, and the parameters in space need to be called, which is generally taken out with a dictionary. Variable = params[name taken by yourself], and a loss function needs to be passed as a reference for tuning , The original framework uses accuracy, but the smaller the parameter adjustment needs, the better, so the minus sign is added
(3) Precautions for model architecture: The input for this model includes training set, intermediate set, and test set. The prediction of the test set is used as the output, the training set adjusts the weight of the model, and the middle set adjusts the parameters
  
### 2. ML main console ML_Construction.py (normal model only needs the ability to find parameter space, shallow neural network model also needs model architecture ability)
(1) Space is the parameter space, space is the parameter space, hp.uniform is the normal distribution containing both sides hp.uniformint is the normal distribution containing both sides, but only the integer hp.choice is selected in a list. It needs to be in the form of a dictionary {name:hp.uniform(name,min,max)} max_eval is the maximum number of attempts for different parameter combinations.
(2) Contains different models, cl_model is a collection of classification models, and reg_model is a collection of regression models. The classification and regression of the voting model are used to weight the model. You can try the integration of the model.
(3) There are two training methods you can choose by yourself, one is for ordinary models, and the other is for models of shallow neural networks like xgboost, lightgbm and catboost. The training methods are different and the architectures are also different.
  
## Log writing:
  
## Please enter your name and changes under var_lib during training (file name: e.g. var_lib.py) 
More about this source textSource text required for additional translation information
Send feedback
Side panels
History
Saved
Contribute




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


