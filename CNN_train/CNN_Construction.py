import numpy as np
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam
from keras import regularizers, initializers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D, AveragePooling2D, \
    GlobalAveragePooling2D, Conv1D
from hyperopt import hp, STATUS_OK
from keras.layers import BatchNormalization
import keras
from data_gen.var_lib import *
from hyperopt import fmin, Trials, tpe
import os
import Backtest_model_1
import pandas as pd

# from ranger21 import Ranger21
# 定义参数空间

space = {'ll_float': hp.uniform('ll_float', 10 ** -99, 10 ** -5),
         'lr': hp.uniform('lr', 10 ** -99, 10 ** -5),
         'beta_1_float': hp.uniform('beta_1_float', 0.8, 1 - (10 ** -10)),
         'beta_2_float': hp.uniform('beta_2_float', 0.8, 1 - (10 ** -10)),
         'epsilon_float': hp.uniform('epsilon_float', 10 ** -100, 10 ** -10),
         'batch_size': hp.uniformint('batch_size', 1000, 2000),
         'epochs': hp.uniformint('epochs', 50, 200),
         'first_num': hp.uniformint('first_dense', 10, 20),
         'first_dense': hp.uniformint('first_num', 5, 10),
         'first_drop': hp.uniform('first_drop', 0.1, 0.8),
         'second_num': hp.uniformint('second_dense', 5, 10),
         'second_dense': hp.uniformint('second_num', 10, 20),
         'second_drop': hp.uniform('second_drop', 0.1, 0.8)
         # 'conv_num': hp.uniformint('conv_num', 64, 128)
         }

max_eval = 20  # 参数最大调整次数
Path = r'C:\\Users\\Admin\\Desktop\\model'  # 模型存储的位置
silent = -1


def CNN_2_FNN(x_train, y_train, x_via, y_via, x_test, y_test, mat, test, test_date, test_code, test_high, test_open,
              test_low, test_close, amount, index_daily):
    def f_NN1(params):
        # device = cuda.get_current_device()
        # device.reset()
        # cuda.select_device(0)

        def back_test(pred_Y):
            pred_Y = (pred_Y * mat).sum(axis=1)
            pred_Y_com = pd.DataFrame([pred_Y, test, test_date.values,
                                       test_code.values, test_high, test_open, test_low, test_close, amount,
                                       index_daily])  # self.index_return, self.test_date, self.test_code])
            pred_Y_com = pred_Y_com.T
            pred_Y_com.columns = ['pred', 'test', 'tradeDate', 'code', 'high', 'open', 'low', 'close', 'amount',
                                  'index_daily']  # 'index_daily', 'tradeDate', 'code']
            pred_Y_com.to_pickle('pred_Y.pkl')

            temp = Backtest_model_1.Back_test(pred_Y_com, params)
            temp.main()
            year_profit = temp.year_profit
            sharpe_ratio = temp.sharpe_ratio
            return year_profit, sharpe_ratio

        if silent <= 0:
            print(params)
        if silent <= -1:
            print('是否含有nan值:', (True in np.isnan(x_train)), (True in np.isnan(x_via)), (True in np.isnan(x_test)))
            print('是否含有inf值:', (True in np.isinf(x_train)), (True in np.isinf(x_via)), (True in np.isinf(x_test)))
        keras.backend.set_image_data_format('channels_first')
        # define params
        ll_float = params["ll_float"]  # 学习率其
        learn_rate_float = params["lr"]
        beta_1_float = params["beta_1_float"]
        beta_2_float = params["beta_2_float"]
        epsilon_float = params["epsilon_float"]
        batch_size_num = params['batch_size']
        epochs_num = params['epochs']

        model = Sequential()
        # init_conv_1 = initializers.he_normal(seed=100)

        model.add(Conv2D(10, (3, 3), padding='same',  # 20 个filter (5,5) 的size 都可以加入上面的space进行调整
                         input_shape=x_train.shape[1:], kernel_regularizer=regularizers.l2(ll_float),
                         kernel_initializer=initializers.initializers_v2.GlorotNormal(),
                  data_format="channels_last"))  # , init=init_conv_1)
        model.add(BatchNormalization())
        model.add(Activation('relu'))  # 可使用LeakyReLU

        model.add(Conv2D(10, (3, 3), padding='same',  # 20 个filter (5,5) 的size 都可以加入上面的space进行调整
                         kernel_regularizer=regularizers.l2(ll_float),
                         kernel_initializer=initializers.initializers_v2.GlorotNormal(),
                         data_format="channels_last"))  # , init=init_conv_1)
        model.add(BatchNormalization())
        model.add(Activation('relu'))  # 可使用LeakyReLU

        # init_conv_2 = initializers.he_normal(seed=101)
        # model.add(Conv2D(10, (5, 5), padding='same',  # 20 个filter (5,5) 的size 都可以加入上面的space进行调整
        #                  kernel_regularizer=regularizers.l2(ll_float),
        #                  data_format="channels_last"))  # , init=init_conv_2)
        # model.add(Activation('relu'))  # 可使用LeakyReLU

        # model.add(AveragePooling2D(20, (5, 5), padding='same'))  # 20 个filter (5,5) 的size 都可以加入上面的space进行调整
        # model.add(Activation('relu'))  # 可使用LeakyReLU
        # # init_layer_1 = initializers.he_normal(seed=102)

        # model.add(Flatten())

        for i in range(params['first_num']):
            # model.add(
            #     Dense(params['first_dense'], activation='relu'))  # , kernel_initializer=init_layer_1)  # 可使用LeakyReLU
            model.add(Conv2D(params['first_dense'], (3, 3), padding='same',  # 20 个filter (5,5) 的size 都可以加入上面的space进行调整
                             kernel_regularizer=regularizers.l2(ll_float),
                             kernel_initializer=initializers.initializers_v2.GlorotNormal(),
                             data_format="channels_last"))  # , init=init_conv_1)
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(params['first_drop']))

        # init_layer_2 = initializers.he_normal(seed=103)
        for i in range(params['second_num']):
            # model.add( Dense(params['second_dense'], activation='relu'))  # , kernel_initializer=init_layer_2)
            # 可使用LeakyReLU

            model.add(Conv2D(params['second_dense'], (3, 3), padding='same',  # 20 个filter (5,5) 的size 都可以加入上面的space进行调整
                             kernel_regularizer=regularizers.l2(ll_float),
                             kernel_initializer=initializers.initializers_v2.GlorotNormal(),
                             data_format="channels_last"))  # , init=init_conv_1)
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(params['second_drop']))

        model.add(Flatten())
        model.add(Dense(cl_or_reg, activation='softmax'))

        # initiate RMSprop optimizer
        adam = Adam(learning_rate=learn_rate_float, beta_1=beta_1_float, beta_2=beta_2_float, epsilon=epsilon_float)

        if cl_or_reg > 1:
            model.compile(loss='categorical_crossentropy',
                          optimizer=adam,
                          metrics=['accuracy'])
        else:
            model.compile(loss='mse',
                          optimizer=adam)

        # callback fun
        # 防止过拟合
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0,
                                       mode='min')  # patience 指的是最多几个没有改变然后进行改变
        # monitor: 需要监视的量;patience:  当 early stop 被激活(如发现loss相比上一个epoch训练没有下降)，则经过patience个epoch后停止训练

        filepath = "model.hdf5"
        model_filepath = os.path.join(Path, filepath)
        checkpoint = ModelCheckpoint(model_filepath, save_weights_only=False, monitor='val_loss', mode='min',
                                     save_best_only='True')
        callback_lists = [early_stopping, checkpoint]
        # checkpoint 路径可以唯一

        history = model.fit(x_train, y_train,
                            batch_size=int(batch_size_num),
                            epochs=int(epochs_num),
                            verbose=0,
                            validation_data=(x_via, y_via),
                            shuffle=True,
                            callbacks=callback_lists,
                            workers=-1,
                            use_multiprocessing=True)

        # Save model and weights
        # model_path = os.path.join(Path, model_name)
        # print(model_path)
        # model.save(model_path)
        # get the best model

        # best_model = load_model(model_path)
        # 验证集
        # Y_via_pre = model.predict(x_via, verbose=0)
        train_score = model.evaluate(x_train, y_train, verbose=0)
        train_score = train_score[0]
        if silent <= 0:
            print('train accuracy:', train_score)

        via_score = model.evaluate(x_via, y_via, verbose=0)
        via_acc = via_score[1]
        if silent <= 0:
            print('Validation accuracy:', via_acc)
        if silent <= -1:
            print(model.predict(x_via))
        model.save('model.h5')

        # 测试集
        test_score = model.evaluate(x_test, y_test, verbose=0)
        Y_test_pre = model.predict(x_test)
        tacc = test_score[1]
        if silent <= 0:
            print('Test accuracy:', tacc)

        if silent <= -1:
            print(Y_test_pre)

        year_profit, sharpe_ratio = back_test(Y_test_pre)

        if which_loss_func == 0:
            loss = -(via_acc + train_score) / (2 * (1 + abs(via_acc - train_score)))  # 让训练集和测试集更加接近，远离要么是过度拟合要么是欠拟合
        elif which_loss_func == 1:
            loss = -(np.sign(year_profit) * year_profit * sharpe_ratio)

        return {"loss": loss, 'year_profit': year_profit, 'sharpe_ratio': sharpe_ratio, "status": STATUS_OK,
                "model": model, 'test_score': test_score,
                'test_pred': Y_test_pre}

    trials = Trials()
    best = fmin(f_NN1, space, algo=tpe.suggest, max_evals=max_eval, trials=trials)  # 指定需要最小化的函数，搜索的空间，最大迭代次数。
    return best, trials
