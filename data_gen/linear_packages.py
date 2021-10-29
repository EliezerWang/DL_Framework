from PyEMD import EEMD
import scipy
from scipy.interpolate import make_interp_spline
from data_gen.filter_var import *
import numpy as np
import cv2
import pandas as pd


# 滤波器只针对一天的数据
# 输入前请转成np.array

def envelope_extraction(signal, which):
    def general_equation(first_x, first_y, second_x, second_y):
        # 斜截式 y = kx + b
        A = second_y - first_y
        B = first_x - second_x
        C = second_x * first_y - first_x * second_y
        k = -1 * A / B
        b = -1 * C / B
        return k, b

    signal.reset_index(drop=True, inplace=True)
    s = signal.astype(float)
    q_u = np.zeros(s.shape)
    q_l = np.zeros(s.shape)

    # 在插值值前加上第一个值。这将强制模型对上包络和下包络模型使用相同的起点。
    # Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0, ]  # 上包络的x序列
    u_y = [s[0], ]  # 上包络的y序列

    l_x = [0, ]  # 下包络的x序列
    l_y = [s[0], ]  # 下包络的y序列

    # 检测波峰和波谷，并分别标记它们在u_x,u_y,l_x,l_中的位置。
    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1, len(s) - 1):
        if (sign(s[k] - s[k - 1]) == 1) and (sign(s[k] - s[k + 1]) == 1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k] - s[k - 1]) == -1) and ((sign(s[k] - s[k + 1])) == -1):
            l_x.append(k)
            l_y.append(s[k])

    u_x.append(len(s) - 1)  # 上包络与原始数据切点x
    u_y.append(s.iloc[-1])  # 对应的值

    l_x.append(len(s) - 1)  # 下包络与原始数据切点x
    l_y.append(s.iloc[-1])  # 对应的值

    # u_x,l_y是不连续的，以下代码把包络转为和输入数据相同大小的数组[便于后续处理，如滤波]
    upper_envelope_y = np.zeros(len(signal))
    lower_envelope_y = np.zeros(len(signal))

    upper_envelope_y[0] = u_y[0]  # 边界值处理
    upper_envelope_y[-1] = u_y[-1]
    lower_envelope_y[0] = l_y[0]  # 边界值处理
    lower_envelope_y[-1] = l_y[-1]

    # 上包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(u_x[0], u_y[0], u_x[1], u_y[1])  # 初始的k,b
    for e in range(1, len(upper_envelope_y) - 1):

        if e not in u_x:
            v = k * e + b
            upper_envelope_y[e] = v
        else:
            idx = u_x.index(e)
            upper_envelope_y[e] = u_y[idx]
            last_idx = u_x.index(e)
            next_idx = u_x.index(e) + 1
            # 求连续两个点之间的直线方程
            k, b = general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])

            # 下包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(l_x[0], l_y[0], l_x[1], l_y[1])  # 初始的k,b
    for e in range(1, len(lower_envelope_y) - 1):

        if e not in l_x:
            v = k * e + b
            lower_envelope_y[e] = v
        else:
            idx = l_x.index(e)
            lower_envelope_y[e] = l_y[idx]
            last_idx = l_x.index(e)
            next_idx = l_x.index(e) + 1
            # 求连续两个切点之间的直线方程
            k, b = general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

            # 也可以使用三次样条进行拟合
    # u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    # l_p = interp1d(l_x,l_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    # for k in range(0,len(s)):
    #   q_u[k] = u_p(k)
    #   q_l[k] = l_p(k)
    if which == 'low':
        return pd.Series(lower_envelope_y)
    else:
        return pd.Series(upper_envelope_y)


def sav_filter(data):  # 输出单个
    feature = scipy.signal.savgol_filter(data, 53, 3, mode='nearest')
    return pd.Series(feature)


def inter_plot(data):  # 注意：插值法会增大数据！！！！！ 倍数可以在filter_lib调整    输出单个
    long = len(data)
    x = np.arange(0, long, 1)
    x_smooth = np.linspace(x.min(), x.max(), long * inter_plot_times)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
    feature = make_interp_spline(x, data)(x_smooth)
    return pd.Series(feature)


def moving_avg(data):  # 输出单个
    window = np.ones(int(windowsize)) / float(windowsize)
    feature = np.convolve(data, window, 'same')
    return pd.Series(feature)


def low_pass_filter(data):  # 低通滤波器 输出单个
    b, a = scipy.signal.butter(8, low_pass_rate, 'lowpass')
    feature = scipy.signal.filtfilt(b, a, data)
    return pd.Series(feature)


def high_pass_filter(data):  # 高通滤波器 输出单个
    b, a = scipy.signal.butter(8, high_pass_rate, 'highpass')
    feature = scipy.signal.filtfilt(b, a, data)
    return pd.Series(feature)


def band_pass_filter(data):  # 通带滤波器 输出单个
    b, a = scipy.signal.butter(8, band_pass_rate, 'bandpass')
    feature = scipy.signal.filtfilt(b, a, data)
    return pd.Series(feature)


def band_stop_filter(data):  # 阻带滤波器 输出单个
    b, a = scipy.signal.butter(8, band_stop_rate, 'bandstop')
    feature = scipy.signal.filtfilt(b, a, data)
    return pd.Series(feature)


# 下面使用cv2库 全部支持二维输入和输出

def gaussian_filter(data):  # 支持单维输入 也支持二维输入
    feature = cv2.GaussianBlur(data.values, gaussian_kernel_size, sigma_x)
    return pd.Series(feature)


def box_filter(data):  # 支持单维输入 也支持二维输入
    feature = cv2.boxFilter(src=data, ddepth=-1, ksize=box_filter_size, normalize=True)
    return pd.Series(feature)


def mean_filter(data):  # 支持单维输入 也支持二维输入
    feature = cv2.blur(src=data, ksize=mean_filter_size)
    return pd.Series(feature)


def median_filter(data):  # 支持任意维度输入 但需要匹配
    feature = scipy.signal.medfilt(data, kernel_size=median_filter_size)
    return pd.Series(feature)


def EMD_filter(data):
    emd = EEMD()
    x = data.values
    y = np.arange(0, len(x), 1, dtype=int)
    imf = emd.emd(x, y)
    feature = imf[imf_num:, :].sum(axis=0)
    return pd.Series(feature)


def trans_EMD(data):
    emd = EEMD()
    temp_data = data.values
    x = temp_data.iloc[:, 0].values
    y = np.arange(0, len(x), 1, dtype=int)
    IMF = pd.DataFrame(emd.emd(x, y)).T
    name_list = []
    for z in range(0, IMF.shape[1]):
        name_list.append('第一个分解' + str(z))
    IMF.columns = name_list
    for i in range(1, temp_data.shape[1]):
        x = temp_data.iloc[:, i].values
        y = np.arange(0, len(x), 1, dtype=int)
        temp_IMF = pd.DataFrame(emd.emd(x, y)).T
        name_list = []
        for z in range(0, temp_IMF.shape[1]):
            name_list.append('第' + str(i + 1) + '个分解' + str(z))
        temp_IMF.columns = name_list
        IMF = pd.concat(objs=[IMF, temp_IMF], axis=1)
    IMF = pd.DataFrame(IMF)
    return IMF


def ewm_process(data):
    feature = data.ewm(span=30).mean()
    return pd.Series(feature)

import pandas as pd
data = pd.read_csv('E:\data\hs300_1分钟\SH#000300.csv',skiprows=2, skipfooter=1,engine='python'
                            ,names =['date','time','开盘','最高','最低','收盘','成交量','成交额'])
data.开盘.plot()