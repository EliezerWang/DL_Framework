# 滤波器参数
inter_plot_times = 100  # 插值法放大几倍   对应函数：inter_plot

windowsize = 50  # 移动平均的日期数量   对应函数：moving_avg

low_pass_rate = 0.8  # 越低滤波效果越强 损失的越多   对应函数：low_pass_filter

high_pass_rate = 0.2  # 越高滤波效果越强 损失的越多   对应函数：high_pass_filter

band_pass_rate = [0.2, 0.8]  # 越窄滤波效果越强 损失的越多   对应函数：band_pass_filter

band_stop_rate = [0.2, 0.8]  # 越宽滤波效果越强 损失的越多   对应函数：band_stop_filter

gaussian_kernel_size = (3, 3)  # 维度不需要匹配 高斯核的大小 越大滤波效果越好 损失越多 只能取正奇数   对应函数：gaussian_filter
sigma_x = 0  # 越高滤波效果越好 但损失的越多   对应函数：gaussian_filter

box_filter_size = (5, 5)  # 维度不需要匹配 核的大小 越大滤波效果越好 损失越多 只能取正奇数   对应函数：box_filter

mean_filter_size = (5, 5)  # 维度不需要匹配 核的大小 越大滤波效果越好 损失越多 只能取正奇数   对应函数：mean_filter

median_filter_size = (3, 3, 3)  # 维度需要匹配 核的大小 越大滤波效果越好 损失越多 只能取正奇数   对应函数：median_filter

imf_num = 2  # emd从第几个之后开始相加（因为EMD是分解波的一种工具，将数据分解之后会得到不同的imf波，这些波相加后等于原波。一般默认为有一些噪音的存在，所以会筛选掉前几个‘噪音波’再相加得到没有噪音的数据）   对应函数：EMD_filter
