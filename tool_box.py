import os
from natsort import natsorted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy import signal
from pylab import *
from sklearn import preprocessing
from scipy.signal import find_peaks
from lxml import etree


# 导入数据
class DataImport():
    def NOVA_Raman(xml_path):
        '''
        ElementTree 对象是 ElementTree 库中的核心对象之一,它代表了整个XML文档的树形结构。
        XML文档由元素(element)组成,这些元素形成了层次结构,而 ElementTree 对象就是对这个结构的抽象表示。
        在使用 etree.parse(xml_path) 时,该函数会返回一个 ElementTree 对象,其中包含了整个XML文档的信息。
        ElementTree 对象提供了一组方法和属性,使得你能够轻松地遍历、查询和修改 XML 文档。
        一些常见的 ElementTree 方法和属性包括：
        getroot(): 返回XML文档的根元素。
        find() 和 findall(): 用于查询特定元素或元素集。
        iter(): 用于迭代文档中的所有元素。
        write(): 将修改后的 XML 数据写回到文件中。
        '''
        tree = etree.parse(xml_path) 
        root = tree.getroot() # 返回XML文档的根元素。
        raman_shift_rows  = root.xpath('//Column[@ID="0"]/Row[@ID]')  # 使用 XPath 表达式查询第一个 Column 的所有带有 ID 属性的 Row 元素
        intensity_rows  = root.xpath('//Column[@ID="1"]/Row[@ID]') # 使用 XPath 表达式查询第二个 Column 的所有带有 ID 属性的 Row 元素
        data = np.zeros((2,len(intensity_rows))) # 创建二维空数组,一行拉曼位移,一行强度
        count = 0
        for raman_shift,intensity in zip(raman_shift_rows,intensity_rows):
            data[0][count] = raman_shift.text
            data[1][count] = intensity.text
            count +=1
        return data




# 基线矫正
class Baseline():
    # SNIP 基线校准
    def SNIP(origin_intensities, niter):
        origin_intensities[origin_intensities < 0] = 0 # 将负值置0
        LLS = np.log(np.log(np.sqrt(origin_intensities + 1)+1)+1) # LLS
        for count in np.arange(1, niter+1):
            a = LLS
            b = (np.roll(LLS, -count) + np.roll(LLS, +count))/2
            reverse_LLS = np.minimum(a, b)
        baseline_intensity = (np.exp(np.exp(reverse_LLS)-1)-1)**2 - 1 # reverseLLS
        after_baselinecorrection_intensity = origin_intensities - baseline_intensity      
        # return after_baselinecorrection_intensity, baseline_intensity
        return after_baselinecorrection_intensity
    

# 滤波
class Filter():
    # # Savitzky-Golay平滑滤波
    def Savitzky_Golay(intensity,window,polynomial):
        return signal.savgol_filter(intensity, window, polynomial)

    # 滑动平均值
    def Moving_Ave(indensity, window):
        window = np.ones(int(window)) / float(window)
        return np.convolve(indensity, window, 'same')
    


# 标准化
class Standardization():
    
    def normalization(indentsity,rate): # 归一化
        min_intensity = np.min(indentsity)
        max_intensity = np.max(indentsity)
        return ((indentsity - min_intensity) / (max_intensity- min_intensity))*rate

    def standardization(indentsity): # Z-Score标准化
        # 建立standardscaler对象
        zscore = preprocessing.StandardScaler()
        # 将一维Numpy数组转换为二维数组,然后进行标准化处理
        indentsity = indentsity.reshape(-1, 1)
        indentsity_zs = zscore.fit_transform(indentsity)
        # 将标准化后的数据重新转换为一维Numpy数组
        indentsity_zs = indentsity_zs.ravel()
        return indentsity_zs
        
    def MSC_MultiplicativeSignalCorrection(mult_indentsitys): # 多元信号矫正
        label_row = mult_indentsitys[:1,] # 标签
        origin_intensitys = mult_indentsitys[1:,] # 删除第一行数据标签
        origin_intensitys = origin_intensitys.T # 转置,将每一行代表一个光谱,不同列代表不同波长
        # print(origin_intensitys)
        # Step 1: 计算每个样本的均值
        sample_means = np.mean(origin_intensitys, axis=1)  # 沿着每行计算均值,每一个光谱的平均值
        # print(sample_means)
        # Step 2: 计算全局均值光谱
        global_mean_spectrum = np.mean(origin_intensitys, axis=0)  # 沿着每列计算均值, 每一个波长的平均值
        # print(global_mean_spectrum)
        # Step 3: 进行MSC校正([:, np.newaxis]将一维数组变成一个列向量)
        MSC_intensitys= (origin_intensitys / sample_means[:, np.newaxis]) * global_mean_spectrum
        # print(MSC_intensitys)
        MSC_intensitys = np.insert(MSC_intensitys.T, 0, label_row, axis=0) # 矩阵转置后在第一行插入标签
        return MSC_intensitys


# import numpy as np

# # 假设你有原始光谱数据 X,每行代表一个样本,每列代表不同的波长
# # 假设 X 是一个二维NumPy数组

# # Step 1: 计算每个样本的均值
# sample_means = np.mean(X, axis=1)  # 沿着每行计算均值

# # Step 2: 计算全局均值光谱
# global_mean_spectrum = np.mean(X, axis=0)  # 沿着每列计算均值

# # Step 3: 进行MSC校正
# X_msc = (X - sample_means[:, np.newaxis]) + global_mean_spectrum

# # 现在,X_msc 包含了MSC校正后的光谱数据

# # 可以继续使用 X_msc 进行进一步的数据分析或建模
