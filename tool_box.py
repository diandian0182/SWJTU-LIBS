import os
from natsort import natsorted
import numpy as np
from scipy import signal
# from pylab import *
from sklearn import preprocessing
from scipy.signal import find_peaks
from lxml import etree

class PeakSearching():
    '''
    函数名: Sliding_Window
    函数简介: 滑动窗口比较法
    参数1:wavelength            原始波长
    参数2:intensity             原始强度
    参数3:find_wave             需要查找的峰对应的波长
    参数4:compare_points        窗口大小
    返回值1: return_wave        找到峰值对应的波长
    返回值2: return_intensity   在指定的波长位置通过滑动窗口找到的峰强
    '''
    def Sliding_Window(wavelength,intensity,find_waves,compare_points):
        peaks = []
        for find_wave in find_waves:
            # 计算每个元素与目标值的绝对差, 找到最小差的索引
            closest_index = np.argmin(np.abs(wavelength - find_wave))
            # 寻峰 x点比较
            window = intensity[closest_index - compare_points : closest_index + compare_points]
            # 最接近的值
            exc_inte = intensity[closest_index]
            count = 0
            while max(window) != exc_inte:
                if count < 20 : # 限制滑动窗口的次数
                    if max(window) > exc_inte:
                        for i in range(len(window)):
                            if window[i] == max(window):
                                closest_index = closest_index + i - compare_points
                            exc_inte = intensity[closest_index] # 重新计算
                    window = intensity[closest_index - compare_points : closest_index + compare_points]
                    count = count + 1
                else:
                    break
            peaks.append([wavelength[closest_index],intensity[closest_index]])
        np_peaks = np.array(peaks).T
        return np_peaks[0], np_peaks[1]

    '''
    函数名: Sliding_Window_all
    函数简介: 滑动窗口比较法(总)
    参数1:wavelength            一列原始波长
    参数2:intensity             多列原始强度
    参数3:find_wave             需要查找的峰对应的波长
    参数4:compare_points        窗口大小
    返回值: return_peaks        第一列为波长 后续为对应的多个峰强 具体波长由最后一列输入决定
    '''        
    def Sliding_Window_all(wavelength,intensity,find_waves,compare_points):
        result_peaks = np.array([])
        peaks = []
        for find_wave in find_waves:
            # 计算每个元素与目标值的绝对差, 找到最小差的索引
            closest_index = np.argmin(np.abs(wavelength - find_wave))
            columns = intensity.shape[1]
            for column in range(columns):
                each_intensity = intensity[column]
                # 寻峰 x点比较
                window = each_intensity[closest_index - compare_points : closest_index + compare_points]
                # 最接近的值
                exc_inte = each_intensity[closest_index]
                count = 0
                while max(window) != exc_inte:
                    if count < 20 : # 限制滑动窗口的次数
                        if max(window) > exc_inte:
                            for i in range(len(window)):
                                if window[i] == max(window):
                                    closest_index = closest_index + i - compare_points
                                exc_inte = each_intensity[closest_index] # 重新计算
                        window = each_intensity[closest_index - compare_points : closest_index + compare_points]
                        count = count + 1
                    else:
                        break
                peaks.append(each_intensity[closest_index])
            peaks.insert(0, wavelength[closest_index])
            result_peaks = np.vstack([result_peaks, peaks]) if result_peaks.size else np.array([peaks])
            peaks = []
        return result_peaks
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
    
    def normalization(indentsity,rate):
        '''
        归一化
        '''
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
    
class data_processing():
    '''
    函数名: mad_based_outlier
    函数简介: 绝对中位差（根据阈值大小删除异常数据）
    参数1:intens            一组一维numpy数据
    参数2:thresh             阈值
    返回值: 返回异常值的索引
    '''       
    def mad_based_outlier(intens, thresh=1.5): #这里设定的阈值1.5
        if len(intens.shape) == 1:
            intens = intens[:,None]
        median = np.median(intens, axis=0)
        diff = np.sum((intens - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return np.where(modified_z_score > thresh)[0]
