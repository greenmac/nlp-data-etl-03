import numpy as np
from numpy import *

def laodDataSet(filename, delim='\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # print(stringArr) # 二維數組
    dataArr = [list(map(float, line)) for line in stringArr]
    # print(dataArr)
    return mat(dataArr)

'''將NaN替換成平均值函數'''
def replaceNanWithMean(dataArr):
    numfeat = shape(dataArr)
    for i in range(numfeat[1]-1):
        # print(nonzero(~isnan(dataArr[:, i].A))[0], i)
        meanVal = mean(dataArr[nonzero(~isnan(dataArr[:, i].A))[0], i])
        dataArr[nonzero(isnan(dataArr[:, i].A))[0], i] = meanVal
    return dataArr

'''數值標準化:特徵值轉化為0-1之間:
newValue = (oldValue-min) / (max - min)
'''
def norm_dataset(dataset):
    # 參數0是取得列中的最小值, 而不是行中最小值
    minValue = dataset.min(0)
    maxValue = dataset.max(0)
    ranges = maxValue-minValue

    normdataset = zeros(shape(dataset))
    m = dataset.shape[0]

    # tile:複製同樣大小的矩陣
    molecular = dataset-tile(minValue, (m, 1)) # 分子
    denominator = tile(ranges, (m, 1)) # 分母
    
    normdataset = molecular/denominator # 標準化結果

    print('標準化結果:\n' + str(normdataset))
    return normdataset

if __name__ == "__main__":
    dataArr = laodDataSet(r'files\dataset.data', '    ')
    dataset = replaceNanWithMean(dataArr)
    # print(dataset)

    normdataset = norm_dataset(dataset[:, :-1])
