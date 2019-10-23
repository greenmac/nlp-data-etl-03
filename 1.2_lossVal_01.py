import numpy as np
from numpy import *

'''加載數據集'''
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

if __name__ == "__main__":
    # 加載數據集
    dataArr = laodDataSet(r'files\dataset.data', '    ')
    # print(dataArr)

    # 均職填補缺失值
    dataMat = replaceNanWithMean(dataArr)
    print(dataMat)