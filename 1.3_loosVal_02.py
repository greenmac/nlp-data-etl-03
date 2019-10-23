import pandas as pd
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


dataMat = laodDataSet(r'files\dataset.data', '    ')
df = pd.DataFrame(dataMat)

# 1 重構矩陣
df = df.reindex(range(dataMat.shape[0]+5))
# print(df)

# 3 均值填充法:NaN視為0, 若數據是NaN和是NaN
lossVs = [df[col].mean() for col in range(dataMat.shape[1])]
# print(lossVs)
lists = [list(df[i].fillna(lossVs[i])) for i in range(len(lossVs))]
# print(mat(lists).T) # T是轉置矩陣

# 4 其他缺失值處理方法
# 4.1 用標量值替換NaN
# print(df.fillna(0))

# 4.2 前進和後退:pad/fill 和 bfill/backfill
# print(df.fillna(method='pad'))
# print(df.fillna(method='backfill'))

# 4.3 丟失缺少的值:axis=0在行上應用, axis=1在列上應用
# print(df.dropna(axis=0))

# 4.4 忽略無效值法
print("df.dropna():\n{}\n".format(df.dropna()))