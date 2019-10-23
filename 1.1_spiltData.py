'''
特徵數據集:
特徵1: 年均投入時間(min)
特徵2:玩遊戲站時間百分比
特徵3:每天看綜藝的時間(h)

標籤集:
1.學習專注
2.學習正常
3.比較貪玩
'''
import os
from numpy import *

def file_matrix(filepath):
    file = open(filepath)
    arrayLines = file.readlines()

    returnMat =zeros((len(arrayLines), 3)) # 特徵數據集
    classLabelMat = [] # 標籤集
    index = 0
    for line in arrayLines:
        listFromLine = line.strip().split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelMat.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelMat

if __name__ == '__main__':
    filepath = os.path.abspath('files\dataset.txt')
    returnMat, classLabelMat = file_matrix(filepath)
    print('數據集:\n', returnMat, '\n標籤集:\n', classLabelMat)