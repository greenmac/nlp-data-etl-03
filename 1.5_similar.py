from numpy import linalg as la
from numpy import *

'''
列向量為商品的類別
行向量為用戶對商品的評分
可以根據相似度推薦商品
'''
def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 3, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]

'''基於歐式距離相似度計算
假定inA和inB 都是列向量
norm:范氏計算, 默認是2范數, 即:sqrt(a^2+b^2+...)
相似度=1/(1+距離), 相似度介於0-1之間
'''
def ecludSim(inA, inB):
    return 1.0/(1.0+la.norm(inA-inB)) # +1.0是因為分母不能為0

'''計算餘弦相似度
如果夾角為90度相似度為0, 兩個像量的方向相同, 相似度為1.0
餘弦值取值-1到1之間, 標準化到0與1之間即:相似度=0.5 + 0.5*cosθ
餘弦相似度cosθ=(A*B/|A|*|B|)
'''
def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

'''皮爾遜相關係數
範圍[-1, 1], 標準化[0, 1]即0.5+0.5*values
'''
def pearSim(inA, inB):
    return 0.5+0.5*corrcoef(inA, inB, rowvar=0)[0][1]



if __name__ == "__main__":
    myMat = mat(loadExData())
    # 計算歐式距離, 比較第一列商品A和第四列商品D的相似率
    print('歐式距離計算相似度:\n',ecludSim(myMat[:, 0], myMat[:, 3]))
    print('餘弦相似度:\n', cosSim(myMat[:, 0], myMat[:, 3]))
    print('皮爾遜距離計算相似度:\n', pearSim(myMat[:, 0], myMat[:, 3]))
