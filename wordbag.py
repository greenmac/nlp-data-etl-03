from numpy import *
import numpy as np

# 創建數據集, 單詞列表postingList, 標籤列表classVec
def loadDataSet():
    # corpus參數樣利數據如下
    dataSet = []
    sports = ['姚明', '我來', '承擔', '連敗', '巨人', '宣言', '酷似', '當年', '麥蒂', '新浪', '體育訊', '北京' , '時間', '消息', '休斯敦', '紀事報', '專欄', '記者', '喬納森', '費根', '報導', '姚明', '渴望', '一場', '勝利', '當年', '隊友', '麥蒂', '慣用', '句式']
    entertainment = ['謝婷婷', '模特', '酬勞', '僅夠', '生活', '風光', '背後', '慘遭', '拖薪', '新浪', '娛樂', '金融', '海嘯', 'blog', '席捲', '全球', '模特兒', '酬勞', '被迫', '打折', '全職', 'Model', '謝婷婷', '業界', '工作量', '有增無減', '收入', '僅夠', '糊口', '拖薪']
    education = ['名師', '解讀', '四六級', '閱讀', '真題', '技巧', '考前', '複習', '重點', '歷年', '真題', '閱讀' , '聽力', '完形', '提升', '空間', '天中', '題為', '主導', '考過', '六級', '四級', '題為', '主導', '真題', '告訴', '方向', '會考', '題材', '包括']
    political = ['美國', '軍艦', '抵達', '越南', '聯合', '軍演', '中新社', '北京', '日電', '楊剛', '美國', '海軍', '第七', '艦隊', '三艘', '軍艦', '抵達', '越南', '峴港', '為期', '七天', '美越', '南海', '聯合', '軍事訓練', '拉開序幕', '美國', '海軍', '官方網站', '消息']
    dataSet.append(sports)
    dataSet.append(entertainment)
    dataSet.append(education)
    dataSet.append(political)

    classLab = ['體育', '娛樂', '教育', '政治']

    return dataSet, classLab

# 獲取所有詞的集合: 返回不重複元素的單詞列表
def createVocatList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # 運算符號 "|" 求集合聯集
        vocabSet = vocabSet | set(document)
    vocabList = list(vocabSet)
    return vocabList

# 特徵詞轉特徵向量:詞集模型
def setOfWords2Vec(vocabList, dataSet):
    setvec = []
    for document in dataSet:
        returnVec = [0]*len(vocabList)
        # 如果特徵詞在詞彙列表中, 修正為1, 反之為0
        for word in document:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
        # 追加所有文檔詞向量列表
        setvec.append(returnVec)
    return setvec

# 特徵詞轉特徵向量:詞袋模型
def bagOfWords2Vec(vocabList, dataSet):
    bagvec = []
    for document in dataSet:
        returnVec = [0]*len(vocabList)
        # 如果特徵詞在詞彙列表中, 修正為1, 反之為0
        for word in document:
            if word in vocabList:
                returnVec[vocabList.index(word)] += 1
        # 追加所有文檔詞向量列表
        bagvec.append(returnVec)
    return bagvec

# 詞集模型轉化為tf-idf計算
def TfIdf(bagvec):
    # 詞頻TF = 某個詞在文章中出現中出現次數/文字總次數
    tf = [word/sum(word) for word in bagvec]
    # 逆詞頻IDF = log(與料庫中文檔總數/(包含該詞文檔數+1))
    m = len(bagvec) # 語料文檔總數
    ndw = sum(mat(bagvec).T != 0, axis=1).T # 包含該詞文檔數
    idf = [log(m/(t+1)) for t in ndw]
    tfidf = tf*np.array(idf)
    return tfidf

if __name__ == "__main__":
    # 1 印出數據集和標籤集
    dataSet, classLab = loadDataSet()
    print('數據集:\n', mat(dataSet), '\n標籤集:\n', classLab)

    # 2 獲取所有文檔的詞集合
    vocabList = createVocatList(dataSet)
    print('\n詞彙列表:\n', vocabList)

    # 3 特徵詞轉特徵向量:詞集模型
    setvec = setOfWords2Vec(vocabList, dataSet)
    print('詞集模型:\n', mat(setvec))

    # 3 特徵詞轉特徵向量:詞袋模型
    bagvec = bagOfWords2Vec(vocabList, dataSet)
    print('詞袋模型:\n', mat(bagvec))

    # 4 計算TF-IDF
    tfidf = TfIdf(bagvec)
    print('TF-IDF:\n', tfidf)