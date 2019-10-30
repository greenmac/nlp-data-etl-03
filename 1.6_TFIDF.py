# 利用sklearn計算tfidf值特徵
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import re, jieba, sys
from zhtools.langconv import *

# 讀取文本信息
def readFile(path):
    str_doc = ''
    with open(path, 'r', encoding='utf-8') as f:
        str_doc = f.read()
    return str_doc

# 正則對字符串清洗
def textParse(str_doc):
    # 去掉字符
    str_doc = re.sub('\u3000', '', str_doc)
    # 去除空格
    str_doc = re.sub('\s+', ' ', str_doc)
    # 去除換行符
    str_doc = str_doc.replace('\n', ' ')
    # 正則過濾掉特殊符號, 標點, 英文, 數字等
    r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+'
    str_doc = re.sub(r1, ' ', str_doc)
    return str_doc

# 創建停用詞列表
def get_stop_words(path=r'dataSet/StopWord/NLPIR_stopwords.txt'):
    file = open(path, 'r', encoding='utf-8').read().split('\n')
    return set(file)

# 去掉一些停用詞和數字
def rm_tokens(words, stwlist):
    word_list = list(words)
    stop_words = stwlist
    for i in range(word_list.__len__())[::-1]:
        if word_list[i] in stop_words: # 去除停用詞
            word_list.pop(i)
        elif word_list[i].isdigit(): # 去除數字
            word_list.pop(i)
        elif len(word_list[i]) == 1 : # 去除單個字符
            word_list.pop(i)
        elif word_list[i] == ' ' : # 去除空字符
            word_list.pop(i)
    return word_list

# 利用jieba對文本進行分詞, 返回切詞後的list
def seg_doc(str_doc):
    # 1.正則處理原文本
    sent_list = str_doc.split('\n')
    sent_list = map(textParse, sent_list)
    # 2.獲取停用詞
    stwlist = get_stop_words()
    # 3.分詞並去除停用詞
    word_2dlist = [rm_tokens(jieba.cut(part, cut_all=False), stwlist) for part in sent_list]
    # 4.合併列表
    word_list = sum(word_2dlist, [])
    return word_list

# 利用sklearn計算tfidf值特徵
def sklearn_tfidf_feature(corpus=None):
    # 構建詞彙表
    vectorizer = CountVectorizer()
    # print(vectorizer)
    # 該類下統計每個詞 tfidf權值
    transformer = TfidfTransformer()
    # fit_transform 是計算tf-idf, fit_transform()先凝和數據，再標準化
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 獲取詞袋模型中的所有詞語
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    # 遍歷文本, 讀取每一類的特徵詞權重
    for i in range(len(weight)):
        print('------這裡輸出第', i, '類文本的詞語tfidf權重值')
        for j in range(len(word)):
            print(word[j], weight[i][j])

if __name__ == '__main__':
    # corpus參數樣列數據如下
    corpus = [
            "我 來到 成都 成都 春熙路 很開心",
            "今天 在 寬窄巷子 耍 了 一天",
            "成都 整體 來說 還是 挺 安逸 的",
            "成都 的 美食 真 舒適 慘 了"]
    sklearn_tfidf_feature(corpus)