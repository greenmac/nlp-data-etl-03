import os, time
from StopWords import *
from wordbag import *

# **********高效讀取文件**********
class LoadFolders(object):
    def __init__(self, par_path):
        self.par_path = par_path
    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abdpath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abdpath):
                yield file_abdpath # return

class Loadfiles(object):
    def __init__(self, par_path):
        self.par_path = par_path
    def __iter__(self):
        folders = LoadFolders(self.par_path)
        for folder in folders: # level directory
            catg = folder.split(os.sep)[-1]
            for file in os.listdir(folder): # secondary directory
                file_path = os.path.join(folder, file)
                # 文件具體操作
                if os.path.isfile(file_path):
                    this_file = open(file_path, 'rb')
                    content = this_file.read().decode('utf8')
                yield catg, content
                this_file.close()


if __name__ == "__main__":
    start = time.time()

    filepath = os.path.abspath(r'dataSet/CSCMNews')
    files = Loadfiles(filepath)
    n = 5 # 表示抽樣率
    for i, msg in enumerate(files):
        if i % n == 0:
            catg = msg[0]
            content = msg[1]
            zh_content = Converter('zh-hant').convert(content)
            # 每個文檔TFIDF向量化
            word_list = seg_doc(zh_content)
            vocabList = createVocatList(word_list)
            bagvec = bagOfWords2Vec(vocabList, word_list)
            tfidf = TfIdf(bagvec)
            if int(i/n)%1000 == 0:
                print('{t}****{i} \t docs has been dealed'.format(i=i, t=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())), '\n', catg, '\t', tfidf)

    end = time.time()
    print('Total spent times:%.2f' % (end-start))