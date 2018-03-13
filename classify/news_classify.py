"""
Created on 2018/3/13
@Author: Jeff Yang
"""
import xlrd
import xlwt
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC


def read_data(file, sheet_index=0):
    """读取文件内容"""
    workbook = xlrd.open_workbook(file)
    sheet = workbook.sheet_by_index(sheet_index)
    data = []
    for i in range(0, sheet.nrows):
        data.append([x for x in sheet.row_values(i) if x.strip()])
    return data


def get_classified_sample():
    """返回手动分类的新闻"""
    data = read_data('test.xls')

    return {
        '经济': data[1] + data[14] + data[20],
        '社会': data[2] + data[3] + data[4] + data[9] + data[17] + data[18],
        '政法': data[5] + data[6] + data[7] + data[8] + data[11] + data[13] + data[15] + data[16] + data[19],
        '军事': data[10],
        '娱乐': data[12],
    }


def classify():
    """进行分类"""

    # 一共分成5类，并且类别的标识定为0，1，2，3，4
    category_ids = range(0, 5)
    category = {}
    category[0] = '经济'
    category[1] = '社会'
    category[2] = '政法'
    category[3] = '军事'
    category[4] = '娱乐'

    corpus = []  # 语料库
    classified_sample = get_classified_sample()
    for k, v in classified_sample.items():
        line = ' '.join(classified_sample[k])
        corpus.append(line)

    data = read_data('test.xls')

    # 把未分类的文章追加到语料库末尾行
    # 21开始是因为我手动分类了前20条
    for lst in data[21:]:
        line = ' '.join(lst)
        corpus.append(line)

    # 计算tf-idf
    vectorizer = CountVectorizer()
    csr_mat = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(csr_mat)
    y = np.array(category_ids)

    # 用前5行已标分类的数据做模型训练
    model = SVC()
    model.fit(tfidf[0:5], y)

    # 对5行以后未标注分类的数据做分类预测
    predicted = model.predict(tfidf[5:])

    # 结果
    # print(len(predicted))
    for i in range(len(predicted) - 1):
        print(corpus[i + 5], '============》', category[predicted[i]])


if __name__ == '__main__':
    classify()
