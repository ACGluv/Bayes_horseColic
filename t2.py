# -*- coding: UTF-8 -*-
import numpy as np
from functools import reduce


"""
函数说明:创建实验样本

Parameters:
	无
Returns:
	postingList - 实验样本切分的词条
	classVec - 类别标签向量
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-11
"""


def loadDataSet(file):
    postingList = []  # 创建数据列表
    classVec = []  # 创建标签列表
    fr = open(file)  # 打开文件
    for line in fr.readlines():  # 逐行读取
        lineArr = line.strip().split()  # 去回车，放入列表
        # print('lineArr',type(lineArr))
        # print('lineArr[0]', type(lineArr[0]))
        # print(lineArr)
        # print(list(map(int, lineArr[0:1])))

        if float(lineArr[0]) - 1 <= 0.00001:
            one = 'A'
        else:
            one = 'a'
        if float(lineArr[1]) - 1 <= 0.00001:
            two = 'B'
        else:
            two = 'b'

        '''
        if float(lineArr[2]) - 37.8 < 0.5:
            three = 'C'
        elif float(lineArr[2]) - 1 <= 0.00001:
            three = '3'
        else:
            three = 'c'
        '''


        if float(lineArr[5]) - 2 < 0.0001:
            four = 'D'
        else:
            four = 'd'
        if float(lineArr[6]) - 2 < 0.0001:
            five = 'E'
        else:
            five = 'e'

        if float(lineArr[8]) - 1 < 0.0001:
            six = 'F'
        elif float(lineArr[8]) - 3 <= 0.00001:
            six = '6'
        elif float(lineArr[8]) - 0 <= 0.00001:
            six = 'Ff'
        else:
            six = 'f'

        if float(lineArr[19]) - 0 < 0.0001:
            seven = 'Gg'
        elif float(lineArr[19]) - 1 < 0.0001:
            seven = 'g'
        elif float(lineArr[19]) - 2 < 0.0001:
            seven = 'G'
        else:
            seven = '7'

        '''
        if float(lineArr[13]) - 0 < 0.0001:
            eight = 'Hh'
        elif float(lineArr[13]) - 1 < 0.0001:
            eight = 'H'
        elif float(lineArr[13]) - 2 < 0.0001:
            eight = 'h'
        else:
            eight = '8'
        '''

        postingList.append([one, two, four, five, six,seven])  # 添加数据

        classVec.append(float(lineArr[21]))  # 添加标签
    fr.close()  # 关闭文件
    classVec = list(map(int, classVec))  # float转为int型
    return postingList, classVec

"""
函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

Parameters:
	dataSet - 整理的样本数据集
Returns:
	vocabSet - 返回不重复的词条列表，也就是词汇表
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-11
"""


def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)


"""
函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

Parameters:
	vocabList - createVocabList返回的列表
	inputSet - 切分的词条列表
Returns:
	returnVec - 文档向量,词集模型
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-11
"""


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  # 返回文档向量


"""
函数说明:根据vocabList词汇表，构建词袋模型

Parameters:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表
Returns:
    returnVec - 文档向量,词袋模型
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-08-14
"""


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
    return returnVec


"""
函数说明:朴素贝叶斯分类器训练函数

Parameters:
	trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
	trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
	p0Vect - 侮辱类的条件概率数组
	p1Vect - 非侮辱类的条件概率数组
	pAbusive - 文档属于侮辱类的概率
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-12
"""


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 计算训练的文档数目
    numWords = len(trainMatrix[0])  # 计算每篇文档的词条数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于侮辱类的概率
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)  # 创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0;
    p1Denom = 2.0  # 分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)  # 取对数，防止下溢出
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive  # 返回属于侮辱类的条件概率数


"""
函数说明:朴素贝叶斯分类器分类函数

Parameters:
	vec2Classify - 待分类的词条数组
	p0Vec - 侮辱类的条件概率数组
	p1Vec -非侮辱类的条件概率数组
	pClass1 - 文档属于侮辱类的概率
Returns:
	0 - 属于非侮辱类
	1 - 属于侮辱类
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-12
"""


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


"""
函数说明:测试朴素贝叶斯分类器

Parameters:
	无
Returns:
	无
Author:
	Jack Cui
Blog:
	http://blog.csdn.net/c406495762
Modify:
	2017-08-12
"""


def testingNB():
    listOPosts, listClasses = loadDataSet('horseColicTraining.txt')  # 创建实验样本
    myVocabList = createVocabList(listOPosts)  # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练朴素贝叶斯分类器



    lo,lc=loadDataSet('horseColicTest.txt')

    numTrainDocs = len(lo)  # 计算训练的文档数目
    numWords = len(lo[0])  # 计算每篇文档的词条数
    i=0
    errorCount = 0  # 错误分类计数
    for i in range(numTrainDocs):
        print(lo[i])
        thisDoc = np.array(setOfWords2Vec(myVocabList, lo[i]))  # 测试样本向量化
        #testEntry = ['a', 'B', 'C', 'D', 'E', 'f']  # 测试样本1
        if classifyNB(thisDoc, p0V, p1V, pAb) !=lc[i]:
            errorCount += 1  # 错误计数加1
            print("分类错误的测试集：", lo[i])
    print('错误率：%.2f%%' % (float(errorCount) / numTrainDocs * 100))

if __name__ == '__main__':
    testingNB()
