# -*- coding: utf-8 -*-
import codecs
from sklearn.model_selection import train_test_split


labels = ['unacc', 'acc', 'good', 'vgood']
def readDate(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        dataSet = []

        for line in f.readlines():
            line = line.strip().split(',')
            dataSet.append(line)
    labels = ['unacc', 'acc', 'good', 'vgood']
    return dataSet, labels


def randSplit(dataSet, rate):
    train, test = train_test_split(dataSet, test_size=rate)
    return train, test


def valueLabel(train, num, label, value, n):  # 计算某个标签下，该属性的概率
    value_label = 0
    for line in train:
        if line[n] == value and line[-1] == label:
            value_label += 1
    x = labels.index(label)
    value_label = value_label / num[x]

    return value_label


def Statistics(train, tag, n):  # 计算属性的概率
    tag_value = 0
    b = [i[n] for i in train]
    for j in b:
        if j == tag:
            tag_value += 1
    return tag_value/len(train)


def classify(data, labels, train, num):
    a = 1   # 用于计算概率
    i = 0   # 用于取得label下标
    max = 0     # 记录最大的概率
    x = 0   # 记录最大概率对应的label下标
    precision = 0
    for line in data:
        while i < 4:
            n = 0   # 属性下标
            for value in line[:-1]:
                a *= valueLabel(train, num, labels[i], value, n)
                a = a/(Statistics(train, value, n))
                n += 1
                if max < a:
                    max = a
                    x = i
            i += 1
        if labels[x] == line[-1]:
            precision += 1
    return precision/len(data)


num = [1210, 384, 69, 65]
prob = [0.70023, 0.22222, 0.3993, 0.3762]
dataSet, labels = readDate('car.data')
train, test = randSplit(dataSet, 0.2)
print("测试集准确率为：", classify(test, labels, train, num))
