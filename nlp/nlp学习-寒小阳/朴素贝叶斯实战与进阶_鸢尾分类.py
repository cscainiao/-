# -*- coding: utf-8 -*-
# @Time : 2019/10/10 0010 下午 4:15
# @Author : xing.xiong
# @Email : 980784252@qq.com
# @File : 朴素贝叶斯实战与进阶.py
# @Project : Data-Analysis-learning-notes

from sklearn import datasets
iris = datasets.load_iris()

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
right_num = (iris.target == y_pred).sum()
print("Total testing num :%d , naive bayes accuracy :%f" %(iris.data.shape[0], float(right_num)/iris.data.shape[0]))
# Total testing num :150 , naive bayes accuracy :0.960000