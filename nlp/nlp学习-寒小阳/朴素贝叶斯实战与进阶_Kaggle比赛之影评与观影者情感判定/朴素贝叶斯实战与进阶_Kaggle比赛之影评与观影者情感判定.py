# -*- coding: utf-8 -*-
# @Time : 2019/10/15 0015 上午 10:38
# @Author : xing.xiong
# @Email : 980784252@qq.com
# @File : 朴素贝叶斯实战与进阶_Kaggle比赛之影评与观影者情感判定.py
# @Project : Data-Analysis-learning-notes

import re  # 正则表达式
from bs4 import BeautifulSoup  # html标签处理
import pandas as pd


def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    '''
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review).get_text()
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 小写化所有的词，并转成词list
    words = review_text.lower().split()
    # 返回words
    return words


# 使用pandas读入训练和测试csv文件
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t",quoting=3)
test = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)
# 取出情感标签，positive/褒 或者 negative/贬
y_train = train['sentiment']
# 将训练和测试数据都转成词list
train_data = []
for i in range(0, len(train['review'])):
    train_data.append(" ".join(review_to_wordlist(train['review'][i])))
test_data = []
for i in range(0, len(test['review'])):
    test_data.append(" ".join(review_to_wordlist(test['review'][i])))


from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
# 初始化TFIV对象，去停用词，加2元语言模型
tfv = TFIV(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = 'english')
# 合并训练和测试集以便进行TFIDF向量化操作
X_all = train_data + test_data
len_train = len(train_data)

# 这一步有点慢，去喝杯茶刷会儿微博知乎歇会儿...
tfv.fit(X_all)
X_all = tfv.transform(X_all)
# 恢复成训练集和测试集部分
X = X_all[:len_train]
X_test = X_all[len_train:]


# 多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.externals import joblib

model_NB = MNB()
model_NB.fit(X, y_train) #特征数据直接灌进来
MNB(alpha=1.0, class_prior=None, fit_prior=True)
# # joblib.dump(model_NB, 'model_NB.m')
# X_p = model_NB.predict(X_test)
# print(X_p)
# test['sentiment'] = X_p
# test.to_csv('test_predict.csv')
from sklearn.model_selection import cross_val_score
import numpy as np

print("多项式贝叶斯分类器20折交叉验证得分: ", np.mean(cross_val_score(model_NB, X, y_train, cv=10, scoring='roc_auc')))
#多项式贝叶斯分类器20折交叉验证得分: 0.950837239



#折腾一下逻辑回归，恩
from sklearn.linear_model import LogisticRegression as LR, LogisticRegression
from sklearn.model_selection import GridSearchCV

# 设定grid search的参数
grid_values = {'C':[30]}
# 设定打分为roc_auc
model_LR = GridSearchCV(LR(penalty='L2', dual = True, random_state=0), grid_values, scoring='roc_auc', cv=20)
# 数据灌进来
model_LR.fit(X,y_train)
# 20折交叉验证，开始漫长的等待...
GridSearchCV(cv=10, estimator=LogisticRegression(C=1.0, class_weight=None, dual=True,
             fit_intercept=True, intercept_scaling=1, penalty='L2', random_state=0, tol=0.0001),
         iid=True,  n_jobs=1,
        param_grid={'C': [30]}, pre_dispatch='2*n_jobs', refit=True,
         scoring='roc_auc', verbose=0)
#输出结果
print(model_LR.best_score_)
