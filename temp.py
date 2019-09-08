# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from pandas import DataFrame
import json

data1 = pd.read_json("categorized-comments.jsonl", lines = True)

import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

data1.loc[data1["cat"]=="sports", "cat"]=0
data1.loc[data1["cat"]=="science_and_technology", "cat"]=1
data1.loc[data1["cat"]=="video_games", "cat"]=2
data1.loc[data1["cat"]=="news", "cat"]=3

data1.cat.astype(int).head()

data1.dtypes

data1.head()

ps = PorterStemmer()

# =============================================================================
# corpus = []
# for i in range(0, 2347476, n_jobs = -1):
#         txt = re.sub('[^a-zA-Z]', " ", data1["txt"][0])
#         txt = txt.lower()
#         txt = txt.split()
#         txt = [ps.stem(word) for word in txt if not word in set(stopwords.words("english"))] 
#         txt = " ".join(txt)
#         corpus.append(txt)
#         
# =============================================================================
# =============================================================================
# data1.info(memory_usage = "deep")
# 
# X = data1.iloc[:1].values
# y = data1.iloc[:0].values
# =============================================================================

# =============================================================================
# X = data1.txt
# y = data1.cat
# 
# from sklearn.cross_validation import train_test_split
# 
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 1)
# 
# from sklearn.feature_extraction.text import CountVectorizer
# 
# vect = CountVectorizer(max_features = 1000)
# vect.fit(X_train)
# x_train_dtm = vect.transform(X_train).toarray()
# 
# d = x_train_dtm[0:5, 1:10]
# 
# from sklearn.linear_model import LogisticRegressionCV
# 
# classifier = LogisticRegressionCV(n_jobs = -1, random_state = 0, penalty = "l2")
# classifier.fit(x_train_dtm, y_train.astype("int"))
# =============================================================================

data_0 = data1[data1["cat"]==0].iloc[0:2000]
data_1 = data1[data1["cat"]==1].iloc[0:2000]
data_2 = data1[data1["cat"]==2].iloc[0:2000]
data_3 = data1[data1["cat"]==3].iloc[0:2000]

data_condensed = pd.concat([data_0, data_1], axis=0)
data_condensed = pd.concat([data_condensed, data_2], axis=0)
data_condensed = pd.concat([data_condensed, data_3], axis=0)

corpus = []

for i in range(0, 8000):
        txt = re.sub('[^a-zA-Z]', " ", data1["txt"][i])
        txt = txt.lower()
        txt = txt.split()
        txt = [ps.stem(word) for word in txt if not word in set(stopwords.words("english"))] 
        txt = " ".join(txt)
        corpus.append(txt)


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()
y = data_condensed.iloc[:, 0].values

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 1)

from sklearn.linear_model import LogisticRegressionCV

classifier = LogisticRegressionCV(n_jobs = -1, random_state = 0, penalty = "l2")
classifier.fit(X_train, y_train.astype("int"))


test_predictions = classifier.predict(X_test)
train_predictions = classifier.predict(X_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_predictions)

accuarcy_test = accuracy_score(y_test, test_predictions)
accuarcy_train = accuracy_score(y_train, train_predictions)


