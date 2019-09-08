#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:43:58 2019

@author: ganga
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

ps = PorterStemmer()

def commas(txt):
    txt = re.sub('[^a-zA-Z]', " ", txt)
    return txt

data1["txt"] = data1["txt"].apply(commas)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = {'english'}, max_features = 1000)
X = cv.fit_transform(data1["txt"]).toarray()
y = data1.iloc[:, 0].values


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 1)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(n_jobs = -1, solver = "sag", multi_class = "multinomial", penalty = "l1")
classifier.fit(X_train, y_train.astype("int"))


from sklearn.metrics import accuracy_score

test_predictions = classifier.predict(X_test)
train_predictions = classifier.predict(X_train)

accuarcy_test = accuracy_score(y_test.astype("int"), test_predictions.astype("int"))
accuarcy_train = accuracy_score(y_train.astype("int"), train_predictions)