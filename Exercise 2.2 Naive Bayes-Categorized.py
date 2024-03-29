#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 10:52:59 2019

@author: ganga
"""
# Export the necessary libraries
import pandas as pd
from pandas import DataFrame
import json

# Load the categorized comments json file
data1 = pd.read_json("categorized-comments.jsonl", lines = True)

# Export regular expression library and natural language tool kit
# Exporting the stop words library from natural language tool kit
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Exporting the stemming process library from natural language tool kit
data1.loc[data1["cat"]=="sports", "cat"]=0
data1.loc[data1["cat"]=="science_and_technology", "cat"]=1
data1.loc[data1["cat"]=="video_games", "cat"]=2
data1.loc[data1["cat"]=="news", "cat"]=3


data1.cat.astype(int).head()

ps = PorterStemmer()

# Function to extract only the alphabetical characters
def commas(txt):
    txt = re.sub('[^a-zA-Z]', " ", txt)
    return txt
# Applying the function on the condensed data frame
data1["txt"] = data1["txt"].apply(commas)

# Passing on the processed data on to the count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = {'english'}, max_features = 1000)
X = cv.fit_transform(data1["txt"]).toarray()
y = data1.iloc[:, 0].values

# Splitting the data into test and train data
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 1)

# Applying Naive Bayes Algorithm

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X_train, y_train.astype("int"))

# Calculating the accuracy of the model

from sklearn.metrics import accuracy_score

test_predictions = classifier.predict(X_test)
train_predictions = classifier.predict(X_train)

accuarcy_test = accuracy_score(y_test.astype("int"), test_predictions.astype("int"))
accuarcy_train = accuracy_score(y_train.astype("int"), train_predictions)
