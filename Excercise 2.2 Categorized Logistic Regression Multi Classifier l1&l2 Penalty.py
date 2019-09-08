# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Export the necessary libraries
import pandas as pd
from pandas import DataFrame
import json

# Load the categorized comments json file
data1 = pd.read_json("categorized-comments.jsonl", lines = True)

# Export regular expression library and natural language tool kit
import re
import nltk
nltk.download('stopwords')

# Exporting the stemming process library from natural language tool kit
from nltk.stem.porter import PorterStemmer

# Exporting the stop words library from natural language tool kit
from nltk.corpus import stopwords

# Encode the categorical variable of categorical variable into integers
data1.loc[data1["cat"]=="sports", "cat"]=0
data1.loc[data1["cat"]=="science_and_technology", "cat"]=1
data1.loc[data1["cat"]=="video_games", "cat"]=2
data1.loc[data1["cat"]=="news", "cat"]=3

# looking at the top five rows of the dataset
data1.cat.astype(int).head()

ps = PorterStemmer()

# Extracting 20,000 rows of each category from the total dataset
data_0 = data1[data1["cat"]==0].iloc[0:20000]
data_1 = data1[data1["cat"]==1].iloc[0:20000]
data_2 = data1[data1["cat"]==2].iloc[0:20000]
data_3 = data1[data1["cat"]==3].iloc[0:20000]

# Merging the individual data frames from 2 million rows to 80,000 rows as the classifier has hard time fitting the data
data_condensed = pd.concat([data_0, data_1], axis=0)
data_condensed = pd.concat([data_condensed, data_2], axis=0)
data_condensed = pd.concat([data_condensed, data_3], axis=0)

data1.cat.astype(int).head()

ps = PorterStemmer()


# Function to extract only the alphabetical characters
def commas(txt):
    txt = re.sub('[^a-zA-Z]', " ", txt)
    return txt

# Applying the function on the condensed data frame
data_condensed["txt"] = data_condensed["txt"].apply(commas)

# Passing on the processed data on to the count vectorizer

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = {'english'}, max_features = 1000)
X = cv.fit_transform(data_condensed["txt"]).toarray()
y = data_condensed.iloc[:, 0].values

# Splitting the data into test and train data

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 1)

# Applying logistic regression with l2 penalty

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(n_jobs = -1, solver = "sag", multi_class = "multinomial", penalty = "l2")
classifier.fit(X_train, y_train.astype("int"))

# Calculating the accuracy of the model

from sklearn.metrics import accuracy_score

test_predictions = classifier.predict(X_test)
train_predictions = classifier.predict(X_train)

accuarcy_test = accuracy_score(y_test.astype("int"), test_predictions.astype("int"))
accuarcy_train = accuracy_score(y_train.astype("int"), train_predictions)

# Applying the logistic regression with penalty l1

from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(n_jobs = -1, solver = "saga", multi_class = "multinomial", penalty = "l1")
classifier2.fit(X_train, y_train.astype("int"))

# Calculating the accuracy of the logistic regression with l1 penalty
from sklearn.metrics import accuracy_score

test_predictions2 = classifier.predict(X_test)
train_predictions2 = classifier.predict(X_train)

accuarcy_test2 = accuracy_score(y_test.astype("int"), test_predictions2.astype("int"))
accuarcy_train2 = accuracy_score(y_train.astype("int"), train_predictions2)

Accuracy  = DataFrame({"Accuracy Train (%)": [68.0,69.6], "Accuracy Test (%)":[68.0, 69.6]})

Accuracy.set_index([["Categorized-Logistic L1", "Categorized-Logistic L2"]])