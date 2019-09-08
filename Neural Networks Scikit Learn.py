#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 02:22:52 2019

@author: ganga
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

#ps = PorterStemmer()

# Extracting 20,000 rows of each category from the total dataset
# Extracted the rows in such a way that equal representation is seen for all the categories
# This is an alternative to using all the dataset as it is taking forever
data_0 = data1[data1["cat"]==0].iloc[0:20000]
data_1 = data1[data1["cat"]==1].iloc[0:20000]
data_2 = data1[data1["cat"]==2].iloc[0:20000]
data_3 = data1[data1["cat"]==3].iloc[0:20000]

# Merging the individual data frames from 2 million rows to 80,000 rows as the classifier has hard time fitting the data
data_condensed = pd.concat([data_0, data_1], axis=0)
data_condensed = pd.concat([data_condensed, data_2], axis=0)
data_condensed = pd.concat([data_condensed, data_3], axis=0)

# Function to extract only the alphabetical characters
def commas(txt):
    txt = re.sub('[^a-zA-Z]', " ", txt)
    return txt

# Applying the function on the condensed data frame
data_condensed["txt"] = data_condensed["txt"].apply(commas)

# Passing on the processed data on to the count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words = {'english'}, max_features = 5000)
X = cv.fit_transform(data_condensed["txt"]).toarray()
y = data_condensed.iloc[:, 0].values

# Importing the MLPClassifier and Model selection library to perform test and train split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Performing the test and train split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 1)


# Intializing the neural network MLPClassifier
nn = MLPClassifier(random_state=42)

# Fitting the neural network object on X_train and Y_train
nn.fit(X_train, y_train.astype("int"))

# Getting the predictions for the test set using the predictor function of neural network
test_predictions = nn.predict(X_test)

# Getting the accuracy of the model
from sklearn.metrics import accuracy_score
accuarcy_test = accuracy_score(y_test.astype("int"), test_predictions.astype("int"))

# Calculating the classification report, precision, recall, f1-score
from sklearn.metrics import classification_report
report = classification_report(y_test.astype("int"), test_predictions.astype("int"))
print(report)


# Getting the confusion matrix for the results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.astype("int"),test_predictions.astype("int"))


