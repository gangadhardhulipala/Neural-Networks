#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 05:33:12 2019

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
data1.cat.astype('category').head()

ps = PorterStemmer()

# Extracting 20,000 rows of each category from the total dataset
data_0 = data1[data1["cat"]==0].iloc[0:10000]
data_1 = data1[data1["cat"]==1].iloc[0:10000]
data_2 = data1[data1["cat"]==2].iloc[0:10000]
data_3 = data1[data1["cat"]==3].iloc[0:10000]

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


# Importing the necessary libraries for 1 hot encoding and test and train split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils.np_utils import to_categorical
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# Splitting the data into train and test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 1)

# One-hot encode target vector to create a target matrix
y_training = to_categorical(y_train)
y_testing = to_categorical(y_test)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

import numpy as np

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_shape=(5000,), units=2500))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=2500))

# Adding the output layer
classifier.add(Dense(activation="softmax", units=4))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(X_train, y_training, batch_size = 200, epochs = 5)

# Getting the accuracy and cost function
scores = classifier.evaluate(X_test, y_testing, verbose=0)

# Getting the predictions from the model
y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)

# Generating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_testing.argmax(axis=1),y_pred.argmax(axis=1))

# Calculating the classification report
from sklearn.metrics import classification_report
report = classification_report(y_testing.argmax(axis=1), y_pred.argmax(axis=1))
print(report)

