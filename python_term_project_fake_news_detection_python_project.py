# -*- coding: utf-8 -*-
"""Python Term Project_Fake News Detection_Python Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tqqo7cV8nK6wkz5EWY3nsPGaAOZNcBkT
"""

from google.colab import drive
drive.mount("/content/gdrive")

import pandas as pd
import numpy as np

import re #Regular Expression, searchibg for text data doc
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer #stemmer removes prefix and suffix of the word and return the root word
from sklearn.feature_extraction.text import TfidfVectorizer #TfidfVectorizer to convert the text into feature vectors(numbers)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py

column = ['id','label','text','subject','speaker','speaker\'s job title','state info','party affiliation','barely true','false','half true','mostly true','pants on fire','context']

train = pd.read_csv('/content/gdrive/My Drive/liar_dataset/train.csv',sep='\t',header=None,names=column)
test = pd.read_csv('/content/gdrive/My Drive/liar_dataset/train.csv',sep='\t',header=None,names=column)
valid = pd.read_csv('/content/gdrive/My Drive/liar_dataset/train.csv',sep='\t',header=None,names=column)

train.head()

test.head()

valid.head()

train.shape,test.shape,valid.shape

"""#Data Pre-processing"""

data = pd.concat([train,test,valid])

"""https://pandas.pydata.org/docs/reference/api/pandas.concat.html

"""

df = data.drop(columns=['id'])

df.head(5)

#finding missing values
df.isnull().sum()

df.isnull().sum().sum()

df = df.dropna()
df.isnull().sum()

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
df["label"] = enc.fit_transform(df[["label"]])
df.head()

df['label'] = df['label'].astype('int')

#count of label coulmun
sns.countplot(x='label', data=df, palette='hls')

"""0 - Barely True

1- False

2- Half-true

3- Mostly true

4- Pants on Fire

5- True
"""

#articles based on subject

print(df.groupby(['subject'])['text'].count())

data = df.drop(columns =['subject','speaker\'s job title','state info','party affiliation','barely true','false','half true','mostly true','pants on fire'], axis = 1)

data.to_csv('/content/gdrive/My Drive/liar_dataset/final_dataset.csv', index = False)

finalData = pd.read_csv('/content/gdrive/My Drive/liar_dataset/final_dataset.csv')
finalData.tail(20)

#download stopwords
import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

"""Stopwords are doesn't have much value. hence removing all the stopwords from the dataset."""

#stemming words
port_sem = PorterStemmer()
def stemming(statement):
  stemmed_content = re.sub('[^a-zA-Z]',' ',statement) #searching paragraph or text
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_sem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

finalData['text'] = finalData['text'].apply(stemming)

print(finalData['text'])

finalData.info()

"""true – The statement is correct, and nothing important is lost.

Mostly true – The statement is correct, but further information or clarification is required.

half-true – The statement is partially correct, but it fails to mention key details or places information out of context.

barely true – The statement contains certain truth, but it leaves out important details that would provide a different image.


false – The statement is wrong.


pants-on-fire - The statement is incorrect and makes an outrageous claim. i.e. "Liar, Liar, Pants on Fire!" exclaims the speaker.

"""

#separating data and label
X= finalData.iloc[:,1].values
Y = finalData.iloc[:,0].values
print(X)
print(Y)

print(type(X))

print(type(Y))

"""NumPy is an N-dimensional array type called ndarray. It describes the collection of items of the same type. Items in the collection can be accessed using a zero-based index. Every item in an ndarray takes the same size of block in the memory.

Convert all text into numbers using vectorization
"""

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25,  random_state = 0)

#feature scaling
#from sklearn.preprocessing import StandardScaler
#scale = StandardScaler(with_mean=False)
#scale_Fit = scale.fit(X_train)
#X_train = scale_Fit.transform(X_train)
#X_test = scale_Fit.transform(X_test)

#converting content text into numerical value
vector = TfidfVectorizer()
tf_fit = vector.fit(X_train)
X_train_tf = tf_fit.transform(X_train)
X_test_tf = tf_fit.transform(X_test)

X_train_tf.toarray().shape

#Creating a function for the models.
def models(X_train,y_train):
  
  #Logistic Model
  from sklearn.linear_model import LogisticRegression
  lg = LogisticRegression(random_state=0)
  lg.fit(X_train,y_train)

  #Decision Tree
  from sklearn.tree import DecisionTreeClassifier
  ds = DecisionTreeClassifier(criterion='entropy', random_state = 0)
  ds.fit(X_train,y_train)

  #RandomForest Classifier
  from sklearn.ensemble import RandomForestClassifier
  rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
  rf.fit(X_train,y_train)

  #Print the accuracy on the training data for above models
  print('[0]Logistic Regression training data Accuracy: ', lg.score(X_train, y_train))
  print('[1]Decision Tree Classifier training data Accuracy: ', ds.score(X_train, y_train))
  print('[2]Random Forest Classifier training data Accuracy: ', rf.score(X_train, y_train))

  return lg, ds, rf

#Getting all models
md = models(X_train_tf,y_train)
print(md)

#model accuracy on test data using confusion matrix
from sklearn.metrics import confusion_matrix

for i in range(len(md)):
  print('Model ',i)
  cn = confusion_matrix(y_test, md[i].predict(X_test_tf))

  truePositive = cn[0][0]
  trueNegative = cn[1][1]
  falsePositive = cn[1][0]
  falseNegative = cn[0][1]

  
  print('Testing Accuracy: ', (truePositive+trueNegative)/(truePositive+trueNegative+falsePositive+falseNegative))
  print()
  from sklearn.metrics import plot_confusion_matrix
  
  plot_confusion_matrix(md[i], X_test_tf, y_test) 
 
  plt.show()

#prediction for Random Forest Classification model
prediction = md[2].predict(X_test_tf)
print("Model prediction values for News:", prediction)
print()
print("Actual values for News:",y_test)

def news_data(news):
  input_data = {"text":[news]}
  input = pd.DataFrame(input_data)
  input["text"]=input["text"].apply(stemming)
  text = input["text"]
 

  vectorize_data = vector.transform(text)
  pred = md[2].predict(vectorize_data)
  print(pred)

  if pred == 0:
    print("This news is Barely True ")
  elif pred == 1:
    print("This News is False")
  elif pred == 2:
    print("This News is Half True")
  elif pred == 3:
    print("This News is Mostly True")
  elif pred == 4:
    print("This News is Pants on Fire")
  elif pred == 5:
    print("This News is True")
  else:
    print("This News is not listed in the dataset")

news_data("On residency requirements for public workers	city-government")

X_new = X_test[1]
pred = md[2].predict(X_new)
print(pred)
if pred == 0:
  print("This news is Barely True ")
elif pred == 1:
  print("This News is False")
elif pred == 2:
  print("This News is Half True")
elif pred == 3:
  print("This News is Mostly True")
elif pred == 4:
  print("This News is Pants on Fire")
elif pred == 5:
  print("This News is True")
else:
  print("This News is not listed in the dataset")