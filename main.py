
import pandas as pd
import numpy as np

import re  # Regular Expression, searchibg for text data doc
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer  # stemmer removes prefix and suffix of the word and return the root word
from sklearn.feature_extraction.text import TfidfVectorizer  # TfidfVectorizer to convert the text into feature vectors(numbers)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py

column = ['id', 'label', 'text', 'subject', 'speaker', 'speaker\'s job title', 'state info', 'party affiliation',
          'barely true', 'false', 'half true', 'mostly true', 'pants on fire', 'context']


#root_folder = "C:\Users\Priyanka\OneDrive\Desktop\Term1 Docs\AML-1214 - Python Programming\WorkSpace\whatsappChatBot"
train = pd.read_csv('train.csv',sep='\t',header=None,names=column)
test = pd.read_csv('test.csv',sep='\t',header=None,names=column)
valid = pd.read_csv('valid.csv',sep='\t',header=None,names=column)



"""#Data Pre-processing"""

data = pd.concat([train, test, valid])

"""https://pandas.pydata.org/docs/reference/api/pandas.concat.html

"""

df = data.drop(columns=['id'])



# finding missing values


df = df.dropna()
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
df["label"] = enc.fit_transform(df[["label"]])
df.head()

data = df.drop(
    columns=['subject', 'speaker\'s job title', 'state info', 'party affiliation', 'barely true', 'false', 'half true',
             'mostly true', 'pants on fire'], axis=1)




data.to_csv('final_dataset.csv', index = False)

finalData = pd.read_csv('final_dataset.csv')


# download stopwords
import nltk

nltk.download('stopwords')

#print(stopwords.words('english'))

"""Stopwords are doesn't have much value. hence removing all the stopwords from the dataset."""

# stemming words
port_sem = PorterStemmer()


def stemming(statement):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', statement)  # searching paragraph or text
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_sem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


finalData['text'] = finalData['text'].apply(stemming)
#separating data and label
X= finalData.iloc[:,1].values
Y = finalData.iloc[:,0].values






X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#converting content text into numerical value
vector = TfidfVectorizer()
tf_fit = vector.fit(X_train)
X_train_tf = tf_fit.transform(X_train)
X_test_tf = tf_fit.transform(X_test)

from pandas.core.common import random_state


# Creating a function for the models.
def models(X_train, y_train):
    # Logistical Model
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(random_state=0)
    lg.fit(X_train, y_train)

    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    ds = DecisionTreeClassifier(criterion='entropy', random_state=0)
    ds.fit(X_train, y_train)

    # RandomForest Classifier
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    rf.fit(X_train, y_train)

    # Print the accuracy on the training data for above models
    lg.score(X_train, y_train)
    ds.score(X_train, y_train)
    rf.score(X_train, y_train)

    return lg, ds, rf


# Getting all models
md = models(X_train_tf, y_train)

# model accuracy on test data using confusion matrix
from sklearn.metrics import confusion_matrix

for i in range(len(md)):
    print('Model ', i)
    cn = confusion_matrix(y_test, md[i].predict(X_test_tf))

    truePositive = cn[0][0]
    trueNegative = cn[1][1]
    falsePositive = cn[1][0]
    falseNegative = cn[0][1]



    accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)





'''import pickle


pickle.dump(md[2], open('model.pkl', 'wb'))

# load the model from the disk
load_model = pickle.load(open('model.pkl', 'rb'))'''

from flask import Flask, request
#import pickle


from twilio.twiml.messaging_response import MessagingResponse


app = Flask(__name__)

#load_model = pickle.load(open('model.pkl', 'rb'))



@app.route('/sms', methods=['POST'])
def sms():
    #inboxMsg = [request.values.get('Body').lower()]
    input_data = {"text": [request.values.get('Body').lower()]}
    resp = MessagingResponse()
    msg = resp.message()

    input = pd.DataFrame(input_data)
    input["text"] = input["text"].apply(stemming)
    text = input["text"]

    vectorize_data = vector.transform(text)
    pred = md[2].predict(vectorize_data)

    if pred == 0:
        msg.body("This news is Barely True ")
    elif pred == 1:
        msg.body("This News is False")
    elif pred == 2:
        msg.body("This News is Half True")
    elif pred == 3:
        msg.body("This News is Mostly True")
    elif pred == 4:
        msg.body("This News is Pants on Fire")
    elif pred == 5:
        msg.body("This News is True")
    else:
        msg.body("This News is not listed in the dataset")

    return str(resp)


if __name__ == '__main__':
    app.run()