# -*- coding: utf-8 -*-
"""
Created on Sun May 31 22:52:41 2020

@author: jayni
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


df = pd.read_csv("C:/Users/jayni/.spyder-py3/learning/resources/news.csv")
print(df.head())
print(df.columns)

#defining the variable we wish to find. In this case it would be the label of the news (Binary Category, either Real or Fake)
y = df.label
print(y)

#using test train split to split data into training and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df['text'], y, test_size=0.2, random_state=7)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Normalizing the data.
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)

temp_df = pd.DataFrame(tfidf_train.toarray(), columns=tfidf_vectorizer.get_feature_names())

#building the model
from sklearn.linear_model import PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


#testing the model
from sklearn.metrics import accuracy_score, confusion_matrix
y_pred=pac.predict(tfidf_test)

#evaluating model
#Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
cnf_matrix = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
plot_confusion_matrix(cnf_matrix, classes=['FAKE','REAL'],normalize= False,  title='Confusion matrix')
#f1 score
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average='weighted') )

#jaccard Score
from sklearn.metrics import jaccard_similarity_score
print(jaccard_similarity_score(y_test, y_pred))
