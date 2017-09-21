# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #quoting=3  ignore quoting

# Cleaning the texts
import re #cleaning library

#review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) #keep only Letters
#    review = review.lower()
#    review = review.split()
#    ps = PorterStemmer()#
#    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#    review = ' '.join(review)
#    corpus.append(review)
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #set improves execution speed
    #ps object do stemming ex: love ,loving etc to love
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model / metrix with lot of 0's called sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #max_features :removes unrelavent words like names etc 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


""" 0  1
0   87	10
1   46	57
"""
#0.72
TP=cm.item(1,1)
TN=cm.item(0,0)
FN=cm.item(0,1)
FP=cm.item(1,0)

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1_Score = 2*Precision*Recall / (Precision+Recall)

print("Accuracy =",round(Accuracy, 2)*100,'%')
print("Precision =",round(Precision, 2)*100,'%')
print("Recall =",round(Recall, 2)*100,'%')
print("F1_Score =",round(F1_Score, 2)*100,'%')
