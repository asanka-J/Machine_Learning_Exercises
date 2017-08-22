# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# enable full array display
np.set_printoptions(threshold=np.nan)


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#obj       classs para              
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])

#Encording categorical data - to identify uniquely
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelEncorder_X=LabelEncoder()
X[:,0]=labelEncorder_X.fit_transform(X[:,0])
oneHotEncoder=OneHotEncoder(categorical_features=[0])
X=oneHotEncoder.fit_transform(X).toarray()

labelEncorder_y=LabelEncoder()
y=labelEncorder_y.fit_transform(y)

## Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)