p;[# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
"""allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. Theano features:

-tight integration with NumPy – Use numpy.ndarray in Theano-compiled functions.
-transparent use of a GPU – Perform data-intensive computations much faster than on a CPU.
-efficient symbolic differentiation – Theano does your derivatives for functions with one or many inputs.
-speed and stability optimizations – Get the right answer for log(1+x) even when x is really tiny.
-dynamic C code generation – Evaluate expressions faster.
-extensive unit-testing and self-verification – Detect and diagnose many types of errors.
"""
# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html
"""
-library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges
 represent the multidimensional data arrays (tensors) communicated between them.
 The flexible architecture allows you to deploy 
 computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. """

### For both Tensorflow and theano we need to implement Nural network from scratch.
#
# Installing Keras
# pip install --upgrade keras
"""
Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow,Theano.

-Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
-Supports both convolutional networks and recurrent networks, as well as combinations of the two.
-Runs seamlessly on CPU and GPU.
"""
# Part 1 - Data Preprocessing
#dataset churn_modelling :contains bank customer information for 6 months.@predict leave or stay

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential# required to init ANN
from keras.layers import Dense #required to build layers of ANN

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))#
#Dense function initialize the weights
#output dim=>average #output+input=> (no of input columns+ output )/2=>(11+1)/2

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
 #relu removes the negative parts of the fuction
#no need at this step , just to express how to add another hidden layer

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
#if output is more than 1 catagorical variable apply activation='softmax'and output_dim="#outputs"

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#adam-type of schocastic gradient decent   #loss = 'binary_crossentropy'
# Fitting the ANN to the Training set
#metrics=>output evaluation cretarian 

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
#batch_size =>  
#nb_epoch =>


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #convert to true false threshold=>0.5 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
