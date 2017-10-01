# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # initialize nural network
from keras.layers import Convolution2D # add convolution layers 2D:to images
from keras.layers import MaxPooling2D # do pooling
from keras.layers import Flatten #flattning.
from keras.layers import Dense # add fully connected layer to ANN

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# 32-#feature detectors; 3x3 feature detector width and length
#input_shape = (64, 64, 3) => 64=dementions of 2D array,3= #channels

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#pool size =>2x2 

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
# no of nodes in hidden layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#adam-type of schocastic gradient decent   #loss = 'binary_crossentropy' -> binary output
#metrics=>output evaluation cretarian

#done 