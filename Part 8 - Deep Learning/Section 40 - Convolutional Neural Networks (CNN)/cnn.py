# Convolutional Neural Network

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # initialize nural network
from keras.layers import Conv2D # add convolution layers 2D:to images
from keras.layers import MaxPooling2D # do pooling
from keras.layers import Flatten #flattning.
from keras.layers import Dense # add fully connected layer to ANN

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32,( 3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# 32-#feature detectors; 3x3 feature detector width and length
#input_shape = (64, 64, 3) => 64=dementions of 2D array,3= #channels

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#pool size =>2x2 

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
# no of nodes in hidden layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#adam-type of schocastic gradient decent   #loss = 'binary_crossentropy' -> binary output
#metrics=>output evaluation cretarian

#done 

# Part 2 - Fitting the CNN to the images

#image augmentation code copied from keras documentation..THis avoids overfitting
# make changes in the image batches by rescale, shear,zoom etc
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


#after updated keras 2
classifier.fit_generator(   training_set,
                            steps_per_epoch=8000,
                            epochs=25,
                            validation_data=test_set,
                            validation_steps=2000)


#predicting the image


from keras.preprocessing import image as image_utils
import numpy as np
 
test_image = image_utils.load_img('dataset/test_set/cats/cat.4001.jpg', target_size=(64, 64))
test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
 
result = classifier.predict_on_batch(test_image)
if(result==1):
    print("dog")
else:
    print("cat")
    
from skimage.io import imread
from skimage.transform import resize
img = imread('dataset/test_set/dogs/dog.4001.jpg') #make sure that path_to_file contains the path to the image you want to predict on. 
img = resize(img, (64, 64))
img = np.reshape(img,(1,64,64,3))
img = img/(255.0)
prediction = classifier.predict_classes(img)
print (prediction)    
