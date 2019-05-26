import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
import cv2
import os
from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
import cv2
import os
import h5py
from keras.callbacks import ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
# we always initialize the random number generator to a constant seed #value for reproducibility of results.
seed = 7
np.random.seed(seed)


# load data from the path specified by the user
def data_loader(path_train, path_test):
    train_list = os.listdir(path_train)
    '''
    # Map class names to integer labels
    train_class_labels = { label: index for index, label in enumerate(class_names) } 
    '''
    # Number of classes in the dataset
    num_classes = len(train_list)

    # Empty lists for loading training and testing data images as well as corresponding labels
    x_train = []
    y_train_1 = []
    y_train_2 = []
    y_train_3 = []
    y_train_4 = []
    x_test = []
    y_test_1 = []
    y_test_2 = []
    y_test_3 = []
    y_test_4 = []

    # Loading training data
    for label, elem in enumerate(train_list):

        path1 = path_train + '/' + str(elem)
        images = os.listdir(path1)
        for elem2 in images:
            path2 = path1 + '/' + str(elem2)
            # Read the image form the directory
            img = cv2.imread(path2)
            # Append image to the train data list
            x_train.append(img)
            # Append class-label corresponding to the image
            path_1 = os.path.basename(path1)
            label = np.asarray(path_1.split('_'))
            y_train_1.append(str(label[0]))
            y_train_2.append(str(label[1]))
            y_train_3.append(str(label[2]))
            y_train_4.append(str(label[3]))

        # Loading testing data
        path1 = path_test + '/' + str(elem)
        images = os.listdir(path1)
        for elem2 in images:
            path2 = path1 + '/' + str(elem2)
            # Read the image form the directory
            img = cv2.imread(path2)
            # Append image to the test data list
            x_test.append(img)
            path_2 = os.path.basename(path2)
            label = np.asarray(path_2.split('_'))
            # Append class-label corresponding to the image
            y_test_1.append(str(label[0]))
            y_test_2.append(str(label[1]))
            y_test_3.append(str(label[2]))
            y_test_4.append(str(label[3]))

    # Convert lists into numpy arrays
    x_train = np.asarray(x_train)
    y_train_1 = np.asarray(y_train_1)
    y_train_2 = np.asarray(y_train_2)
    y_train_3 = np.asarray(y_train_3)
    y_train_4 = np.asarray(y_train_4)

    x_test = np.asarray(x_test)
    y_test_1 = np.asarray(y_test_1)
    y_test_2 = np.asarray(y_test_2)
    y_test_3 = np.asarray(y_test_3)
    y_test_4 = np.asarray(y_test_4)

    return x_train, y_train_1,y_train_2,y_train_3,y_train_4 ,x_test, y_test_1, y_test_2, y_test_3, y_test_4


path_train = '/home/supriyo/practice/dataset_1/dataset/train/'
path_test = '/home/supriyo/practice/dataset_1/dataset/test/'

X_train, y_train_1,y_train_2,y_train_3,y_train_4 ,X_test, y_test_1, y_test_2, y_test_3, y_test_4 = data_loader(path_train, path_test)

# print(X_train.shape)
# print(y_train_2.shape)
# print(X_test.shape)
# print(y_test_1.shape)

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
# forcing the precision of the pixel values to be 32 bit
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255.
X_test = X_test / 255.

y_train_1 = np.asarray(y_train_1)
y_test_1 = np.asarray(y_test_1)
y_train_2 = np.asarray(y_train_2)
y_test_2 = np.asarray(y_test_2)
y_train_3 = np_utils.to_categorical(y_train_3)
y_test_3 = np_utils.to_categorical(y_test_3)
y_train_4 = np.asarray(y_train_4)
y_test_4 = np.asarray(y_test_4)



# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 5
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (28, 28, 3)
# Splitting the trining data into training and validation
# X_#rain, X_val_1,y_train_1,y_train_2,y_train_3,y_train_4, y_val = train_test_split(X_train, y_train_1, test_size=0.2, random_state=42)
# X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_train, y_train_2, test_size=0.2, random_state=42)
# X_train_3, X_val_3, y_train_3, y_val_3 = train_test_split(X_train, y_train_3, test_size=0.2, random_state=42)
# X_train_4, X_val_4, y_train_4, y_val_4 = train_test_split(X_train, y_train_4, test_size=0.2, random_state=42)


# define baseline model


def baseline_model(inputs):
    # CONV => RELU => POOL
    x = Conv2D(32, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)

    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # CONV => RELU => POOL
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # define a branch of output layers for the number of different
    # colors (i.e., red, black, blue, etc.)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)
    return x
# build the model


inputShape = (28, 28, 3)
inputs = Input(shape=inputShape)
model = baseline_model(inputs)
length_classification = Dense(1, activation='sigmoid', name='line_output')(model)

width_classification = Dense(1, activation='sigmoid', name='width_output')(model)

angle_classification = Dense(12, activation='softmax', name='angle_output')(model)

color_classification = Dense(1, activation='sigmoid', name='color_output')(model)


losses = {
	"line_output": "binary_crossentropy",
	"width_output": "binary_crossentropy",
    "angle_output": "categorical_crossentropy",
	"color_output": "binary_crossentropy",
}
lossWeights = {"line_output": 1.0, "width_output": 1.0, "angle_output": 1.0 ,"color_output": 1.0}

designed_model = Model(inputs, [length_classification, width_classification, angle_classification,color_classification])

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
#opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
designed_model.compile(optimizer="adam", loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])

callback_1 = keras.callbacks.TensorBoard(log_dir='./graphs', histogram_freq=0, write_graph=True, write_images=True)
filepath="./Model_Assignment_2.2/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# train the network to perform multi-output classification
designed_model.summary()
designed_model.fit(X_train,
	{"line_output": y_train_1,"width_output": y_train_2,"angle_output": y_train_3,"color_output": y_train_4},
	epochs=EPOCHS,batch_size=BS,
	verbose=1,callbacks=callbacks_list)


