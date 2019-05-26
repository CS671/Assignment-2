import os
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
import cv2
import os
from keras.layers import Input
from keras import backend as K
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
import tensorflow as tf
from keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
import cv2
import os
from sklearn.metrics import confusion_matrix
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
    print(y_train_2.shape)
    x_test = np.asarray(x_test)
    y_test_1 = np.asarray(y_test_1)
    y_test_2 = np.asarray(y_test_2)
    y_test_3 = np.asarray(y_test_3)
    y_test_4 = np.asarray(y_test_4)

    return x_train, y_train_1,y_train_2,y_train_3,y_train_4 ,x_test, y_test_1, y_test_2, y_test_3, y_test_4


path_train = './final_dataset/train'
path_test = './final_dataset/test'

X_train, y_train_1,y_train_2,y_train_3,y_train_4 ,X_test, y_test_1, y_test_2, y_test_3, y_test_4 = data_loader(path_train, path_test)

print(X_train.shape)
print(y_train_2.shape)
print(X_test.shape)
print(y_test_1.shape)

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
# forcing the precision of the pixel values to be 32 bit
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255.
X_test = X_test / 255.
# one hot encode outputs using np_utils.to_categorical inbuilt function
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
#X_#rain, X_val_1,y_train_1,y_train_2,y_train_3,y_train_4, y_val = train_test_split(X_train, y_train_1, test_size=0.2, random_state=42)
#X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_train, y_train_2, test_size=0.2, random_state=42)
#X_train_3, X_val_3, y_train_3, y_val_3 = train_test_split(X_train, y_train_3, test_size=0.2, random_state=42)
#X_train_4, X_val_4, y_train_4, y_val_4 = train_test_split(X_train, y_train_4, test_size=0.2, random_state=42)
# define baseline model
#The model is a simple neural network with one hidden layer with the same number of neurons as there are inputs (784)
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
    x = Dropout(0.5)(x)
    # return the color prediction sub-network
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
filename = "/home/ankit/PycharmProjects/Deep_learning/Model_Assignment_2.2/weights-improvement-05-0.2719.hdf5"
designed_model.load_weights(filename)
designed_model.compile(optimizer=Adam(lr=INIT_LR), loss=losses, loss_weights=lossWeights,
	metrics=["accuracy"])
designed_model.summary()
# train the network to perform multi-output classification
# evaluate the model
scores = designed_model.evaluate(X_test, {"line_output": y_test_1,"width_output": y_test_2,"angle_output": y_test_3,"color_output": y_test_4}, verbose=1)
print("%s: %.2f%%" % (designed_model.metrics_names[5], scores[5]*100))
print("%s: %.2f%%" % (designed_model.metrics_names[6], scores[6]*100))
print("%s: %.2f%%" % (designed_model.metrics_names[7], scores[7]*100))
print("%s: %.2f%%" % (designed_model.metrics_names[8], scores[8]*100))


print("%s: %.2f%%" % (designed_model.metrics_names[1], scores[1]*100))
print("%s: %.2f%%" % (designed_model.metrics_names[2], scores[2]*100))
print("%s: %.2f%%" % (designed_model.metrics_names[3], scores[3]*100))
print("%s: %.2f%%" % (designed_model.metrics_names[4], scores[4]*100))


y_pred = designed_model.predict(X_test)

y_pred[0][y_pred[0] > 0.5] = 1
y_pred[0][y_pred[0] <= 0.5] = 0
y_pred_length = np.asarray(y_pred[0])
L = len(y_pred[0])
#length_pred_out = np.argmax(y_pred_length, axis=1)
length_pred_out = np.reshape(y_pred_length, [L, 1])
print('pred_length=' + str(length_pred_out.shape))
print(length_pred_out)


y_pred[1][y_pred[1] > 0.5] = 1
y_pred[1][y_pred[1] <= 0.5] = 0
y_pred_width = np.asarray(y_pred[1])
W = len(y_pred[1])
#width_pred_out = np.argmax(y_pred_width, axis=1)
width_pred_out = np.reshape(y_pred_width, [W, 1])
print('pred_width=' + str(width_pred_out.shape))
print(width_pred_out)


y_pred_angle = np.asarray(y_pred[2])
angle_pred_out = np.argmax(y_pred_angle, axis=1)
print('pred_angle=' + str(angle_pred_out.shape))
print(angle_pred_out)

y_pred[3][y_pred[3] > 0.5] = 1
y_pred[3][y_pred[3] <= 0.5] = 0
y_pred_color = np.asarray(y_pred[3])
C = len(y_pred[3])
#color_pred_out = np.argmax(y_pred_color, axis=1)
color_pred_out = np.reshape(y_pred_color, [C, 1])
print('pred_color=' + str(color_pred_out.shape))
print(color_pred_out)

true_length = np.asarray(y_test_1)
true_length = [int(i) for i in true_length]
L = len(true_length)
true_length = np.reshape(true_length, [L, 1])
print('true_length=' + str(true_length.shape))
print(true_length)

true_width = np.asarray(y_test_2)
true_width = [int(i) for i in true_width]
W = len(true_width)
true_width = np.reshape(true_width, [W, 1])
print('true_width=' + str(true_width.shape))
print(true_width)

true_angle = np.asarray(y_test_3)
true_angle = np.argmax(true_angle, axis=1)
print('true_angle=' + str(true_angle.shape))
print(true_angle)

true_color = np.asarray(y_test_4)
true_color = [int(i) for i in true_color]
C = len(true_color)
true_color = np.reshape(true_color, [C, 1])
print('true_color=' + str(true_color.shape))
print(true_color)


print('Confusion Matrix for length')
print(confusion_matrix(y_true=true_length, y_pred=length_pred_out))

print('Confusion Matrix for width')
print(confusion_matrix(y_true=true_width, y_pred=width_pred_out))

print('Confusion Matrix for angle')
print(confusion_matrix(y_true=true_angle, y_pred=angle_pred_out))

print('Confusion Matrix for color')
print(confusion_matrix(y_true=true_color, y_pred=color_pred_out))


## Question 3

# part 1

def getNsave(layer_name, model, data):
    out = model.input
    for l in model.layers[1:]:
        out = l(out)
        if l.name == layer_name:
            break
    intermediate_layer_model = keras.models.Model(inputs=model.input, outputs=out)
    intermediate = intermediate_layer_model.predict(data)
    intermediate_layer_model.summary()
    for i in range(len(data)):
        print('------------', i, '------------')
        print_filter(intermediate[i], 64, 'output/'+str(i)+'th_imageActivation_'+layer_name)



def print_filter(x, n, savename):
    fig=plt.figure(figsize=(32, 32))
    columns = 8
    rows = 4
    for i in range(0, columns*rows):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(x[:, :, i])
    plt.savefig(savename)
    # plt.show()


getNsave('conv2d_2', designed_model, X_test[2000:2006])
getNsave('conv2d_1', designed_model, X_test[2000:2006])


# part 2

def plot_conv_weights(model, layer_name, depth, filters):
    weights = model.get_layer(name=layer_name).get_weights()[0]
    if len(weights.shape) == 4:
        weights = np.squeeze(weights)
        print(weights.shape)

        fig, axs = plt.subplots(filters, depth, figsize=(depth*2, filters*2))
#         fig.subplots_adjust(hspace = 2, wspace=2)
        axs = axs.ravel()
        for j in range(depth):
            for i in range(filters):
                axs[32*j+i].imshow(weights[:, :, j, i])
                axs[32*j+i].set_title(str(i))
        fig.savefig(layer_name+'conv filters')

plot_conv_weights(designed_model, 'conv2d_1', 3, 32)
plot_conv_weights(designed_model, 'conv2d_2', 32, 64)



