
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator


seed = 7
np.random.seed(seed)

path_train = '/home/souvik/Downloads/Code/Codes_2/Data/train'
path_test = '/home/souvik/Downloads/Code/Codes_2/Data/test'

epoch = 10
batch_size = 400


def data_loader(path_train, path_test):
    train_list = os.listdir(path_train)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for label, element in enumerate(train_list):
        path1 = path_train + '/' + str(element)
        images = os.listdir(path1)
        for element2 in images:
            path2 = path1 + '/' + str(element2)
            img = cv2.imread(path2)
            x_train.append(img)
            y_train.append(str(label))

        path1 = path_test + '/' + str(element)
        images = os.listdir(path1)
        for element2 in images:
            path2 = path1 + '/' + str(element2)
            img = cv2.imread(path2)
            x_test.append(img)
            y_test.append(str(label))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    return x_train, y_train, x_test, y_test


X_train, y_train, X_test, y_test = data_loader(path_train, path_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


num_of_test_samples = X_test.shape[0]
print(num_of_test_samples)

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (7, 7), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2, padding='valid'))
    # model.add(Dropout(0.25))
    #model.add(Conv2D(32, (7, 7), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    #model.add(MaxPooling2D((2, 2), strides=2, padding='valid'))
    # model.add(Dropout(0.25))
    #model.add(Conv2D(32, (7, 7), strides=1, padding='same', input_shape=input_shape, activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=2, padding='valid'))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    keras.layers.BatchNormalization()
    #model.add(Dropout(0.))
    # model.add(Dense(200, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.1))# model.add(Embedding((:,19), (:,19), embeddings_initializer='uniform', embeddings_regularizer=None, input_length=19))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # keras.layers.BatchNormalization()
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
    return model


model = baseline_model()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epoch, batch_size=batch_size, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=2)
y_pred = model.predict_classes(X_test, batch_size=batch_size)

print("Baseline Error: %.2f%%" % (100-scores[1]*100))
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("Y_pred=" + str(y_pred))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_data_gen = ImageDataGenerator(rescale=1. / 255)

validation_generator = test_data_gen.flow_from_directory('/home/souvik/Downloads/Code/Codes_2/Data/test', batch_size=batch_size, class_mode='categorical', target_size=[28, 28], shuffle=False)


def prediction(path_test):
    test_list = os.listdir(path_test)

    x_true = []
    y_true = []
    for label, element in enumerate(test_list):
        path1 = path_test + '/' + str(element)
        images = os.listdir(path1)
        for element2 in images:
            path2 = path1 + '/' + str(element2)
            img = cv2.imread(path2)
            x_true.append(img)
            y_true.append(str(label))


    y_true = np.asarray(y_true)
    return y_true


y_true = prediction(path_test)
y_true = [int(i) for i in y_true]
print("Y_true=" + str(y_true))
print('Confusion Matrix')
print(confusion_matrix(y_true=y_true, y_pred=y_pred, labels=None, sample_weight=None))
