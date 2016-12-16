import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import json

def dataimport():
    dataDir = 'proefrondje1-16dec/'
    #dataDir = 'data/'
    imArr = glob.glob(dataDir + "IMG/*.jpg")
    numImages = len(imArr)
    autodata = dataDir + 'driving_log.csv'
    steerdiff = 0.1  # to add or subtract from steering angle when using left or right camera
    resize = (160,40)

    # build a dictionary with all image timestamps ad keys and steering angles as values
    steer_angle = {}
    with open(autodata) as csvfile:
        lines = csvfile.read().split("\n")  # "\r\n" if needed
        for line in lines:
            if (line[0:3] == "IMG") or (line[0:3] == "/Us"):
                cols = line.split(",")
                steer_angle[(cols[0][len(cols[0]) - 27:-4:])] = float(cols[3])

    '''
    create and fill training data arrays
    resize the images to 160(width)x80(height)
    add/subtract to steering angles when image is from left or right camera
    bound steering angles to range [-1,1]
    add mirrored images with negated steering angle
    '''
    X_simulator = np.zeros((2 * numImages, resize[1], resize[0], 3), dtype='uint8')
    y_simulator = np.zeros(2 * numImages, dtype='float')
    for i in range(numImages):
        if i%1000 == 0:
            print(i,numImages)
        img = plt.imread(imArr[i])  # matplotlib --> imread imports RGB format
        crop_img = img[:,40:120]
        X_simulator[i] = cv2.resize(crop_img, resize)
        X_simulator[numImages + i] = cv2.flip(X_simulator[i], 1)
        if imArr[i][len(imArr[i]) - 30:-29] == 'e':  # e means centEr camera
            y_simulator[i] = steer_angle[imArr[i][len(imArr[i]) - 27:-4]]
        elif imArr[i][len(imArr[i]) - 30:-29] == 'f':  # f means leFt camera
            y_simulator[i] = steer_angle[imArr[i][len(imArr[i]) - 27:-4]] + steerdiff
            if y_simulator[i] > 1:  # if out of bounds set to maximum
                y_simulator[i] = 1
        else:  # this must be the right camera
            y_simulator[i] = steer_angle[imArr[i][len(imArr[i]) - 27:-4]] - steerdiff
            if y_simulator[i] < -1:  # if out of bounds set to maximum
                y_simulator[i] = -1
        # add flipped images and negated steerinmg angle to second half of array
        X_simulator[numImages + i] = cv2.flip(X_simulator[i], 1)
        y_simulator[numImages + i] = -y_simulator[i]
    '''
    random shuffle of both X en y. setting the random.set_state equasl guarantees that both arrays are shuffled
    in the same order --> corresponding X's and y's keep same index number
    '''
    rng_state = np.random.get_state()
    np.random.shuffle(X_simulator)
    np.random.set_state(rng_state)
    np.random.shuffle(y_simulator)
    return X_simulator, y_simulator

def train_val_split(X, y, split):
    cut = int((1-split) * len(X))
    X_train = X[:cut:]
    y_train = y[:cut:]
    X_val = X[cut::]
    y_val = y[cut::]
    return X_train, X_val, y_train, y_val


def define_model():
    model = Sequential()
    # input: 320x160 images with 3 channels -> (320, 160, 3) tensors.
    # this applies 24 convolution filters of size 5x5 each.
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), input_shape=(40, 160, 3), activation='relu'))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2), activation='relu'))
    #model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model

def save_model(model):
    from keras.models import model_from_json
    model_json = model.to_json()
    with open("model.json", "w") as f:
        json.dump(model_json,f)
    model.save_weights("model.h5")

X_sim, y_sim = dataimport()
X_train, X_val, y_train, y_val = train_val_split(X_sim, y_sim, 0.10)
print(X_train.shape, y_train.shape)

model = define_model()
model.fit(X_train, y_train, batch_size=64, nb_epoch=2)
#data_generator = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
#data_train = data_generator.flow(X_train, y_train)

#model.fit_generator(data_train, samples_per_epoch=48216, nb_epoch=5, validation_data=(X_val, y_val))
#model.fit_generator(data_train, samples_per_epoch=48216, nb_epoch=5)

save_model(model)
preds = model.predict(X_train[:1000:])
for i in range(1000):
    print(y_train[i],preds[i])


#score = model.evaluate(X_train,y_train)

