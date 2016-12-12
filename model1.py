import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

model = Sequential()
# input: 320x160 images with 3 channels -> (320, 160, 3) tensors.
# this applies 24 convolution filters of size 5x5 each.
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2,2), input_shape=(160, 80, 3))) #158x78x24
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2,2)))
model.add(Convolution2D(64, 3, 3))
model.add(Convolution2D(64, 3, 3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

'''
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)
'''