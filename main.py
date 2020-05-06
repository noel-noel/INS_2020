import var7
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import FeatureLogger

batch_size = 30
num_classes = 3
epochs = 35
img_size = 25
sample_size = 1000

X, Y = var7.gen_data(size=sample_size, img_size=img_size)
X, Y = shuffle(X, Y)
x_train = X[0:900, :, :]
x_test = X[900:1000, :, :]
y_train = Y[0:900]
y_test = Y[900:1000]
x_train = x_train.reshape(900, img_size, img_size, 1)
x_test = x_test.reshape(100, img_size, img_size, 1)

input_shape = (img_size, img_size, 1)

onehotencoder = OneHotEncoder()
y_train = onehotencoder.fit_transform(y_train).toarray()
y_test = onehotencoder.fit_transform(y_test).toarray()

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer="normal"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(4, 4), activation='relu', input_shape=input_shape, kernel_initializer="normal"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_initializer="normal"))
model.add(Dense(32, activation='relu', kernel_initializer="normal"))
model.add(Dense(16, activation='relu', kernel_initializer="normal"))
model.add(Dense(num_classes, activation='softmax', kernel_initializer="normal"))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])

# ! ONLY 3 LINES BELOW ARE CHANGED SINCE pr6. For custom callback see FeatureLogger.py
print("Input epochs when logging needed:")
epochs_to_log = list(map(int, input().split()))
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),
                    callbacks=[FeatureLogger.FeatureLogger(epochs_to_log=epochs_to_log)])
# !
plt.subplot(211)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.subplot(212)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Couldn't make any better than
# Test loss: 0.3850960233807564
# Test accuracy: 0.8700000047683716