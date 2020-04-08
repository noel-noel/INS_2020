from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.utils import to_categorical
from tensorflow import optimizers
from matplotlib import pyplot

from PIL import Image
from numpy import asarray
import tensorflow as tf
import numpy as np


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def looping_optimizers(opt_list, labels):
    acc_list = []
    history_loss_list = []
    history_acc_list = []

    # create model
    for opt in opt_list:
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # train the model
        res = model.fit(train_images, train_labels, epochs=5, batch_size=128)
        history_loss_list.append(res.history['loss'])
        history_acc_list.append(res.history['accuracy'])
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        acc_list.append(test_acc)

    print('test_acc:', acc_list)
    x = range(1, 6)

    pyplot.subplot(211)
    pyplot.title('Loss')
    for loss in history_loss_list:
        pyplot.plot(x, loss)
    pyplot.legend(labels)

    pyplot.subplot(212)
    pyplot.title('Accuracy')
    for acc in history_acc_list:
        pyplot.plot(x, acc)
    pyplot.legend(labels)
    pyplot.show()
    return model


test = 6
opt_list = []

# comparing default
if test == 0:
    opt_list = ("adam", "adagrad", "rmsprop", "sgd")
    looping_optimizers(opt_list, opt_list)

# tuning sgd
if test == 1:
    opt_list.append(optimizers.SGD())                                                     # default
    opt_list.append(optimizers.SGD(learning_rate=0.1))                                    # conf 1
    opt_list.append(optimizers.SGD(momentum=0.9))                                         # conf 2
    opt_list.append(optimizers.SGD(learning_rate=0.1, momentum=0.9))                      # conf 3
    looping_optimizers(opt_list, ("default", "config 1", "config 2", "config 3"))

# tuning adagrad
if test == 2:
    opt_list.append(optimizers.Adagrad())                      # default (0.01)
    opt_list.append(optimizers.Adagrad(learning_rate=0.001))   # conf 1
    opt_list.append(optimizers.Adagrad(learning_rate=0.1))     # conf 2
    looping_optimizers(opt_list, ("default", "config 1", "config 2"))

# tuning rmsprop
if test == 3:
    opt_list.append(optimizers.RMSprop())                                 # default learning_rate=0.001, rho=0.9
    opt_list.append(optimizers.RMSprop(learning_rate=0.01))               # conf 1
    opt_list.append(optimizers.RMSprop(rho=0.1))                          # conf 2
    opt_list.append(optimizers.RMSprop(learning_rate=0.01, rho=0.1))      # conf 3
    looping_optimizers(opt_list, ("default", "config 1", "config 2", "config 3"))

# tuning adam
if test == 4:
    opt_list.append(optimizers.Adam())                          # default (lr=0.001, beta_1=0.9, beta_2=0.999)
    opt_list.append(optimizers.Adam(amsgrad=True))              # conf 1
    opt_list.append(optimizers.Adam(learning_rate=0.01))        # conf 2
    opt_list.append(optimizers.Adam(beta_1=0.1, beta_2=0.1))    # conf 3
    looping_optimizers(opt_list, ("default", "config 1", "config 2", "config 3"))

# comparing best
if test == 5:
    opt_list.append(optimizers.SGD(learning_rate=0.1, momentum=0.9))
    opt_list.append(optimizers.Adagrad(learning_rate=0.1))
    opt_list.append(optimizers.RMSprop())
    opt_list.append(optimizers.Adam())
    looping_optimizers(opt_list, ("SGD", "Adagrad", "RMSProp", "Adam"))


def read_file(path):
    # load the image
    image = Image.open(path).convert('L')
    # convert image to numpy array and without rgb stuff
    data = asarray(image)
    data = data.reshape((1, 28, 28))
    return data


model = looping_optimizers([optimizers.SGD(learning_rate=0.1, momentum=0.9)], ["SGD"])
filename = "0.jpeg"
img = read_file(filename)
Y = model.predict_classes(img)
print("prediction for file " + filename + " -- " + np.array2string(Y[0]))
img = read_file("1.jpeg")
filename = "1.jpeg"
Y = model.predict_classes(img)
print("prediction for file " + filename + " -- " + np.array2string(Y[0]))
model.save("model.h5")
