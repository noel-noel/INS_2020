import matplotlib as matplotlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models

from keras import layers
from keras.datasets import imdb

N = 1000
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=N)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)


def vectorize(sequences, dimension=N):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


data = vectorize(data)

targets = np.array(targets).astype("float32")
test_x = data[:10000]

test_y = targets[:10000]

train_x = data[10000:]

train_y = targets[10000:]
# Input - Layer
model = models.Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(N, )))

# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
# Output- Layer
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
results = model.fit(train_x, train_y, epochs=2, batch_size=500, validation_data=(test_x, test_y))
print(np.mean(results.history["val_accuracy"]))

plt.subplot(211)
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(212)
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


def custom_review(name):
    f = open(name, 'r')
    f_text = f.read()
    index = imdb.get_word_index()
    txt = []
    for i in f_text:
        if i in index and index[i] < 10000:
            txt.append(index[i])

    txt = vectorize([txt])
    return txt


text = custom_review("pos_review.txt")
print("Testing positive:")
result = model.predict_classes(text)
if result == 1:
    print("Positive review")
else:
    print("Negative review")

print("Testing negative:")
text = custom_review("neg_review.txt")
result = model.predict_classes(text)

if result == 1:
    print("Positive review")
else:
    print("Negative review")
