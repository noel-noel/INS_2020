from keras.datasets import imdb
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
import numpy as np

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vector_length = 32
model_A = Sequential()
model_A.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model_A.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_A.add(Dense(1, activation='sigmoid'))
model_A.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Start fitting A")
model_A.fit(X_train, y_train, epochs=3, batch_size=64)

print("Start evaluating A")
scores_A = model_A.evaluate(X_test, y_test, verbose=0)

model_B = Sequential()
model_B.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model_B.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_B.add(MaxPooling1D(pool_size=2))
model_B.add(LSTM(100))
model_B.add(Dense(1, activation='sigmoid'))
model_B.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Start fitting B")
model_B.fit(X_train, y_train, epochs=3, batch_size=64)

print("Start evaluating B")
scores_B = model_B.evaluate(X_test, y_test, verbose=0)

print("Model_A accuracy: %.2f%%" % (scores_A[1]*100))
print("Model_B accuracy: %.2f%%" % (scores_B[1]*100))

def ensemble_prediction(models, weights, X):
    yhats = [model.predict(X) for model in models]
    yhats = np.array(yhats)
    weighted_res = np.tensordot(yhats, weights, axes=((0),(0)))
    return np.rint(weighted_res)


models = [model_A, model_B]

weights = [0.2, 0.8]
y_hat_not_eq = ensemble_prediction(models, weights, X_test)
ACC = accuracy_score(y_test, y_hat_not_eq)
print("Ensemble accuracy with unequal weights:%.2f%%" % (ACC*100))

weights = [0.3, 0.7]
y_hat_not_eq = ensemble_prediction(models, weights, X_test)
ACC = accuracy_score(y_test, y_hat_not_eq)
print("Ensemble accuracy with unequal weights:%.2f%%" % (ACC*100))

weights = [0.5, 0.5]
y_hat_eq = ensemble_prediction(models, weights, X_test)
ACC = accuracy_score(y_test, y_hat_eq)
print("Ensemble accuracy with equal weights:%.2f%%" % (ACC*100))


def vectorize(sequences, dimension=top_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def custom_review(name):
    f = open(name, 'r')
    f_text = f.read()
    index = imdb.get_word_index()
    txt = []
    for i in f_text:
        if i in index and index[i] < top_words:
            txt.append(index[i])

    txt = vectorize([txt])
    return txt


weights = [0.3, 0.7]
text = custom_review("pos_review.txt")
text = sequence.pad_sequences(text, maxlen=max_review_length)
print("Testing positive:")
result = ensemble_prediction(models, weights, text)
if result == 1:
    print("Positive review")
else:
    print("Negative review")

print("Testing negative:")
text = custom_review("neg_review.txt")
text = sequence.pad_sequences(text, maxlen=max_review_length)
result = ensemble_prediction(models, weights, text)

if result == 1:
    print("Positive review")
else:
    print("Negative review")