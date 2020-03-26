from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np


def naive_relu(x):
    assert len(x.shape) == 2    # проверка размерности 2
    x = x.copy()                # копирования от защиты изменения исходного тензора
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape   # проверка, что x и y двумерные тензоры с одинаковой формой
    x = x.copy()                # копирования для защиты от изменения исходного тензора
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


def naive_matrix_mult(A, B):
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i, j] = naive_vector_dot(A[i, :], B[:, j])
    return C


def gen_data():
    a = b = c = (False, True)
    n = 0
    X = np.zeros((2**3, 3))
    Y = np.zeros((2**3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                X[n, ] = (a[i], b[j], c[k])
                Y[n] = (a[i] and not b[j]) or (c[k] != b[j])
                n += 1
    return X, Y


X, Y = gen_data()
print(X)
print(Y)
model = Sequential()
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def el_wise_fit(input_tens, w_list):
    tmp = naive_matrix_mult(np.transpose(w_list[0]), np.transpose(input_tens))
    tmp = naive_relu(tmp)

    tmp = naive_matrix_mult(np.transpose(w_list[2]), tmp)
    tmp = naive_relu(tmp)

    tmp = naive_matrix_mult(np.transpose(w_list[4]), tmp)
    res = 1/(1+np.exp(-tmp))

    return res


def np_fit(input_tens, w_list):
    tmp = np.dot(np.transpose(w_list[0]), np.transpose(input_tens))
    tmp = np.maximum(tmp, 0)

    tmp = np.dot(np.transpose(w_list[2]), tmp)
    tmp = np.maximum(tmp, 0)

    tmp = np.dot(np.transpose(w_list[4]), tmp)
    res = 1/(1+np.exp(-tmp))
    return res


el_wise_res = el_wise_fit(X, model.get_weights())
np_res = np_fit(X, model.get_weights())
print("Element-wise result (untrained):", el_wise_res)
print("Result using numpy lib (untrained):", np_res)

model.fit(X, Y, epochs=50, batch_size=1)

el_wise_res = el_wise_fit(X, model.get_weights())
np_res = np_fit(X, model.get_weights())
print("Element-wise result:", el_wise_res)
print("Result using numpy lib:", np_res)
