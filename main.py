import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import boston_housing
from matplotlib import pyplot

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

print(test_targets)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def calc(k, train_data, train_targets):
    num_val_samples = len(train_data) // k
    num_epochs = 75
    all_scores = []

    for i in range(k):
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
        model = build_model()
        model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0,
                        validation_data=(val_data,val_targets))
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)

        # loss = res.history['loss']
        # mae = res.history['mae']
        # v_loss = res.history['val_loss']
        # v_mae = res.history['val_mae']
        # x = range(1, num_epochs + 1)
        # pyplot.subplot(211)
        # pyplot.plot(x, loss)
        # pyplot.plot(x, v_loss)
        # pyplot.title('Loss')
        # pyplot.ylabel('loss')
        # pyplot.xlabel('epochs')
        # pyplot.legend(['train', 'test'], loc='upper left')
        # pyplot.subplot(212)
        # pyplot.title('Accuracy')
        # pyplot.plot(x, mae)
        # pyplot.plot(x, v_mae)
        # pyplot.legend(['train', 'test'], loc='upper left')
        # pyplot.show()
        return np.mean(all_scores)


mae_vec = []
for i in range(10):             # for each k we can calculate mean of mae and then plot it
    mae_vec.append(calc(i+2, train_data, train_targets))

pyplot.plot(range(10)[2:11], mae_vec)
pyplot.show()
