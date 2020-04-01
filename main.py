from keras.layers import Input, Dense
from keras.models import Model
from matplotlib import pyplot
import pandas as pd
import numpy as np

# data parameters
nrow = 500
ncol = n_in = 6
mu = -5
sigma = np.sqrt(10)
noise_sigma = np.sqrt(0.3)
noise_mu = 0

# generating data
X = np.zeros((nrow, ncol))
Y = np.zeros(nrow)

for i in range(nrow):
    x = sigma * np.random.randn(1) + mu
    e = noise_sigma*np.random.randn(1) + noise_mu
    X[i, :] = (-x**3 + e, np.sin(3*x) + e, np.exp(x) + e, x + 4 + e, -x + np.sqrt(np.abs(x)) + e, x + e)
    Y[i] = np.log(np.abs(x)) + e

# normalizing data.
mean = X.mean(axis=0)
X -= mean
std = X.std(axis=0)
X /= std

# for encoder
main_input = Input(shape=(n_in,))
encoded = Dense(60, activation='relu')(main_input)
encoded = Dense(60, activation='relu')(encoded)
encoded = Dense(35, activation='relu')(encoded)
encoded = Dense(3, activation='linear')(encoded)       # a-e bottleneck

# for decoder
decoded = Dense(35, activation='relu', kernel_initializer='normal', name='dec_1')(encoded)
decoded = Dense(60, activation='relu', name='dec_2')(decoded)
decoded = Dense(60, activation='relu', name='dec_3')(decoded)
decoded = Dense(n_in, name="out_aux")(decoded)

# for regression
predicted = Dense(64, activation='relu', kernel_initializer='normal')(encoded)
predicted = Dense(64, activation='relu')(predicted)
predicted = Dense(64, activation='relu')(predicted)
predicted = Dense(64, activation='relu')(predicted)
predicted = Dense(1, name="out_main")(predicted)

# tie it together
model = Model(inputs=main_input, outputs=[decoded, predicted])
model.compile(optimizer='adam', loss='mse')
num_epochs = 125
res = model.fit(X, [X, Y], epochs=num_epochs, batch_size=5, verbose=1, validation_split=0.2)

# coders & regression model separately
encoder = Model(main_input, encoded)
regr = Model(main_input, predicted)

dec_input = Input(shape=(3,))               # I would love to write Model(encoded, decoded) but I cannot
decoder = model.get_layer('dec_1')(dec_input)
decoder = model.get_layer('dec_2')(decoder)
decoder = model.get_layer('dec_3')(decoder)
decoder = model.get_layer('out_aux')(decoder)

decoder = Model(dec_input, decoder)

encoder.save('encoder.h5')
decoder.save('decoder.h5')
regr.save('regression.h5')

# saving to csv various required data
pd.DataFrame(np.round(X, 3)).to_csv("initial_X.csv")

enc = encoder.predict(X)
pd.DataFrame(enc).to_csv("encoded_X.csv")

dec = decoder.predict(enc)
pd.DataFrame(np.round(dec, 3)).to_csv("decoded_X.csv")

y_hat = regr.predict(X, verbose=0)
pd.DataFrame(np.round(Y, 3)).to_csv("initial_Y.csv")
pd.DataFrame(np.round(y_hat, 3)).to_csv("predicted_Y.csv")

# just to look at what we've got
loss_idk, dec_mse, regr_mse = model.evaluate(X, [X, Y])
print(dec_mse, regr_mse)

# training results
loss = res.history['loss']
v_loss = res.history['val_loss']
x = range(1, num_epochs+1)

pyplot.plot(x, loss)
pyplot.plot(x, v_loss)
pyplot.title('Loss')
pyplot.ylabel('loss')
pyplot.xlabel('epochs')
pyplot.show()
