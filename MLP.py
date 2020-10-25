
# Disable tensorflow INFO messages in console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


# ================= Configure GPU ====================
config = tf.compat.v1.ConfigProto(gpu_options=
                                  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                  # device_count = {'GPU': 1}
                                  )
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# ================= import data =================
dataFile = 'audi.csv'
data = pd.read_csv(dataFile, sep=',')
print(data)

# TODO: Shorten training time
data = data[:500]

# ============= EDA and Preprocessing ======================

print(data.isnull().sum())

# compute age of car by subtracting 2020 from the 'year' field
data["age_of_car"] = 2020 - data["year"]
data = data.drop(columns=["year"])

corr_matrix = data.corr()
print(corr_matrix['price'].sort_values(ascending=False))

# one-hot encoding for categorical attributes
data_onehot = pd.get_dummies(data, columns=['model', 'transmission', 'fuelType'])

# separate features and target variable
X = data_onehot.drop(['price'], axis=1)
Y = data_onehot[['price']]

# split training and test set
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full, test_size=0.2, random_state=42)

print(X_train.shape)

scalerX = StandardScaler()
X_train_scaled = scalerX.fit_transform(X_train)
X_valid_scaled = scalerX.transform(X_valid)
X_test_scaled = scalerX.transform(X_test)

scalerY = StandardScaler()
Y_train_scaled = scalerY.fit_transform(Y_train)
Y_valid_scaled = scalerY.transform(Y_valid)
Y_test_scaled = scalerY.transform(Y_test)


# ========================= Modeling ==============================

input_shape = X_train.shape[1:]


def build_model(n_hidden=1, n_neurons=30, learning_rate=0.01, init='glorot_uniform', momentum=0, activation='relu'):
    model = keras.models.Sequential()

    # input layer
    model.add(layers.Dense(38, activation=activation, kernel_initializer=init, input_shape=input_shape))

    # hidden layers
    for layer in range(n_hidden):
        model.add(layers.Dense(n_neurons, activation=activation, kernel_initializer=init))

    # output layer
    model.add(layers.Dense(1, kernel_initializer=init))
    optimizer = keras.optimizers.SGD(learning_rate, momentum=momentum)
    model.compile(loss='mse', optimizer=optimizer)
    return model


keras_reg = KerasRegressor(build_model)


# ==================== Hyper-parameters Tuning ===================

hidden_layers = [5, 6, 7, 8]
neurons = list(range(19, 100))
learn_rate = [0.1, 0.01, 0.001, 0.02, 0.002, 0.03, 0.003]
init_mode = ['uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
momen = [0, 0.5, 0.6, 0.7, 0.8, 0.9]
activate = ['relu', 'elu']


param_grid = dict(n_hidden=hidden_layers, n_neurons=neurons, learning_rate=learn_rate,
                  init=init_mode, momentum=momen, activation=activate)

rnd_search_cv = RandomizedSearchCV(keras_reg, param_grid, cv=5)
rnd_search_cv.fit(X_train_scaled, Y_train_scaled, epochs=100, validation_data=(X_valid_scaled, Y_valid_scaled),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

print(rnd_search_cv.best_params_)

# Plot loss vs epochs of best estimator after randomized search CV
history = rnd_search_cv.best_estimator_.model.history

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.grid(True)
plt.show()

y_train_pred_scaled = rnd_search_cv.predict(X_train_scaled)
y_train_pred = scalerY.inverse_transform(y_train_pred_scaled)

results = X_train.copy()
results['actual'] = Y_train
results['predicted'] = y_train_pred
results = results[['predicted', 'actual']]
print(results)

# ==================== Accuracy & Evaluation ===========================
y_test_pred_scaled = rnd_search_cv.predict(X_test_scaled)
y_test_pred = scalerY.inverse_transform(y_test_pred_scaled)

test_rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
print(test_rmse)


# ==================== No Hyper-parameters Tuning ==========================

# leaky_relu = keras.layers.LeakyReLU(alpha=0.2)

# model = keras.models.Sequential([
#     layers.Dense(38, input_shape=X_train_scaled.shape[1:], activation='relu', kernel_initializer='normal'),
#     layers.Dense(38, activation='relu', kernel_initializer='normal'),
#     layers.Dense(38, activation='relu', kernel_initializer='normal'),
#     layers.Dense(38, activation='relu', kernel_initializer='normal'),
#     layers.Dense(38, activation='relu', kernel_initializer='normal'),
#     layers.Dense(38, activation='relu', kernel_initializer='normal'),
#     layers.Dense(1, kernel_initializer='normal')
# ])
#
# model.summary()
#
# model.compile(loss='mse', optimizer='sgd')
# history = model.fit(X_train_scaled, Y_train_scaled, epochs=150, validation_data=(X_valid_scaled, Y_valid_scaled),
#                     callbacks=[keras.callbacks.EarlyStopping(patience=20)])
#
# y_train_pred_scaled = model.predict(X_train_scaled)
# y_train_pred = scalerY.inverse_transform(y_train_pred_scaled)
#
# results = X_train.copy()
# results['actual'] = Y_train
# results['predicted'] = y_train_pred
# results = results[['predicted', 'actual']]
# print(results)
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.grid(True)
# plt.show()
#
# # =================== Accuracy and Evaluation =====================
#
# y_test_pred_scaled = model.predict(X_test_scaled)
# y_test_pred = scalerY.inverse_transform(y_test_pred_scaled)
#
# test_rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
# print(test_rmse)
