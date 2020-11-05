
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
from sklearn.metrics import mean_absolute_error
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

# ============= EDA and Preprocessing ======================

# Detect if there is null value
print(data.isnull().sum())

# check feature-target correlation
corr_matrix = data.corr()
print(corr_matrix['price'].sort_values(ascending=False))

# heatmap
sns.heatmap(corr_matrix, annot=True)
plt.show()

# one-hot encoding for categorical attributes
data_onehot = pd.get_dummies(data, columns=['model', 'transmission', 'fuelType'])

corr_matrix_onehot = data_onehot.corr()
print(corr_matrix_onehot['price'].sort_values(ascending=False))

# separate features and target variable
X = data_onehot.drop(['price'], axis=1)
Y = data_onehot[['price']]

# split training, validation, test set as 60% : 20% : 20%
X_train_full, X_test, Y_train_full, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full, test_size=0.25, random_state=42)

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

# standardization
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


def build_model(n_hidden=1, n_neurons=38, learning_rate=0.01, init='glorot_uniform', activation='relu'):
    model = keras.models.Sequential()

    # input layer
    model.add(layers.Dense(38, activation=activation, kernel_initializer=init, input_shape=input_shape))

    # hidden layers
    for layer in range(n_hidden):
        model.add(layers.Dense(n_neurons, activation=activation, kernel_initializer=init))

    # output layer
    model.add(layers.Dense(1, kernel_initializer=init))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


keras_reg = KerasRegressor(build_model)


# ==================== Hyper-parameters Tuning ===================

hidden_layers = [5, 6, 7, 8]
neurons = list(range(20, 100))
learn_rate = [0.01, 0.001, 0.002]
init_mode = ['he_normal', 'random_normal', 'glorot_normal']
activate = ['relu', 'elu']


param_grid = dict(n_hidden=hidden_layers, n_neurons=neurons, learning_rate=learn_rate,
                  init=init_mode, activation=activate)

rnd_search_cv = RandomizedSearchCV(keras_reg, param_grid, cv=5)
rnd_search_cv.fit(X_train_scaled, Y_train_scaled, epochs=100, validation_data=(X_valid_scaled, Y_valid_scaled),
                  callbacks=[keras.callbacks.EarlyStopping(patience=30)])

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


# =================== Accuracy and Evaluation =====================

# train set performance
Y_train_pred_scaled = rnd_search_cv.predict(X_train_scaled)
Y_train_pred = scalerY.inverse_transform(Y_train_pred_scaled)

results = X_train.copy()
results['actual'] = Y_train
results['predicted'] = Y_train_pred
results = results[['predicted', 'actual']]
results['predicted'] = results['predicted'].round(0)
print(results)

results = pd.DataFrame.reset_index(results, drop=True)

plt.plot(results['predicted'].head(100), label='predicted')
plt.plot(results['actual'].head(100), label='actual')
plt.xlabel('index in train set')
plt.ylabel('price')
plt.title('Predicted vs Actual in Train set')
plt.legend()
plt.show()

# test set performance
Y_test_pred_scaled = rnd_search_cv.predict(X_test_scaled)
Y_test_pred = scalerY.inverse_transform(Y_test_pred_scaled)

test_results = X_test.copy()
test_results['actual'] = Y_test
test_results['predicted'] = Y_test_pred
test_results = test_results[['predicted', 'actual']]
test_results['predicted'] = test_results['predicted'].round(0)
print(test_results)

test_results = pd.DataFrame.reset_index(test_results, drop=True)

plt.plot(test_results['predicted'].head(100), label='predicted')
plt.plot(test_results['actual'].head(100), label='actual')
plt.xlabel('index in test set')
plt.ylabel('price')
plt.title('Predicted vs Actual in test set')
plt.legend()
plt.show()

# Error Metric summary
model_performance = pd.DataFrame(columns=['Train MAE', 'Train RMSE', 'Test MAE', 'Test RMSE'])

Train_MAE = mean_absolute_error(Y_train, Y_train_pred).round(0)
Train_RMSE = np.sqrt(mean_squared_error(Y_train, Y_train_pred)).round(0)

Test_MAE = mean_absolute_error(Y_test, Y_test_pred).round(0)
Test_RMSE = np.sqrt(mean_squared_error(Y_test, Y_test_pred)).round(0)

model_performance = model_performance.append({'Train MAE': Train_MAE,
                        'Train RMSE': Train_RMSE,
                        'Test MAE': Test_MAE,
                        'Test RMSE': Test_RMSE},
                        ignore_index=True)

print(model_performance)
