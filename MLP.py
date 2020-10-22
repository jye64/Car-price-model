
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
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


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

print(data.isnull().sum())

# ============= EDA and Preprocessing ======================

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

model = keras.models.Sequential([
    layers.Dense(38, input_shape=X_train_scaled.shape[1:], kernel_initializer='normal', activation='relu'),
    layers.Dense(175, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
history = model.fit(X_train_scaled, Y_train_scaled, epochs=100, validation_data=(X_valid_scaled, Y_valid_scaled))

y_train_pred_scaled = model.predict(X_train_scaled)
y_train_pred = scalerY.inverse_transform(y_train_pred_scaled)

results = X_train.copy()
results['actual'] = Y_train
results['predicted'] = y_train_pred
results = results[['predicted', 'actual']]
print(results)


# =================== Accuracy and Evaluation =====================

y_test_pred_scaled = model.predict(X_test_scaled)
y_test_pred = scalerY.inverse_transform(y_test_pred_scaled)

rmse = np.sqrt(mean_squared_error(Y_test, y_test_pred))
print(rmse)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
