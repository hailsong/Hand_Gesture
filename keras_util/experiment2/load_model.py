import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras.models.load_model(
    'model_save/my_model.h5'
)
print(model.summary())


df = pd.read_csv("Threshold Analysis table.csv")
print(df.tail())
size = df.shape[0]
print(size, "num of data")

df = df.sample(frac=1).reset_index(drop=True) #shuffle

test_ratio = 0.2
test_num = int(size - size * test_ratio)

train = df[:test_num]
test = df[test_num:]

train_x = train[['FA1', 'FA2', 'FA3', 'FA4', 'FA5', 'FD1', 'FD2', 'FD3', 'FD4', 'FD5', 'HA1', 'HA2', 'HA3', 'HA4', 'HA5']].to_numpy()
train_y = train['real'].to_numpy()

test_x = test[['FA1', 'FA2', 'FA3', 'FA4', 'FA5', 'FD1', 'FD2', 'FD3', 'FD4', 'FD5', 'HA1', 'HA2', 'HA3', 'HA4', 'HA5']].to_numpy()
test_y = test['real'].to_numpy()

prediction = model.predict(test_x[[5]])
print(prediction[0])
print(test_y[5])