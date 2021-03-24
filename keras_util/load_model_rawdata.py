import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras.models.load_model(
    'model_save/my_model_21.h5'
)
print(model.summary())


df = pd.read_csv("output.csv")
print(df.tail())
size = df.shape[0]
print(size, "num of data")

df = df.sample(frac=1).reset_index(drop=True) #shuffle

test_ratio = 0.2
test_num = int(size - size * test_ratio)

train = df[:test_num]
test = df[test_num:]

col_name = [str(i) for i in range(0, 63)]

train_x = train[col_name].to_numpy()
train_y = train['FILENAME'].to_numpy()

test_x = test[col_name].to_numpy()
test_y = test['FILENAME'].to_numpy()

prediction = model.predict(test_x[[5]])
print(len(test_x[[5]][0]))
print(prediction[0])
print(test_y[5])
