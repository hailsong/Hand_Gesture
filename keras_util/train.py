import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)

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

# print(train_x)
# print(train_y)
# print(train_x.shape)
# print(len(train_y)) #1에서 14사이 정수 label

test_x = test[['FA1', 'FA2', 'FA3', 'FA4', 'FA5', 'FD1', 'FD2', 'FD3', 'FD4', 'FD5', 'HA1', 'HA2', 'HA3', 'HA4', 'HA5']].to_numpy()
test_y = test['real'].to_numpy()

print(len(train_y), len(test_y)) #1에서 14사이 정수 label

model = keras.Sequential([
    keras.layers.Dense(15, activation = 'relu'),
    keras.layers.Dense(25, activation = 'relu'),
    #keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dense(15, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(train_x, train_y, epochs=100)

test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

print('\n테스트 정확도:', test_acc)

prediction = model.predict(test_x[[5]])
print(prediction[0])
print(test_y[5])




fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
#loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
#acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

model.save('model_save/my_model.h5')
print('new model saved')
