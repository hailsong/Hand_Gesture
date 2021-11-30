import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)

df = pd.read_csv("../video_output/지문자/output_21.csv")
df = df[1:]
print(df.tail())
size = df.shape[0]
print(size, "num of data")

df = df.sample(frac=1).reset_index(drop=True) #shuffle

test_ratio = 0.2
test_num = int(size - size * test_ratio)

train = df[:test_num]
test = df[test_num:]


col_name = [str(i) for i in range(0, 21)]
print(col_name)

print(train['FILENAME'].to_numpy())

train_x = train[col_name].to_numpy()
train_y = train['FILENAME'].to_numpy()
train_y = train_y.astype(np.int64)

# print(train_x)
# print(train_y)
# print(train_x.shape)
# print(len(train_y)) #1에서 14사이 정수 label

test_x = test[col_name].to_numpy()
test_y = test['FILENAME'].to_numpy()
test_y = test_y.astype(np.int64)

print(len(train_y), len(test_y)) #1에서 14사이 정수 label

model = keras.Sequential([
    keras.layers.Dense(21, activation = 'relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(50, activation = 'relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(50, activation='relu'),
    #keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dense(32, activation='softmax')
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

model.save('model_save/my_model_21_KSL.h5')
print('new model saved')
