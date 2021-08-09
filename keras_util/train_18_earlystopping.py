import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)

df = pd.read_csv("../video_output/output_sum_18.csv")

# TODO Gesture num 15 일단 제외하고 학습
df = df[df['FILENAME'] < 15]

df = df[1:]
print(df.tail())
size = df.shape[0]
print(size, "num of data")

df = df.sample(frac=1).reset_index(drop=True) #shuffle

test_ratio = 0.2
test_num = int(size - size * test_ratio)

train = df[:test_num]
test = df[test_num:]

col_name = [str(i) for i in range(0, 18)]
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

from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)

print(len(train_y), len(test_y)) #1에서 14사이 정수 label

print(train_x)
print(train_y)

model = keras.Sequential([
    keras.layers.Dense(18, activation = 'relu'),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(20, activation='relu'),
    # keras.layers.Dropout(0.2),
    keras.layers.Dense(15, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss', patience=5)
filename = 'model_save/my_model_18_2.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

hist = model.fit(train_x, train_y, epochs=100, batch_size=10, validation_data=(valid_x, valid_y),
                 callbacks=[early_stop, checkpoint])

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

# model.save('model_save/my_model_21.h5')
# print('new model saved')
