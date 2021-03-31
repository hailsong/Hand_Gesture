from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from keras.preprocessing import sequence

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models
import pickle

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

def load_data(filename):
    f = open(filename, 'r')
    label = []
    _list = []
    while True:
        line = f.readline()
        if not line: break
        label_1 = line.split('/')[-2]
        label.append(int(label_1.split('_')[-1]))
        _list.append(line[:-1])
    f.close()
    print(_list)

    allData = [] # 읽어 들인 csv파일 내용을 저장할 빈 리스트를 하나 만든다
    for file in _list:
        print(file)
        file = '../video_output/' + file
        df = pd.read_csv(file) # for구문으로 csv파일들을 읽어 들인다
        df = df[1:]
        df2 = df[[str(i) for i in range(63)]]
        df_list = df2.values.tolist()
        allData.append(np.array(df_list)) # 빈 리스트에 읽어 들인 내용을 추가한다

    #print(allData)
    #print(len(allData[0]))

    df = allData


    size = len(df)
    print(size, "num of data")

    label_list = [label for _ in range(size)]
    return df, label_list

df, label = load_data("../video_output/LSTM_DATASET/csv_list_LSTM.txt")
label = np.array(label[0])
print(df[0].shape, label.shape)

print(len(df), len(label))

frame_size = len(df[0])
data_size = len(df[0][0])
print(frame_size, data_size)

test_ratio = 0.2
test_num = int(len(df) * 0.2)

train_x = []
train_y = []
test_x = []
test_y = []

# for i in range(0, data_size):
#     if i % 5 == 0:
#         test_x.append(df[i])
#         test_y.append(label[i])
#     else:
#         train_x.append(df[i])
#         train_y.append(label[i])
#
# print(len(train_x))
# print(train_x[0].shape)
# print(len(train_y))
# print(train_y[49])
#
# print(np.array(train_x))
# exit()
#
# col_name = [str(i) for i in range(0, 63)]
# print(col_name)
#
# print(train['FILENAME'].to_numpy())
#
# train_x = train[col_name].to_numpy()
# train_y = train['FILENAME'].to_numpy()
# train_y = train_y.astype(np.int64)
#
# # print(train_x)
# # print(train_y)
# # print(train_x.shape)
# # print(len(train_y)) #1에서 14사이 정수 label
#
# test_x = test[col_name].to_numpy()
# test_y = test['FILENAME'].to_numpy()
# test_y = test_y.astype(np.int64)

class Data:
    def __init__(self):
        # X = []
        # Y = []
        # maxlength = 0
        # with open(pklname, 'rb') as fin:
        #     frames = pickle.load(fin)
        #     for i, frame in enumerate(frames):
        #         features = frame[0]
        #         maxlength = len(features)
        #         word = frame[1]

                # X.append(np.array(features))
                # Y.append(word)
        X = df
        Y = label
        # X = np.array(X)
        # Y = np.array(Y)

        maxlength = 0
        for x in X:
            if x.shape[0] > maxlength:
                maxlength = x.shape[0]

        new_x = []
        for i in range(len(X)): #각각의 동영상
            new = np.zeros((maxlength, 63))
            for j in range(len(X[i])): #각각의 프레임
                new[j] = X[i][j]
            #print('shape', new.shape)
            new_x.append(new)
        X = new_x

        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in range(len(X)):
            if i%5 == 0:
                test_x.append(X[i])
                test_y.append(Y[i])
            else:
                train_x.append(X[i])
                train_y.append(Y[i])
        X = train_x
        Y = train_y
        #
        # t = Tokenizer()
        # t.fit_on_texts(Y)
        # # print(t.word_index)
        # # Y = to_categorical(Y,len(t.word_index))
        # encoded = t.texts_to_sequences(Y)
        # # print(encoded)
        # one_hot = to_categorical(encoded)
        # # print(one_hot)
        #
        # #(x_train, y_train) = X, one_hot
        # (x_train, y_train) = X, one_hot

        X = tf.stack(X)
        Y = tf.stack(Y)
        X_t = tf.stack(test_x)
        Y_t = tf.stack(test_y)

        (x_train, y_train) = X, Y
        (x_test, y_test) = X_t, Y_t

        print('ytrain', y_train)

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.length = maxlength

class RNN_LSTM(models.Model):
    def __init__(self, maxlen):
        x = layers.Input((maxlen,))
        h = layers.Embedding(maxlen, 128)(x)
        h = layers.LSTM(128,input_shape=(maxlen, 63), dropout=0.2, recurrent_dropout=0.2)(h)
        y = layers.Dense(2, activation='softmax')(h)
        super().__init__(x, y)

        # try using different optimizers and different optimizer configs
        self.compile(loss='binary_crossentropy',
                     optimizer='adam', metrics=['accuracy'])

class Machine:
    def __init__(self):
        self.data = Data()
        self.model = RNN_LSTM(self.data.length)

    def run(self, epochs=3, batch_size=32):
        data = self.data
        model = self.model
        print('Training stage')
        print('==============')

        print(data.x_train.shape)
        print(data.y_train.shape)
        print(data.x_test.shape)
        print(data.y_test.shape)

        model.fit(data.x_train, data.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(data.x_test, data.y_test))

        score, acc = model.evaluate(data.x_test, data.y_test,
                                    batch_size=batch_size)
        print('Test performance: accuracy={0}, loss={1}'.format(acc, score))

def main():
    m = Machine()
    m.run()

if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='run Model')
    # parser.add_argument("--pkl_data_path", help=" ")
    # args = parser.parse_args()
    # pkl_data_path = args.pkl_data_path
    main()

#############
print(len(train_y), len(test_y)) #1에서 14사이 정수 label

print('ㅇㅇㅇㅇ', train_x)

model = keras.Sequential([
    # keras.layers.Dense(63, activation = 'relu'),
    # keras.layers.Dense(400, activation = 'relu'),
    # keras.layers.Dense(100, activation='relu'),
    # keras.layers.Dense(60, activation='relu'),
    # keras.layers.Dense(30, activation='relu'),
    # #keras.layers.Dense(30, activation = 'relu'),
    # keras.layers.Dense(15, activation='softmax')
    keras.layers.Embedding(2, 100),
    keras.layers.LSTM(200,
                      input_shape=(21, 63),
                      activation = 'relu',
                      return_sequences=False),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(2, activation = 'softmax')
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# print(train_x[0].shape, train_y[0])
# exit()

hist = model.fit(train_x, train_y, epochs=10)

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

model.save('model_save/my_model_63.h5')
print('new model saved')
