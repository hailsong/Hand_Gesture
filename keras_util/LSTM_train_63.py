import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_data(filename):
    f = open(filename, 'r')
    label = []
    _list = []
    while True:
        line = f.readline()
        if not line: break
        print(line)
        label_1 = line.split('/')[-2]
        label.append(int(label_1.split('_')[-1]))
        _list.append(line[:-1])
    f.close()
    print(_list)

    allData = [] # 읽어 들인 csv파일 내용을 저장할 빈 리스트를 하나 만든다
    for file in _list:
        file = '../video_output/' + file
        df = pd.read_csv(file) # for구문으로 csv파일들을 읽어 들인다
        df = df[1:]
        df2 = df[[str(i) for i in range(63)]]
        df_list = df2.values.tolist()
        allData.append(df_list) # 빈 리스트에 읽어 들인 내용을 추가한다

    print(allData)
    print(len(allData[0]))

    df = allData

    size = len(df)
    print(size, "num of data")

    label_list = [label for _ in range(size)]
    return df, label_list

df, label = load_data("../video_output/LSTM_DATASET/csv_list_LSTM.txt")
label = label[0]

print(len(df), len(label))

padded = pad_sequences(df, dtype = 'float64', padding = 'post')
df = padded

frame_size = len(df[0])
data_size = len(df[0][0])
#print(frame_size, data_size)

test_ratio = 0.2
test_num = int(len(df) * 0.2)

train_x = []
train_y = []
test_x = []
test_y = []



col_name = [str(i) for i in range(0, 63)]
#print(col_name)

print(train['FILENAME'].to_numpy())
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

#print(len(train_y), len(test_y)) #1에서 14사이 정수 label


train_x = np.array(train_x)

print(train_x.shape)
#print(test_x.shape)

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
                      input_shape=(20, 63),
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
