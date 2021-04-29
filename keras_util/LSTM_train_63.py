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

    #print(allData)
    print(len(allData[0]))

    df = allData

    size = len(df)
    print(size, "num of data")

    label_list = [label for _ in range(size)]
    return df, label_list

def derivative(df):
    new_df = []
    for data in df:
        # Dataframe은 전체 데이터셋으로 데이터 수 * (동영상 프레임 수)
        data = np.array(data)
        new_data = []
        for i in range(1, len(data)):
            result = data[i] - data[i - 1]
            result.tolist()
            new_data.append(result)
        new_df.append(new_data)
    return new_df

def concat_origin_d(origin, d):
    for data_n in range(len(origin)):
        for frame_n in range(len(origin[0])):
            try:
                origin[data_n][frame_n].append(d[data_n][frame_n])
            except:
                pass
    return origin

def frame15(df, label):
    new_label = np.array(label)
    mod_size =15
    data_i = 0
    target = []
    for data in df:
        if len(data) < mod_size:
            target.append(data_i)
        data_i += 1
    print('삭제한 데이터 index :', target)
    for i in target[::-1]:
        del df[i]
        new_label = np.delete(new_label, i)

    # # padded = pad_sequences(df, dtype = 'float64', padding = 'pre')
    # # df = padded

    new_df = []
    for data in df:
        max_diff = 0
        local_i = 0
        for i in range(len(data[:-mod_size])):
            target_15 = [[0.] * 63] * mod_size
            local_diff = 0
            frame_before = np.array(data[i])
            if i > 5 or i < len(data[:-mod_size]) - 5:
                for frame in data[i:i + mod_size]:
                    frame = np.array(frame)
                    local_diff += np.sum(np.abs(frame - frame_before))
                    frame_before = frame
                if local_diff > max_diff:
                    max_diff = local_diff
                    target_15 = data[i:i + mod_size]
                    local_i = i
        # print(local_i)
        # print(max_diff)
        new_df.append(target_15)
    return new_df, new_label

    print(len(new_df))
    print(len(new_df[0][0]))

    df = np.array(new_df)


df, label = load_data("../video_output/LSTM_DATASET2/csv_list_LSTM.txt")
label = label[0]

# min_frame = 60
# for i in df:
#     if min_frame > len(i):
#         min_frame = len(i)
#         print(min_frame)
# print(min_frame)
# exit()

print(len(df), len(label))

data_i = 0
target = []
for data in df:
    if len(data) < 35 or len(data) > 50:
        target.append(data_i)
    data_i += 1
print('삭제한 데이터 index :', target)
for i in target[::-1]:
    del df[i]
    del label[i]

label = np.array(label)
label = label - 1

#df = derivative(df)
#df = concat_origin_d(df, d_df)
#df, label = frame15(df, label)

padded = pad_sequences(df, dtype = 'float64', padding = 'post')

df = padded

frame_size = len(df[0])
data_size = len(df[0][0])



df = np.array(df)
#print(df)
print(df.shape) # 82 * 46 * 63

test_ratio = 0.2
test_num = int(len(df) * 0.2)

#df = df

index = []
for i in range(len(df)):
    if i % 5 != 0:
        index.append(i)
train_x = df[index, :, :]
train_y = label[index]

test_x = df[::5, :, :]
test_y = label[::5]

col_name = [str(i) for i in range(0, 63)]
#print(col_name)

test_y = test_y.astype(np.int64)
train_y = train_y.astype(np.int64)
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

model = keras.Sequential([
    # keras.layers.Dense(63, activation = 'relu'),
    # keras.layers.Dense(400, activation = 'relu'),
    # keras.layers.Dense(100, activation='relu'),
    # keras.layers.Dense(60, activation='relu'),
    # keras.layers.Dense(30, activation='relu'),
    # #keras.layers.Dense(30, activation = 'relu'),
    # keras.layers.Dense(15, activation='softmax')

    #keras.layers.Embedding(500, 4, mask_zero=True),

    #keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None),

    keras.layers.LSTM(150,
                      input_shape=(frame_size, 63),
                      activation = 'relu',
                      return_sequences=False),
    # keras.layers.Dense(300, activation='relu', kernel_initializer='he_normal'),
    keras.layers.Dense(150, activation='relu', kernel_initializer='he_normal'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(70, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(4, activation = 'softmax')

    # keras.layers.LSTM(128,
    #                   input_shape=(frame_size, 63),
    #                   activation='relu',
    #                   return_sequences=True),
    # keras.layers.LSTM(64, return_sequences=True),
    # keras.layers.LSTM(32, return_sequences=True),
    # keras.layers.LSTM(16),
    # keras.layers.Dense(4, activation = 'softmax')
    ])

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              #loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

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
