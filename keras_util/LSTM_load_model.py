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

df, label = load_data("../video_output/LSTM_DATASET2/csv_list_LSTM.txt")
label = label[0]

print(len(df), len(label))



# TODO padding ver
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
padded = pad_sequences(df, dtype = 'float64', padding = 'pre')
df = padded

# TODO 15프레임 ver
'''
프레임별로 데이터 들어오는데...
'''
df = np.array(df)

label = np.array(label)
label = label - 1



# frame_size = len(df[0])
# data_size = len(df[0][0])
df = np.array(df)
print(df)
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
model = keras.models.load_model(
    'model_save/my_model_63.h5'
)

prediction = model.predict(test_x[[67]])
print(test_x[[67]].shape)
print(prediction[0])
print(test_y[67])



