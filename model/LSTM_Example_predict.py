import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pybithumb

data = pybithumb.get_ohlcv('ETH', interval="hour6")
df = pd.DataFrame(data)
print(df)

df = df[-800:]
plt.figure(figsize=(16, 9))
sns.lineplot(y=df['close'], x=df.index)
plt.xlabel('time')
plt.ylabel('price')\

#plt.show()

from sklearn.preprocessing import MinMaxScaler

df.sort_index(ascending=False).reset_index(drop=True)

scaler = MinMaxScaler()
scale_cols = ['open', 'high', 'low', 'close', 'volume']
df_scaled = scaler.fit_transform(df[scale_cols])
df_scaled = pd.DataFrame(df_scaled)
df_scaled.columns = scale_cols

TEST_SIZE = 200
WINDOW_SIZE = 20

train = df_scaled[:-TEST_SIZE]
test = df_scaled[-TEST_SIZE:]

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

from sklearn.model_selection import train_test_split

feature_cols = ['open', 'high', 'low', 'volume']
label_cols = ['close']

train_feature = train[feature_cols]
train_label = train[label_cols]

train_feature, train_label = make_dataset(train_feature, train_label)

x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
print(x_train.shape, x_valid.shape)

test_feature = test[feature_cols]
test_label = test[label_cols]

print(test_feature.shape, test_label.shape)

test_feature, test_label = make_dataset(test_feature, test_label, 20)
print(test_feature.shape, test_label.shape)

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from numpy import argmax
from keras.models import load_model


model = load_model('tmp_checkpoint.h5')

pred = model.predict(test_feature)

pred = np.transpose(pred)
test_label = np.transpose(test_label)
print(pred[0].shape)
print(test_label[0].shape)

pred = pred[0]
test_label = test_label[0]

prod = 1

before_day = 7
period = 4

for i in range(before_day * period):
    #print(pred[i+1], pred[i])
    if pred[i+1]/pred[i] > 1:
        temp = (test_label[i+1]/test_label[i]) * (1 - 0.05 / 100)
        prod *= temp
        print(temp, prod)

plt.figure(figsize=(12, 9))
plt.plot(test_label[-before_day * period:], label = 'actual')
plt.plot(pred[-before_day * period:], label = 'prediction')
plt.legend()
plt.show()