import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python import keras
import matplotlib.pyplot as plt

model = keras.models.load_model(
    'keras_util/model_save/my_model.h5'
)

def predict_static(input): #(1,15) shape의 numpy array 넣어주기
    input = input[np.newaxis]
    print(input.shape)
    print(input)
    try:
        prediction = model.predict(input)
        return np.argmax(prediction[0])
    except:
        return 0
