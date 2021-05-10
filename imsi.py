from keras import applications
# import tensorflow.keras.applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, concatenate, core, Input, RepeatVector, Reshape, Conv2D,BatchNormalization , Activation, MaxPooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping, ProgbarLogger
import os
import numpy as np
import time
# from tensorflow.python.keras.applications.efficientnet import *
from sklearn.metrics import confusion_matrix
import cv2
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

img_width, img_height = 300, 375
batch_size= 8
n_epochs = 30
num_classes = 2
lr = 0.000005

path = '/home/junankim18/Desktop/aaof/data/augmented_data_sex/'
# path = '../data/tmp/'

dir_xray_train = path + 'train/'
dir_xray_val = path + 'val/'
dir_xray_test = path + 'test/'

labels = ['0','1']

train_num =0
val_num = 0

for label in labels:
    num1 = len(os.listdir(dir_xray_train+label))
    num2 = len(os.listdir(dir_xray_val+label))
    train_num +=num1
    val_num +=num2

print('train_num = ', train_num)
print('val_num = ', val_num)

model1 = applications.densenet.DenseNet121(input_shape = (img_height,img_width,3), include_top=False, weights='imagenet',pooling = 'avg')
# model1 = EfficientNetB0(input_shape = (img_height,img_width,3), include_top=False, weights='imagenet',pooling = 'avg')

x = model1.output
# x = layers.Dense(node, activation='swish')(x)
x = Dense(32, activation='relu')(x)

x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=model1.input, outputs=x)
model.summary()


optimizer = "Adam"
opt = optimizers.Adam(lr=lr)

model.compile(loss = "binary_crossentropy", optimizer=opt, metrics=["accuracy"])

if not os.path.exists("/home/junankim18/Desktop/aaof/hdf5/sex/"):
    os.makedirs("/home/junankim18/Desktop/aaof/hdf5/sex/")

# Save the model according to the conditions
progbar = ProgbarLogger(count_mode='steps', stateful_metrics=None)

checkpoint = ModelCheckpoint(
    "/home/junankim18/Desktop/aaof/hdf5/sex/sex.{epoch:02d}-{val_accuracy:.3f}.hdf5",
    monitor='val_accuracy',
    verbose=1,
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    save_frequency='epoch',
    period=1)


train_idg = ImageDataGenerator(rescale=1./255)
val_idg = ImageDataGenerator(rescale=1./255)
test_idg = ImageDataGenerator(rescale=1./255)

train_generator = train_idg.flow_from_directory(
    dir_xray_train,
    target_size=(img_height,img_width),
    batch_size = batch_size,
    shuffle=True,
    class_mode='binary')

val_generator = val_idg.flow_from_directory(
    dir_xray_val,
    target_size= (img_height,img_width),
    batch_size = batch_size,
    shuffle=True,
    class_mode='binary')

test_generator = test_idg.flow_from_directory(
    dir_xray_test,
    target_size= (img_height,img_width),
    batch_size = batch_size,
    shuffle=True,
    class_mode='binary')

model.fit_generator(
        train_generator,
        epochs=n_epochs,
        steps_per_epoch=train_num//batch_size,
        validation_data=val_generator,
        validation_steps=val_num//batch_size,
        callbacks = [progbar, checkpoint])



if not os.path.exists('/home/junankim18/Desktop/aaof/test/sex/'):
    os.makedirs('/home/junankim18/Desktop/aaof/test/sex/')
f = open('/home/junankim18/Desktop/aaof/test/sex/1.txt','a')
save_path = '/home/junankim18/Desktop/aaof/test/sex/fig/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

test_num = 0

for label in labels:
    num3 = len(os.listdir(dir_xray_test+label))
    test_num +=num3

print('test_num = ',test_num)


y_true = []


images0 = os.listdir(dir_xray_test + '0')
for jj, image0 in enumerate(images0):
    if jj == 0:
        x_test = np.expand_dims(cv2.imread(dir_xray_test + '0/'+image0),axis=0)
        y_true.append(0)
    else:
        x_test = np.append(x_test, np.expand_dims(cv2.imread(dir_xray_test + '0/'+image0),axis=0),axis=0)
        y_true.append(0)


images1 = os.listdir(dir_xray_test + '1')
for image1 in images1:
    x_test = np.append(x_test, np.expand_dims(cv2.imread(dir_xray_test + '1/'+image1),axis=0),axis=0)
    y_true.append(1)




x_test = x_test /255.


aaa = time.time()

hdf5 = '/home/junankim18/Desktop/aaof/hdf5/sex/'

weights = os.listdir(hdf5)
weights.sort()
acc = []
for ii, weight in enumerate(weights):
    aaa = time.time()
    print(weight)
    model = load_model(hdf5 +'/'+ weight)


    # y_pred_test = model.predict(x_test)[:,1].ravel()
    y_pred_test = model.predict(x_test).ravel()
    lw = 2



    fpr_test, tpr_test, threshold_test = roc_curve(y_true, y_pred_test)
    auc_test = auc(fpr_test, tpr_test)

    plt.figure(1, figsize=(6,6))
    plt.plot(fpr_test, tpr_test,
            label='Periodentitis (AUC = {0:0.3f})'
                ''.format(auc_test),
            color='tab:blue')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-specificity',fontsize= 20)
    plt.ylabel('Sensitivity',fontsize= 20)
    # plt.title('ROC curve')
    plt.legend(loc="lower right",fontsize=15)
    # plt.show()

    plt.savefig(save_path + str(ii)+'.jpg', bbox_inches='tight')
    plt.close()


    normed_con_mat = confusion_matrix(y_true, model.predict(x_test).argmax(axis=1))/confusion_matrix(y_true,model.predict(x_test).argmax(axis=1)).sum(axis=1).astype('float')

    print('acc = ',  sum(model.predict(x_test).argmax(axis=1) == y_true) / float(len(y_true)))
    # print normed_con_mat
    # quit()
    f.write('weight: {}\n'.format(weight))
    f.write('acc: {}\n'.format(sum(model.predict(x_test).argmax(axis=1) == y_true) / float(len(y_true))))
    f.write('con_mat: {}\n'.format(normed_con_mat))
    f.write('pred: {}\n'.format(model.predict(x_test).argmax(axis=1)))
    f.write('y_true: {}\n'.format(y_true))



    print('time = ',time.time() -aaa)
    # quit()
    acc.append(sum(model.predict(x_test).argmax(axis=1) == y_true) / float(len(y_true)))
    k.clear_session()
f.write('total_weights : {}'.format(weights))
f.write('total_acc: {}'.format(acc))
f.close()