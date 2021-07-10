import tensorflow
from numpy.random import seed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
from keras.metrics import top_k_categorical_accuracy
from keras.models import load_model
import numpy as np
from tensorflow.keras.models import Model
import itertools
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image

seed(101)
matplotlib.use('Agg')
image_size = 224
predict_dir = './static/predict_image'
'''
predict_dir下面必须再建一层文件夹放要预测的图片
真正图片存放的位置是这样的./static/predict_image/unpredict/predict.jpg
'''


# src:predict_dir
def get_inputs(src):
    datagen = ImageDataGenerator(
        preprocessing_function=tensorflow.keras.applications.mobilenet.preprocess_input)

    predict_batch = datagen.flow_from_directory(predict_dir,
                                                target_size=(image_size, image_size),
                                                batch_size=1,
                                                shuffle=False)
    return predict_batch


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def predict(predict_batch):
    # 加载模型h5文件
    model = load_model("D:/Models/model2.h5", custom_objects={'top_2_accuracy': top_2_accuracy,
                                                              'top_3_accuracy': top_3_accuracy})

    pre_y = model.predict(predict_batch, steps=1, verbose=1)

    labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    # 返回最大数的索引
    y_pred = np.argmax(pre_y)
    label_of_y = labels[y_pred]
    return label_of_y

# 预测

# predict_batch = get_inputs(predict_dir)
# label_of_y = predict(predict_batch)
# label_of_y
