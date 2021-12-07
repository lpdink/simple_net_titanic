# coding=UTF-8
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pdb

from model import GreatNet
from dataset import TitanicDataSet
'''
手写一个单层dense+激活函数的神经网络。
需要：
model;
dataset;
loss;
train_code;
'''



if __name__=="__main__":
    model = GreatNet()
    model._build()
    dataset = TitanicDataSet()
    optimizer = tf.keras.optimizers.Adam()
    tmp_loss=0
    for epoch in range(200):
        for flag, data in enumerate(dataset):
            with tf.GradientTape() as tape:
                out = model(data["x_label"])
                loss = tf.losses.mean_squared_error(data["y_label"], out)
            tmp_loss+=loss
            if flag%100==0:
                print(epoch, flag, tmp_loss)
                tmp_loss=0
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    pdb.set_trace()
