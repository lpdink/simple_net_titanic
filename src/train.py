# coding=UTF-8
import sys
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
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
    checkpoint, directory="../resource/ckpts_dense512/", max_to_keep=5)
    status = checkpoint.restore(manager.latest_checkpoint)

    tmp_loss=0
    try:
        for epoch in range(1, 201):
            #dataset = dataset.shuffle(10, reshuffle_each_iteration=True)
            for flag, data in enumerate(dataset):
                with tf.GradientTape() as tape:
                    out = model(data["x_label"])
                    loss = loss_function(data["y_label"], out)
                    print(loss)
                tmp_loss+=loss
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print("epoch: {} loss:{}".format(epoch, tmp_loss))
            tmp_loss = 0
            if epoch%50==0:
                manager.save()
                print("epoch: {}ckpts saved in /resource/ckpts".format(epoch))
    except KeyboardInterrupt:
        manager.save()

    pdb.set_trace()
