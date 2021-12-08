# coding=UTF-8
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import pdb

from model import GreatNet
from dataset import TitanicDataSet

if __name__=="__main__":
    model = GreatNet()
    model._build()
    dataset = TitanicDataSet().get_test_set()
    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
    checkpoint, directory="../resource/ckpts_dense512/", max_to_keep=5)
    status = checkpoint.restore(manager.latest_checkpoint)
    flag = 0
    rst =0
    rst_array=[]
    for data in dataset["x_label"]:
        out = int(float(model(data.reshape([1,-1])).numpy())+0.5)
        rst_array.append(out)
        if out==dataset["y_label"][flag]:
            rst+=1
        flag+=1
        
    print(rst/flag, flag)
    answer_csv = "../resource/data/gender_submission.csv"
    answer = pd.read_csv(answer_csv)
    answer["Survived"] = rst_array
    answer.to_csv("submission.csv", index=False)