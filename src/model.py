import tensorflow as tf
import numpy as np

input_shape=(7,)
unit_nums = 32

# model
class GreatNet(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        # 输入
        # x = tf.keras.layers.Input(shape=input_shape, name="/simpleDense/Input", dtype=tf.float32)
        layers = []
        layers +=[
            tf.keras.layers.InputLayer(input_shape=input_shape, dtype=tf.float32, name="input_layer"),
            tf.keras.layers.Dense(
                unit_nums, activation='ReLU', name="dense_32"
            ),
            tf.keras.layers.Dense(
                1, activation=None, use_bias=False
            )
        ]
        self.great_net = tf.keras.models.Sequential(layers)
        print(self.great_net.summary())
    
    def call(self, x):
        return self.great_net(x)

    def _build(self):
        fake_input = tf.random.uniform(shape=[1,7], dtype=tf.float32)
        self(fake_input)