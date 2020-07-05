import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform

# TODO? add dropout?
class MLP(Model):
    __seed = 1

    def set_next_seed(n):
        MLP.__seed = n

    def __init__(self, layer_shapes, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.dense_layers = []
        for k in range(len(layer_shapes) - 1):
            self.dense_layers.append(Dense(layer_shapes[k], activation='relu', kernel_initializer=GlorotUniform(seed=MLP.__seed)))
            MLP.__seed += 1
        self.dense_layers.append(Dense(layer_shapes[-1], activation='softplus', kernel_initializer=GlorotUniform(seed=MLP.__seed)))
        MLP.__seed += 1

    def call(self, inputs):
        x = inputs
        for k in range(len(self.dense_layers)):
            x = self.dense_layers[k](x)
        return x
