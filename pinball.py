import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike, FloatTensorLike
from constants import LOSS_SIZE, DTYPE, QUANTILES, HORIZON_LENGTH
from tensorflow.keras.metrics import Metric
from tensorflow.keras.losses import Loss, Reduction

TAU = tf.expand_dims(tf.cast(QUANTILES, DTYPE), 0)

class PinballLoss(Loss):
    def __init__(self, name='pinball_loss'):
        super().__init__(reduction=Reduction.SUM, name=name)

    def call(self, y_true, y_pred):
        return pinball(y_true, y_pred)

# input shape: (batch_size, LOSS_SIZE, HORIZON_LENGTH, Q)
# return shape: (batch_size)
def pinball(y_true, y_pred): # TODO apply weight 
    def modify(y):
        return tf.cast(y, DTYPE)
    y_true = modify(y_true)
    y_pred = modify(y_pred)

    one = tf.cast(1, DTYPE)
    delta_y = y_true - y_pred
    result = tf.math.maximum(TAU * delta_y, (TAU - one) * delta_y)
    result = tf.reduce_mean(result, axis=(-3, -2, -1))
    # return tf.divide(result, tf.constant(12.0))
    return result


class PinballMetric(Metric):
    def __init__(self,  name='pinball_metric', **kwargs):
        super(PinballMetric, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='pinball_sum', initializer='zeros', dtype=DTYPE)
        self.count = self.add_weight(name='num_samples', initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None): # SAMPLE WEIGHTS DON'T WORK
        values = pinball(y_true, y_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            # sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)
        else:
            self.count.assign_add(tf.shape(values)[0])
            
        self.sum.assign_add(tf.reduce_sum(values))

    def result(self):
        divisor = tf.math.maximum(tf.constant(1, dtype=DTYPE), tf.cast(self.count, DTYPE))
        return tf.divide(self.sum, divisor)

    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)
