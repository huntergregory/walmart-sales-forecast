import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike, FloatTensorLike
from constants import LOSS_SIZE, DTYPE, QUANTILES, HORIZON_LENGTH
from tensorflow.keras.metrics import Metric
from tensorflow.keras.losses import Loss, Reduction

TAU = tf.expand_dims(tf.cast(QUANTILES, DTYPE), 0)

# input shape: (batch_size, LOSS_SIZE, HORIZON_LENGTH, Q)
# return shape: (batch_size)

AVG_FORECASTS = True
AVG_HORIZON_LENGTH = True
AVG_QUANTILES = True

# def get_axes(avg_forecasts=False, avg_horizon_length=False, avg_quantiles=False):
#     divide_bys = [avg_forecasts, avg_horizon_length, avg_quantiles]
#     mean_axes = [k+1 for k in range(3) if divide_bys[k]]
#     sum_axes = [-(k+1) for k in range(3 - sum(divide_bys))]
#     return mean_axes, sum_axes

# reduce_mean_axes, reduce_sum_axes = get_axes(AVG_FORECASTS,AVG_HORIZON_LENGTH, AVG_QUANTILES)
# should_reduce_mean = tf.convert_to_tensor(len(reduce_mean_axes) > 0)
# should_reduce_sum = tf.convert_to_tensor(len(reduce_sum_axes) > 0)

class PinballLoss(Loss):
    def __init__(self, name='pinball_loss'):
        super().__init__(reduction=Reduction.SUM, name=name)

    def call(self, y_true, y_pred):
        return pinball(y_true, y_pred)

def pinball(y_true, y_pred): # TODO apply weight 
    batch_size = tf.shape(y_true)[0]
    def modify(y):
        return tf.cast(y, DTYPE) # tf.reshape(tf.cast(y, DTYPE), (batch_size * LOSS_SIZE * HORIZON_LENGTH, len(QUANTILES)))
    y_true = modify(y_true)
    y_pred = modify(y_pred)

    one = tf.cast(1, DTYPE)
    delta_y = y_true - y_pred
    result = tf.math.maximum(TAU * delta_y, (TAU - one) * delta_y)
    # result = tf.reshape(result, (batch_size, LOSS_SIZE, HORIZON_LENGTH, len(QUANTILES)))

    # result = tf.cond(should_reduce_mean, lambda: tf.reduce_mean(result, axis=reduce_mean_axes), lambda: result)
    # result = tf.cond(should_reduce_sum, lambda: tf.reduce_sum(result, axis=reduce_sum_axes), lambda: result)

    result = tf.reduce_mean(result, axis=(-3, -2, -1))

    # result = tf.multiply(result, sample_weight) FIXME uncomment?
    # result = tf.reduce_sum(result)

    # return tf.divide(result, tf.constant(12.0))
    return result
    # return tf.multiply(result, tf.cast(batch_size, DTYPE))


class PinballMetric(Metric):
    def __init__(self,  name='pinball_metric', **kwargs):
        super(PinballMetric, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='pinball_sum', initializer='zeros', dtype=DTYPE)
        self.count = self.add_weight(name='num_samples', initializer='zeros', dtype=tf.int32)

    def update_state(self, y_true, y_pred): # SAMPLE WEIGHTS DON'T WORK
        values = pinball(y_true[:,-1], y_pred[:,-1])

        # if sample_weight is not None: # FIXME never take this branch
        #     sample_weight = tf.cast(sample_weight, self.dtype)
        #     # sample_weight = tf.broadcast_weights(sample_weight, values)
        #     values = tf.multiply(values, sample_weight)

        self.sum.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.shape(values)[0])

    def result(self):
        return tf.divide(self.sum, tf.cast(self.count, DTYPE))

    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)

    
## OLD LOGIC
# if avg_forecasts:
#     if avg_horizon_length:
#         if avg_quantiles:
#             reduce_mean_axes = (1,2,3)
#             reduce_sum_axes = ()
#         else:
#             reduce_mean_axes = (1,2)
#             reduce_sum_axes = (3,)
#     else:
#         if avg_quantiles:
#             reduce_mean_axes = (1,3)
#             reduce_sum_axes = (-1,)
#         else:
#             reduce_mean_axes = (1,)
#             reduce_sum_axes = (-2,-1)
# else:
#     if avg_horizon_length:
#         if avg_quantiles:
#             reduce_mean_axes = (2,3)
#             reduce_sum_axes = (1,)
#         else:
#             reduce_mean_axes = (2,)
#             reduce_sum_axes = (-2,-1)
#     else:
#         if avg_quantiles:
#             reduce_mean_axes = (3,)
#             reduce_sum_axes = (1,2)
#         else:
#             reduce_mean_axes = ()
#             reduce_sum_axes = (1,2,3)

## ORIGINAL
# @tf.function
# def pinball_loss(
#     y_true: TensorLike, y_pred: TensorLike, tau: FloatTensorLike = 0.5
# ) -> tf.Tensor:
#     """
#     Args:
#       y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`
#       y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
#       tau: (Optional) Float in [0, 1] or a tensor taking values in [0, 1] and
#         shape = `[d0,..., dn]`.  It defines the slope of the pinball loss. In
#         the context of quantile regression, the value of tau determines the
#         conditional quantile level. When tau = 0.5, this amounts to l1
#         regression, an estimator of the conditional median (0.5 quantile).

#     Returns:
#         pinball_loss: 1-D float `Tensor` with shape [batch_size].
#     """
#     y_pred = tf.convert_to_tensor(y_pred)
#     y_true = tf.cast(y_true, y_pred.dtype)

#     # Broadcast the pinball slope along the batch dimension
#     tau = tf.expand_dims(tf.cast(tau, y_pred.dtype), 0)
#     one = tf.cast(1, tau.dtype)

#     delta_y = y_true - y_pred
#     pinball = tf.math.maximum(tau * delta_y, (tau - one) * delta_y)
#     return tf.reduce_sum(pinball, axis=-1)
