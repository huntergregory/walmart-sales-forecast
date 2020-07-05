from tensorflow.keras import Model
from tensorflow.python.eager import def_function

# verbatim with Model source code except critical change preventing use of massive amounts of memory
class CustomModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Override
    def make_predict_function(self):
        if self.predict_function is not None:
          return self.predict_function

        def predict_function(iterator):
          data = next(iterator)
          quantile_forecasts, _ = self.distribute_strategy.run(self.predict_step, args=(data,)) # critical change: drop training forecasts output
          quantile_forecasts = reduce_per_replica(quantile_forecasts, self.distribute_strategy, reduction='concat')
          return quantile_forecasts

        if not self.run_eagerly:
          predict_function = def_function.function(predict_function, experimental_relax_shapes=True)

        self.predict_function = predict_function
        return self.predict_function


######      FROM TF SOURCE CODE (Model)    #######
######      Need for reduce_per_replica    #######

######################################################################
# Functions below exist only as v1 / v2 compatibility shims.
######################################################################

from tensorflow.python.util import nest
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.distribute import distributed_training_utils as dist_utils

def reduce_per_replica(values, strategy, reduction='first'):
  """Reduce PerReplica objects.
  Arguments:
    values: Structure of `PerReplica` objects or `Tensor`s. `Tensor`s are
      returned as-is.
    strategy: `tf.distribute.Strategy` object.
    reduction: One of 'first', 'concat'.
  Returns:
    Structure of `Tensor`s.
  """

  def _reduce(v):
    """Reduce a single `PerReplica` object."""
    if not isinstance(v, ds_values.PerReplica):
      return v
    elif reduction == 'first':
      return strategy.unwrap(v)[0]
    elif reduction == 'concat':
      if _is_tpu_multi_host(strategy):
        return _tpu_multi_host_concat(v, strategy)
      else:
        return concat(strategy.unwrap(v))
    else:
      raise ValueError('`reduction` must be "first" or "concat".')

  return nest.map_structure(_reduce, values)


def concat(tensors, axis=0):
  """Concats `tensor`s along `axis`."""
  if isinstance(tensors[0], sparse_tensor.SparseTensor):
    return sparse_ops.sparse_concat_v2(axis=axis, sp_inputs=tensors)
  if isinstance(tensors[0], ragged_tensor.RaggedTensor):
    return ragged_concat_ops.concat(tensors, axis=axis)
  return array_ops.concat(tensors, axis=axis)


def _is_tpu_multi_host(strategy):
  return (dist_utils.is_tpu_strategy(strategy) and
          strategy.extended.num_hosts > 1)


def _tpu_multi_host_concat(v, strategy):
  """Correctly order TPU PerReplica objects."""
  replicas = strategy.unwrap(v)
  # When distributed datasets are created from Tensors / NumPy,
  # TPUStrategy.experimental_distribute_dataset shards data in
  # (Replica, Host) order, and TPUStrategy.unwrap returns it in
  # (Host, Replica) order.
  # TODO(b/150317897): Figure out long-term plan here.
  num_replicas_per_host = strategy.extended.num_replicas_per_host
  ordered_replicas = []
  for replica_id in range(num_replicas_per_host):
    ordered_replicas += replicas[replica_id::num_replicas_per_host]
  return concat(ordered_replicas)