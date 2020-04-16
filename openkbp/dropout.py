from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn

class CustomSpatialDropout3D(Dropout):
  """Spatial 3D version of Dropout.
  This version performs the same function as Dropout, however it drops
  entire 3D feature maps instead of individual elements. If adjacent voxels
  within feature maps are strongly correlated (as is normally the case in
  early convolution layers) then regular dropout will not regularize the
  activations and will otherwise just result in an effective learning rate
  decrease. In this case, SpatialDropout3D will help promote independence
  between feature maps and should be used instead.
  Arguments:
    rate: Float between 0 and 1. Fraction of the input units to drop.
    data_format: 'channels_first' or 'channels_last'.
        In 'channels_first' mode, the channels dimension (the depth)
        is at index 1, in 'channels_last' mode is it at index 4.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
  Call arguments:
    inputs: A 5D tensor.
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).
  Input shape:
    5D tensor with shape:
    `(samples, channels, dim1, dim2, dim3)` if data_format='channels_first'
    or 5D tensor with shape:
    `(samples, dim1, dim2, dim3, channels)` if data_format='channels_last'.
  Output shape:
    Same as input.
  References:
    - [Efficient Object Localization Using Convolutional
      Networks](https://arxiv.org/abs/1411.4280)
  """

  def __init__(self, rate, data_format=None, **kwargs):
    super(CustomSpatialDropout3D, self).__init__(rate, **kwargs)
    if data_format is None:
      data_format = K.image_data_format()
    if data_format not in {'channels_last', 'channels_first'}:
      raise ValueError('data_format must be in '
                       '{"channels_last", "channels_first"}')
    self.data_format = data_format
    self.input_spec = InputSpec(ndim=5)

  def _get_noise_shape(self, inputs):
    input_shape = array_ops.shape(inputs)
    if self.data_format == 'channels_first':
      return (input_shape[0], input_shape[1], 1, 1, 1)
    elif self.data_format == 'channels_last':
      return (input_shape[0], 1, 1, 1, input_shape[4])

  def call(self, inputs):
    return nn.dropout(inputs, noise_shape=self._get_noise_shape(inputs), seed=self.seed, rate=self.rate)
