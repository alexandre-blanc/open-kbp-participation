from tensorflow.keras.layers import Layer

class MultiplyByScalar(Layer):

    def __init__(self, initial_value=0, **kwargs):
        super(MultiplyByScalar, self).__init__(**kwargs)
        self.initial_value = initial_value

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(),
                                      initializer=tf.constant_initializer(value=self.initial_value),
                                      dtype="float32",
                                      trainable=True)
        super(MultiplyByScalar, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {'initial_value':self.initial_value}
