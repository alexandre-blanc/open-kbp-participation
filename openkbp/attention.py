from tensorflow.keras.layers import Layer, Reshape, Conv3D, AveragePooling3D, Activation
import tensorflow.keras.backend as K
import tensorflow as tf

from openkbp.MultiplyByScalar import MultiplyByScalar
from openkbp.spectral_normalization import ConvSN3D



class SelfAttention(Layer):

    number_of_attention_blocks = 0

    def __init__(self, use_sn=False, downsampling_factor=2, channel_reduction_factor = 8):
        super(SelfAttention, self).__init__()
        self.use_sn = use_sn
        self.downsampling_factor = downsampling_factor
        self.channel_reduction_factor = channel_reduction_factor
        self.number_of_attention_blocks += 1

    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x_input):
    batch_size, x, y, z, num_channels = K.int_shape(x_input)
    location_num = x*y*z
    downsampled_num = location_num // self.downsampling_factor**3 # **3 car on est en 3D
     
    if use_sn:
        # f path (cf paper)
        f = ConvSN3D(num_channels // self.channel_reduction_factor , 1 , strides=1 , padding="same", use_bias=True)(x_input)
        f = Reshape((location_num , num_channels // self.channel_reduction_factor))(f)
        
        # g path (cf paper)
        g = ConvSN3D(num_channels // self.channel_reduction_factor , 1 , strides=1 , padding="same", use_bias=True)(x_input)
        g = AveragePooling3D(pool_size=self.downsampling_factor)(g)
        g = Reshape((downsampled_num , num_channels // 8))(g)
        
        # attention
        attn = tf.matmul(f , g , transpose_b = True)
        attn = Activation('softmax', name='attention_activation_{}'.format(self.number_of_attention_blocks))(attn)
        
        # h path (cf paper)
        h = ConvSN3D(num_channels // self.channel_reduction_factor , 1 , strides=1 , padding="same", use_bias=True)(x_input)
        h = AveragePooling3D(pool_size=self.downsampling_factor)(h)
        h = Reshape((downsampled_num , num_channels // self.channel_reduction_factor))(h)
        
        # attention * h
        attn_h = tf.matmul(attn, h)
        attn_h = Reshape(( x, y , z , num_channels // self.channel_reduction_factor))(attn_h)
        attn_h = ConvSN3D(num_channels, 1 , strides=1 , padding="same", use_bias=True)(attn_h)
    else:
        # f path (cf paper)
        f = Conv3D(num_channels // self.channel_reduction_factor , 1 , strides=1 , padding="same", use_bias=True)(x_input)
        f = Reshape((location_num , num_channels // self.channel_reduction_factor))(f)
        
        # g path (cf paper)
        g = Conv3D(num_channels // self.channel_reduction_factor , 1 , strides=1 , padding="same", use_bias=True)(x_input)
        g = AveragePooling3D(pool_size=self.downsampling_factor)(g)
        g = Reshape((downsampled_num , num_channels // 8))(g)
        
        # attention
        attn = tf.matmul(f , g , transpose_b = True)
        attn = Activation('softmax', name='attention_activation_{}'.format(self.number_of_attention_blocks))(attn)
        
        # h path (cf paper)
        h = Conv3D(num_channels // self.channel_reduction_factor , 1 , strides=1 , padding="same", use_bias=True)(x_input)
        h = AveragePooling3D(pool_size=self.downsampling_factor)(h)
        h = Reshape((downsampled_num , num_channels // self.channel_reduction_factor))(h)

        # attention * h
        attn_h = tf.matmul(attn, h)
        attn_h = Reshape(( x, y , z , num_channels // self.channel_reduction_factor))(attn_h)
        attn_h = Conv3D(num_channels, 1 , strides=1 , padding="same", use_bias=True)(attn_h)
        
        # attention * h
        attn_h = tf.matmul(attn, h)
        attn_h = Reshape(( x, y , z , num_channels // 2))(attn_h)
        attn_h = Conv(num_channels, 1 , strides=1 , padding="same", use_bias=True)(attn_h)
        
    # define sigma 
    attn_h = MultiplyByScalar(name='attention_multiplier_{}'.format(self.number_of_attention_blocks))(attn_h)
    
    return x_input + attn_h

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {'use_sn':self.use_sn, 'downsampling_factor':self.downsampling_factor, 'channel_reduction_factor':self.channel_reduction_factor}
