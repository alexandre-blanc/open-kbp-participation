import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def apply_spectral_normalization(layer):
    to_be_normalized = getattr(layer, 'apply_spectral_normalization', False)
    if to_be_normalized:
        layer.apply_spectral_normalization()
   
    has_sub_layers = getattr(layer, 'layers', False)
    if has_sub_layers:
        for sub_layer in layer.layers:
            apply_spectral_normalization(sub_layer)
            
class DenseSN(Dense):
    def build(self, input_shape):
        super(DenseSN, self).build(input_shape)
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)

    def power_iteration(self, W, u):
        _u = u
        _v = K.l2_normalize(K.dot(_u, K.transpose(W)))
        _u = K.l2_normalize(K.dot(_v, W))
        return _u, _v
        
    def apply_spectral_normalization(self):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = self.power_iteration(W_reshaped, self.u)
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        W_bar = W_reshaped / sigma
        W_bar = K.reshape(W_bar, W_shape)
        self.kernel.assign(W_bar)
    
class ConvSN2D(Conv2D):

    def build(self, input_shape):
        super(ConvSN2D, self).build(input_shape)            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)

    def power_iteration(self, W, u):
        _u = u
        _v = K.l2_normalize(K.dot(_u, K.transpose(W)))
        _u = K.l2_normalize(K.dot(_v, W))
        return _u, _v

    def apply_spectral_normalization(self):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = self.power_iteration(W_reshaped, self.u)
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        W_bar = W_reshaped / sigma
        W_bar = K.reshape(W_bar, W_shape)
        self.kernel.assign(W_bar)

class ConvSN2DTranspose(Conv2DTranspose):

    def build(self, input_shape):
        super(ConvSN2DTranspose, self).build(input_shape)            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)

    def power_iteration(self, W, u):
        _u = u
        _v = K.l2_normalize(K.dot(_u, K.transpose(W)))
        _u = K.l2_normalize(K.dot(_v, W))
        return _u, _v

    def apply_spectral_normalization(self):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = self.power_iteration(W_reshaped, self.u)
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        W_bar = W_reshaped / sigma
        W_bar = K.reshape(W_bar, W_shape)
        self.kernel.assign(W_bar)

class ConvSN3D(Conv3D):

    def build(self, input_shape):
        super(ConvSN3D, self).build(input_shape)            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)

    def power_iteration(self, W, u):
        _u = u
        _v = K.l2_normalize(K.dot(_u, K.transpose(W)))
        _u = K.l2_normalize(K.dot(_v, W))
        return _u, _v

    def apply_spectral_normalization(self):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = self.power_iteration(W_reshaped, self.u)
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        W_bar = W_reshaped / sigma
        W_bar = K.reshape(W_bar, W_shape)
        self.kernel.assign(W_bar)

class ConvSN3DTranspose(Conv3DTranspose):

    def build(self, input_shape):
        super(ConvSN3DTranspose, self).build(input_shape)            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)

    def power_iteration(self, W, u):
        _u = u
        _v = K.l2_normalize(K.dot(_u, K.transpose(W)))
        _u = K.l2_normalize(K.dot(_v, W))
        return _u, _v

    def apply_spectral_normalization(self):
        W_shape = self.kernel.shape.as_list()
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = self.power_iteration(W_reshaped, self.u)
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        W_bar = W_reshaped / sigma
        W_bar = K.reshape(W_bar, W_shape)
        self.kernel.assign(W_bar)