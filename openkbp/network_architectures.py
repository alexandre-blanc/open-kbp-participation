''' Neural net architectures '''

from tensorflow.keras.layers import Input, LeakyReLU, BatchNormalization, Layer
from tensorflow.keras.layers import Conv3D, concatenate, Activation, Flatten, Dense
from tensorflow.keras.layers import AveragePooling3D, Conv3DTranspose, Multiply
from tensorflow.keras.layers import Reshape, MaxPooling3D, UpSampling3D, Lambda, Add, ActivityRegularization
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.activations import softmax
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import mean_absolute_error, binary_crossentropy
from tensorflow.keras.backend import int_shape

from openkbp.dropout import CustomSpatialDropout3D
from openkbp.spectral_normalization import DenseSN, ConvSN3D, ConvSN3DTranspose
from openkbp.cones import Cones3D, cone3D_coordinates_normalization, cone3D_aperture_regularization, ConeCoordinatesRegularization
from openkbp.MultiplyByScalar import MultiplyByScalar
from openkbp.attention import SelfAttention

# The two functions below are helper functions which condense and control with their parameters succession of layers that are very often used.

def downsampling_block(x, number_of_filters, use_batchnorm=True, use_sn=False, relu_leak_rate=0.2, filter_size=(4,4,4), stride_size=(2,2,2)):
    if use_sn:
        x = ConvSN3D(number_of_filters, filter_size, strides=stride_size, padding="same", use_bias=True)(x)
    else:
        x = Conv3D(number_of_filters, filter_size, strides=stride_size, padding="same", use_bias=True)(x)
    if use_batchnorm:
        x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
    x = LeakyReLU(alpha=relu_leak_rate)(x)
    return x

def upsampling_block(x, nodes, use_dropout = True, use_batchnorm=True, skip_x=None, use_upsampling=False,\
                                    use_sn=False, dropout_rate = 0.5, relu_leak_rate=0.0, filter_size=(4,4,4), stride_size=(2,2,2)):
    if skip_x is not None:
        x = concatenate([x, skip_x])
    if use_upsampling:
        x = UpSampling3D()(x)
        if use_sn:
            x = ConvSN3D(nodes, kernel_size=filter_size, padding='same', use_bias=True)(x)
        else:
            x = Conv3D(nodes, kernel_size=filter_size, padding='same', use_bias=True)(x)
    else:
        if use_sn:
            x = ConvSN3DTranspose(nodes, filter_size, strides=stride_size, padding="same", use_bias=True)(x)
        else:
            x = Conv3DTranspose(nodes, filter_size, strides=stride_size, padding="same", use_bias=True)(x)
   
    if use_batchnorm:
        x  = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
   
    if use_dropout:
        x = CustomSpatialDropout3D(dropout_rate)(x)
   
    x = LeakyReLU(alpha=relu_leak_rate)(x)  # Use LeakyReLU(alpha = 0) instead of ReLU because ReLU is buggy when saved

    return x


class ConesGenerator:

    def __init__(self):
        self.use_possible_dose = True
        self.use_upsampling = True
        self.use_energy_conservation = True
        self.use_sn = True

        self.number_of_cones = 9

        self.aperture_regularization = .1
        self.direction_regularization = .1

        self.dropout_rate = 0.5
        self.use_dropout = True

        self.initial_number_of_filters = 8
        self.use_attention = True

        self.model = None

    def get_model(self):
        if self.model:
            return self.model

        # Define inputs
        ct_image = Input(self.ct_shape, name='ct_input')
        roi_masks = Input(self.roi_masks_shape, name='roi_input')
        possible_dose = Input(self.dose_shape, name='possible_dose_input')

        # Build Model starting with Conv3D layers                                                                          # output shape
        x = concatenate([ct_image, roi_masks])                                                                             # bs*128*128*128*11
        x1 = downsampling_block(x, self.initial_number_of_filters, use_batchnorm=False, use_sn=self.use_sn)                           # bs*64*64*64*64
        x2 = downsampling_block(x1, 2 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*32*32*32*128
        if self.use_attention_layer:
            x2 =  SelfAttention(use_sn=self.use_sn)(x2)
        x3 = downsampling_block(x2, 4 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*16*16*16*256
        x4 = downsampling_block(x3, 8 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*8*8*8*512
        x5 = downsampling_block(x4, 16 * self.initial_number_of_filters, use_sn=self.use_sn)                                           # bs*4*4*4*1024

        # Transform latent representation into cone coordinates
        c1 = AveragePooling3D(pool_size=(4,4,4))(x5)
        c2 = Flatten()(c1)
        c3 = DenseSN(8*self.initial_number_of_filters)(c2)
        c3 = LeakyReLU(alpha=0.2)(c3)
        c4 = DenseSN(4*self.initial_number_of_filters)(c3)
        c4 = LeakyReLU(alpha=0.2)(c4)
        c5 = DenseSN(2*self.initial_number_of_filters)(c4)
        c5 = LeakyReLU(alpha=0.2)(c5)

        cone_coefficients = Dense(self.number_of_cones,activation='softmax')(c5)
        cone_coefficients = Reshape([1,1,1,self.number_of_cones], name='reshape_cone_coefficients')(cone_coefficients)

        cone_coordinates = Dense(7*self.number_of_cones)(c5)
        cone_coordinates = Reshape([self.number_of_cones,7],name='cone_coordinates_reshape')(cone_coordinates)
        cone_coordinates = Activation(cone3D_coordinates_normalization, name='normalize_cone_coordinates')(cone_coordinates)
        cone_coordinates = ConeCoordinatesRegularization(aperture=self.aperure_regularization, direction=self.direction_regularization)(cone_coordinates)

        cones = []
        for i in range(self.number_of_cones):
            single_cone_coordinates = Lambda(lambda x: x[:,i,:], name='extract_single_cone_coordinates_{}'.format(i))(cone_coordinates)
            cones.append(Cones3D(128,use_energy_conservation=self.use_energy_conservation)(single_cone_coordinates))
        cones = concatenate(cones)



        combined_cones = Lambda(lambda x: x[0]*x[1], name='multiply_cones_by_coefficients')([cone_coefficients, cones])
        combined_cones = Lambda(lambda x: K.sum(x, axis=-1,keepdims=True), name='combine_cones')(combined_cones)

        final_dose = MultiplyByScalar()combined_cones

        if self.use_possible_dose:
            final_dose = Multiply()([final_dose, possible_dose])                                                           # bs*128*128*128*1

        self.model = Model(inputs=[ct_image, roi_masks, possible_dose], outputs = final_dose, name="generator")
        self.model.summary()

        return self.model

class UnetGenerator:
    def __init__(self):
        self.generator_ = None
        self.discriminator_ = None
        self.adversarial_model_ = None
        self.discriminator_model_ = None
        self.use_possible_dose = True
        self.use_upsampling = True
        self.use_sn = True

        self.dropout_rate = 0.5
        self.use_dropout = True
        self.initial_number_of_filters = 8
        self.use_attention_layer = True

        self.model = None

    def get_model(self):
        """Makes a generator that takes a CT image as input to generate a dose distribution of the same dimensions"""

        if self.model:
            return self.model

        # Define inputs
        ct_image = Input(self.ct_shape, name='ct_input')
        roi_masks = Input(self.roi_masks_shape, name='roi_input')
        possible_dose = Input(self.dose_shape, name='possible_dose_input')

        # Build Model starting with Conv3D layers                                                                          # output shape
        x = concatenate([ct_image, roi_masks])                                                                             # bs*128*128*128*11
        x1 = downsampling_block(x, self.initial_number_of_filters, use_batchnorm=False, use_sn=self.use_sn)                           # bs*64*64*64*64
        x2 = downsampling_block(x1, 2 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*32*32*32*128
        if self.use_attention:
            x2 =  SelfAttention(use_sn=self.use_sn)(x2)
        x3 = downsampling_block(x2, 4 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*16*16*16*256
        x4 = downsampling_block(x3, 8 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*8*8*8*512
        x5 = downsampling_block(x4, 16 * self.initial_number_of_filters, use_sn=self.use_sn)                                           # bs*4*4*4*1024

        # Build model back up from bottleneck through upsampling
        x4b = upsampling_block(x5, 16 * self.initial_number_of_filters, use_sn=self.use_sn, use_upsampling=self.use_upsampling, use_dropout=self.use_dropout, dropout_rate=self.dropout_rate)                                # bs*8*8*8*1024
        x3b = upsampling_block(x4b, 8 * self.initial_number_of_filters, skip_x=x4, use_sn=self.use_sn, use_upsampling=self.use_upsampling, use_dropout=self.use_dropout, dropout_rate=self.dropout_rate)                     # bs*16*16*16*512
        x2b = upsampling_block(x3b, 4 * self.initial_number_of_filters, skip_x=x3, use_sn=self.use_sn, use_upsampling=self.use_upsampling, use_dropout=self.use_dropout, dropout_rate=self.dropout_rate)                     # bs*32*32*32*256
        if self.use_attention:
            x2b =  SelfAttention(use_sn=self.use_sn)(x2b)
        x1b = upsampling_block(x2b, 2 * self.initial_number_of_filters, skip_x=x2, use_dropout=False, use_sn=self.use_sn, use_upsampling=self.use_upsampling, use_dropout=self.use_dropout, dropout_rate=self.dropout_rate)  # bs*64*64*64*128
        x0b = upsampling_block(x1b, 1, skip_x=x1, use_dropout=False, use_batchnorm=False, use_sn=self.use_sn, use_upsampling=self.use_upsampling, use_dropout=self.use_dropout, dropout_rate=self.dropout_rate)  # bs*64*64*64*128
        x_final = Activation("sigmoid")(x0b)# AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding="same")(x0b)                                      # bs*128*128*128*1
                      # Average pooling is commented out to preserve high frequency detail
        final_dose = MultiplyByScalar(x_final)                                                                           # bs*128*128*128*1

        if self.use_possible_dose:
            final_dose = Multiply()([final_dose, possible_dose])                                                           # bs*128*128*128*1

        self.model = Model(inputs=[ct_image, roi_masks, possible_dose], outputs = final_dose, name="generator")
        self.model.summary()

        return self.model

class UnetAndConesGenerator:
    """This class defines the architecture for a U-NET and must be inherited by a child class that
    executes various functions like training or predicting"""
    def __init__(self):
        self.use_possible_dose = True
        self.use_upsampling = True
        self.use_energy_conservation = True
        self.use_sn = True
        self.number_of_cones = 9
        self.aperure_regularization = .1
        self.direction_regularization = .1

        self.dropout_rate = 0.5
        self.use_dropout = True

        self.initial_number_of_filters = 8
        self.use_attention = True

        self.model = None



    def generator(self):
        """Makes a generator that takes a CT image as input to generate a dose distribution of the same dimensions"""

        if self.generator_:
            return self.generator_

        # Define inputs
        ct_image = Input(self.ct_shape, name='ct_input')
        roi_masks = Input(self.roi_masks_shape, name='roi_input')
        possible_dose = Input(self.dose_shape, name='possible_dose_input')

        # Build Model starting with Conv3D layers                                                                          # output shape
        x = concatenate([ct_image, roi_masks])                                                                             # bs*128*128*128*11
        x1 = downsampling_block(x, self.initial_number_of_filters, use_batchnorm=False, use_sn=self.use_sn)                           # bs*64*64*64*64
        x2 = downsampling_block(x1, 2 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*32*32*32*128
        if self.use_attention:
            x2 =  SelfAttention(use_sn=self.use_sn)(x2)
        x3 = downsampling_block(x2, 4 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*16*16*16*256
        x4 = downsampling_block(x3, 8 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*8*8*8*512
        x5 = downsampling_block(x4, 16 * self.initial_number_of_filters, use_sn=self.use_sn)                                           # bs*4*4*4*1024

        # Build model back up from bottleneck through upsampling
        x4b = upsampling_block(x5, 16 * self.initial_number_of_filters, use_sn=self.use_sn, use_upsampling=self.use_upsampling)                                # bs*8*8*8*1024
        x3b = upsampling_block(x4b, 8 * self.initial_number_of_filters, skip_x=x4, use_sn=self.use_sn, use_upsampling=self.use_upsampling)                     # bs*16*16*16*512
        x2b = upsampling_block(x3b, 4 * self.initial_number_of_filters, skip_x=x3, use_sn=self.use_sn, use_upsampling=self.use_upsampling)                     # bs*32*32*32*256
        if self.use_attention:
            x2b =  SelfAttention(use_sn=self.use_sn)(x2b)
        x1b = upsampling_block(x2b, 2 * self.initial_number_of_filters, skip_x=x2, use_dropout=False, use_sn=self.use_sn, use_upsampling=self.use_upsampling)  # bs*64*64*64*128
        x0b = upsampling_block(x1b, 1, skip_x=x1, use_dropout=False, use_batchnorm=False, use_sn=self.use_sn, use_upsampling=self.use_upsampling)  # bs*64*64*64*128
        x_final = x0b # AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding="same")(x0b)                                      # bs*128*128*128*1
                      # Average pooling is commented out to preserve high frequency detail
        unet_final_dose = Activation("sigmoid")(x_final)                                                                           # bs*128*128*128*1

        # Transform latent representation into cone coordinates
        c1 = AveragePooling3D(pool_size=(4,4,4))(x5)
        c2 = Flatten()(c1)
        c3 = DenseSN(8*self.initial_number_of_filters)(c2)
        c3 = LeakyReLU(alpha=0.2)(c3)
        c4 = DenseSN(4*self.initial_number_of_filters)(c3)
        c4 = LeakyReLU(alpha=0.2)(c4)
        c5 = DenseSN(2*self.initial_number_of_filters)(c4)
        c5 = LeakyReLU(alpha=0.2)(c5)

        # Compute the cone superposition
        cone_coefficients = Dense(self.number_of_cones,activation='softmax')(c5)
        cone_coefficients = Reshape([1,1,1,self.number_of_cones], name='reshape_cone_coefficients')(cone_coefficients)

        cone_coordinates = Dense(7*self.number_of_cones)(c5)
        cone_coordinates = Reshape([self.number_of_cones,7],name='cone_coordinates_reshape')(cone_coordinates)
        cone_coordinates = Activation(cone3D_coordinates_normalization, name='normalize_cone_coordinates')(cone_coordinates)
        cone_coordinates = ConeCoordinatesRegularization(aperture=self.aperure_regularization, direction=self.direction_regularization)(cone_coordinates)

        cones = []
        for i in range(self.number_of_cones):
            single_cone_coordinates = Lambda(lambda x: x[:,i,:], name='extract_single_cone_coordinates_{}'.format(i))(cone_coordinates)
            cones.append(Cones3D(128,use_energy_conservation=self.use_energy_conservation)(single_cone_coordinates))
        cones = concatenate(cones)



        combined_cones = Lambda(lambda x: x[0]*x[1], name='multiply_cones_by_coefficients')([cone_coefficients, cones])
        combined_cones = Lambda(lambda x: K.sum(x, axis=-1,keepdims=True), name='combine_cones')(combined_cones)

        # Superpose the output of the Unet and the cones.
        scaled_unet_dose = MultiplyByScalar(initial_value=1.0)(unet_final_dose)
        scaled_cone_dose = MultiplyByScalar(initial_value=1.0)(combined_cones)
        final_dose = Add()([scaled_cone_dose, scaled_unet_dose])

        final_dose = Reshape([-1])(final_dose)
        final_dose = Lambda(lambda x: x/K.max(x,axis=-1,keepdims=True))(final_dose)
        final_dose = Reshape([128,128,128,1])(final_dose)

        if self.use_possible_dose:
            final_dose = Multiply()([final_dose, possible_dose])                                                           # bs*128*128*128*1

        self.model = Model(inputs=[ct_image, roi_masks, possible_dose], outputs = final_dose, name="generator")
        self.model.summary()

        return self.model


class Discriminator:
    """This class defines the architecture for a U-NET and must be inherited by a child class that
    executes various functions like training or predicting"""
    def __init__(self):
        self.initial_number_of_filters = 8
        self.use_attention = True
        self.model = None

    def discriminator(self):
        if self.discriminator_:
            return self.discriminator_
        # Define inputs
        dose_to_examine = Input(self.dose_shape)
        ct_image = Input(self.ct_shape)
        roi_masks = Input(self.roi_masks_shape)

        # The jump is the distance between to consecutive centers of a receptive field
        # The receptive field size is the width of the receptive field
        # For more details see https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807

        # Build Model starting with Conv3D layers                                                   # output shape           # jump           # receptive field
        x = concatenate([dose_to_examine, ct_image, roi_masks])                                     # bs*128*128*128*12      # 1              # 1
        x1 = downsampling_block(x, self.initial_number_of_filters, use_batchnorm=False, use_sn=self.use_sn)    # bs*64*64*64            # 2              # 1 + 3 * 1 = 4
        x2 = downsampling_block(x1, 2 * self.initial_number_of_filters, use_sn=self.use_sn)                     # bs*32*32*32*128        # 4             # 4 + 3 * 2 = 10
        if self.use_attention:
            x2 =  elfAttention(use_sn=self.use_sn, use_sn=self.use_sn)(x2)
        x3 = downsampling_block(x2, 4 * self.initial_number_of_filters, use_sn=self.use_sn)                     # bs*16*16*256           # 8              # 10 + 3 * 4 = 22
        x4 = downsampling_block(x3, 8 * self.initial_number_of_filters, use_sn=self.use_sn)                     # bs*8*8*8*512           # 16             # 22 + 3 * 8 = 46
        x5 = Conv3D(1, 1)(x4)
        x6 = BatchNormalization(momentum=0.99, epsilon=1e-3)(x5)
        x7 = LeakyReLU(alpha = 0.2)(x6)
        x8 = Flatten()(x7)
        dose_score = Dense(1,activation='sigmoid')(x8)

        self.discriminator_ = Model(inputs=[dose_to_examine, ct_image, roi_masks], outputs=dose_score, name="discriminator")
        self.model.summary()
        return self.model