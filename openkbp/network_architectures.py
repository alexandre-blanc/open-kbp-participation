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

from provided_code.dropout import CustomSpatialDropout3D
from provided_code.spectral_normalization import DenseSN, ConvSN3D, ConvSN3DTranspose
from provided_code.Cones import Cones3D, cone3D_coordinates_normalization, cone3D_aperture_regularization, ConeCoordinatesRegularization


class DefineDoseFromCT:
    """This class defines the architecture for a U-NET and must be inherited by a child class that
    executes various functions like training or predicting"""
    def __init__(self):
        self.generator_ = None
        self.discriminator_ = None
        self.adversarial_model_ = None
        self.discriminator_model_ = None
        self.use_possible_dose = True
        self.use_upsampling = True
        self.use_energy_conservation = True
        self.use_sn = True
        self.use_cones = True
        self.use_unet = False
        self.number_of_cones = 9
        self.aperure_regularization = .1
        self.direction_regularization = .1

        self.gen_deconv_drop_out_rate = 0.5
        self.gen_deconv_use_dropout = True
        self.l1_lambda = 100
        self.initial_number_of_filters = 8
        self.gen_use_attention_layer = True

        # for internal use only
        self.number_of_attention_blocks = 0
        
    def generator_convolution(self, x, number_of_filters, use_batchnorm=True, use_sn=False, relu_leak_rate=0.2):
        """Convolution block used for generator"""
        if use_sn:
            x = ConvSN3D(number_of_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=True)(x)
        else:
            x = Conv3D(number_of_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=True)(x)
        if use_batchnorm:
            x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = LeakyReLU(alpha=relu_leak_rate)(x)
        return x

    def generator_convolution_transpose(self, x, nodes, use_dropout = None, use_batchnorm=True, skip_x=None, use_upsampling=False, use_sn=False, drop_out_rate = None, relu_leak_rate=0.0):
        """Convolution transpose block used for generator"""
        
        if use_dropout is None:
            use_dropout = self.gen_deconv_use_dropout
        if drop_out_rate is None:
            drop_out_rate = self.gen_deconv_drop_out_rate

        if skip_x is not None:
            x = concatenate([x, skip_x])
        if use_upsampling:
            x = UpSampling3D()(x)
            if use_sn:
                x = ConvSN3D(nodes, kernel_size=self.filter_size, padding='same', use_bias=True)(x)
            else:
                x = Conv3D(nodes, kernel_size=self.filter_size, padding='same', use_bias=True)(x)
        else:
            if use_sn:
                x = ConvSN3DTranspose(nodes, self.filter_size, strides=self.stride_size, padding="same", use_bias=True)(x)
            else:
                x = Conv3DTranspose(nodes, self.filter_size, strides=self.stride_size, padding="same", use_bias=True)(x)
       
        if use_batchnorm:
            x  = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
       
        if use_dropout:
            x = CustomSpatialDropout3D(drop_out_rate)(x)
       
        x = LeakyReLU(alpha=relu_leak_rate)(x)  # Use LeakyReLU(alpha = 0) instead of ReLU because ReLU is buggy when saved

        return x

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
        x1 = self.generator_convolution(x, self.initial_number_of_filters, use_batchnorm=False, use_sn=self.use_sn)                           # bs*64*64*64*64
        x2 = self.generator_convolution(x1, 2 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*32*32*32*128
        if self.gen_use_attention_layer:
            x2 =  self.attention_block(x2, use_sn=self.use_sn)
        x3 = self.generator_convolution(x2, 4 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*16*16*16*256
        x4 = self.generator_convolution(x3, 8 * self.initial_number_of_filters, use_sn=self.use_sn)                                            # bs*8*8*8*512
        x5 = self.generator_convolution(x4, 16 * self.initial_number_of_filters, use_sn=self.use_sn)                                           # bs*4*4*4*1024

        if self.use_unet:
            # Build model back up from bottleneck through upsampling
            x4b = self.generator_convolution_transpose(x5, 16 * self.initial_number_of_filters, use_sn=self.use_sn, use_upsampling=self.use_upsampling)                                # bs*8*8*8*1024
            x3b = self.generator_convolution_transpose(x4b, 8 * self.initial_number_of_filters, skip_x=x4, use_sn=self.use_sn, use_upsampling=self.use_upsampling)                     # bs*16*16*16*512
            x2b = self.generator_convolution_transpose(x3b, 4 * self.initial_number_of_filters, skip_x=x3, use_sn=self.use_sn, use_upsampling=self.use_upsampling)                     # bs*32*32*32*256
            if self.gen_use_attention_layer:
                x2b, _ =  self.attention_block(x2b, use_sn=self.use_sn)
            x1b = self.generator_convolution_transpose(x2b, 2 * self.initial_number_of_filters, skip_x=x2, use_dropout=False, use_sn=self.use_sn, use_upsampling=self.use_upsampling)  # bs*64*64*64*128
            x0b = self.generator_convolution_transpose(x1b, 1, skip_x=x1, use_dropout=False, use_batchnorm=False, use_sn=self.use_sn, use_upsampling=self.use_upsampling)  # bs*64*64*64*128
            x_final = x0b # AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding="same")(x0b)                                      # bs*128*128*128*1
                          # Average pooling is commented out to preserve high frequency detail
            unet_final_dose = Activation("sigmoid")(x_final)                                                                           # bs*128*128*128*1

        if self.use_cones:
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
            
        if self.use_unet and self.use_cones:
            scaled_unet_dose = MultiplyByScalar(initial_value=1.0)(unet_final_dose)
            scaled_cone_dose = MultiplyByScalar(initial_value=1.0)(combined_cones)
            final_dose = Add()([scaled_cone_dose, scaled_unet_dose])

            final_dose = Reshape([-1])(final_dose)
            final_dose = Lambda(lambda x: x/K.max(x,axis=-1,keepdims=True))(final_dose)
            final_dose = Reshape([128,128,128,1])(final_dose)
        
        elif self.use_unet:
            final_dose = unet_final_dose
        
        elif self.use_cones:
            final_dose = combined_cones

        else:
            raise ValueError("One of self.use_unet, self.use_cone must be set to True.")

        if self.use_possible_dose:
            final_dose = Multiply()([final_dose, possible_dose])                                                           # bs*128*128*128*1


        # Compile model for use
        self.generator_ = Model(inputs=[ct_image, roi_masks, possible_dose], outputs = final_dose, name="generator")
        self.generator_.compile(self.gen_optimizer, loss='mean_absolute_error')
        self.generator_.summary()

        return self.generator_

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
        x1 = self.generator_convolution(x, self.initial_number_of_filters, use_batchnorm=False)    # bs*64*64*64            # 2              # 1 + 3 * 1 = 4
        x2 = self.generator_convolution(x1, 2 * self.initial_number_of_filters)                     # bs*32*32*32*128        # 4             # 4 + 3 * 2 = 10
        if self.gen_use_attention_layer:
            x2 =  self.attention_block(x2, use_sn=self.use_sn)
        x3 = self.generator_convolution(x2, 4 * self.initial_number_of_filters)                     # bs*16*16*256           # 8              # 10 + 3 * 4 = 22
        x4 = self.generator_convolution(x3, 8 * self.initial_number_of_filters)                     # bs*8*8*8*512           # 16             # 22 + 3 * 8 = 46
        x5 = Conv3D(1, 1)(x4)
        x6 = BatchNormalization(momentum=0.99, epsilon=1e-3)(x5)
        x7 = LeakyReLU(alpha = 0.2)(x6)
        x8 = Flatten()(x7)
        dose_score = Dense(1,activation='sigmoid')(x8)

        self.discriminator_ = Model(inputs=[dose_to_examine, ct_image, roi_masks], outputs=dose_score, name="discriminator")
        self.discriminator_.summary()
        return self.discriminator_

    def adversarial_model(self):
        if self.adversarial_model_:
            return self.adversarial_model_
    
        self.discriminator_.trainable = False
        ct_image = Input(self.ct_shape)
        roi_masks = Input(self.roi_masks_shape)
        possible_dose = Input(self.dose_shape)
        
        dose_to_examine = self.generator_([ct_image, roi_masks, possible_dose])
        adversarial_score = self.discriminator_([dose_to_examine, ct_image, roi_masks])
        self.adversarial_model_ = Model(inputs=[ct_image, roi_masks, possible_dose], outputs=[dose_to_examine, adversarial_score])
        self.adversarial_model_.compile(loss=['mean_absolute_error', 'binary_crossentropy'], loss_weights = [self.l1_lambda, 1.0], optimizer=self.gen_optimizer)
        
        self.discriminator_.trainable = True

        
        return self.adversarial_model_

    def discriminator_model(self):
        if self.discriminator_model_:
            return self.discriminator_model_
        dose_to_examine = Input(self.dose_shape)
        ct_image = Input(self.ct_shape)
        roi_masks = Input(self.roi_masks_shape)
        final_dose = self.discriminator_([dose_to_examine, ct_image, roi_masks])
        self.discriminator_model_ = Model(inputs=[dose_to_examine, ct_image, roi_masks], outputs=final_dose)
        self.discriminator_model_.compile(loss='binary_crossentropy', loss_weights=[0.5] , optimizer=self.disc_optimizer, metrics=['accuracy'])
        return self.discriminator_model_ 


    def attention_block(self, x_input, use_sn=False):
        self.number_of_attention_blocks += 1
        batch_size, x, y, z, num_channels = K.int_shape(x_input)
        location_num = x*y*z
        downsampled_num = location_num // 8 # 8 car on est en 3D
         
        if use_sn:
            # f path (cf paper)
            f = ConvSN3D(num_channels // 8 , 1 , strides=1 , padding="same", use_bias=True)(x_input)
            f = Reshape((location_num , num_channels // 8))(f)
            
            # g path (cf paper)
            g = ConvSN3D(num_channels // 8 , 1 , strides=1 , padding="same", use_bias=True)(x_input)
            g = AveragePooling3D(strides = 2)(g)
            g = Reshape((downsampled_num , num_channels // 8))(g)
            
            # attention
            attn = tf.matmul(f , g , transpose_b = True)
            attn = Activation('softmax', name='attention_activation_{}'.format(self.number_of_attention_blocks))(attn)
            
            # h path (cf paper)
            h = ConvSN3D(num_channels // 2 , 1 , strides=1 , padding="same", use_bias=True)(x_input)
            h = AveragePooling3D(strides = 2)(h)
            h = Reshape((downsampled_num , num_channels // 2))(h)
            
            # attention * h
            attn_h = tf.matmul(attn, h)
            attn_h = Reshape(( x, y , z , num_channels // 2))(attn_h)
            attn_h = ConvSN3D(num_channels, 1 , strides=1 , padding="same", use_bias=True)(attn_h)
        else:
            # f path (cf paper)
            f = Conv3D(num_channels // 8 , 1 , strides=1 , padding="same", use_bias=True)(x_input)
            f = Reshape((location_num , num_channels // 8))(f)
            
            # g path (cf paper)
            g = Conv3D(num_channels // 8 , 1 , strides=1 , padding="same", use_bias=True)(x_input)
            g = AveragePooling3D(strides = 2)(g)
            g = Reshape((downsampled_num , num_channels // 8))(g)
            
            # attention
            attn = tf.matmul(f , g , transpose_b = True)
            attn = Activation('softmax', name='attention_activation_{}'.format(self.number_of_attention_blocks))(attn)
            
            # h path (cf paper)
            h = Conv3D(num_channels // 2 , 1 , strides=1 , padding="same", use_bias=True)(x_input)
            h = AveragePooling3D(strides = 2)(h)
            h = Reshape((downsampled_num , num_channels // 2))(h)
            
            # attention * h
            attn_h = tf.matmul(attn, h)
            attn_h = Reshape(( x, y , z , num_channels // 2))(attn_h)
            attn_h = Conv3D(num_channels, 1 , strides=1 , padding="same", use_bias=True)(attn_h)
            
        # define sigma 
        attn_h = MultiplyByScalar(name='attention_multiplier_{}'.format(self.number_of_attention_blocks))(attn_h)
        
        return x_input + attn_h

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















