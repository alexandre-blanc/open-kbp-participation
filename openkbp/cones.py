import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import Regularizer
import tensorflow as tf

def trace_cone(center, direction, ouverture, Nx):
  center = Nx*center
  ouverture = np.pi*ouverture
  
  X = np.zeros((Nx,Nx,2))
  for i in range(Nx):
    for j in range(Nx) :
      X[i,j,:] = np.array([i,j])

  im = np.zeros((Nx,Nx))

  smooth_sign = lambda x: (1/(1+np.exp(-Nx*x)))

  for i in range(Nx):
    for j in range(Nx):
      u = X[i,j,:] - center
      u = u/(1e-8+np.linalg.norm(u))
      angle = np.dot(u, direction)
      angle = np.arccos(angle)
      im[i,j] = smooth_sign(ouverture-angle)

  return im

class Cones(Layer):
    def __init__(self, output_dim, tolerance=1e-4, use_energy_conservation=False, smooth_sign_width=None, **kwargs):
        super(Cones, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.tolerance = tolerance
        self.use_energy_conservation = use_energy_conservation

        coordinates = np.zeros((self.output_dim, self.output_dim, 2))
        for i in range(self.output_dim):
           for j in range(self.output_dim):
              coordinates[i,j,:] = np.array([i,j])

        coordinates = K.constant(coordinates)
        self.coordinates = tf.Variable(initial_value=coordinates, trainable=False)
        if smooth_sign_width == None:
            self.smooth_sign_width = tf.Variable(initial_value=output_dim, dtype=tf.float32, trainable=False)
        else:
            self.smooth_sign_width = tf.Variable(initial_value=smooth_sign_width, dtype=tf.float32, trainable=False)
        self.grid_width = tf.Variable(initial_value=output_dim, dtype=tf.float32, trainable=False)


    def build(self, input_shape):
        super(Cones, self).build(input_shape)
    
    def call(self, x):
        center = self.grid_width*x[:,:2]
        center = K.expand_dims(center, axis=1)
        center = K.expand_dims(center, axis=1)

        direction = x[:,2:4]
        direction = K.expand_dims(direction,1)
        direction = K.expand_dims(direction,1)
        direction = K.l2_normalize(direction, axis=-1)

        aperture = np.pi*x[:,4:]
        aperture = K.expand_dims(aperture)

        u = self.coordinates - center
        normalized_u = K.l2_normalize(u, axis=-1)

        angle = self.heron_angle_from_vectors(normalized_u, direction)

        output = self.smooth_sign(aperture-angle)

        if self.use_energy_conservation:
            distance_to_origin = tf.norm(u, axis=-1)
            output = output/distance_to_origin
        
        output = K.expand_dims(output, -1)
        return output

    def smooth_sign(self, x):
        return tf.math.sigmoid(self.smooth_sign_width*x)

    def heron_angle_from_vectors(self, x, y):
        c = tf.norm(x - y, axis=-1)
        x_norm = tf.norm(x, axis=-1)
        y_norm = tf.norm(y, axis=-1)

        c = tf.reshape(c,[-1,self.output_dim**2])
        x_norm = tf.reshape(x_norm,[-1, self.output_dim**2])
        y_norm = tf.reshape(y_norm,[-1,1])

        y_norm = tf.repeat(y_norm, repeats=[self.output_dim**2], axis=1)
        angle_refill = tf.zeros_like(c)


        close_to_0 = c < self.tolerance       # to be clipped to 0
        close_to_pi = c > 2.0-self.tolerance  # to be clipped to pi
        close_to_center = x_norm < self.tolerance  # to be clipped to 0
        out_of_range = tf.math.logical_or(tf.math.logical_or(close_to_0, close_to_pi), close_to_center)
        in_range = tf.math.logical_not(out_of_range)
        in_range_indices = tf.cast(tf.where(in_range), tf.int32)

        pis = tf.where(close_to_pi, np.pi*tf.ones_like(c), tf.zeros_like(c))

        c = tf.boolean_mask(c, in_range)
        x_norm = tf.boolean_mask(x_norm, in_range)
        y_norm = tf.boolean_mask(y_norm, in_range)

        a = tf.math.maximum(x_norm, y_norm)
        b = tf.math.minimum(x_norm, y_norm)

        max_b_c = tf.math.maximum(b, c)
        min_b_c = tf.math.minimum(b, c)

        mu = min_b_c - (a - max_b_c)

        num = ((a-b)+c)*mu
        den = (a+(b+c))*((a-c)+b)

        angle = 2*tf.math.atan(tf.math.sqrt(num/den))

        angle = tf.tensor_scatter_nd_add(angle_refill, in_range_indices, angle)
        angle = angle + pis
        angle = tf.reshape(angle, [-1, self.output_dim, self.output_dim])

        return angle

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, self.output_dim, 1)

class Cones3D(Layer):

    coordinates_dict = {}

    def __init__(self, output_dim, tolerance=1e-2, use_energy_conservation=False, smooth_sign_width=None, **kwargs):
        super(Cones3D, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.tolerance = tolerance
        self.use_energy_conservation = use_energy_conservation

        if not(output_dim in self.coordinates_dict):
            coordinates = np.zeros((output_dim, output_dim, output_dim, 3))
            for i in range(output_dim):
              for j in range(output_dim):
                  for k in range(output_dim):
                      coordinates[i,j,k,:] = np.array([i,j,k]) - np.array([output_dim, output_dim, output_dim])/2
            self.coordinates_dict[output_dim] = K.constant(coordinates)

        if smooth_sign_width == None:
            self.smooth_sign_width = tf.Variable(initial_value=output_dim, dtype=tf.float32, trainable=False, name='smooth_sign_width')
        else:
            self.smooth_sign_width = tf.Variable(initial_value=smooth_sign_width, dtype=tf.float32, trainable=False, name = 'smooth_sign_width')
        self.grid_width = tf.Variable(initial_value=output_dim/2, dtype=tf.float32, trainable=True, name='grid_width')


    def build(self, input_shape):
        super(Cones3D, self).build(input_shape)
    
    def call(self, x):
        coordinates = self.coordinates_dict[self.output_dim]

        center = self.grid_width*x[:,:3]
        center = K.expand_dims(center, axis=1)
        center = K.expand_dims(center, axis=1)
        center = K.expand_dims(center, axis=1)

        direction = x[:,3:6]
        direction = K.expand_dims(direction,1)
        direction = K.expand_dims(direction,1)
        direction = K.expand_dims(direction,1)
        direction = K.l2_normalize(direction, axis=-1)

        aperture = np.pi*x[:,6:]
        aperture = K.expand_dims(aperture)
        aperture = K.expand_dims(aperture)

        u = coordinates - center
        normalized_u = K.l2_normalize(u, axis=-1)

        angle = self.heron_angle_from_vectors(normalized_u, direction)
        output = self.smooth_sign(aperture-angle)

        if self.use_energy_conservation:
            distance_to_origin = tf.norm(u, axis=-1)
            output = output/distance_to_origin**2
        
        output = K.expand_dims(output, -1)
        return output

    def smooth_sign(self, x):
        return tf.math.sigmoid(self.smooth_sign_width*x)

    def heron_angle_from_vectors(self, x, y):
        c = tf.norm(x - y, axis=-1)
        x_norm = tf.norm(x, axis=-1)
        y_norm = tf.norm(y, axis=-1)

        c = tf.reshape(c,[-1,self.output_dim**3])
        x_norm = tf.reshape(x_norm,[-1, self.output_dim**3])
        y_norm = tf.reshape(y_norm,[-1,1])

        y_norm = tf.repeat(y_norm, repeats=[self.output_dim**3], axis=1)
        angle_refill = tf.zeros_like(c)


        close_to_0 = c < self.tolerance       # to be clipped to 0
        close_to_pi = c > 2.0-self.tolerance  # to be clipped to pi
        close_to_center = x_norm < self.tolerance  # to be clipped to 0
        out_of_range = tf.math.logical_or(tf.math.logical_or(close_to_0, close_to_pi), close_to_center)
        in_range = tf.math.logical_not(out_of_range)
        in_range_indices = tf.cast(tf.where(in_range), tf.int32)

        pis = tf.where(close_to_pi, np.pi*tf.ones_like(c), tf.zeros_like(c))

        c = tf.boolean_mask(c, in_range)
        x_norm = tf.boolean_mask(x_norm, in_range)
        y_norm = tf.boolean_mask(y_norm, in_range)

        a = tf.math.maximum(x_norm, y_norm)
        b = tf.math.minimum(x_norm, y_norm)

        max_b_c = tf.math.maximum(b, c)
        min_b_c = tf.math.minimum(b, c)

        mu = min_b_c - (a - max_b_c)

        num = ((a-b)+c)*mu
        den = (a+(b+c))*((a-c)+b)

        angle = 2*tf.math.atan(tf.math.sqrt(num/den))

        angle = tf.tensor_scatter_nd_add(angle_refill, in_range_indices, angle)
        angle = angle + pis
        angle = tf.reshape(angle, [-1, self.output_dim, self.output_dim, self.output_dim])

        return angle

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, self.output_dim, 1)

    def get_config(self):
        return {'output_dim':self.output_dim,'tolerance':self.tolerance,\
                'use_energy_conservation':self.use_energy_conservation}

def cone3D_coordinates_normalization(coordinates):
    centers = coordinates[:,:,:3]
    directions = coordinates[:,:,3:6]
    apertures = coordinates[:,:,6:]

    return tf.concat([0.5*tf.math.tanh(centers),tf.math.l2_normalize(directions, axis=-1),tf.math.sigmoid(apertures)], axis=-1)

def cone3D_aperture_regularization(coordinates):
    apertures = coordinates[:,:,6]
    return tf.norm(apertures)/tf.cast(tf.size(apertures), dtype=tf.float32)

def cone3D_direction_regularization(coordinates):
    centers = coordinates[:,:,:3]
    center_to_origin = -tf.math.l2_normalize(centers)
    directions = coordinates[:,:,3:6]
    scalar_products = K.sum(center_to_origin*directions, axis=-1)

    return tf.norm(tf.ones_like(scalar_products)-scalar_products)/tf.cast(tf.size(scalar_products), dtype=tf.float32)

class aperture_direction(Regularizer):
  def __init__(self, aperture=0., direction=0.):
    self.aperture = K.cast_to_floatx(aperture)
    self.direction = K.cast_to_floatx(direction)

  def __call__(self, x):
    if not self.aperture and not self.direction:
      return K.constant(0.)
    regularization = 0.
    if self.aperture:
      regularization += self.aperture * cone3D_aperture_regularization(x)
    if self.direction:
      regularization += self.direction * cone3D_direction_regularization(x)
    return regularization

  def get_config(self):
    return {'aperture': float(self.aperture), 'direction': float(self.direction)}

class ConeCoordinatesRegularization(Layer):
  def __init__(self, aperture=0., direction=0., **kwargs):
    super(ConeCoordinatesRegularization, self).__init__(
        activity_regularizer=aperture_direction(aperture=aperture, direction=direction), **kwargs)
    self.supports_masking = True
    self.aperture = aperture
    self.direction = direction

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'aperture': self.aperture, 'direction': self.direction}
    base_config = super(ConeCoordinatesRegularization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))












