import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, activations, initializers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
#------------------------------------------------------------------------------------------------------
@register_keras_serializable(package="MyLayers")
class GeneSelection(Layer):
    def __init__(self, mask, n_status, **kwargs):
        super(GeneSelection, self).__init__(**kwargs)
        self.mask = mask
        self.n_status = n_status
        if isinstance(mask, list):
            self.mask_array = np.array(mask)
        else:
            self.mask_array = mask
            
        self.selected_indices = np.where(np.repeat(self.mask_array, n_status) == 1)[0]
       
    def call(self, inputs):
        selected = tf.gather(inputs, self.selected_indices, axis=-1)     
        return selected
    
    def get_config(self):
        config = super(GeneSelection, self).get_config()
        mask_config = self.mask.tolist() if isinstance(self.mask, np.ndarray) else self.mask
        config.update({'mask': mask_config, 'n_status': self.n_status})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.rep_mask.shape[1])
#------------------------------------------------------------------------------------------------------
class Diagonal(Layer):
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', W_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(Diagonal, self).__init__(**kwargs)
        self.units = units
        self.activation_fn = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = bias_constraint
        
 
    def build(self, input_shape):
        input_dimension = input_shape[1]
        self.kernel_shape = (input_dimension, self.units)
        self.n_inputs_per_node = int(input_dimension / self.units)
        rows = np.arange(input_dimension)
        cols = np.repeat(np.arange(self.units), self.n_inputs_per_node)
        self.nonzero_ind = np.column_stack((rows, cols))
        self.kernel = self.add_weight(name='kernel', shape=(input_dimension,),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True, constraint=self.kernel_constraint)
        
        self.branch_weights = self.add_weight(name='branch_weights', shape=(self.units,),
                                              initializer='ones', trainable=True)
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,), initializer=self.bias_initializer,
                                        name='bias', regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        
        else:
            self.bias = None
        super(Diagonal, self).build(input_shape)

    def call(self, x):
        n_features = x.shape[1]
        kernel = K.reshape(self.kernel, (1, n_features))
        mult = x * kernel
        shape_integers = tuple(int(dim) for dim in (-1, self.n_inputs_per_node))
        mult = K.reshape(mult, shape_integers)
        mult = K.sum(mult, axis=1)
        output = K.reshape(mult, (-1, self.units))
        

        output = output * self.branch_weights
        
        if self.use_bias:
            output = K.bias_add(output, self.bias)
   
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(Diagonal, self).get_config()
        config.update({'units': self.units, 'activation': activations.serialize(self.activation_fn),
                       'use_bias': self.use_bias, 'kernel_initializer': initializers.serialize(self.kernel_initializer),
                       'bias_initializer': initializers.serialize(self.bias_initializer),
                       'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                       'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                       'kernel_constraint': constraints.serialize(self.kernel_constraint),
                       'bias_constraint': constraints.serialize(self.bias_constraint)})
        return config
#------------------------------------------------------------------------------------------------------
class SparseTF(Layer):
    def __init__(self, units, map, kernel_initializer='glorot_uniform', W_regularizer=None, 
                 activation=None, use_bias=True, bias_initializer='zeros', bias_regularizer=None, 
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(SparseTF, self).__init__(**kwargs)
        self.units = units
        self.activation_fn = activations.get(activation)
        self.map = map
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.kernel_shape = (input_dim, self.units)  # Define kernel_shape
        self.map = self.map.astype(np.float32)
        self.nonzero_ind = np.array(np.nonzero(np.array(self.map))).T
        nonzero_count = self.nonzero_ind.shape[0]
        self.kernel_vector = self.add_weight(name='kernel_vector', shape=(nonzero_count,),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer, trainable=True,
                                             constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,), initializer=self.bias_initializer,
                                        name='bias', regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        super(SparseTF, self).build(input_shape)

    def call(self, inputs):
        tt = tf.scatter_nd(self.nonzero_ind, self.kernel_vector, self.kernel_shape)
        output = K.dot(inputs, tt)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output

    def get_config(self):
        config = super(SparseTF, self).get_config()
        config.update({'units': self.units, 'activation': activations.serialize(self.activation_fn),
                       'use_bias': self.use_bias, 'nonzero_ind': self.nonzero_ind.tolist(),
                       'kernel_initializer': initializers.serialize(self.kernel_initializer),
                       'W_regularizer': regularizers.serialize(self.kernel_regularizer),
                       'bias_initializer': initializers.serialize(self.bias_initializer),
                       'bias_regularizer': regularizers.serialize(self.bias_regularizer)})
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
#------------------------------------------------------------------------------------------------------
@register_keras_serializable()
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
#------------------------------------------------------------------------------------------------------
def LeCunUniform(seed=None):
    from tensorflow.keras.initializers import VarianceScaling
    return VarianceScaling(scale=1., mode='fan_in', distribution='uniform', seed=seed)
#------------------------------------------------------------------------------------------------------