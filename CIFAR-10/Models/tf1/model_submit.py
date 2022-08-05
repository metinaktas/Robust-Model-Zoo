# based on https://github.com/tensorflow/models/tree/master/resnet
import numpy as np
import tensorflow as tf

class Model(object):
  """ResNet model."""

  def __init__(self, mode,x_input,y_input,quantize_active=True):
    """ResNet constructor.

    Args:
      mode: One of 'train' and 'eval'.
    """
    self.mode = mode
    self.x_input = x_input
    self.y_input = y_input
    self.filter_size = 3
    self.class_num = 10
    self.activation = "swish"
    self.batch_normalization_active = True
    self.skip_connection_active = True
    self.residual_sizes = 2
    self.quantize_active = quantize_active
    
    self._build_model()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    assert self.mode == 'train' or self.mode == 'eval'
    """Build the core model within the graph."""
    step_num = 10    
    global_max_var = None
    alpha = 100000000.0
    with tf.compat.v1.variable_scope('input',reuse=tf.compat.v1.AUTO_REUSE):
      x = self.x_input

    # LAYER-1 BLOCK-1
    in_filter = 3
    out_filter = 32
    x = self.block_1(x,in_filter,out_filter,"block_1_layer_1",num=1)
    # LAYER-1 BLOCK-2
    x = self.block_2(x,step_num,alpha,out_filter, in_filter,"block_2_layer_1")
    # LAYER-1 BLOCK-3
    x = self.block_3(x,in_filter,out_filter,"block_3_layer_1",num=1)

    # STRIDE-1
    in_filter = 32
    out_filter = 64
    x = self.block_4(x,in_filter,out_filter,'stride_1')

    in_filter = 64
    out_filter = 64
    for i in range(self.residual_sizes):
      # LAYER-2 BLOCK-1
      x = self.block_1(x,in_filter,out_filter,"block_1_layer_2_%d"%i,num=1)
      # LAYER-2 BLOCK-2
      x = self.block_2(x,step_num,alpha,out_filter,in_filter,"block_2_layer_2_%d"%i)
      # LAYER-2 BLOCK-3
      x = self.block_3(x,in_filter,out_filter,"block_3_layer_2_%d"%i,num=1)

    # STRIDE-2
    in_filter = 64
    out_filter = 128
    x = self.block_4(x,in_filter,out_filter,'stride_2')

    # LAYER-3 BLOCK-1
    in_filter = 128
    out_filter = 128
    x = self.block_1(x,in_filter,out_filter,"block_1_layer_3_%d"%i,num=1)
    # LAYER-3 BLOCK-2
    x = self.block_2(x,step_num,alpha,out_filter,in_filter,"block_2_layer_3_%d"%i)
    # LAYER-3 BLOCK-3
    x = self.block_3(x,in_filter,out_filter,"block_3_layer_3_%d"%i,num=1)

    # STRIDE-3
    in_filter = 128
    out_filter = 256
    x = self.block_4(x,in_filter,out_filter,'stride_3')

    in_filter = 256
    out_filter = 256
    for i in range(self.residual_sizes):
      # LAYER-4 BLOCK-1
      x = self.block_1(x,in_filter,out_filter,"block_1_layer_4_%d"%i,num=1)
      # LAYER-4 BLOCK-2
      x = self.block_2(x,step_num,alpha,out_filter,in_filter,"block_2_layer_4_%d"%i)
      # LAYER-4 BLOCK-3
      x = self.block_3(x,in_filter,out_filter,"block_3_layer_4_%d"%i,num=1)

    with tf.compat.v1.variable_scope('unit_last',reuse=tf.compat.v1.AUTO_REUSE):
      x = self._batch_norm('final_bn', x)
      x = self.activation_function(x, 0.1)
      x = self._global_avg_pool(x)

    self.last_latent = x
    with tf.compat.v1.variable_scope('logit',reuse=tf.compat.v1.AUTO_REUSE):
      self.pre_softmax,self.weight_class_w,self.weight_class_b = self._fully_connected(x, self.class_num)

    self.predictions = tf.argmax(input=self.pre_softmax, axis=1)
    self.correct_prediction = tf.equal(self.predictions, self.y_input)
    self.num_correct = tf.reduce_sum(
        input_tensor=tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(
        input_tensor=tf.cast(self.correct_prediction, tf.float32))

  def block_1(self,x,in_filter,out_filter,name,num=1):
    with tf.compat.v1.variable_scope('%s_sub_0' %name,reuse=tf.compat.v1.AUTO_REUSE):
      x = self._residual(x, in_filter, out_filter, self._stride_arr(1), False)
    for i in range(1,num,1):
      with tf.compat.v1.variable_scope('%s_sub_%d'%(name,i),reuse=tf.compat.v1.AUTO_REUSE):
        x = self._residual(x, out_filter, out_filter, self._stride_arr(1), False)

    return x

  def block_2(self,x,step_num,alpha,in_filter, out_filter,name):
    if self.quantize_active == True:
      x = quantize(x,step_num,None,scale_active=True,limit_active=True,q=1.0,alpha=alpha)
    with tf.compat.v1.variable_scope('%s' %(name),reuse=tf.compat.v1.AUTO_REUSE):
      x = self.block_2_1(x,in_filter, out_filter, self._stride_arr(1))        

    return x

  def block_3(self,x,in_filter,out_filter,name,num=1):
    with tf.compat.v1.variable_scope('%s_sub_0' %name,reuse=tf.compat.v1.AUTO_REUSE):
      x = self._residual(x, in_filter, out_filter, self._stride_arr(1), False)
    for i in range(1,num,1):
      with tf.compat.v1.variable_scope('%s_sub_%d'%(name,i),reuse=tf.compat.v1.AUTO_REUSE):
        x = self._residual(x, out_filter, out_filter, self._stride_arr(1), False)

    return x

  def block_4(self,x,in_filter,out_filter,name):
    with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
      x = self._residual(x, in_filter, out_filter, self._stride_arr(2), False)

    return x

  def _batch_norm(self, name, x):
    """Batch normalization."""
    if self.batch_normalization_active == True:
      with tf.compat.v1.name_scope(name):
        with tf.compat.v1.variable_scope('BN',reuse=tf.compat.v1.AUTO_REUSE):
          shape = x.get_shape().as_list()
          mean_1,variance_1 = tf.nn.moments(x, axes=[0,1,2],keepdims=True,name="moments")
          x = (x - mean_1)/tf.sqrt(variance_1+1e-9)
          dn_beta = tf.compat.v1.get_variable('DN_Mean', [1, 1, 1, shape[3]],tf.float32, initializer=tf.compat.v1.keras.initializers.Constant(value=0, dtype=tf.float32))
          dn_scale = tf.compat.v1.get_variable('DN_Variance', [1, 1, 1, shape[3]],tf.float32, initializer=tf.compat.v1.keras.initializers.Constant(value=1, dtype=tf.float32))
          x = tf.multiply(x,dn_scale) + dn_beta

    return x

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.compat.v1.variable_scope('shared_activation',reuse=tf.compat.v1.AUTO_REUSE):
        x = self._batch_norm('init_bn', x)
        x = self.activation_function(x, 0.1)
        orig_x = x
    else:
      with tf.compat.v1.variable_scope('residual_only_activation',reuse=tf.compat.v1.AUTO_REUSE):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self.activation_function(x, 0.1)

    with tf.compat.v1.variable_scope('sub1',reuse=tf.compat.v1.AUTO_REUSE):
      x,_ = self._conv('conv1', x, self.filter_size, in_filter, out_filter, stride)

    with tf.compat.v1.variable_scope('sub2',reuse=tf.compat.v1.AUTO_REUSE):
      x = self._batch_norm('bn2', x)
      x = self.activation_function(x, 0.1)
      x,_ = self._conv('conv2', x, self.filter_size, out_filter, out_filter, [1, 1, 1, 1])

    with tf.compat.v1.variable_scope('sub_add',reuse=tf.compat.v1.AUTO_REUSE):
      if stride[1] > 1:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
      if in_filter != out_filter:
        orig_x,_=self._conv('conv_1x1', orig_x, 1, in_filter, out_filter, self._stride_arr(1))

      if self.skip_connection_active == True:
        x += orig_x

    tf.compat.v1.logging.debug('image after unit %s', x.get_shape())
    return x

  def block_2_1(self,x,in_filter, out_filter, stride):
    with tf.compat.v1.variable_scope('sub1',reuse=tf.compat.v1.AUTO_REUSE):
      x,_ = self._conv('conv1', x, self.filter_size, in_filter, in_filter*2, stride)
      x = self.activation_function(x, 0.1)
    with tf.compat.v1.variable_scope('sub2',reuse=tf.compat.v1.AUTO_REUSE):
      x,_ = self._conv('conv2', x, self.filter_size, in_filter*2, in_filter*2, [1, 1, 1, 1])
      x = self.activation_function(x, 0.1)
    with tf.compat.v1.variable_scope('sub3',reuse=tf.compat.v1.AUTO_REUSE):
      x,_ = self._conv('conv3', x, self.filter_size, in_filter*2, in_filter, stride)
      x = self.activation_function(x, 0.1)
    with tf.compat.v1.variable_scope('sub4',reuse=tf.compat.v1.AUTO_REUSE):
      x,_ = self._conv('conv4', x, self.filter_size, in_filter, out_filter, [1, 1, 1, 1])

    return x

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.compat.v1.variable_scope(name,reuse=tf.compat.v1.AUTO_REUSE):
      n = filter_size * filter_size * out_filters
      kernel = tf.compat.v1.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.compat.v1.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(input=x, filters=kernel, strides=strides, padding='SAME'),kernel

  def activation_function(self, x, leakiness=0.0, theda=1.0):
    if self.activation == 'tanh':
      return tf.nn.tanh(x)
    elif self.activation == 'sigmoid':
      return tf.nn.sigmoid(x)
    elif self.activation == 'relu':
      return tf.nn.relu(x)
    elif self.activation == 'swish':
      return x * tf.nn.sigmoid(theda *x)
    elif self.activation == 'lrelu':
      return tf.compat.v1.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
    elif self.activation == "avg_pool":
      return tf.nn.avg_pool(x, ksize=[1,5,5,1], strides=[1,1,1,1], padding="SAME")
    else:
      err_not_defined_activation

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(input=x)[0], -1])
    with tf.compat.v1.variable_scope('FC',reuse=tf.compat.v1.AUTO_REUSE):
      w = tf.compat.v1.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, distribution="uniform"))
      b = tf.compat.v1.get_variable('biases', [out_dim],
                        initializer=tf.compat.v1.constant_initializer())
    return tf.compat.v1.nn.xw_plus_b(x, w, b),w,b

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(input_tensor=x, axis=[1, 2])
  
def sig(x,alpha,b):
    return 1.0/(1.0 + tf.math.exp(-alpha*(x-b)))

def round(x,step_length,alpha):
    b = tf.round(x/step_length)*step_length
    y = step_length*sig(x,alpha,b)+b
 
    return y

def quantize(var,step_num,global_max_var,scale_active=True,limit_active=True,q=1.0,alpha=1000.0):
    step_num = tf.cast(step_num,dtype=tf.float32)    
    if scale_active == True:
        if global_max_var is None:
            global_max_var = tf.reduce_max(input_tensor=tf.abs(var))
        scaled_var = tf.truediv(var,global_max_var+1e-9)
        scaled_var = tf.multiply(tf.pow(tf.abs(scaled_var),q), tf.sign(scaled_var))
        if limit_active == True:
            cond = tf.cast(tf.abs(scaled_var) > 1.0, dtype=tf.float32)
            scaled_var = tf.multiply(cond,tf.sign(scaled_var)) + tf.multiply(1.0-cond,scaled_var)
        scaled_var = tf.multiply(tf.pow(tf.abs(scaled_var),1.0/q), tf.sign(scaled_var))
        step_length = 1.0 / step_num
        quantized_var = tf.multiply(global_max_var,round(scaled_var,step_length,alpha))
    else:
        step_length = global_max_var / step_num
        quantized_var = round(var,step_length,alpha)
        
    return quantized_var
