import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import ops
import utils

class Generator:
  def __init__(self, name, is_training, ngf=64, norm='instance', image_size=128, is_dropout=False):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size = image_size
    self.is_dropout = is_dropout
    # if image_size == 40:
    #   self.noise = tf.random_normal([1, 10, 10, 10, 128],mean=0.0, stddev=1.0,dtype=tf.float32)
    # else:
    #   self.noise = tf.random_normal([1, 7, 7, 7, 128],mean=0.0, stddev=1.0,dtype=tf.float32)
    self.noise = None

  def __call__(self, input, noise=None):
    """
    Args:
      input: batch_size x width x height x 1
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
      # conv layers
      c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='c7s1_32')                             # (?, w, h, 32)
      d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d64')                                 # (?, w/2, h/2, 64)
      d128 = ops.dk(d64, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='d128')                                # (?, w/4, h/4, 128)

      if self.image_size <= 128:
        # use 6 residual blocks for 128x128 images
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, norm=None, n=6)      # (?, w/4, h/4, 128)
      else:
        # 9 blocks for higher resolution
        res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9)      # (?, w/4, h/4, 128)
      
      if (self.noise==None):
        noise = tf.random_normal(res_output.get_shape().as_list(),mean=0.0, stddev=1.0,dtype=tf.float32)
      else:
        noise = self.noise
      res_output = tf.concat([res_output, noise], 4)
      # fractional-strided convolution
      u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u64')                                 # (?, w/2, h/2, 64)
      u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
          reuse=self.reuse, name='u32', output_size=self.image_size)         # (?, w, h, 32)

      # conv layer
      # Note: the paper said that ReLU and _norm were used
      # but actually tanh was used and no _norm here


      output = ops.c7s1_k(u32, 1, norm=None,
          activation='tanh', reuse=self.reuse, name='output')           # (?, d, w, h, 1)
          
      if self.is_dropout:
        output = tf.nn.dropout(output, keep_prob=0.9)
    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output

  def sample(self, input):
    image = utils.batch_convert2int(self.__call__(input))
    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return image
