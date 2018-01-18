import tensorflow as tf 
tf.set_random_seed(1234)
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import numpy as np
np.random.seed(1234)
#from tensorflow.contrib.keras.python.keras.applications.mobilenet import _conv_block, _depthwise_conv_block
from tensorflow.contrib.keras.python.keras.applications.mobilenet import DepthwiseConv2D
from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Input
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import Dropout
from tensorflow.contrib.keras.python.keras.layers import Conv2D
from tensorflow.contrib.keras.python.keras.layers import Reshape
from tensorflow.contrib.keras.python.keras.layers import add
from tensorflow.contrib.keras.python.keras.layers import Activation, Flatten, MaxPooling2D,BatchNormalization
from tensorflow.contrib.keras.python.keras.layers import GaussianNoise 
from tensorflow.contrib.keras.python.keras.losses import categorical_crossentropy
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.models import Sequential
import pdb
depth_multiplier = 1
class Mentee(object):

    def relu6(self, x):
        return K.relu(x, max_value=6)

    def build(self,alpha, img_input, num_classes):

        shape = (1, 1, int(1024 * alpha))
	"""
	This looks dangerous. Not sure how the model would get affected with the laarning_phase variable set to True.
	"""
        
        K.set_learning_phase(True)

	with tf.name_scope('student') as scope:

	    self.conv1 = Conv2D(
                        int(32*alpha),
                        (3,3),
                        padding='same',
                        use_bias=False,
                        strides=(1,1),
                        name='student_conv1')(img_input)
            self.conv2 = BatchNormalization(axis=-1, name='student_conv1_bn')(self.conv1)
            self.conv3 = Activation(self.relu6, name='student_conv1_relu')(self.conv2)

	    self.conv4 = self._depthwise_conv_block(self.conv3, 64, alpha, depth_multiplier, block_id =15)
	    self.conv5 = self._depthwise_conv_block(self.conv4, 128, alpha, depth_multiplier,strides=(2, 2), block_id =16)
	    self.conv6 = self._depthwise_conv_block(self.conv5, 128, alpha, depth_multiplier,block_id =17)
	    self.conv7 = self._depthwise_conv_block(self.conv6, 256, alpha, depth_multiplier, strides=(2,2),block_id =18)
	    self.conv8 = self._depthwise_conv_block(self.conv7, 256, alpha, depth_multiplier, block_id =19)
	    self.conv9 = self._depthwise_conv_block(self.conv8, 512, alpha, depth_multiplier, strides = (2,2), block_id =20)
	    self.conv10 = self._depthwise_conv_block(self.conv9, 512, alpha, depth_multiplier, block_id =21)
	    self.conv11 = self._depthwise_conv_block(self.conv10, 512, alpha, depth_multiplier, block_id =22)
	    self.conv12 = self._depthwise_conv_block(self.conv11, 512, alpha, depth_multiplier, block_id =23)
	    self.conv13 = self._depthwise_conv_block(self.conv12, 512, alpha, depth_multiplier, block_id =24)
	    self.conv14 = self._depthwise_conv_block(self.conv13, 512, alpha, depth_multiplier, block_id =25)
	    self.conv15 = self._depthwise_conv_block(self.conv14, 1024, alpha, depth_multiplier,strides=(2,2), block_id =26)
	    self.conv16 = self._depthwise_conv_block(self.conv15, 1024, alpha, depth_multiplier, block_id =27)

            self.conv17 = GlobalAveragePooling2D()(self.conv16)
            self.conv18 = Reshape(shape, name='student_reshape_1')(self.conv17)
	
            self.conv19 = Dropout(0.5, name='student_dropout')(self.conv18)
            self.conv20 = Conv2D(num_classes, (1, 1), padding='same', name='student_conv_preds')(self.conv18)
            self.conv21 = Activation('softmax', name='student_act_softmax')(self.conv20)
            self.conv22 = Reshape((num_classes,), name='student_reshape_2')(self.conv21)

        return self
    def loss(self, labels):
        labels = tf.to_int64(labels)
        """
            softmax cross entropy with logits takes logits as input instead of probabilities. check the input for caution.
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = self.conv20, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    def training(self, loss, lr, global_step):
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op
    def _depthwise_conv_block(self, inputs, pointwise_conv_filters, alpha,depth_multiplier=1, strides=(1, 1), block_id=1):
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    	x = DepthwiseConv2D((3, 3),
			    padding='same',
			    depth_multiplier=depth_multiplier,
			    strides=strides,
			    use_bias=False,
			    name='student_conv_dw_%d' % block_id)(inputs)
	x = BatchNormalization(axis=-1, name='student_conv_dw_%d_bn' % block_id)(x)
	x = Activation(self.relu6, name='student_conv_dw_%d_relu' % block_id)(x)

	x = Conv2D(pointwise_conv_filters, (1, 1),
		   padding='same',
		   use_bias=False,
		   strides=(1, 1),
		   name='student_conv_pw_%d' % block_id)(x)
	x = BatchNormalization(axis=-1, name='student_conv_pw_%d_bn' % block_id)(x)
        return Activation(self.relu6, name='student_conv_pw_%d_relu' % block_id)(x)

