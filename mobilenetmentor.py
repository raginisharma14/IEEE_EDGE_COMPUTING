import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import numpy as np
from tensorflow.contrib.keras.python.keras.applications.mobilenet import _conv_block, _depthwise_conv_block
from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.layers import Input
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import Dropout
from tensorflow.contrib.keras.python.keras.layers import Conv2D
from tensorflow.contrib.keras.python.keras.layers import Reshape
from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers import Activation, Flatten, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers.noise import GaussianNoise

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.models import Sequential
import pdb
depth_multiplier = 1
class Mentor(object):



    def build(self, img_input, alpha, train_mode, num_classes, num_channels, seed, trainable) :
        shape = (1, 1, int(1024 * alpha))
	with tf.name_scope('mentor_conv1_1') as scope:
            
            kernel = tf.Variable(tf.truncated_normal([3, 3, num_channels, int(32* alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable,name = 'mentor_weights')

            #biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable = trainable , name = 'mentor_biases')
            #img_input = tf.keras.layers.InputLayer(shape = (32, 32, 3))
  	    self.conv1 = tf.nn.conv2d(img_input, kernel, [1, 2, 2, 1], "SAME")
            #self.conv1 = tf.nn.bias_add(self.conv1, biases)
            self.conv1 = tf.nn.relu(self.conv1)
        
	with tf.name_scope('mentor_conv2_1') as scope:
  	    #conv2 = tf.layers.batch_normalization(img_input,training = train_mode, name='conv1_bn_mentor')
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(32*alpha), 1], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable,name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*32), int(64 * alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            self.conv2 = tf.nn.separable_conv2d(self.conv1,depthwise_filter, pointwise_filter, [1, 1, 1, 1], "SAME")
#            self.conv2 = tf.contrib.layers.batch_norm(self.conv2, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            
            self.conv2 = tf.nn.relu(self.conv2)

	with tf.name_scope('mentor_conv3_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(alpha*64), 1], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*64), int(128 * alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            self.conv3 = tf.nn.separable_conv2d(self.conv2, depthwise_filter, pointwise_filter, [1, 2, 2, 1], "SAME")
 #           self.conv3 = tf.contrib.layers.batch_norm(self.conv3, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            

            self.conv3 = tf.nn.relu(self.conv3)
	with tf.name_scope('mentor_conv4_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(alpha*128), 1], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*128), int(128 *alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            self.conv4 = tf.nn.separable_conv2d(self.conv3, depthwise_filter, pointwise_filter, [1, 1, 1, 1], "SAME")
  #          self.conv4 = tf.contrib.layers.batch_norm(self.conv4, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            

            self.conv4 = tf.nn.relu(self.conv4)
	with tf.name_scope('mentor_conv5_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(alpha*128), 1], dtype= tf.float32, stddev = 1e-2, seed=seed),trainable = trainable, name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*128), int(256* alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            self.conv5 = tf.nn.separable_conv2d(self.conv4, depthwise_filter, pointwise_filter, [1,2,2,1], "SAME")
   #         self.conv5 = tf.contrib.layers.batch_norm(self.conv5, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            

            self.conv5 = tf.nn.relu(self.conv5)
	with tf.name_scope('mentor_conv6_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(alpha*256), 1], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*256), int(256* alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            self.conv6 = tf.nn.separable_conv2d(self.conv5, depthwise_filter, pointwise_filter, [1, 1, 1, 1], "SAME")
    #        self.conv6 = tf.contrib.layers.batch_norm(self.conv6, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            
            self.conv6 = tf.nn.relu(self.conv6)
	with tf.name_scope('mentor_conv7_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(256*alpha), 1], dtype= tf.float32, stddev = 1e-2, seed=seed),trainable = trainable, name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*256), int(alpha*512)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            self.conv7 = tf.nn.separable_conv2d(self.conv6, depthwise_filter, pointwise_filter, [1, 2, 2,1], "SAME")
     #       self.conv7 = tf.contrib.layers.batch_norm(self.conv7, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            
            self.conv7 = tf.nn.relu(self.conv7)
	with tf.name_scope('mentor_conv8_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(512* alpha), 1], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable,name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*512), int(alpha*512)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            self.conv8 = tf.nn.separable_conv2d(self.conv7, depthwise_filter, pointwise_filter, [1, 1, 1, 1], "SAME")
      #      self.conv8 = tf.contrib.layers.batch_norm(self.conv8, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            
            self.conv8 = tf.nn.relu(self.conv8)
	with tf.name_scope('mentor_conv9_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(alpha*512), 1], dtype= tf.float32, stddev = 1e-2, seed=seed),trainable = trainable, name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*512), int(512* alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            self.conv9 = tf.nn.separable_conv2d(self.conv8,depthwise_filter, pointwise_filter,  [1, 1, 1, 1], "SAME")
       #     self.conv9 = tf.contrib.layers.batch_norm(self.conv9, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            
            self.conv9 = tf.nn.relu(self.conv9)
	with tf.name_scope('mentor_conv10_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(512* alpha), 1], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable,name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*512), int(512* alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed),trainable = trainable, name = 'mentor_weights')
            self.conv10 = tf.nn.separable_conv2d(self.conv9, depthwise_filter, pointwise_filter,  [1, 1, 1, 1], "SAME")
        #    self.conv10 = tf.contrib.layers.batch_norm(self.conv10, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            
            self.conv10 = tf.nn.relu(self.conv10)
	with tf.name_scope('mentor_conv11_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(512 *alpha), 1], dtype= tf.float32, stddev = 1e-2, seed=seed),trainable = trainable, name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*512), int(512 *alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable,name = 'mentor_weights')
            self.conv11= tf.nn.separable_conv2d(self.conv10, depthwise_filter, pointwise_filter,  [1, 1, 1, 1], "SAME")
         #   self.conv11 = tf.contrib.layers.batch_norm(self.conv11, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            
            self.conv11 = tf.nn.relu(self.conv11)
	with tf.name_scope('mentor_conv12_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(512*alpha), 1], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable,name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*512), int(512*alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            self.conv12= tf.nn.separable_conv2d(self.conv11, depthwise_filter, pointwise_filter,  [1, 1, 1, 1], "SAME")
          #  self.conv12 = tf.contrib.layers.batch_norm(self.conv12, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            
            self.conv12 = tf.nn.relu(self.conv12)
	with tf.name_scope('mentor_conv13_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(512 *alpha), 1], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable ,name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*512), int(1024 *alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable,name = 'mentor_weights')
            self.conv13 = tf.nn.separable_conv2d(self.conv12, depthwise_filter, pointwise_filter,  [1, 2, 2, 1], "SAME")
           # self.conv13 = tf.contrib.layers.batch_norm(self.conv13, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            
            self.conv13 = tf.nn.relu(self.conv13)
	with tf.name_scope('mentor_conv14_1') as scope:
            depthwise_filter = tf.Variable(tf.truncated_normal([3, 3, int(1024 *alpha), 1], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            pointwise_filter = tf.Variable(tf.truncated_normal([1, 1, int(alpha*1024), int(1024 *alpha)], dtype= tf.float32, stddev = 1e-2, seed=seed), trainable = trainable, name = 'mentor_weights')
            self.conv14 = tf.nn.separable_conv2d(self.conv13, depthwise_filter, pointwise_filter,  [1, 1, 1, 1], "SAME")
            #self.conv14 = tf.contrib.layers.batch_norm(self.conv14, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            
            self.conv14 = tf.nn.relu(self.conv14)
        with tf.name_scope('mentor_conv15_1') as scope:
            
            self.conv15 = tf.reduce_mean(self.conv14, [1,2])
            
            #self.conv15 = tf.layers.average_pooling2d(self.conv14, [2, 2], [1, 1])

            #shape = int(np.prod(conv15.get_shape()[1:]))
            #shape = [1, 1, int(1024 * alpha)]
            self.conv16 = tf.reshape(self.conv15,  [-1, 1, 1, int(1024*alpha)])

	    kernel = tf.Variable(tf.truncated_normal([1, 1, int(1024 * alpha), num_classes], dtype = tf.float32, stddev = 1e-2, seed = seed),trainable = trainable, name = "mentor_weights")
            #self.conv17 = tf.nn.dropout(self.conv16, 0.5)
            self.conv18 = tf.nn.conv2d(self.conv16, kernel, [1, 1, 1, 1], padding='SAME')
            #self.conv18 = tf.contrib.layers.batch_norm(self.conv18, decay = 0.9, center = True, scale = False, updates_collections = None, is_training= train_mode)            
            self.conv19 = tf.nn.softmax(self.conv18)
            self.conv20 = tf.reshape(self.conv19, [-1, num_classes])
	   # conv = GaussianNoise(0.5)(conv)
	
        #model = Model(img_input, conv22)
        """    
	for l in model.layers:
	    l.trainable = False
        """
        data_dict = {}
        data_dict = self.fill_data_dict(data_dict, self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, self.conv9, self.conv10, self.conv11, self.conv12, self.conv13, self.conv14, self.conv15, self.conv16, self.conv18, self.conv19, self.conv20)

        return  data_dict
    def loss(self, labels):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits= self.conv20, name = 'xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    def training(self, loss, lr, global_step):
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.MomentumOptimizer(lr,0.9, use_locking = True, use_nesterov = True)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def fill_data_dict(self, data_dict, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15,conv17, conv18, conv19, conv20):
        data_dict['conv1'] = conv1
        data_dict['conv2'] = conv2
        data_dict['conv3'] = conv3
        data_dict['conv4'] = conv4
        data_dict['conv5'] = conv5
        data_dict['conv6'] = conv6
        data_dict['conv7'] = conv7
        data_dict['conv8'] = conv8
        data_dict['conv9'] = conv9
        data_dict['conv10'] = conv10
        data_dict['conv11'] = conv11
        data_dict['conv12'] = conv12
        data_dict['conv13'] = conv13
        data_dict['conv14'] = conv14
        data_dict['conv15'] = conv15
        #data_dict['conv16'] = conv16
        data_dict['conv17'] = conv17
        data_dict['conv18'] = conv18
        data_dict['conv19'] = conv19
        data_dict['conv20'] = conv20

        return data_dict
