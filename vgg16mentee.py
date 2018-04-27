import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
from tensorflow.contrib.keras.python.keras import backend as K
"""
Removed all kinds of regularizers such as dropout and batch_normalization

"""
class Mentee(object):

	def __init__(self, num_channels, trainable=True):
		self.trainable = trainable
		self.parameters = []
                self.num_channels = num_channels
        
        
        """
            This function is not being used currently; if we need regularization we call it.
            as mentioned below.
        """

        def extra_regularization(self, out):
            out = tf.contrib.layers.batch_norm(out,  decay=0.999,
                                    center=True,
                                    scale=False,
                                    updates_collections= None, is_training= train_mode)
            mean, var = tf.nn.moments(out, axes=[0])
            out = tf.nn.batch_normalization(out, mean, var)
            out = (out - mean) / tf.sqrt(var + tf.constant(1e-10))

            return out


	def build(self, rgb, num_classes, temp_softmax, seed,train_mode):
                
                K.set_learning_phase(True)
    		# conv1_1
		with tf.name_scope('mentee_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, self.num_channels, 64], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable= self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
                        #out = self.extra_regularization(out)

			self.conv1_1 = tf.nn.relu(out, name=scope)
                        #self.conv1_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv1_1')(self.conv1_1)
			self.parameters += [kernel, biases]
			
		self.pool1 = tf.nn.max_pool(self.conv1_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool1')
                #conv2_1
		with tf.name_scope('mentee_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
								trainable= self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv2_1 = tf.nn.relu(out, name=scope)
                        #self.conv2_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv2_1')(self.conv2_1)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv2_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool2')
		with tf.name_scope('mentee_conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								trainable= self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv3_1 = tf.nn.relu(out, name=scope)
                        #self.conv3_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv3_1')(self.conv3_1)
			self.parameters += [kernel, biases]

		self.pool3 = tf.nn.max_pool(self.conv3_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool3')

		with tf.name_scope('mentee_conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
													 stddev=1e-2, seed= seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv4_1 = tf.nn.relu(out, name=scope)
                        #self.conv4_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv4_1')(self.conv4_1)
			self.parameters += [kernel, biases]

		self.pool4 = tf.nn.max_pool(self.conv4_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool4')
		with tf.name_scope('mentee_conv5_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv5_1 = tf.nn.relu(out, name=scope)
                        #self.conv5_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv5_1')(self.conv5_1)
			self.parameters += [kernel, biases]
		
                self.pool5 = tf.nn.max_pool(self.conv5_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool5')
		with tf.name_scope('mentee_conv6_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-2, seed = seed), trainable = self.trainable, name='mentee_weights')
			conv = tf.nn.conv2d(self.pool5, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								trainable=self.trainable, name='mentee_biases')
			out = tf.nn.bias_add(conv, biases)
			self.conv6_1 = tf.nn.relu(out, name=scope)
                        #self.conv6_1 = BatchNormalization(axis = -1, name= 'mentee_bn_conv6_1')(self.conv6_1)
			self.parameters += [kernel, biases]
		
                self.pool6 = tf.nn.max_pool(self.conv6_1,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='pool6')

                # fc1
		with tf.name_scope('mentee_fc1') as scope:
			shape = int(np.prod(self.pool6.get_shape()[1:]))
			fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
														 dtype=tf.float32, stddev=1e-2, seed = seed), trainable = self.trainable,name='mentee_weights')
			fc1b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
								 trainable=self.trainable, name='mentee_biases')
			pool6_flat = tf.reshape(self.pool6, [-1, shape])
			fc1l = tf.nn.bias_add(tf.matmul(pool6_flat, fc1w), fc1b)
			self.fc1 = tf.nn.relu(fc1l)
                        #self.fc1 = BatchNormalization(axis = -1, name= 'mentee_bn_fc1')(self.fc1)
                        #if train_mode == True:
                        print("Traine_mode is true")
                        #self.fc1 = tf.nn.dropout(self.fc1, 0.5, seed = seed)
			self.parameters += [fc1w, fc1b]
		
                with tf.name_scope('mentee_fc2') as scope:
			fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
														 dtype=tf.float32, stddev=1e-2, seed = seed), trainable = self.trainable,name='mentee_weights')
			fc2b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
								 trainable=self.trainable, name='mentee_biases')
			fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
			self.fc2 = tf.nn.relu(fc2l)
                        #self.fc2 = BatchNormalization(axis = -1, name= 'mentee_bn_fc2')(self.fc2)

                        """
                            Dropout and BatchNormalization are added to regularize the network and perform better by generalizing well.
                            However, to demonstrate knowledge transfer effectiveness, no other regularizers are added.
                        """
                        #if train_mode == True:
                        #self.fc2 = tf.nn.dropout(self.fc2, 0.5, seed = seed)
			self.parameters += [fc2w, fc2b]
                
                with tf.name_scope('mentee_fc3') as scope:
			fc3w = tf.Variable(tf.truncated_normal([4096, num_classes],
														 dtype=tf.float32, stddev=1e-2, seed = seed), trainable = self.trainable,name='mentee_weights')
			fc3b = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
								 trainable=self.trainable, name='mentee_biases')
			self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
			#self.fc3 = tf.nn.relu(fc3l)
			self.parameters += [fc3w, fc3b]
                
                self.softmax = tf.nn.softmax(self.fc3l/temp_softmax)
                return self

	def loss(self, labels):
		#labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
			logits=self.fc3l, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')


	def training(self, loss, learning_rate, global_step):
		tf.summary.scalar('loss', loss)
		#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		
		train_op = optimizer.minimize(loss, global_step=global_step
                        )
		

		return train_op
