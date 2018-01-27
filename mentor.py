import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import pdb
import numpy as np
import h5py as h5 
from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
from tensorflow.contrib.keras.python.keras.layers import Dropout
from tensorflow.contrib.keras.python.keras import backend as K
VGG_MEAN = [103.939, 116.779, 123.68]

class Teacher(object):

	def __init__(self, trainable=True, dropout=0.5):
		self.trainable = trainable
        	self.dropout = dropout
		self.parameters = []

	def build(self, rgb, num_classes, temp_softmax, train_mode):
                K.set_learning_phase(True)
		# conv1_1
		with tf.name_scope('mentor_conv1_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable, name='mentor_weights')
			conv = tf.nn.conv2d(rgb, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
		#	out = tf.nn.bias_add(conv, biases)
                 #       mean , var = tf.nn.moments(out, axes= [0])
                  #      out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv1_1 = tf.nn.relu(conv, name=scope)
                        self.conv1_1 = BatchNormalization(axis = -1, name= 'mentor_bn_conv1_1')(self.conv1_1)
                        #self.conv1_1 = Dropout((0.4))(self.conv1_1)
			self.parameters += [kernel, biases]
			
		with tf.name_scope('mentor_conv1_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable, name='mentor_weights')
			conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                        #mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv1_2 = tf.nn.relu(conv, name=scope)
                        self.conv1_2 = BatchNormalization(axis = -1, name= 'mentor_bn_conv1_2')(self.conv1_2)
			self.parameters += [kernel, biases]
			
		self.pool1 = tf.nn.max_pool(self.conv1_2,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='mentor_pool1')
		with tf.name_scope('mentor_conv2_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable,name='mentor_weights')
			conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
								trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                        #mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv2_1 = tf.nn.relu(conv, name=scope)
                        self.conv2_1 = BatchNormalization(axis = -1, name= 'mentor_bn_conv2_1')(self.conv2_1)
                        #self.conv2_1 = Dropout((0.4))(self.conv2_1)
			self.parameters += [kernel, biases]

		with tf.name_scope('mentor_conv2_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable, name='mentor_weights')
			conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
								trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                        #mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv2_2 = tf.nn.relu(conv, name=scope)
                        self.conv2_2 = BatchNormalization(axis = -1, name= 'mentor_bn_conv2_2')(self.conv2_2)
			self.parameters += [kernel, biases]

		self.pool2 = tf.nn.max_pool(self.conv2_2,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='mentor_pool2')
		
                with tf.name_scope('mentor_conv3_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable, name='mentor_weights')
			conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                       # mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv3_1 = tf.nn.relu(conv, name=scope)
                        self.conv3_1 = BatchNormalization(axis = -1, name= 'mentor_bn_conv3_1')(self.conv3_1)
                        #self.conv3_1 = Dropout((0.4))(self.conv3_1)
			self.parameters += [kernel, biases]

                with tf.name_scope('mentor_conv3_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable, name='mentor_weights')
			conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                        #mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv3_2 = tf.nn.relu(conv, name=scope)
                        self.conv3_2 = BatchNormalization(axis = -1, name= 'mentor_bn_conv3_2')(self.conv3_2)
                        #self.conv3_2 = Dropout((0.4))(self.conv3_2)
			self.parameters += [kernel, biases]


		with tf.name_scope('mentor_conv3_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable,name='mentor_weights')
			conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                        #mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv3_3 = tf.nn.relu(conv, name=scope)
                        self.conv3_3 = BatchNormalization(axis = -1, name= 'mentor_bn_conv3_3')(self.conv3_3)
			self.parameters += [kernel, biases]
		self.pool3 = tf.nn.max_pool(self.conv3_3,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='mentor_pool3')
		
                with tf.name_scope('mentor_conv4_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable,name='mentor_weights')
			conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                        #mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv4_1 = tf.nn.relu(conv, name=scope)
                        self.conv4_1 = BatchNormalization(axis = -1, name= 'mentor_bn_conv4_1')(self.conv4_1)
                        #self.conv4_1 = Dropout((0.4))(self.conv4_1)
			self.parameters += [kernel, biases]

		# conv5_1
		with tf.name_scope('mentor_conv4_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable, name='mentor_weights')
			conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                        #mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv4_2 = tf.nn.relu(conv, name=scope)
                        self.conv4_2 = BatchNormalization(axis = -1, name= 'mentor_bn_conv4_2')(self.conv4_2)
                        #self.conv4_2 = Dropout((0.4))(self.conv4_2)
			self.parameters += [kernel, biases]

		with tf.name_scope('mentor_conv4_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable, name='mentor_weights')
			conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                        #mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv4_3 = tf.nn.relu(conv, name=scope)
                        self.conv4_3 = BatchNormalization(axis = -1, name= 'mentor_bn_conv4_3')(self.conv4_3)
			self.parameters += [kernel, biases]
		self.pool4 = tf.nn.max_pool(self.conv4_3,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='mentor_pool4')
		with tf.name_scope('mentor_conv5_1') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable, name='mentor_weights')
			conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                        #mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv5_1 = tf.nn.relu(conv, name=scope)
                        self.conv5_1 = BatchNormalization(axis = -1, name= 'mentor_bn_conv5_1')(self.conv5_1)
                        #self.conv5_1 = Dropout((0.4))(self.conv5_1)
			self.parameters += [kernel, biases]
		with tf.name_scope('mentor_conv5_2') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable, name='mentor_weights')
			conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                        #mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv5_2 = tf.nn.relu(conv, name=scope)
                        self.conv5_2 = BatchNormalization(axis = -1, name= 'mentor_batch_norm_conv5_2')(self.conv5_2)
                        #self.conv5_2 = Dropout((0.4))(self.conv5_2)
			self.parameters += [kernel, biases]

		with tf.name_scope('mentor_conv5_3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
													 stddev=1e-2), trainable = self.trainable, name='mentor_weights')
			conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
			#out = tf.nn.bias_add(conv, biases)
                        #mean , var = tf.nn.moments(out, axes= [0])
                        #out = (out - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.conv5_3 = tf.nn.relu(conv, name=scope)
                        self.conv5_3 = BatchNormalization(axis = -1, name= 'mentor_batch_norm_conv5_3')(self.conv5_3)
			self.parameters += [kernel, biases]
                        
		self.pool5 = tf.nn.max_pool(self.conv5_3,
									ksize=[1, 2, 2, 1],
									strides=[1, 2, 2, 1],
									padding='SAME',
									name='mentor_pool5')
		# fc1
		with tf.name_scope('mentor_fc1') as scope:
			shape = int(np.prod(self.pool5.get_shape()[1:]))
			fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
														 dtype=tf.float32, stddev=1e-2), trainable = self.trainable,name='mentor_weights')
			fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
			pool5_flat = tf.reshape(self.pool5, [-1, shape])
			fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
                        #mean , var = tf.nn.moments(fc1l, axes= [0])
                        #fc1l = (fc1l - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.fc1 = tf.nn.relu(fc1l)
                        self.fc1 = BatchNormalization(axis = -1, name= 'mentor_batch_norm_fc1')(self.fc1)
                       # self.fc1 = Dropout((0.4))(self.fc1)
                        #self.fc1 = tf.nn.dropout(self.fc1, 0.5)
			self.parameters += [fc1w, fc1b]
                
                
		with tf.name_scope('mentor_fc2') as scope:
			fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
					 dtype=tf.float32, stddev=1e-2), trainable = self.trainable,name='mentor_weights')
			fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
			fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
                        #mean , var = tf.nn.moments(fc2l, axes= [0])
                        #fc2l = (fc2l - mean)/tf.sqrt(var + tf.Variable(1e-10))
			self.fc2 = tf.nn.relu(fc2l)
                        self.fc2 = BatchNormalization(axis = -1, name= 'mentor_batch_norm_fc2')(self.fc2)
                        if train_mode == True:
                            self.fc2 = tf.nn.dropout(self.fc2, 0.5)
			self.parameters += [fc2w, fc2b]
            
                
		with tf.name_scope('mentor_fc3') as scope:
			fc3w = tf.Variable(tf.truncated_normal([4096, num_classes],
					dtype=tf.float32, stddev=1e-2), trainable =self.trainable,name='mentor_weights')
			fc3b = tf.Variable(tf.constant(1.0, shape=[num_classes], dtype=tf.float32),
								 trainable=self.trainable, name='mentor_biases')
			self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
			#self.fc3l = tf.nn.relu(fc3l)
			self.parameters += [fc3w, fc3b]
                
            
                return self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2,self.conv3_1, self.conv3_2, self.conv3_3, self.conv4_1, self.conv4_2, self.conv4_3, self.conv5_1, self.conv5_2, self.conv5_3, self.fc3l, tf.nn.softmax(self.fc3l/temp_softmax)

	def loss(self, labels):
		labels = tf.to_int64(labels)
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
			logits=self.fc3l, name='xentropy')
		return tf.reduce_mean(cross_entropy, name='xentropy_mean')


	def training(self, loss, learning_rate, global_step):
		tf.summary.scalar('loss', loss)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		
		train_op = optimizer.minimize(loss, global_step=global_step)
		

		return train_op
