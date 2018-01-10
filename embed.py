import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import math
import numpy as np
import pdb
NUM_CLASSES = 102

VGG_MEAN = [103.939, 116.779, 123.68]
EMBED_UNITS = 64

class Embed(object):

	def __init__(self, trainable=False, dropout=0.5):
		self.trainable = trainable
		self.dropout = dropout
		self.parameters = []

	def build(self, mentor_dict, mentee_dict, train_mode=None):
                with tf.name_scope('mentor_embed_1'):
                        shape = int(np.prod(mentor_dict['conv1'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv1_flat = tf.reshape(mentor_dict['conv1'], [-1, shape])
                        embed_mentor_1 = tf.nn.bias_add(tf.matmul(mentor_conv1_flat, weights), biases)
                
                with tf.name_scope('mentor_embed_2'):

                        shape = int(np.prod(mentor_dict['conv2'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv2_flat = tf.reshape(mentor_dict['conv2'], [-1, shape])
                        embed_mentor_2 = tf.nn.bias_add(tf.matmul(mentor_conv2_flat, weights), biases)

                with tf.name_scope('mentor_embed_3'):
                        shape = int(np.prod(mentor_dict['conv3'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv3_flat = tf.reshape(mentor_dict['conv3'], [-1, shape])
                        embed_mentor_3 = tf.nn.bias_add(tf.matmul(mentor_conv3_flat, weights), biases)

                with tf.name_scope('mentor_embed_4'):
                        shape = int(np.prod(mentor_dict['conv4'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv4_flat = tf.reshape(mentor_dict['conv4'], [-1, shape])
                        embed_mentor_4 = tf.nn.bias_add(tf.matmul(mentor_conv4_flat, weights), biases)

                with tf.name_scope('mentor_embed_5'):
                        shape = int(np.prod(mentor_dict['conv5'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv5_flat = tf.reshape(mentor_dict['conv5'], [-1, shape])
                        embed_mentor_5 = tf.nn.bias_add(tf.matmul(mentor_conv5_flat, weights), biases)

                with tf.name_scope('mentor_embed_6'):
                        shape = int(np.prod(mentor_dict['conv6'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv6_flat = tf.reshape(mentor_dict['conv6'], [-1, shape])
                        embed_mentor_6 = tf.nn.bias_add(tf.matmul(mentor_conv6_flat, weights), biases)

                with tf.name_scope('mentor_embed_7'):
                        shape = int(np.prod(mentor_dict['conv7'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv7_flat = tf.reshape(mentor_dict['conv7'], [-1, shape])
                        embed_mentor_7 = tf.nn.bias_add(tf.matmul(mentor_conv7_flat, weights), biases)

                with tf.name_scope('mentor_embed_8'):
                        shape = int(np.prod(mentor_dict['conv8'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv8_flat = tf.reshape(mentor_dict['conv8'], [-1, shape])
                        embed_mentor_8 = tf.nn.bias_add(tf.matmul(mentor_conv8_flat, weights), biases)

                with tf.name_scope('mentor_embed_9'):
                        shape = int(np.prod(mentor_dict['conv9'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv9_flat = tf.reshape(mentor_dict['conv9'], [-1, shape])
                        embed_mentor_9 = tf.nn.bias_add(tf.matmul(mentor_conv9_flat, weights), biases)

                with tf.name_scope('mentor_embed_10'):
                        shape = int(np.prod(mentor_dict['conv10'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv10_flat = tf.reshape(mentor_dict['conv10'], [-1, shape])
                        embed_mentor_10 = tf.nn.bias_add(tf.matmul(mentor_conv10_flat, weights), biases)
                
                with tf.name_scope('mentor_embed_11'):
                        shape = int(np.prod(mentor_dict['conv11'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv11_flat = tf.reshape(mentor_dict['conv11'], [-1, shape])
                        embed_mentor_11 = tf.nn.bias_add(tf.matmul(mentor_conv11_flat, weights), biases)

                with tf.name_scope('mentor_embed_12'):
                        shape = int(np.prod(mentor_dict['conv12'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv12_flat = tf.reshape(mentor_dict['conv12'], [-1, shape])
                        embed_mentor_12 = tf.nn.bias_add(tf.matmul(mentor_conv12_flat, weights), biases)

                with tf.name_scope('mentor_embed_13'):
                        shape = int(np.prod(mentor_dict['conv13'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv13_flat = tf.reshape(mentor_dict['conv13'], [-1, shape])
                        embed_mentor_13 = tf.nn.bias_add(tf.matmul(mentor_conv13_flat, weights), biases)

                with tf.name_scope('mentor_embed_14'):
                        shape = int(np.prod(mentor_dict['conv14'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv14_flat = tf.reshape(mentor_dict['conv14'], [-1, shape])
                        embed_mentor_14 = tf.nn.bias_add(tf.matmul(mentor_conv14_flat, weights), biases)

                with tf.name_scope('mentor_embed_15'):
                        shape = int(np.prod(mentor_dict['conv15'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                        mentor_conv15_flat = tf.reshape(mentor_dict['conv15'], [-1, shape])
                        embed_mentor_15 = tf.nn.bias_add(tf.matmul(mentor_conv15_flat, weights), biases)

                with tf.name_scope('mentee_embed_1'):
                        shape = int(np.prod(mentee_dict['conv1'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable, name = 'biases_mentee_embed')
                        mentee_conv1_flat = tf.reshape(mentee_dict['conv1'], [-1, shape])
                        embed_mentee_1 = tf.nn.bias_add(tf.matmul(mentee_conv1_flat, weights), biases)

                with tf.name_scope('mentee_embed_2'):
                        shape = int(np.prod(mentee_dict['conv2'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv2_flat = tf.reshape(mentee_dict['conv2'], [-1, shape])
                        embed_mentee_2 = tf.nn.bias_add(tf.matmul(mentee_conv2_flat, weights), biases)
                        
                with tf.name_scope('mentee_embed_3'):
                        shape = int(np.prod(mentee_dict['conv3'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv3_flat = tf.reshape(mentee_dict['conv3'], [-1, shape])
                        embed_mentee_3 = tf.nn.bias_add(tf.matmul(mentee_conv3_flat, weights), biases)

                with tf.name_scope('mentee_embed_4'):
                        shape = int(np.prod(mentee_dict['conv4'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv4_flat = tf.reshape(mentee_dict['conv4'], [-1, shape])
                        embed_mentee_4 = tf.nn.bias_add(tf.matmul(mentee_conv4_flat, weights), biases)

                with tf.name_scope('mentee_embed_5'):
                        shape = int(np.prod(mentee_dict['conv5'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv5_flat = tf.reshape(mentee_dict['conv5'], [-1, shape])
                        embed_mentee_5 = tf.nn.bias_add(tf.matmul(mentee_conv5_flat, weights), biases)

                with tf.name_scope('mentee_embed_6'):
                        shape = int(np.prod(mentee_dict['conv6'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv6_flat = tf.reshape(mentee_dict['conv6'], [-1, shape])
                        embed_mentee_6 = tf.nn.bias_add(tf.matmul(mentee_conv6_flat, weights), biases)

                with tf.name_scope('mentee_embed_7'):
                        shape = int(np.prod(mentee_dict['conv7'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv7_flat = tf.reshape(mentee_dict['conv7'], [-1, shape])
                        embed_mentee_7 = tf.nn.bias_add(tf.matmul(mentee_conv7_flat, weights), biases)

                with tf.name_scope('mentee_embed_8'):
                        shape = int(np.prod(mentee_dict['conv8'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv8_flat = tf.reshape(mentee_dict['conv8'], [-1, shape])
                        embed_mentee_8 = tf.nn.bias_add(tf.matmul(mentee_conv8_flat, weights), biases)

                with tf.name_scope('mentee_embed_9'):
                        shape = int(np.prod(mentee_dict['conv9'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv9_flat = tf.reshape(mentee_dict['conv9'], [-1, shape])
                        embed_mentee_9 = tf.nn.bias_add(tf.matmul(mentee_conv9_flat, weights), biases)

                with tf.name_scope('mentee_embed_10'):
                        shape = int(np.prod(mentee_dict['conv10'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv10_flat = tf.reshape(mentee_dict['conv10'], [-1, shape])
                        embed_mentee_10 = tf.nn.bias_add(tf.matmul(mentee_conv10_flat, weights), biases)

                with tf.name_scope('mentee_embed_11'):
                        shape = int(np.prod(mentee_dict['conv11'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv11_flat = tf.reshape(mentee_dict['conv11'], [-1, shape])
                        embed_mentee_11 = tf.nn.bias_add(tf.matmul(mentee_conv11_flat, weights), biases)

                with tf.name_scope('mentee_embed_12'):
                        shape = int(np.prod(mentee_dict['conv12'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv12_flat = tf.reshape(mentee_dict['conv12'], [-1, shape])
                        embed_mentee_12 = tf.nn.bias_add(tf.matmul(mentee_conv12_flat, weights), biases)

                with tf.name_scope('mentee_embed_13'):
                        shape = int(np.prod(mentee_dict['conv13'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv13_flat = tf.reshape(mentee_dict['conv13'], [-1, shape])
                        embed_mentee_13 = tf.nn.bias_add(tf.matmul(mentee_conv13_flat, weights), biases)

                with tf.name_scope('mentee_embed_14'):
                        shape = int(np.prod(mentee_dict['conv14'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv14_flat = tf.reshape(mentee_dict['conv14'], [-1, shape])
                        embed_mentee_14 = tf.nn.bias_add(tf.matmul(mentee_conv14_flat, weights), biases)

                with tf.name_scope('mentee_embed_15'):
                        shape = int(np.prod(mentee_dict['conv15'].get_shape()[1:]))
                        weights = tf.Variable(tf.random_normal([shape, EMBED_UNITS], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')

                        biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                        mentee_conv15_flat = tf.reshape(mentee_dict['conv15'], [-1, shape])
                        embed_mentee_15 = tf.nn.bias_add(tf.matmul(mentee_conv15_flat, weights), biases)


                self.loss_embed_1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_1, embed_mentee_1))))
                self.loss_embed_2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_2, embed_mentee_2))))
                self.loss_embed_3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_3, embed_mentee_3))))
                self.loss_embed_4 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_4, embed_mentee_4))))
                self.loss_embed_5 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_5, embed_mentee_5))))
                self.loss_embed_6 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_6, embed_mentee_6))))
                self.loss_embed_7 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_7, embed_mentee_7))))
                self.loss_embed_8 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_8, embed_mentee_8))))
                self.loss_embed_9 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_9, embed_mentee_9))))
                self.loss_embed_10 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_10, embed_mentee_10))))
                self.loss_embed_11 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_11, embed_mentee_11))))
                self.loss_embed_12 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_12, embed_mentee_12))))
                self.loss_embed_13 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_13, embed_mentee_13))))
                self.loss_embed_14 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_14, embed_mentee_14))))
                self.loss_embed_15 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(embed_mentor_15, embed_mentee_15))))

                return self
