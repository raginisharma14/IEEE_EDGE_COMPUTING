import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import random
import math
import numpy as np
import pdb
EMBED_UNITS = 64

class Embed(object):

	def __init__(self, trainable=True):
                """ 
                    Args: trainable: The embed layers are made trainable by setting the boolean variable trainable to "True"
                                     The embed layers can be frozen by setting the boolean variable trainable to "False"
                """
		self.trainable = trainable

	def build(self, mentor_dict, mentee_dict, embed_type):

                """ 
                    Agrs: mentor_dict: returned from mentor class and contains output of all the layers
                          mentee_dict: returned from mentee class and contains output of all layers 
                          embed_type : the layers of different width can be either connected with fully connected or convolutional layers
                                       if embed_type = 'fc', embed layers are connected as fully connected layers
                                       if embed_type = 'conv' embed layers are connected as convolution layers
                """


                ### mentor's 3rd layer is connected with embed layer to produce 64 (embed_units) feature maps
                with tf.name_scope('mentor_embed_3'):
                        if(embed_type == 'conv'):
                            print("cov")
                            shape = mentor_dict.conv2_1.get_shape()[3:].as_list()[0]
                            weights = tf.Variable(tf.random_normal([3,3,shape,64], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                            biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                            embed_mentor_3 = tf.nn.conv2d(mentor_dict.conv2_1, weights, [1,1,1,1] , padding = 'SAME')
                            embed_mentor_3 = tf.nn.bias_add(embed_mentor_3, biases)

                        elif(embed_type == 'fc'):
                            shape = int(np.prod(mentor_dict.conv2_1.get_shape()[1:]))
                            mentor_conv3_flat = tf.reshape(mentor_dict.conv2_1, [-1, shape])
                            weights = tf.Variable(tf.truncated_normal([shape, EMBED_UNITS],
                                                                       dtype=tf.float32, stddev=1e-2), trainable = self.trainable,name='mentor_weights')

                            biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                            embed_mentor_3 = tf.nn.bias_add(tf.matmul(mentor_conv3_flat, weights), biases)


                ### mentor's 5th layer is connected with embed layer to produce 64 (embed_units) feature maps
                with tf.name_scope('mentor_embed_4'):
                        if(embed_type == 'conv'):
                            shape = mentor_dict.conv3_1.get_shape()[3:].as_list()[0]
                            weights = tf.Variable(tf.random_normal([3,3,shape,64], stddev = 1e-2), trainable = self.trainable, name = 'weights_mentor_embed')
                            biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                            embed_mentor_4 = tf.nn.conv2d(mentor_dict.conv3_1, weights, [1,1,1,1] , padding = 'SAME')
                            embed_mentor_4 = tf.nn.bias_add(embed_mentor_4, biases)

                        elif (embed_type == 'fc'):
                            shape = int(np.prod(mentor_dict.conv3_1.get_shape()[1:]))
                            mentor_conv4_flat = tf.reshape(mentor_dict.conv3_1, [-1, shape])
                            weights = tf.Variable(tf.truncated_normal([shape, EMBED_UNITS],
                                                                       dtype=tf.float32, stddev=1e-2), trainable = self.trainable,name='mentor_weights')
                            biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                            embed_mentor_4 = tf.nn.bias_add(tf.matmul(mentor_conv4_flat, weights), biases)
            
                ### mentee's 1st layer is connected with the embed layer to produce 64 (embed_units) feature maps
                with tf.name_scope('mentee_embed_3'):
                        if (embed_type == 'conv'):
                            shape = mentee_dict.conv1_1.get_shape()[3:].as_list()[0]
                            weights = tf.Variable(tf.random_normal([3,3,shape,64], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')
                            biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                            embed_mentee_3 = tf.nn.conv2d(mentee_dict.conv1_1, weights, [1,1,1,1] , padding = 'SAME')
                            embed_mentee_3 = tf.nn.bias_add(embed_mentee_3, biases)

                        elif (embed_type == 'fc'):
                            shape = int(np.prod(mentee_dict.conv1_1.get_shape()[1:]))
                            mentee_conv3_flat = tf.reshape(mentee_dict.conv1_1, [-1, shape])
                            weights = tf.Variable(tf.truncated_normal([shape, EMBED_UNITS],
                                                                       dtype=tf.float32, stddev=1e-2), trainable = self.trainable,name='mentor_weights')
                            biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                            embed_mentee_3 = tf.nn.bias_add(tf.matmul(mentee_conv3_flat, weights), biases)

                ### mentee's 2nd layer is connected with the embed layer to produce 64 (embed_units) feature maps
                with tf.name_scope('mentee_embed_4'):
                        if(embed_type == 'conv'):
                            shape = mentee_dict.conv2_1.get_shape()[3:].as_list()[0]
                            weights = tf.Variable(tf.random_normal([3, 3, shape,64], stddev = 1e-2), trainable = self.trainable,name = 'weights_mentee_embed')
                            biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentee_embed')
                            embed_mentee_4 = tf.nn.conv2d(mentee_dict.conv2_1, weights, [1,1,1,1] , padding = 'SAME')
                            embed_mentee_4 = tf.nn.bias_add(embed_mentee_4, biases)

                        elif(embed_type == 'fc'):
                            shape = int(np.prod(mentee_dict.conv2_1.get_shape()[1:]))
                            mentee_conv4_flat = tf.reshape(mentee_dict.conv2_1, [-1, shape])
                            weights = tf.Variable(tf.truncated_normal([shape, EMBED_UNITS],
                                                                       dtype=tf.float32, stddev=1e-2), trainable = self.trainable,name='mentor_weights')
                            biases = tf.Variable(tf.zeros(EMBED_UNITS), trainable = self.trainable,name = 'biases_mentor_embed')
                            embed_mentee_4 = tf.nn.bias_add(tf.matmul(mentee_conv4_flat, weights), biases)

                ### embed_mentor_3 and embed_mentee_3 contain feature maps of different sizes
                ### tf.reduce_mean(embed_mentor_3, axis = [1,2]) makes the size of the feature maps the same (1X1) by taking the mean of all the elements of each feature map
                ### RMSE is calculated on 1X1 feature maps
                if(embed_type == 'conv'):
                    self.loss_embed_3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf.reduce_mean(embed_mentor_3, axis = [1,2]), tf.reduce_mean(embed_mentee_3, axis = [1,2])))))
                    self.loss_embed_4 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf.reduce_mean(embed_mentor_4, axis = [1,2]),  tf.reduce_mean(embed_mentee_4, axis = [1,2])))))
                elif(embed_type == 'fc'):
                    self.loss_embed_3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf.reduce_mean(embed_mentor_3), tf.reduce_mean(embed_mentee_3)))))
                    self.loss_embed_4 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf.reduce_mean(embed_mentor_4),  tf.reduce_mean(embed_mentee_4)))))
                return self
