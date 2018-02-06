import tensorflow as tf
import numpy as np
import random
from DataInput import DataInput
from mobilenetmentee import Mentee
from mobilenetmentor import Mentor
from embed import Embed
import os
import pdb
import sys
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.keras.python.keras.losses import sparse_categorical_crossentropy
import argparse
from tensorflow.contrib.keras.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.keras.python.keras.layers import Reshape
dataset_path = "./"
tf.reset_default_graph()
NUM_ITERATIONS = 100000
SUMMARY_LOG_DIR="./summary-log"
LEARNING_RATE_DECAY_FACTOR = 0.9809
NUM_EPOCHS_PER_DECAY = 1.0
validation_accuracy_list = []
test_accuracy_list = []
seed = 1234
alpha = 0.2
class Mobilenet(object):

    def load_and_preprocess_data(self):
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.y_train = keras.utils.to_categorical(self.y_train, num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, num_classes)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        """
        datagen = ImageDataGenerator(
                featurewise_center = False, 
                samplewise_center = False,
                featurewise_std_normalization = False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip = True,
                vertical_flip = False
                )

        return datagen

    def read_mnist_data(self):
        mnist = read_data_sets(FLAGS.mnist_data_dir)
        return mnist      

    def placeholder_inputs(self, batch_size):
            images_placeholder = tf.placeholder(tf.float32, 
                                                                    shape=(FLAGS.batch_size, FLAGS.image_height, 
                                                                               FLAGS.image_width, FLAGS.num_channels))
            labels_placeholder = tf.placeholder(tf.int32,
                                                               shape=(FLAGS.batch_size))

            return images_placeholder, labels_placeholder


    def fill_feed_dict(self, data_input, images_pl, labels_pl, sess, mode, phase_train):
            images_feed, labels_feed = sess.run([data_input.example_batch, data_input.label_batch])
        
            if mode == 'Train':
                feed_dict = {
                    images_pl: images_feed,
                    labels_pl: labels_feed,
                    phase_train: True
                }

            if mode == 'Test':
                feed_dict = {
                    images_pl: images_feed,
                    labels_pl: labels_feed,
                    phase_train: False
                }

            if mode == 'Validation':
                feed_dict = {
                    images_pl: images_feed,
                    labels_pl: labels_feed,
                    phase_train: False
                }	
            return feed_dict

    def do_eval(self, sess,
                            eval_correct,
                            logits,
                            images_placeholder,
                            labels_placeholder,
                            dataset,mode, phase_train):

            true_count =0
            if mode == 'Test':
                steps_per_epoch = FLAGS.num_testing_examples //FLAGS.batch_size 
                num_examples = steps_per_epoch * FLAGS.batch_size
            if mode == 'Train':
                steps_per_epoch = FLAGS.num_training_examples //FLAGS.batch_size 
                num_examples = steps_per_epoch * FLAGS.batch_size
            if mode == 'Validation':
                steps_per_epoch = FLAGS.num_validation_examples //FLAGS.batch_size 
                num_examples = steps_per_epoch * FLAGS.batch_size
            
            for step in xrange(steps_per_epoch):
                if FLAGS.dataset == 'mnist':
                    feed_dict = {images_placeholder: np.reshape(dataset.test.next_batch(FLAGS.batch_size)[0], [FLAGS.batch_size, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels]), labels_placeholder: dataset.test.next_batch(FLAGS.batch_size)[1]}
                else:
                    feed_dict = self.fill_feed_dict(dataset, images_placeholder,
                                                            labels_placeholder,sess, mode,phase_train)
                count = sess.run(eval_correct, feed_dict=feed_dict)
                true_count = true_count + count

            precision = float(true_count) / num_examples
            print ('  Num examples: %d, Num correct: %d, Precision @ 1: %0.04f' %
                            (num_examples, true_count, precision))
            if mode == 'Validation':
                validation_accuracy_list.append(precision)
            if mode == 'Test':
                test_accuracy_list.append(precision)

    def evaluation(self, logits, labels):
            """
            zero = tf.constant(0, dtype=tf.int32)
            non_zero = tf.not_equal(labels, zero)
            indices = tf.where(non_zero)
            labels = indices[:, 1]
            """
            if FLAGS.top_1_accuracy: 
                correct = tf.nn.in_top_k(logits, labels, 1)
            elif FLAGS.top_3_accuracy:
                correct = tf.nn.in_top_k(logits, labels, 3)
            elif FLAGS.top_5_accuracy:
                correct = tf.nn.in_top_k(logits, labels, 5)
            #pred = tf.argmax(logits, 1)
            return tf.reduce_sum(tf.cast(correct, tf.int32))

    def get_mentor_variables_to_restore(self, variables_to_restore):
           # variables_to_restore.append([var for var in tf.global_variables() if var.op.name.endswith("kernel")])
            return ([var for var in tf.global_variables() if ( var.op.name.endswith("kernel") and var.op.name.startswith("teacher"))])
            #return variables_to_restore

    def get_variables_for_HT(self, variables_for_HT):
        
        variables_for_HT.append([var for var in tf.global_variables() if var.op.name=="student_conv1/kernel"][0])
        
        variables_for_HT.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_15/depthwise_kernel"][0])

        variables_for_HT.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_16/depthwise_kernel"][0])

        return variables_for_HT
    def get_variables_for_KD(self, variables_for_KD):

        variables_for_KD.append([var for var in tf.global_variables() if var.op.name=="student_conv1/kernel"][0])
        
        variables_for_KD.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_15/depthwise_kernel"][0])
        
        variables_for_KD.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_16/depthwise_kernel"][0])

        return variables_for_KD

    def l0_weights_of_mentee(self, l0_mentee_weights):
        l0_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv1/kernel"][0])

        return l0_mentee_weights

    def l1_weights_of_mentee(self, l1_mentee_weights):
        l1_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_15/depthwise_kernel"][0])

        return l1_mentee_weights

    def l2_weights_of_mentee(self, l2_mentee_weights):
        l2_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_16/depthwise_kernel"][0])

        return l2_mentee_weights
    def l3_weights_of_mentee(self, l3_mentee_weights):
        l3_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_17/depthwise_kernel"][0])
        return l3_mentee_weights

    def l4_weights_of_mentee(self, l4_mentee_weights):
        l4_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_18/depthwise_kernel"][0])
        return l4_mentee_weights

    def l5_weights_of_mentee(self, l5_mentee_weights):
        l5_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_19/depthwise_kernel"][0])
        return l5_mentee_weights

    def l6_weights_of_mentee(self, l6_mentee_weights):
        l6_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_20/depthwise_kernel"][0])
        return l6_mentee_weights

    def l7_weights_of_mentee(self, l7_mentee_weights):
        l7_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_21/depthwise_kernel"][0])
        return l7_mentee_weights

    def l8_weights_of_mentee(self, l8_mentee_weights):
        l8_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_22/depthwise_kernel"][0])
        return l8_mentee_weights

    def l9_weights_of_mentee(self, l9_mentee_weights):
        l9_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_23/depthwise_kernel"][0])
        return l9_mentee_weights

    def l10_weights_of_mentee(self, l10_mentee_weights):
        l10_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_24/depthwise_kernel"][0])
        return l10_mentee_weights

    def l11_weights_of_mentee(self, l11_mentee_weights):
        l11_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_25/depthwise_kernel"][0])
        return l11_mentee_weights

    def l12_weights_of_mentee(self, l12_mentee_weights):
        l12_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_26/depthwise_kernel"][0])
        return l12_mentee_weights

    def l13_weights_of_mentee(self, l13_mentee_weights):
        l13_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="student_conv_dw_27/depthwise_kernel"][0])
        return l13_mentee_weights


    def cosine_similarity(self, mentee_conv1_1, mentor_conv1_1, mentor_conv1_2, mentee_conv2_1,mentor_conv2_1, mentor_conv2_2,mentee_conv3_1,mentor_conv3_1, mentor_conv3_2, mentor_conv3_3,mentee_conv4_1, mentor_conv4_1, mentor_conv4_2, mentor_conv4_3, mentee_conv5_1, mentor_conv5_1, mentor_conv5_2, mentor_conv5_3):
        normalize_a_1 = tf.nn.l2_normalize(mentee_conv1_1,0)        
        normalize_b_1 = tf.nn.l2_normalize(mentor_conv1_1,0)
        normalize_a_2 = tf.nn.l2_normalize(mentee_conv1_1,0)        
        normalize_b_2 = tf.nn.l2_normalize(mentor_conv1_2,0)
                        
        normalize_a_3 = tf.nn.l2_normalize(mentee_conv5_1, 0)        
        normalize_b_3 = tf.nn.l2_normalize(mentor_conv5_1,0)
        normalize_a_4 = tf.nn.l2_normalize(mentee_conv5_1, 0)        
        normalize_b_4 = tf.nn.l2_normalize(mentor_conv5_2,0)
        normalize_a_5 = tf.nn.l2_normalize(mentee_conv5_1, 0)        
        normalize_b_5 = tf.nn.l2_normalize(mentor_conv5_3,0)
                   
        normalize_a_6 = tf.nn.l2_normalize(mentee_conv4_1,0)        
        normalize_b_6 = tf.nn.l2_normalize(mentor_conv4_1,0)
        normalize_a_7 = tf.nn.l2_normalize(mentee_conv4_1,0)        
        normalize_b_7 = tf.nn.l2_normalize(mentor_conv4_2,0)
        normalize_a_8 = tf.nn.l2_normalize(mentee_conv4_1,0)        
        normalize_b_8 = tf.nn.l2_normalize(mentor_conv4_3,0)

        normalize_a_9 = tf.nn.l2_normalize(mentee_conv2_1,0)        
        normalize_b_9 = tf.nn.l2_normalize(mentor_conv2_1,0)
        normalize_a_10 = tf.nn.l2_normalize(mentee_conv2_1,0)        
        normalize_b_10= tf.nn.l2_normalize(mentor_conv2_2,0)
                        
        normalize_a_11 = tf.nn.l2_normalize(mentee_conv3_1,0)        
        normalize_b_11= tf.nn.l2_normalize(mentor_conv3_1,0)
        normalize_a_12 = tf.nn.l2_normalize(mentee_conv3_1,0)        
        normalize_b_12 = tf.nn.l2_normalize(mentor_conv3_2,0)        
        normalize_a_13= tf.nn.l2_normalize(mentee_conv3_1,0)
        normalize_b_13= tf.nn.l2_normalize(mentor_conv3_3,0)

        cosine1=tf.reduce_sum(tf.multiply(normalize_a_1,normalize_b_1))
        cosine2=tf.reduce_sum(tf.multiply(normalize_a_2,normalize_b_2))
        cosine3=tf.reduce_sum(tf.multiply(normalize_a_3,normalize_b_3))
        cosine4=tf.reduce_sum(tf.multiply(normalize_a_4,normalize_b_4))
        cosine5=tf.reduce_sum(tf.multiply(normalize_a_5,normalize_b_5))
        cosine6=tf.reduce_sum(tf.multiply(normalize_a_6,normalize_b_6))
        cosine7=tf.reduce_sum(tf.multiply(normalize_a_7,normalize_b_7))
        cosine8=tf.reduce_sum(tf.multiply(normalize_a_8,normalize_b_8))
        cosine9=tf.reduce_sum(tf.multiply(normalize_a_9,normalize_b_9))
        cosine10=tf.reduce_sum(tf.multiply(normalize_a_10,normalize_b_10))
        cosine11=tf.reduce_sum(tf.multiply(normalize_a_11,normalize_b_11))
        cosine12=tf.reduce_sum(tf.multiply(normalize_a_12,normalize_b_12))
        cosine13=tf.reduce_sum(tf.multiply(normalize_a_13,normalize_b_13))


        return cosine1, cosine2, cosine3, cosine4, cosine5, cosine6, cosine7, cosine8, cosine9, cosine10, cosine11, cosine12, cosine13

    def calculate_loss_with_multiple_optimizers(self, train_op0, l0, train_op1, l1, train_op2, l2, train_op3, l3,train_op4, l4,train_op5, l5,train_op6, l6,train_op7, l7,train_op8, l8,train_op9, l9,train_op10, l10,train_op11, l11,train_op12, l12,train_op13, l13, train_op, loss, feed_dict, sess):
        if FLAGS.multiple_optimizers_l0:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            """
            covalue1 = sess.run(cosine1, feed_dict=feed_dict)
            covalue2 = sess.run(cosine2, feed_dict=feed_dict)
            covalue3 = sess.run(cosine3, feed_dict=feed_dict)
            covalue4 = sess.run(cosine4, feed_dict=feed_dict)
            covalue5 = sess.run(cosine5, feed_dict=feed_dict)
            covalue6 = sess.run(cosine6, feed_dict=feed_dict)
            covalue7 = sess.run(cosine7, feed_dict=feed_dict)
            covalue8 = sess.run(cosine8, feed_dict=feed_dict)
            covalue9 = sess.run(cosine9, feed_dict=feed_dict)
            covalue10 = sess.run(cosine10, feed_dict=feed_dict)
            covalue11 = sess.run(cosine11, feed_dict=feed_dict)
            covalue12= sess.run(cosine12, feed_dict=feed_dict)
            covalue13 = sess.run(cosine13, feed_dict=feed_dict)
            """
        elif FLAGS.multiple_optimizers_l1:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)

        elif FLAGS.multiple_optimizers_l2:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
        
        elif FLAGS.multiple_optimizers_l3:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)

        elif FLAGS.multiple_optimizers_l4:                                        
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)

        elif FLAGS.multiple_optimizers_l5:
                                     
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([train_op4, l4], feed_dict=feed_dict)

        elif FLAGS.multiple_optimizers_l6:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([train_op4, l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([train_op5, l5], feed_dict=feed_dict)

        elif FLAGS.multiple_optimizers_l7:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([train_op4, l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([train_op5, l5], feed_dict=feed_dict)
            _, self.loss_value6 = sess.run([train_op6, l6], feed_dict=feed_dict)

        elif FLAGS.multiple_optimizers_l8: 
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([train_op4, l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([train_op5, l5], feed_dict=feed_dict)
            _, self.loss_value6 = sess.run([train_op6, l6], feed_dict=feed_dict)
            _, self.loss_value7 = sess.run([train_op7, l7], feed_dict=feed_dict)
        
        elif FLAGS.multiple_optimizers_l9:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([train_op4, l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([train_op5, l5], feed_dict=feed_dict)
            _, self.loss_value6 = sess.run([train_op6, l6], feed_dict=feed_dict)
            _, self.loss_value7 = sess.run([train_op7, l7], feed_dict=feed_dict)
            _, self.loss_value8 = sess.run([train_op8, l8], feed_dict=feed_dict)

        elif FLAGS.multiple_optimizers_l10:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([train_op4, l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([train_op5, l5], feed_dict=feed_dict)
            _, self.loss_value6 = sess.run([train_op6, l6], feed_dict=feed_dict)
            _, self.loss_value7 = sess.run([train_op7, l7], feed_dict=feed_dict)
            _, self.loss_value8 = sess.run([train_op8, l8], feed_dict=feed_dict)
            _, self.loss_value9 = sess.run([train_op9, l9], feed_dict=feed_dict)

        elif FLAGS.multiple_optimizers_l11:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([train_op4, l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([train_op5, l5], feed_dict=feed_dict)
            _, self.loss_value6 = sess.run([train_op6, l6], feed_dict=feed_dict)
            _, self.loss_value7 = sess.run([train_op7, l7], feed_dict=feed_dict)
            _, self.loss_value8 = sess.run([train_op8, l8], feed_dict=feed_dict)
            _, self.loss_value9 = sess.run([train_op9, l9], feed_dict=feed_dict)
            _, self.loss_value10 = sess.run([train_op10, l10], feed_dict=feed_dict)

        elif FLAGS.multiple_optimizers_l12:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([train_op4, l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([train_op5, l5], feed_dict=feed_dict)
            _, self.loss_value6 = sess.run([train_op6, l6], feed_dict=feed_dict)
            _, self.loss_value7 = sess.run([train_op7, l7], feed_dict=feed_dict)
            _, self.loss_value8 = sess.run([train_op8, l8], feed_dict=feed_dict)
            _, self.loss_value9 = sess.run([train_op9, l9], feed_dict=feed_dict)
            _, self.loss_value10 = sess.run([train_op10, l10], feed_dict=feed_dict)
            _, self.loss_value11 = sess.run([train_op11, l11], feed_dict=feed_dict)
        
        elif FLAGS.multiple_optimizers_l13:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([train_op4, l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([train_op5, l5], feed_dict=feed_dict)
            _, self.loss_value6 = sess.run([train_op6, l6], feed_dict=feed_dict)
            _, self.loss_value7 = sess.run([train_op7, l7], feed_dict=feed_dict)
            _, self.loss_value8 = sess.run([train_op8, l8], feed_dict=feed_dict)
            _, self.loss_value9 = sess.run([train_op9, l9], feed_dict=feed_dict)
            _, self.loss_value10 = sess.run([train_op10, l10], feed_dict=feed_dict)
            _, self.loss_value11 = sess.run([train_op11, l11], feed_dict=feed_dict)
            _, self.loss_value12 = sess.run([train_op12, l12], feed_dict=feed_dict)

        elif FLAGS.multiple_optimizers_l14:
            _, self.loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            _, self.loss_value0 = sess.run([train_op0, l0], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([train_op4, l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([train_op5, l5], feed_dict=feed_dict)
            _, self.loss_value6 = sess.run([train_op6, l6], feed_dict=feed_dict)
            _, self.loss_value7 = sess.run([train_op7, l7], feed_dict=feed_dict)
            _, self.loss_value8 = sess.run([train_op8, l8], feed_dict=feed_dict)
            _, self.loss_value9 = sess.run([train_op9, l9], feed_dict=feed_dict)
            _, self.loss_value10 = sess.run([train_op10, l10], feed_dict=feed_dict)
            _, self.loss_value11 = sess.run([train_op11, l11], feed_dict=feed_dict)
            _, self.loss_value12 = sess.run([train_op12, l12], feed_dict=feed_dict)
            _, self.loss_value13 = sess.run([train_op13, l13], feed_dict=feed_dict)
                                        

    def calculate_loss_with_single_optimizer(self, train_op, loss, feed_dict, sess):

        _, self.loss_value = sess.run([train_op, loss] , feed_dict=feed_dict)
        

    def train_op_for_multiple_optimizers(self, lr, loss, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13):

        l0_var_list = []
        l1_var_list = []
        l2_var_list =[]
        l3_var_list = []
        l4_var_list = []
        l5_var_list = []
        l6_var_list = []
        l7_var_list = []
        l8_var_list =[]
        l9_var_list = []
        l10_var_list = []
        l11_var_list = []
        l12_var_list = []
        l13_var_list = []

        self.train_op0 = tf.train.AdamOptimizer(lr).minimize(l0, var_list = self.l0_weights_of_mentee(l0_var_list))
        self.train_op1 = tf.train.AdamOptimizer(lr).minimize(l1, var_list = self.l1_weights_of_mentee(l1_var_list))
        self.train_op2 = tf.train.AdamOptimizer(lr).minimize(l2, var_list = self.l2_weights_of_mentee(l2_var_list))
        
        self.train_op3 = tf.train.AdamOptimizer(lr).minimize(l3, var_list = self.l3_weights_of_mentee(l3_var_list))
        self.train_op4 = tf.train.AdamOptimizer(lr).minimize(l4, var_list = self.l4_weights_of_mentee(l4_var_list))
        self.train_op5 = tf.train.AdamOptimizer(lr).minimize(l5, var_list = self.l5_weights_of_mentee(l5_var_list))
        self.train_op6 = tf.train.AdamOptimizer(lr).minimize(l6, var_list = self.l6_weights_of_mentee(l6_var_list))
        self.train_op7 = tf.train.AdamOptimizer(lr).minimize(l7, var_list = self.l7_weights_of_mentee(l7_var_list))
        self.train_op8 = tf.train.AdamOptimizer(lr).minimize(l8, var_list = self.l8_weights_of_mentee(l8_var_list))
        
        self.train_op9 = tf.train.AdamOptimizer(lr).minimize(l9, var_list = self.l9_weights_of_mentee(l9_var_list))
        self.train_op10 = tf.train.AdamOptimizer(lr).minimize(l10, var_list = self.l10_weights_of_mentee(l10_var_list))
        self.train_op11 = tf.train.AdamOptimizer(lr).minimize(l11, var_list = self.l11_weights_of_mentee(l11_var_list))
        self.train_op12 = tf.train.AdamOptimizer(lr).minimize(l12, var_list = self.l12_weights_of_mentee(l12_var_list))
        self.train_op13 = tf.train.AdamOptimizer(lr).minimize(l13, var_list = self.l13_weights_of_mentee(l13_var_list))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)


    def train_op_for_single_optimizer(self, lr, loss, l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, data_dict_mentor, data_dict_mentee):
        if FLAGS.single_optimizer_l1:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0)


        if FLAGS.single_optimizer_l1:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1)

        if FLAGS.single_optimizer_l2:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2)

        if FLAGS.single_optimizer_l3:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2 + l3)

        if FLAGS.single_optimizer_l4:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2 + l3 + l4)

        if FLAGS.single_optimizer_l5:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2 + l3 + l4 + l5)
        if FLAGS.single_optimizer_l6:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2 + l3 + l4 + l5 + l6)
        if FLAGS.single_optimizer_l7:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7)

        if FLAGS.single_optimizer_l8:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8)

        if FLAGS.single_optimizer_l9:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9)

        if FLAGS.single_optimizer_l10:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10
                    )
        if FLAGS.single_optimizer_l11:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10 + l11)

        if FLAGS.single_optimizer_l12:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10 + l11 + l12)

        if FLAGS.single_optimizer_l13:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss + l0 + l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10 + l11 + l12 + l13)

        if FLAGS.hard_logits:
            logits = Reshape((FLAGS.num_classes,))(data_dict_mentor.conv20)
            ind_max = tf.argmax(logits, axis = 1)
            hard_logits = tf.one_hot(ind_max, FLAGS.num_classes)
            hard_loss = tf.reduce_sum(tf.square(tf.subtract(hard_logits, data_dict_mentee.conv22)))
            self.train_op = tf.train.AdamOptimizer(lr).minimize(hard_loss)
        if FLAGS.soft_logits:
            soft_loss = tf.reduce_sum(tf.square(tf.subtract(data_dict_mentor.conv22, data_dict_mentee.conv22))) 
            self.train_op = tf.train.AdamOptimizer(lr).minimize(alpha*loss + soft_loss)

        if FLAGS.fitnets_HT:
            variables_for_HT = []
            self.train_op = tf.train.AdamOptimizer(lr).minimize(l9, var_list = self.get_variables_for_HT(variables_for_HT))

    def main(self, _):

            with tf.Graph().as_default():

                    config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
                    datagen = self.load_and_preprocess_data()
                    if FLAGS.dataset == 'mnist':
                        mnist = read_mnist_data()
                    tf.set_random_seed(seed)

                    data_input_train = DataInput(dataset_path, FLAGS.train_dataset, FLAGS.batch_size, FLAGS.num_training_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)

                    data_input_test = DataInput(dataset_path, FLAGS.test_dataset,FLAGS.batch_size, FLAGS.num_testing_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)
                    data_input_validation = DataInput(dataset_path, FLAGS.validation_dataset,FLAGS.batch_size, FLAGS.num_validation_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)
                    images_placeholder, labels_placeholder = self.placeholder_inputs(FLAGS.batch_size)
                    sess = tf.Session(config = config)
                    
                    summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    phase_train = tf.placeholder(tf.bool, name = 'phase_train')
                    summary = tf.summary.merge_all()

                    if FLAGS.teacher:
                        if FLAGS.dataset == 'cifar10' or 'mnist':
                            print("Teacher")
                            mentor = Mentor(True, FLAGS.num_classes)
                        if FLAGS.dataset == 'caltech101':
                            mentor = Mentor(True, FLAGS.num_classes)
                        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
                        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                        data_dict_mentor = mentor.build(FLAGS.teacher_alpha, images_placeholder, FLAGS.temp_softmax)
                        loss = mentor.loss(labels_placeholder)
                        lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
                        if FLAGS.dataset == 'caltech101':
                            self.train_op = mentor.training(loss, lr, global_step)
                        if FLAGS.dataset == 'cifar10':
                            self.train_op = mentor.training(loss, lr, global_step)
                        softmax = data_dict_mentor.conv22
                        init = tf.global_variables_initializer()
                        sess.run(init)
                        saver = tf.train.Saver()

                    elif FLAGS.student:
                        student = Mentee(FLAGS.num_classes)
                        print("Independent student")
                        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
                        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                        data_dict = student.build(FLAGS.student_alpha, images_placeholder, FLAGS.temp_softmax)
                        #data_dict = student.build(images_placeholder,FLAGS.student_alpha, phase_train, FLAGS.num_classes, FLAGS.num_channels, seed, True)
                        loss = student.loss(labels_placeholder)
                        lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
                        self.train_op = student.training(loss,lr, global_step)
                        softmax = data_dict.conv22
                        init = tf.initialize_all_variables()
                        sess.run(init)
                        saver = tf.train.Saver()

                        
                    elif FLAGS.dependent_student:
                        vgg16_mentor = Mentor(False, FLAGS.num_classes)
                        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
                        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                        vgg16_mentee = Mentee(FLAGS.num_classes)
                        data_dict_mentor = vgg16_mentor.build(FLAGS.teacher_alpha, images_placeholder, FLAGS.temp_softmax)

                        data_dict_mentee = vgg16_mentee.build(FLAGS.student_alpha, images_placeholder, FLAGS.temp_softmax)

                        softmax = data_dict_mentee.conv22
                        mentor_variables_to_restore = []
                        mentor_variables_to_restore = self.get_mentor_variables_to_restore(mentor_variables_to_restore)
                        loss = vgg16_mentee.loss(labels_placeholder)
                            
                        embed  = Embed()
                        embed  =  embed.build(data_dict_mentor, data_dict_mentee)
                        lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
                        if FLAGS.single_optimizer:
                            self.train_op_for_single_optimizer(lr, loss, embed.loss_embed_3, embed.loss_embed_4, embed.loss_embed_5, embed.loss_embed_6, embed.loss_embed_7, embed.loss_embed_8, embed.loss_embed_9, embed.loss_embed_10, embed.loss_embed_11, embed.loss_embed_12, embed.loss_embed_13, embed.loss_embed_14, embed.loss_embed_15, embed.loss_embed_16, data_dict_mentor, data_dict_mentee)
                        if FLAGS.multiple_optimizers:
                            self.train_op_for_multiple_optimizers(lr, loss, embed.loss_embed_3, embed.loss_embed_4, embed.loss_embed_5, embed.loss_embed_6, embed.loss_embed_7, embed.loss_embed_8, embed.loss_embed_9, embed.loss_embed_10, embed.loss_embed_11, embed.loss_embed_12, embed.loss_embed_13, embed.loss_embed_14, embed.loss_embed_15, embed.loss_embed_16)

                        init = tf.initialize_all_variables()
                        sess.run(init)
                        if FLAGS.fitnets_KD:
                            variables_for_KD = []
                            saver = tf.train.Saver(self.get_variables_for_KD(variables_for_KD))
                            saver.restore(sess, "./summary-log/new_method_dependent_student_weights_filename_mobilenet_caltech101")
                        saver = tf.train.Saver(mentor_variables_to_restore)
                        saver.restore(sess, "./summary-log/new_method_teacher_weights_filename_mobilenet_caltech101")
                    eval_correct= self.evaluation(softmax, labels_placeholder)
                    count = 0
                    try:
                            for i in range(NUM_ITERATIONS):
                                count = count + 1
                                feed_dict = self.fill_feed_dict(data_input_train, images_placeholder, labels_placeholder, sess, 'Train', phase_train)

                                if FLAGS.student or FLAGS.teacher:
                                    _, loss_value = sess.run([self.train_op, loss], feed_dict=feed_dict)
                                    if FLAGS.dataset == 'mnist':
                                        batch = mnist.train.next_batch(FLAGS.batch_size)
                                        _, loss_value = sess.run([self.train_op, loss], feed_dict = {images_placeholder: np.reshape(batch[0], [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, FLAGS.num_channels]), labels_placeholder: batch[1]})

                                    if i % 10 == 0:
                                        print ('Step %d: loss_value = %.20f' % (i, loss_value))

                                if FLAGS.dependent_student and FLAGS.single_optimizer:
                                                    
                                    self.calculate_loss_with_single_optimizer(self.train_op, loss, feed_dict, sess)
                                    if i % 10 == 0:
                                        print ('Step %d: loss_value = %.20f' % (i, self.loss_value))
                                if FLAGS.dependent_student and FLAGS.multiple_optimizers:

                                    self.calculate_loss_with_multiple_optimizers(self.train_op0,embed.loss_embed_3, self.train_op1, embed.loss_embed_4, self.train_op2, embed.loss_embed_5, self.train_op3, embed.loss_embed_6, self.train_op4, embed.loss_embed_7, self.train_op5, embed.loss_embed_8, self.train_op6, embed.loss_embed_9, self.train_op7, embed.loss_embed_10, self.train_op8, embed.loss_embed_11, self.train_op9, embed.loss_embed_12, self.train_op10, embed.loss_embed_13, self.train_op11, embed.loss_embed_14, self.train_op12, embed.loss_embed_15, self.train_op13, embed.loss_embed_16, self.train_op,loss, feed_dict, sess)
                                                    
                                    if i % 10 == 0:
                                        print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value))
                                        """
                                        print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                                        print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                                        print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                                                        
                                        print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                                        print ('Step %d: loss_value4 = %.20f' % (i, self.loss_value4))
                                        print ('Step %d: loss_value5 = %.20f' % (i, self.loss_value5))
                                        print ('Step %d: loss_value6 = %.20f' % (i, self.loss_value6))
                                        """
                                        #summary_str = sess.run(summary, feed_dict=feed_dict)
                                        #summary_writer.add_summary(summary_str, i)
                                        #summary_writer.flush()
                                if (i) %(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN//FLAGS.batch_size)  == 0 or (i) == NUM_ITERATIONS:
                                                        
                                    checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')
                                    if FLAGS.teacher:
                                        saver.save(sess, FLAGS.teacher_weights_filename)

                                    elif FLAGS.student:
                                        saver.save(sess, FLAGS.student_filename)
                                    """
                                    elif FLAGS.dependent_student:
                                        saver = tf.train.Saver()
                                        saver.save(sess, FLAGS.dependent_student_filename)

                                    """
                                    """
                                    print sess.run(cosine1, feed_dict = feed_dict)
                                    print sess.run(cosine2, feed_dict = feed_dict)
                                    print sess.run(cosine3, feed_dict = feed_dict)
                                    print sess.run(cosine4, feed_dict = feed_dict)
                                    print sess.run(cosine5, feed_dict = feed_dict)
                                    print sess.run(cosine6, feed_dict = feed_dict)
                                    print sess.run(cosine7, feed_dict = feed_dict)
                                    print sess.run(cosine8, feed_dict = feed_dict)
                                    print sess.run(cosine9, feed_dict = feed_dict)
                                    print sess.run(cosine10, feed_dict = feed_dict)
                                    print sess.run(cosine11, feed_dict = feed_dict)
                                    print sess.run(cosine12, feed_dict = feed_dict)
                                    print sess.run(cosine13, feed_dict = feed_dict)
                                    """
                                    if FLAGS.dataset == 'mnist':
                                        print('validation accuracy::MNIST')
                                        self.do_eval(sess, 
                                        eval_correct,
                                        softmax,
                                        images_placeholder,
                                        labels_placeholder,
                                        mnist, 
                                        'Validation', phase_train)

                                        print('test accuracy::MNIST')
                                        self.do_eval(sess, 
                                        eval_correct,
                                        softmax,
                                        images_placeholder,
                                        labels_placeholder,
                                        mnist, 
                                        'Test', phase_train)
                                                            
                                    else:
                                        print ("Training Data Eval:")
                                                            
                                        self.do_eval(sess, 
                                        eval_correct,
                                        softmax,
                                        images_placeholder,
                                        labels_placeholder,
                                        data_input_train, 
                                        'Train', phase_train)
                                                            
                                                            
                                        print ("Test  Data Eval:")
                                        self.do_eval(sess, 
                                        eval_correct,
                                        softmax,
                                        images_placeholder,
                                        labels_placeholder,
                                        data_input_test, 
                                        'Test', phase_train)
                                    
                            coord.request_stop()
                            coord.join(threads)
                    except Exception as e:
                            print(e)
                    if count == NUM_ITERATIONS:
                        index = np.argmax(validation_accuracy_list)
                        print("Model accuracy::", test_accuracy_list[index])
            sess.close()

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--teacher',
            type = bool,
            help = 'train teacher',
            default = False
        )
        parser.add_argument(
            '--dependent_student',
            type = bool,
            help = 'train dependent student',
            default = False
        )
        parser.add_argument(
            '--student',
            type = bool,
            help = 'train independent student',
            default = False
        )
        parser.add_argument(
            '--teacher_weights_filename',
            type = str,
            default = "./summary-log/new_method_teacher_weights_filename_mobilenet_caltech101"
        )
        parser.add_argument(
            '--student_filename',
            type = str,
            default = "./summary-log/new_method_student_weights_filename_mobilenet_caltech101"
        )
        parser.add_argument(
            '--dependent_student_filename',
            type = str,
            default = "./summary-log/new_method_dependent_student_weights_filename_mobilenet_caltech101"
        )

        parser.add_argument(
            '--learning_rate',
            type = float,
            default = 0.0001
        )

        parser.add_argument(
            '--batch_size',
            type = int,
            default = 45                                   
        )
        parser.add_argument(
            '--image_height',
            type = int,
            default = 224                                   
        )
        parser.add_argument(
            '--image_width',
            type = int,
            default = 224                                  
        )
        parser.add_argument(
            '--train_dataset',
            type = str,
            default = "caltech101-train.txt"                                   
        )
        parser.add_argument(
            '--test_dataset',
            type = str,
            default = "caltech101-test.txt"                                   
        )
        parser.add_argument(
            '--validation_dataset',
            type = str,
            default = "caltech101-validation.txt"                                   
        )
        parser.add_argument(
            '--temp_softmax',
            type = int,
            default = 5                                   
        )
        parser.add_argument(
            '--num_classes',
            type = int,
            default = 102                                   
        )
        parser.add_argument(
            '--learning_rate_pretrained',
            type = float,
            default = 0.0001                                   
        )
        parser.add_argument(
            '--NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN',
            type = int,
            default = 8186                                   
        )
        parser.add_argument(
            '--num_training_examples',
            type = int,
            default = 8186                                    
        )
        parser.add_argument(
            '--num_testing_examples',
            type = int,
            default = 958                                    
        )
        parser.add_argument(
            '--num_validation_examples',
            type = int,
            default = 1463                                    
        )
        parser.add_argument(
            '--single_optimizer',
            type = bool,
            help = 'single_optimizer',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers',
            type = bool,
            help = 'multiple_optimizers',
            default = False
        )
        parser.add_argument(
            '--dataset',
            type = str,
            help = 'name of the dataset',
            default = 'cifar10'
        )
        parser.add_argument(
            '--mnist_data_dir',
            type = str,
            help = 'name of the dataset',
            default = './mnist_data'
        )
        parser.add_argument(
            '--teacher_alpha',
            type = float,
            help = 'width_multiplier',
            default =1.0
        )
        parser.add_argument(
            '--student_alpha',
            type = float,
            help = 'width_multiplier',
            default =0.05
        )
        parser.add_argument(
            '--num_channels',
            type = int,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = '3'
        )
        parser.add_argument(
            '--single_optimizer_l1',
            type = bool,
            help = 'optimizer for l1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l2',
            type = bool,
            help = 'optimizer for l2',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l3',
            type = bool,
            help = 'optimizer for l3',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l4',
            type = bool,
            help = 'optimizer for l4',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l5',
            type = bool,
            help = 'optimizer for l5',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l6',
            type = bool,
            help = 'optimizer for l6',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l7',
            type = bool,
            help = 'optimizer for l7',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l8',
            type = bool,
            help = 'optimizer for l8',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l9',
            type = bool,
            help = 'optimizer for l9',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l10',
            type = bool,
            help = 'optimizer for l10',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l11',
            type = bool,
            help = 'optimizer for l11',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l12',
            type = bool,
            help = 'optimizer for l12',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l13',
            type = bool,
            help = 'optimizer for l13',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l0',
            type = bool,
            help = 'optimizer for l0',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l1',
            type = bool,
            help = 'optimizer for l1',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l2',
            type = bool,
            help = 'optimizer for l2',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l3',
            type = bool,
            help = 'optimizer for l3',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l4',
            type = bool,
            help = 'optimizer for l4',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l5',
            type = bool,
            help = 'optimizer for l5',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l6',
            type = bool,
            help = 'optimizer for l6',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l7',
            type = bool,
            help = 'optimizer for l7',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l8',
            type = bool,
            help = 'optimizer for l8',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l9',
            type = bool,
            help = 'optimizer for l9',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l10',
            type = bool,
            help = 'optimizer for l10',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l11',
            type = bool,
            help = 'optimizer for l11',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l12',
            type = bool,
            help = 'optimizer for l12',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l13',
            type = bool,
            help = 'optimizer for l13',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l14',
            type = bool,
            help = 'optimizer for l14',
            default = False
        )
        parser.add_argument(
            '--top_3_accuracy',
            type = bool,
            help = 'top-3-accuracy',
            default = False
        )
        parser.add_argument(
            '--top_5_accuracy',
            type = bool,
            help = 'top-5-accuracy',
            default = False
        )
        parser.add_argument(
            '--top_1_accuracy',
            type = bool,
            help = 'top-1-accuracy',
            default = False
        )
        parser.add_argument(
            '--hard_logits',
            type = bool,
            help = 'hard_logits',
            default = False
        )
        parser.add_argument(
            '--soft_logits',
            type = bool,
            help = 'soft_logits',
            default = False
        )
        parser.add_argument(
            '--fitnets_KD',
            type = bool,
            help = 'fitnets_KD',
            default = False
        )
        parser.add_argument(
            '--fitnets_HT',
            type = bool,
            help = 'fitnets_HT',
            default = False
        )



        FLAGS, unparsed = parser.parse_known_args()
        ex = Mobilenet()
        tf.app.run(main=ex.main, argv = [sys.argv[0]] + unparsed)


