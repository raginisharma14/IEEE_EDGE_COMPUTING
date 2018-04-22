import tensorflow as tf
import numpy as np
import random
from DataInput import DataInput
from vgg16mentee import Mentee
from vgg16mentor import Mentor
from vgg16embed import Embed
from mentor import Teacher
import os
import time
import pdb
import sys
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from PIL import Image
import argparse
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
class VGG16(object):
    ### read_mnist_data is used to read mnist data. This function is not currently used as the code supports only caltech101 and CIFAR-10 for now.
    def read_mnist_data(self):
        mnist = read_data_sets(FLAGS.mnist_data_dir)
        return mnist      

    ### placeholders to hold iamges and their labels of certain datasets 
    def placeholder_inputs(self, batch_size):
            """
                Args:
                    batch_size: batch size used to train the network
                
                Returns:
                    images_placeholder: images_placeholder holds images of either caltech101 or cifar10 datasets
                    labels_placeholder: labels_placeholder holds labels of either caltech101 or cifar10 datasets

            """
            images_placeholder = tf.placeholder(tf.float32, 
                                                                    shape=(FLAGS.batch_size, FLAGS.image_height, 
                                                                               FLAGS.image_width, FLAGS.num_channels))
            labels_placeholder = tf.placeholder(tf.int32,
                                                                    shape=(FLAGS.batch_size))

            return images_placeholder, labels_placeholder

    
    ### placeholders are filled with actual images and labels which are fed to the network while training.
    def fill_feed_dict(self, data_input, images_pl, labels_pl, sess, mode, phase_train):
            """
            Based on the mode whether it is train, test or validation; we fill the feed_dict with appropriate images and labels.
            Args:
                data_input: object instantiated for DataInput class
                images_pl: placeholder to hold images of the datasets
                labels_pl: placeholder to hold labels of the datasets
                mode: mode is either train or test or validation


            Returns: 
                feed_dict: dictionary consists of images placeholder, labels placeholder and phase_train as keys
                           and images, labels and a boolean value phase_train as values.

            """

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

    ## In this function, accuracy is calculated for the training set, test set and validation set
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
            if FLAGS.top_1_accuracy: 
                correct = tf.nn.in_top_k(logits, labels, 1)
            elif FLAGS.top_3_accuracy:
                correct = tf.nn.in_top_k(logits, labels, 3)
            elif FLAGS.top_5_accuracy:
                correct = tf.nn.in_top_k(logits, labels, 5)

            return tf.reduce_sum(tf.cast(correct, tf.int32))



    ### while training dependent student, weights of the teacher network trained prior to the dependent student are loaded on to the teacher network to carry out inference
    
    def get_mentor_variables_to_restore(self):
            """
            Returns:: names of the weights and biases of the teacher model
            """
            return [var for var in tf.global_variables() if var.op.name.startswith("mentor") and (var.op.name.endswith("biases") or var.op.name.endswith("weights")) and (var.op.name != ("mentor_fc3/mentor_weights") and  var.op.name != ("mentor_fc3/mentor_biases"))]


    ### returns 1st layer weight variable of mentee network
    def l1_weights_of_mentee(self, l1_mentee_weights):
        l1_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv1_1/mentee_weights"][0])
    #    l1_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv1_1/mentee_biases"][0])

        return l1_mentee_weights


    ### returns 2nd layer weight variable of mentee network
    def l2_weights_of_mentee(self, l2_mentee_weights):
        l2_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_weights"][0])
     #   l2_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_biases"][0])
        return l2_mentee_weights

    ### returns 3rd layer weight variable of mentee network
    def l3_weights_of_mentee(self, l3_mentee_weights):
        l3_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_weights"][0])
      #  l3_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_biases"][0])
        return l3_mentee_weights

    
    ### returns 4th layer weight variable of mentee network
    def l4_weights_of_mentee(self, l4_mentee_weights):
        l4_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv4_1/mentee_weights"][0])
       # l4_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv4_1/mentee_biases"][0])
        return l4_mentee_weights

    ### returns 5th layer weight variable of mentee network
    def l5_weights_of_mentee(self, l5_mentee_weights):
        l5_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv5_1/mentee_weights"][0])
        #l5_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv5_1/mentee_biases"][0])
        return l5_mentee_weights

    ### returns 6th layer (fully connected) weight variable of mentee network
    def l6_weights_of_mentee(self, l6_mentee_weights):
        l6_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_fc3/mentee_weights"][0])
        #l6_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_fc3/mentee_biases"][0])
        return l6_mentee_weights

    def get_variables_for_HT(self, variables_for_HT):
        
        variables_for_HT.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_weights"][0])
        variables_for_HT.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_weights"][0])

        variables_for_HT.append([var for var in tf.global_variables() if var.op.name=="mentee_conv1_1/mentee_weights"][0])

        return variables_for_HT
    
    def get_variables_for_KD(self, variables_for_KD):
        #return [var for var in tf.global_variables() if (var.op.name.startswith("mentee") and var.op.name.endswith("weights"))]
        variables_for_KD.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_weights"][0])
        variables_for_KD.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_weights"][0])

        variables_for_KD.append([var for var in tf.global_variables() if var.op.name=="mentee_conv1_1/mentee_weights"][0])
        return variables_for_KD

    def cosine_similarity_of_same_width(self, mentee_data_dict, mentor_data_dict):

        """
            cosine similarity is calculated between 1st layer of mentee and 1st layer of mentor.
            Similarly, cosine similarity is calculated between 1st layer of mentee and 2nd layer of mentor.
        """
        normalize_a_1 = tf.nn.l2_normalize(mentee_data_dict.conv1_1,0)        
        normalize_b_1 = tf.nn.l2_normalize(mentor_data_dict.conv1_1,0)
        normalize_a_2 = tf.nn.l2_normalize(mentee_data_dict.conv1_1,0)        
        normalize_b_2 = tf.nn.l2_normalize(mentor_data_dict.conv1_2,0)

        """
            cosine similarity is calculated between 5th layer of mentee and 11th layer of mentor.
            Similarly, cosine similarity is calculated between 5th layer of mentee and 12th layer of mentor.
            Similarly, cosine similarity is calculated between 5th layer of mentee and 13th layer of mentor.

        """
                        
        normalize_a_3 = tf.nn.l2_normalize(mentee_data_dict.conv5_1, 0)        
        normalize_b_3 = tf.nn.l2_normalize(mentor_data_dict.conv5_1,0)
        normalize_a_4 = tf.nn.l2_normalize(mentee_data_dict.conv5_1, 0)        
        normalize_b_4 = tf.nn.l2_normalize(mentor_data_dict.conv5_2,0)
        normalize_a_5 = tf.nn.l2_normalize(mentee_data_dict.conv5_1, 0)        
        normalize_b_5 = tf.nn.l2_normalize(mentor_data_dict.conv5_3,0)

        """
            cosine similarity is calculated between 4th layer of mentee and 8th layer of mentor.
            Similarly, cosine similarity is calculated between 4th layer of mentee and 9th layer of mentor.
            Similarly, cosine similarity is calculated between 4th layer of mentee and 10th layer of mentor.

        """
                   
        normalize_a_6 = tf.nn.l2_normalize(mentee_data_dict.conv4_1,0)        
        normalize_b_6 = tf.nn.l2_normalize(mentor_data_dict.conv4_1,0)
        normalize_a_7 = tf.nn.l2_normalize(mentee_data_dict.conv4_1,0)        
        normalize_b_7 = tf.nn.l2_normalize(mentor_data_dict.conv4_2,0)
        normalize_a_8 = tf.nn.l2_normalize(mentee_data_dict.conv4_1,0)        
        normalize_b_8 = tf.nn.l2_normalize(mentor_data_dict.conv4_3,0)

        """
            cosine similarity is calculated between 2nd layer of mentee and 3rd layer of mentor.
            Similarly, cosine similarity is calculated between 2nd layer of mentee and 4th layer of mentor.

        """

        normalize_a_9 = tf.nn.l2_normalize(mentee_data_dict.conv2_1,0)        
        normalize_b_9 = tf.nn.l2_normalize(mentor_data_dict.conv2_1,0)
        normalize_a_10 = tf.nn.l2_normalize(mentee_data_dict.conv2_1,0)        
        normalize_b_10= tf.nn.l2_normalize(mentor_data_dict.conv2_2,0)

        """
            cosine similarity is calculated between 3rd layer of mentee and 5th layer of mentor.
            Similarly, cosine similarity is calculated between 3rd layer of mentee and 6th layer of mentor.
            Similarly, cosine similarity is calculated between 3rd layer of mentee and 7th layer of mentor.

        """
                        
        normalize_a_11 = tf.nn.l2_normalize(mentee_data_dict.conv3_1,0)        
        normalize_b_11= tf.nn.l2_normalize(mentor_data_dict.conv3_1,0)
        normalize_a_12 = tf.nn.l2_normalize(mentee_data_dict.conv3_1,0)        
        normalize_b_12 = tf.nn.l2_normalize(mentor_data_dict.conv3_2,0)        
        normalize_a_13= tf.nn.l2_normalize(mentee_data_dict.conv3_1,0)
        normalize_b_13= tf.nn.l2_normalize(mentor_data_dict.conv3_3,0)
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

    def normalize_the_outputs_of_mentor_mentee_of_different_widths(self, sess, feed_dict):

        ## normalize mentee's outputs
        normalize_mentee_1 = tf.nn.l2_normalize(self.mentee_data_dict.conv1_1, 0)
        normalize_mentee_2 = tf.nn.l2_normalize(self.mentee_data_dict.conv2_1, 0)
        normalize_mentee_3 = tf.nn.l2_normalize(self.mentee_data_dict.conv3_1, 0)
        normalize_mentee_4 = tf.nn.l2_normalize(self.mentee_data_dict.conv4_1, 0)
        normalize_mentee_5 = tf.nn.l2_normalize(self.mentee_data_dict.conv5_1, 0)

        ## normalize mentor's outputs

        normalize_mentor = {}
        normalize_mentor["1"] = tf.nn.l2_normalize(self.mentor_data_dict.conv1_1, 0)
        normalize_mentor["2"] = tf.nn.l2_normalize(self.mentor_data_dict.conv1_2, 0)
        normalize_mentor["3"] = tf.nn.l2_normalize(self.mentor_data_dict.conv2_1, 0)
        normalize_mentor["4"] = tf.nn.l2_normalize(self.mentor_data_dict.conv2_2, 0)
        normalize_mentor["5"] = tf.nn.l2_normalize(self.mentor_data_dict.conv3_1, 0)
        normalize_mentor["6"] = tf.nn.l2_normalize(self.mentor_data_dict.conv3_2, 0)
        normalize_mentor["7"] = tf.nn.l2_normalize(self.mentor_data_dict.conv3_3, 0)
        normalize_mentor["8"] = tf.nn.l2_normalize(self.mentor_data_dict.conv4_1, 0)
        normalize_mentor["9"] = tf.nn.l2_normalize(self.mentor_data_dict.conv4_2, 0)
        normalize_mentor["10"] = tf.nn.l2_normalize(self.mentor_data_dict.conv4_3, 0)
        normalize_mentor["11"] = tf.nn.l2_normalize(self.mentor_data_dict.conv5_1, 0)
        normalize_mentor["12"] = tf.nn.l2_normalize(self.mentor_data_dict.conv5_2, 0)
        normalize_mentor["13"] = tf.nn.l2_normalize(self.mentor_data_dict.conv5_3, 0)

        
        idx = tf.constant(list(xrange(0, 64)))
        self.cosine_similarity_of_different_widths(normalize_mentee_1, normalize_mentor, idx, sess, feed_dict)

        self.cosine_similarity_of_different_widths(normalize_mentee_2, normalize_mentor, idx, sess, feed_dict)

        self.cosine_similarity_of_different_widths(normalize_mentee_3, normalize_mentor, idx, sess, feed_dict)

        self.cosine_similarity_of_different_widths(normalize_mentee_4, normalize_mentor, idx, sess, feed_dict)

        self.cosine_similarity_of_different_widths(normalize_mentee_5, normalize_mentor, idx, sess, feed_dict)



    def cosine_similarity_of_different_widths(self, normalize_mentee, normalize_mentor, idx, sess, feed_dict):

        """
            cosine similarity between every layer of mentee and mentor is calculated
            normalize_mentor:: dictionary containing output of each layer of mentor
            tf.gather picks the feature maps from the mentor output such that the number of feature maps picked equals the number of feature maps of the mentee's output
            idx indicate the indices of the feature maps that need to be picked form mentor's output

        """

        cosine1 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["1"], idx, axis = 3))))
        cosine2 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["2"], idx, axis = 3))))
        cosine3 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["3"], idx, axis = 3))))
        cosine4 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["4"], idx, axis = 3))))
        cosine5 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["5"], idx, axis = 3))))
        cosine6 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["6"], idx, axis = 3))))
        cosine7 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["7"], idx, axis = 3))))
        cosine8 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["8"], idx, axis = 3))))
        cosine9 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["9"], idx, axis = 3))))
        cosine10 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["10"], idx, axis = 3))))
        cosine11 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["11"], idx, axis = 3))))
        cosine12 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["12"], idx, axis = 3))))
        cosine13 = tf.reduce_sum(tf.multiply(tf.reduce_mean(normalize_mentee), tf.reduce_mean(tf.gather(normalize_mentor["13"], idx, axis = 3))))
        pdb.set_trace()
        print("start")
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
        print("ended")

    def loss_with_different_layer_widths(self, embed_data_dict, mentor_data_dict, mentee_data_dict):
        """
        Here layers of different widths are mapped together.

        """
        
        ## loss between the embed layers connecting mentor's 3rd layer and mentee's 1st layer
        self.l1 = embed_data_dict.loss_embed_3
        ## loss between the embed layers connecting mentor's 5th layer and mentee's 2nd layer
        self.l2 = embed_data_dict.loss_embed_4
        self.l3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv3_2, mentee_data_dict.conv3_1))))
        self.l4 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv4_2, mentee_data_dict.conv4_1))))
        self.l5 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv5_2, mentee_data_dict.conv5_1))))
        ## loss between mentor-mentee last layers before softmax
        self.l6 = (tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.fc3l, mentee_data_dict.fc3l))))

    

    def visualization_of_filters(self, sess):
        
        mentor_filter = sess.run(self.mentor_data_dict.conv1_1)
        mentee_filter = sess.run(self.mentee_data_dict.conv1_1) 
        img1 = Image.fromarray(mentor_filter, 'RGB')
        img2 = Image.fromarray(mentee_filter, 'RGB')
        img1.save('mentor_filter.png')
        img2.save('mentee_filter.png')


    def rmse_loss(self, mentor_data_dict, mentee_data_dict):
        
        """
        Here layers of same width are mapped together. 

        """

        ## loss between mentor's 1st layer and mentee's 1st layer
        self.l1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv1_1, mentee_data_dict.conv1_1))))
        
        ## loss between mentor's 4th layer and mentee's 2nd layer
        self.l2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv2_2, mentee_data_dict.conv2_1))))
        ## loss between mentor's 5th layer and mentee's 3rd layer
        self.l3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv3_1, mentee_data_dict.conv3_1))))
        ## loss between mentor's 9th layer and mentee's 4th layer
        self.l4 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv4_2, mentee_data_dict.conv4_1))))
        ## loss between mentor's 12th layer and mentee's 5th layer
        self.l5 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv5_2, mentee_data_dict.conv5_1))))

        ## loss between mentor-mentee last layers before softmax
        self.l6 = (tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.fc3l, mentee_data_dict.fc3l))))
        ## loss between mentor-mentee softmax layers
        self.l7 = (tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.softmax, mentee_data_dict.softmax))))
        
        ind_max = tf.argmax(mentor_data_dict.fc3l, axis = 1)
        hard_logits = tf.one_hot(ind_max, FLAGS.num_classes)
        ## hard_logits KT technique ::: where hard_logits of teacher are transferred to student softmax output
        self.l8 = (tf.reduce_mean(tf.square(tf.subtract(hard_logits, mentee_data_dict.softmax))))

        ## intermediate representations KT technique (single layer):: HT stands for hint based training which is phase 1 of intermediate representations KT technique.
        ## knowledge from 7th layer of mentor is given to 3rd layer of mentee.
        self.HT = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_data_dict.conv3_3, mentee_data_dict.conv3_1))))

    def calculate_loss_with_multiple_optimizers(self, feed_dict, sess):

        if FLAGS.multiple_optimizers_l0:
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
        
        elif FLAGS.multiple_optimizers_l1:
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            
            
        elif FLAGS.multiple_optimizers_l2:
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
                                    
        elif FLAGS.multiple_optimizers_l3:
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
        
        elif FLAGS.multiple_optimizers_l4:
            _, self.loss_value0 = sess.run([self.train_op0, loss], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([self.train_op4, self.l4], feed_dict=feed_dict)
        

        elif FLAGS.multiple_optimizers_l5:                             
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([self.train_op4, self.l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([self.train_op5, self.l5], feed_dict=feed_dict)
        
        elif FLAGS.multiple_optimizers_l6:                             
            _, self.loss_value0 = sess.run([self.train_op0, self.loss], feed_dict=feed_dict)
            _, self.loss_value1 = sess.run([self.train_op1, self.l1], feed_dict=feed_dict)
            _, self.loss_value2 = sess.run([self.train_op2, self.l2], feed_dict=feed_dict)
            _, self.loss_value3 = sess.run([self.train_op3, self.l3], feed_dict=feed_dict)
            _, self.loss_value4 = sess.run([self.train_op4, self.l4], feed_dict=feed_dict)
            _, self.loss_value5 = sess.run([self.train_op5, self.l5], feed_dict=feed_dict)
            _, self.loss_value6 = sess.run([self.train_op6, self.l6], feed_dict=feed_dict)
                                         
    """
        This is the traditional KT technique where all the layer weights get updated.
        if the var_list is not mentioned explicitely it means by default all the layer weights get updated

    """
    
    def calculate_loss_with_single_optimizer(self,feed_dict, sess):
        _, self.loss_value = sess.run([self.train_op, self.loss] , feed_dict=feed_dict)
        

    """
        This is the new KT method:: where only the weights of the target layer get updated as opposed
        to traditional KT techniques where all the layer weights of the student network get updated. 
        
    """
    def train_op_for_multiple_optimizers(self, lr):

        l1_var_list = []
        l2_var_list =[]
        l3_var_list = []
        l4_var_list = []
        l5_var_list = []
        l6_var_list = []
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        ## its a loss between softmax layer and dataset
        self.train_op0 = tf.train.AdamOptimizer(lr).minimize(self.loss)
        ## here only mentee's 1st layer weights get updated 
        self.train_op1 = tf.train.AdamOptimizer(lr).minimize(self.l1, var_list = self.l1_weights_of_mentee(l1_var_list))
        ## here only mentee's 2nd layer weights get updated 
        self.train_op2 = tf.train.AdamOptimizer(lr).minimize(self.l2, var_list = self.l2_weights_of_mentee(l2_var_list))
        ## here only mentee's 3rd layer weights get updated
        self.train_op3 = tf.train.AdamOptimizer(lr).minimize(self.l3, var_list = self.l3_weights_of_mentee(l3_var_list))
        ## here only mentee's 4th layer weights get updated
        self.train_op4 = tf.train.AdamOptimizer(lr).minimize(self.l4, var_list = self.l4_weights_of_mentee(l4_var_list))
        ## here only mentee's 5th layer weights get updated
        self.train_op5 = tf.train.AdamOptimizer(lr).minimize(self.l5, var_list = self.l5_weights_of_mentee(l5_var_list))
        ## here only mentee's 6th layer weights get updated
        self.train_op6 = tf.train.AdamOptimizer(lr).minimize(self.l6, var_list = self.l6_weights_of_mentee(l6_var_list))

        

    def train_op_for_single_optimizer(self, lr):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        if FLAGS.single_optimizer_l1:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1)
        if FLAGS.single_optimizer_l2:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1 + self.l2)
        if FLAGS.single_optimizer_l3:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1 + self.l2 + self.l3)
        if FLAGS.single_optimizer_l4:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1 + self.l2 + self.l3 + self.l4)
        if FLAGS.single_optimizer_l5:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1 + self.l2 + self.l3 + self.l4 + self.l5)
        if FLAGS.single_optimizer_l6:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l1 + self.l2 + self.l3 + self.l4 + self.l5 + self.l6)
        if FLAGS.single_optimizer_last_layer:
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss + self.l6)

        #####Hard Logits KT technique
        if FLAGS.hard_logits:

            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.l8)
        
        #####Soft Logits KT technique
        if FLAGS.single_optimizer_last_layer_with_temp_softmax:

            self.train_op = tf.train.AdamOptimizer(lr).minimize(alpha * self.loss + self.l7)

        #####Phase 1 of intermediate representations KT technique
        if FLAGS.fitnets_HT:
            variables_for_HT = []
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.HT, var_list = self.get_variables_for_HT(variables_for_HT))

    def train_independent_student(self, images_placeholder, labels_placeholder, seed, phase_train, global_step, sess):

        """
            Student is trained without taking knowledge from teacher

            Args:
                images_placeholder: placeholder to hold images of dataset
                labels_placeholder: placeholder to hold labels of the images of the dataset
                seed: seed value to have sequence in the randomness
                phase_train: determines test or train state of the network
        """

        student = Mentee(FLAGS.num_channels)
        print("Independent student")
        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        ## number of steps after which learning rate should decay
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        
        mentee_data_dict = student.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax,seed, phase_train)
        self.loss = student.loss(labels_placeholder)
        ## learning rate is decayed exponentially with a decay factor of 0.9809 after every epoch
        lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
        self.train_op = student.training(self.loss,lr, global_step)
        self.softmax = mentee_data_dict.softmax
        # initialize all the variables of the network
        init = tf.initialize_all_variables()
        sess.run(init)
        ## saver object is created to save all the variables to a file
        saver = tf.train.Saver()

    
    def train_teacher(self, images_placeholder, labels_placeholder, phase_train, global_step, sess):

        """
            1. Train teacher prior to student so that knowledge from teacher can be transferred to train student.
            2. Teacher object is trained by importing weights from a pretrained vgg 16 network
            3. Mentor object is a network trained from scratch. We did not find the pretrained network with the same architecture for cifar10. 
               Thus, trained the network from scratch on cifar10

        """

        if FLAGS.dataset == 'cifar10' or 'mnist':
            print("Teacher")
            mentor = Teacher()
        if FLAGS.dataset == 'caltech101':
            mentor = Mentor()
        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        mentor_data_dict = mentor.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, phase_train)
        self.loss = mentor.loss(labels_placeholder)
        lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
        if FLAGS.dataset == 'caltech101':
            ## restore all the weights 
            variables_to_restore = self.get_mentor_variables_to_restore()
            self.train_op = mentor.training(loss, FLAGS.learning_rate_pretrained,lr, global_step, variables_to_restore,mentor.get_training_vars())
        if FLAGS.dataset == 'cifar10':
            self.train_op = mentor.training(loss, FLAGS.learning_rate, global_step)
        self.softmax = mentor_data_dict.softmax
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

    def train_dependent_student(self, images_placeholder, labels_placeholder, phase_train, seed, global_step, sess):

        """
        Student is trained by taking supervision from teacher for every batch of data
        Same batch of input data is passed to both teacher and student for every iteration

        """
        if FLAGS.dataset == 'cifar10' or 'mnist':
            print("Teacher")
            vgg16_mentor = Teacher(False)
        if FLAGS.dataset == 'caltech101':
            vgg16_mentor = Mentor(False)
        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        vgg16_mentee = Mentee(FLAGS.num_channels)
        self.mentor_data_dict = vgg16_mentor.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, phase_train)

        self.mentee_data_dict = vgg16_mentee.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)

        """
        The below code is to calculate the cosine similarity between the outputs of the mentor-mentee layers.
        The layers with highest cosine similarity value are mapped together.
        """
        cosine1, cosine2, cosine3, cosine4, cosine5, cosine6, cosine7, cosine8, cosine9, cosine10, cosine11, cosine12, cosine13 = self.cosine_similarity_of_same_width(self.mentee_data_dict, self.mentor_data_dict)
        self.softmax = self.mentee_data_dict.softmax
        mentor_variables_to_restore = self.get_mentor_variables_to_restore()
        self.loss = vgg16_mentee.loss(labels_placeholder)
        lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
        if FLAGS.single_optimizer:
            self.rmse_loss(self.mentor_data_dict, self.mentee_data_dict)
            self.train_op_for_single_optimizer(lr)

        if FLAGS.layers_with_same_width:
            self.rmse_loss(self.mentor_data_dict, self.mentee_data_dict)
            self.train_op_for_multiple_optimizers(lr)

        if FLAGS.layers_with_different_widths:
            embed = Embed()
            embed_data_dict  = embed.build(self.mentor_data_dict, self.mentee_data_dict, FLAGS.embed_type)
            self.loss_with_different_layer_widths(embed_data_dict, self.mentor_data_dict, self.mentee_data_dict)
            self.train_op_for_multiple_optimizers(lr)
            init = tf.initialize_all_variables()
            sess.run(init)
        if FLAGS.fitnets_KD:
            variables_for_KD = []
            saver = tf.train.Saver(self.get_variables_for_KD(variables_for_KD))
            saver.restore(sess, "./summary-log/new_method_dependent_student_weights_filename_cifar10")
        saver = tf.train.Saver(mentor_variables_to_restore)
        saver.restore(sess, "./summary-log/new_method_teacher_weights_filename_caltech101")



    def train_model(self, data_input_train, data_input_test, images_placeholder, labels_placeholder, sess, phase_train):
        
        try:
            eval_correct= self.evaluation(self.softmax, labels_placeholder)
            count = 0
            for i in range(NUM_ITERATIONS):
                start_time = time.time()
                count = count + 1
                feed_dict = self.fill_feed_dict(data_input_train, images_placeholder,
                                                                    labels_placeholder, sess, 'Train', phase_train)
                if FLAGS.student or FLAGS.teacher:
                    _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    if FLAGS.dataset == 'mnist':
                        batch = mnist.train.next_batch(FLAGS.batch_size)
                        _, loss_value = sess.run([self.train_op, self.loss], feed_dict = {images_placeholder: np.reshape(batch[0], [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, FLAGS.num_channels]), labels_placeholder: batch[1]})

                    if i % 10 == 0:
                        print ('Step %d: loss_value = %.20f' % (i, loss_value))

                if FLAGS.dependent_student and FLAGS.single_optimizer:
                                        
                    self.calculate_loss_with_single_optimizer(feed_dict, sess)
                    if i % 10 == 0:
                        print ('Step %d: loss_value = %.20f' % (i, self.loss_value))

                if FLAGS.dependent_student and FLAGS.multiple_optimizers:
                    #self.normalize_the_outputs_of_mentor_mentee_of_different_widths(sess, feed_dict)
                    #self.visualization_of_filters(sess)
                    self.calculate_loss_with_multiple_optimizers(feed_dict, sess)
                                            
                    if i % 10 == 0:
                        if FLAGS.multiple_optimizers_l0:
                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                        elif FLAGS.multiple_optimizers_l1:

                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                        elif FLAGS.multiple_optimizers_l2:
                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                            print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                        elif FLAGS.multiple_optimizers_l3:
                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                            print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                            print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                        elif FLAGS.multiple_optimizers_l4:
                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                            print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                            print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                            print ('Step %d: loss_value4 = %.20f' % (i, self.loss_value4))
                        elif FLAGS.multiple_optimizers_l5:
                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                            print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                            print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                            print ('Step %d: loss_value4 = %.20f' % (i, self.loss_value4))
                            print ('Step %d: loss_value5 = %.20f' % (i, self.loss_value5))
                        elif FLAGS.multiple_optimizers_l6:
                            print ('Step %d: loss_value0 = %.20f' % (i, self.loss_value0))
                            print ('Step %d: loss_value1 = %.20f' % (i, self.loss_value1))
                            print ('Step %d: loss_value2 = %.20f' % (i, self.loss_value2))
                            print ('Step %d: loss_value3 = %.20f' % (i, self.loss_value3))
                            print ('Step %d: loss_value4 = %.20f' % (i, self.loss_value4))
                            print ('Step %d: loss_value5 = %.20f' % (i, self.loss_value5))
                            print ('Step %d: loss_value6 = %.20f' % (i, self.loss_value6))
                                                
                        """
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, i)
                        summary_writer.flush()
                        """
                if (i) %(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN//FLAGS.batch_size)  == 0 or (i) == NUM_ITERATIONS:
                                            
                    checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')
                    if FLAGS.teacher:
                        saver.save(sess, FLAGS.teacher_weights_filename)
                    """
                    elif FLAGS.student:
                        saver.save(sess, FLAGS.student_filename)
                    """
                    """                                            
                    elif FLAGS.dependent_student:
                        saver_new = tf.train.Saver()
                        saver_new.save(sess, FLAGS.dependent_student_filename)
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
                            self.softmax,
                            images_placeholder,
                            labels_placeholder,
                            data_input_train, 
                            'Train', phase_train)

                        print ("Test  Data Eval:")
                        self.do_eval(sess, 
                            eval_correct,
                            self.softmax,
                            images_placeholder,
                            labels_placeholder,
                            data_input_test, 
                            'Test', phase_train)
                        print ("max accuracy % f", max(test_accuracy_list))
                        #print ("test accuracy", test_accuracy_list)
                #print( "--- %s seconds ---" % (time.time() - start_time))    
            coord.request_stop()
            coord.join(threads)
        except Exception as e:
            print(e)
    def main(self, _):

            with tf.Graph().as_default():
                    ## This line allows the code to use only sufficient memory and does not block entire GPU
                    config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
                    if FLAGS.dataset == 'mnist':
                        mnist = read_mnist_data()

                    ## set the seed so that we have same loss values and initializations for every run.
                    tf.set_random_seed(seed)
                    
                    data_input_train = DataInput(dataset_path, FLAGS.train_dataset, FLAGS.batch_size, FLAGS.num_training_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)

                    data_input_test = DataInput(dataset_path, FLAGS.test_dataset,FLAGS.batch_size, FLAGS.num_testing_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)

                    data_input_validation = DataInput(dataset_path, FLAGS.validation_dataset,FLAGS.batch_size, FLAGS.num_validation_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)
                    images_placeholder, labels_placeholder = self.placeholder_inputs(FLAGS.batch_size)
                    sess = tf.Session(config = config)
                    ## this line is used to enable tensorboard debugger
                    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'localhost:6064') 
                    summary_writer = tf.summary.FileWriter(SUMMARY_LOG_DIR, sess.graph)
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    phase_train = tf.placeholder(tf.bool, name = 'phase_train')
                    summary = tf.summary.merge_all()

                    if FLAGS.student:
                        self.train_independent_student(images_placeholder, labels_placeholder, seed, phase_train, global_step, sess)

                    elif FLAGS.teacher:
                        self.train_teacher(images_placeholder, labels_placeholder, phase_train, global_step, sess)

                    elif FLAGS.dependent_student:
                        self.train_dependent_student(images_placeholder, labels_placeholder, phase_train, seed, global_step, sess) 
                    
                    self.train_model(data_input_train, data_input_test, images_placeholder, labels_placeholder, sess, phase_train)

            sess.close()
            summary_writer.close()

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
            default = "./summary-log/new_method_teacher_weights_filename_cifar10"
        )
        parser.add_argument(
            '--student_filename',
            type = str,
            default = "./summary-log/new_method_student_weights_filename_cifar10"
        )
        parser.add_argument(
            '--dependent_student_filename',
            type = str,
            default = "./summary-log/new_method_dependent_student_weights_filename_cifar10"
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
            default = 5853                                    
        )
        parser.add_argument(
            '--num_training_examples',
            type = int,
            default = 5853
        )
        parser.add_argument(
            '--num_testing_examples',
            type = int,
            default = 1829                                    
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
            default = 'caltech101'
        )
        parser.add_argument(
            '--mnist_data_dir',
            type = str,
            help = 'name of the dataset',
            default = './mnist_data'
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
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l2',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l3',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l4',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l5',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_l6',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_last_layer',
            type = bool,
            help = 'last layer loss from mentor only the logits',
            default = False
        )
        parser.add_argument(
            '--single_optimizer_last_layer_with_temp_softmax',
            type = bool,
            help = 'last layer loss from mentor with temp softmax',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l0',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l1',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l2',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l3',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l4',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l5',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--multiple_optimizers_l6',
            type = bool,
            help = 'number of channels in the initial layer if it is RGB it will 3 , if it is gray scale it will be 1',
            default = False
        )
        parser.add_argument(
            '--top_1_accuracy',
            type = bool,
            help = 'top-1-accuracy',
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
            '--hard_logits',
            type = bool,
            help = 'hard_logits',
            default = False
        )
        parser.add_argument(
            '--fitnets_HT',
            type = bool,
            help = 'fitnets_HT',
            default = False
        )
        parser.add_argument(
            '--fitnets_KD',
            type = bool,
            help = 'fitnets_KD',
            default = False
        )
        parser.add_argument(
            '--layers_with_different_widths',
            type = bool,
            help = 'different width layers mapping',
            default = False
        )
        parser.add_argument(
            '--embed_type',
            type = str,
            help = 'embed type can be either fully connected or convolutional layers',
            default = 'fc'
        )
        parser.add_argument(
            '--layers_with_same_width',
            type = bool,
            help = 'same width layers mapping',
            default = False
        )
        
        FLAGS, unparsed = parser.parse_known_args()
        ex = VGG16()
        tf.app.run(main=ex.main, argv = [sys.argv[0]] + unparsed)


