import tensorflow as tf
import numpy as np
import random
from DataInput import DataInput
from vgg16mentee import Mentee
from vgg16mentor import Mentor
#from embed import Embed
import os
import pdb
import sys
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
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
def read_mnist_data():
    mnist = read_data_sets(FLAGS.mnist_data_dir)
    return mnist      

def placeholder_inputs(batch_size):
	images_placeholder = tf.placeholder(tf.float32, 
								shape=(FLAGS.batch_size, FLAGS.image_height, 
									   FLAGS.image_width, FLAGS.num_channels))
	labels_placeholder = tf.placeholder(tf.int32,
								shape=(FLAGS.batch_size))

	return images_placeholder, labels_placeholder

def fill_feed_dict(data_input, images_pl, labels_pl, sess, mode, phase_train):
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

def do_eval(sess,
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
	        feed_dict = fill_feed_dict(dataset, images_placeholder,
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

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, 1)
	pred = tf.argmax(logits, 1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))

def get_mentor_variables_to_restore(variables_to_restore):
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv1_1/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv1_2/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv2_1/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv2_2/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv3_1/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv3_2/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv3_3/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv4_1/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv4_2/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv4_3/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv5_1/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv5_2/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv5_3/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_fc1/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_fc2/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv1_1/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv1_2/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv2_1/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv2_2/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv3_1/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv3_2/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv3_3/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv4_1/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv4_2/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv4_3/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv5_1/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv5_2/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_conv5_3/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_fc2/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_fc1/mentor_biases"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_fc3/mentor_weights"][0])
        variables_to_restore.append([var for var in tf.global_variables() if var.op.name=="mentor_fc3/mentor_biases"][0])
        return variables_to_restore

def l1_weights_of_mentee(l1_mentee_weights):
    l1_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv1_1/mentee_weights"][0])
#    l1_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv1_1/mentee_biases"][0])

    return l1_mentee_weights

def l2_weights_of_mentee(l2_mentee_weights):
    l2_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_weights"][0])
 #   l2_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv2_1/mentee_biases"][0])
    return l2_mentee_weights

def l3_weights_of_mentee(l3_mentee_weights):
    l3_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_weights"][0])
  #  l3_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv3_1/mentee_biases"][0])
    return l3_mentee_weights

def l4_weights_of_mentee(l4_mentee_weights):
    l4_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv4_1/mentee_weights"][0])
   # l4_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv4_1/mentee_biases"][0])
    return l4_mentee_weights

def l5_weights_of_mentee(l5_mentee_weights):
    l5_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv5_1/mentee_weights"][0])
    #l5_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_conv5_1/mentee_biases"][0])
    return l5_mentee_weights

def l6_weights_of_mentee(l6_mentee_weights):
    l6_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_fc3/mentee_weights"][0])
    #l6_mentee_weights.append([var for var in tf.global_variables() if var.op.name=="mentee_fc3/mentee_biases"][0])
    return l6_mentee_weights

def cosine_similarity(mentee_conv1_1, mentor_conv1_1, mentor_conv1_2, mentee_conv2_1,mentor_conv2_1, mentor_conv2_2,mentee_conv3_1,mentor_conv3_1, mentor_conv3_2, mentor_conv3_3,mentee_conv4_1, mentor_conv4_1, mentor_conv4_2, mentor_conv4_3, mentee_conv5_1, mentor_conv5_1, mentor_conv5_2, mentor_conv5_3):
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

def rmse_loss(mentor_conv1_2, mentee_conv1_1, mentee_conv2_1, mentor_conv2_2, mentor_conv3_1,mentee_conv3_1,mentor_conv4_3,mentee_conv4_1, mentor_conv5_2,mentee_conv5_1, logits_mentor, logits_mentee):

    l1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_conv1_2, mentee_conv1_1))))
    l2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_conv2_2, mentee_conv2_1))))
    l3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_conv3_1, mentee_conv3_1))))
    l4 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_conv4_3, mentee_conv4_1))))
    l5 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(mentor_conv5_2, mentee_conv5_1))))
    l6 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(logits_mentor, logits_mentee))))

    return l1, l2, l3 , l4, l5, l6


def calculate_loss_with_multiple_optimizers(train_op0, loss, train_op1, l1, train_op2, l2, train_op3, l3, train_op4, l4, train_op5, l5, train_op6, l6, feed_dict, sess):

    _, loss_value0 = sess.run([train_op0, loss], feed_dict=feed_dict)
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
    _, loss_value1 = sess.run([train_op1, l1], feed_dict=feed_dict)
    _, loss_value2 = sess.run([train_op2, l2], feed_dict=feed_dict)
                                
    _, loss_value3 = sess.run([train_op3, l3], feed_dict=feed_dict)
                                    
    _, loss_value4 = sess.run([train_op4, l4], feed_dict=feed_dict)
                                 
    _, loss_value5 = sess.run([train_op5, l5], feed_dict=feed_dict)
    _, loss_value6 = sess.run([train_op6, l6], feed_dict=feed_dict)
                                     
    return loss_value0, loss_value1, loss_value2, loss_value3, loss_value4, loss_value5, loss_value6 

def calculate_loss_with_single_optimizer(train_op, loss, feed_dict, sess):
    _, loss_value = sess.run([train_op, loss] , feed_dict=feed_dict)
    return loss_value

def train_op_for_multiple_optimizers(lr, loss, l1, l2, l3, l4, l5, l6):

    l1_var_list = []
    l2_var_list =[]
    l3_var_list = []
    l4_var_list = []
    l5_var_list = []
    l6_var_list = []
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    train_op0 = tf.train.AdamOptimizer(lr).minimize(loss)
    train_op1 = tf.train.AdamOptimizer(lr).minimize(l1, var_list = l1_weights_of_mentee(l1_var_list))
    train_op2 = tf.train.AdamOptimizer(lr).minimize(l2, var_list = l2_weights_of_mentee(l2_var_list))
    train_op3 = tf.train.AdamOptimizer(lr).minimize(l3, var_list = l3_weights_of_mentee(l3_var_list))
    train_op4 = tf.train.AdamOptimizer(lr).minimize(l4, var_list = l4_weights_of_mentee(l4_var_list))
    train_op5 = tf.train.AdamOptimizer(lr).minimize(l5, var_list = l5_weights_of_mentee(l5_var_list))
    train_op6 = tf.train.AdamOptimizer(lr).minimize(l6, var_list = l6_weights_of_mentee(l6_var_list))

    return train_op0, train_op1, train_op2, train_op3, train_op4, train_op5, train_op6

def train_op_for_single_optimizer(lr, loss, l1, l2, l3, l4, l5, l6):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(lr).minimize(loss + l1 + l2 + l3 + l4 + l5 + l6)

    return train_op

def main(_):

	with tf.Graph().as_default():
                config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))
               	if FLAGS.dataset == 'mnist':
                    mnist = read_mnist_data()
                tf.set_random_seed(seed)

		data_input_train = DataInput(dataset_path, FLAGS.train_dataset, FLAGS.batch_size, FLAGS.num_training_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)

		data_input_test = DataInput(dataset_path, FLAGS.test_dataset,FLAGS.batch_size, FLAGS.num_testing_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)

		data_input_validation = DataInput(dataset_path, FLAGS.validation_dataset,FLAGS.batch_size, FLAGS.num_validation_examples, FLAGS.image_width, FLAGS.image_height, FLAGS.num_channels, seed, FLAGS.dataset)
		images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
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
		        mentor = Mentor()
                    if FLAGS.dataset == 'caltech101':
                        mentor = Mentor()
                    num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
                    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                    mentor.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, phase_train)
                    loss = mentor.loss(labels_placeholder)
                    lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
                    variables_to_restore = []
                    variables_to_restore = get_mentor_variables_to_restore(variables_to_restore)
                    if FLAGS.dataset == 'caltech101':
                        train_op = mentor.training(loss, FLAGS.learning_rate_pretrained,lr, global_step, variables_to_restore,mentor.get_training_vars())
                    if FLAGS.dataset == 'cifar10':
                        #train_op = mentor.training(loss, lr, global_step)
                        train_op = mentor.training(loss, FLAGS.learning_rate_pretrained,lr, global_step, variables_to_restore,mentor.get_training_vars())
                    softmax = mentor.fc3l
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    saver = tf.train.Saver()

                elif FLAGS.student:
                    student = Mentee(FLAGS.num_channels)
                    print("Independent student")
                    num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
                    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                    _,_,_,_,_,_,_,softmax_mentee = student.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax,seed, phase_train)
                    loss = student.loss(labels_placeholder)
                    lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
                    train_op = student.training(loss,lr, global_step)
                    softmax = softmax_mentee
                    init = tf.initialize_all_variables()
                    sess.run(init)
                    saver = tf.train.Saver()

                    
                elif FLAGS.dependent_student:
		    vgg16_mentor = Mentor(trainable = False)
                    num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
                    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
                    vgg16_mentee = Mentee(FLAGS.num_channels)
		    mentor_conv1_1, mentor_conv1_2, mentor_conv2_1, mentor_conv2_2, mentor_conv3_1, mentor_conv3_2, mentor_conv3_3, mentor_conv4_1, mentor_conv4_2, mentor_conv4_3,  mentor_conv5_1, mentor_conv5_2, mentor_conv5_3, logits_mentor, softmax_mentor = vgg16_mentor.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, phase_train)

                    mentee_conv1_1, mentee_conv2_1, mentee_conv3_1, mentee_conv4_1, mentee_conv5_1, mentee_conv6_1, logits_mentee, softmax_mentee = vgg16_mentee.build(images_placeholder, FLAGS.num_classes, FLAGS.temp_softmax, seed, phase_train)

                    """
                    The below code is to calculate the cosine similarity between the outputs of the mentor-mentee layers.
                    """
                    """
                    cosine1, cosine2, cosine3, cosine4, cosine5, cosine6, cosine7, cosine8, cosine9, cosine10, cosine11, cosine12, cosine13 = cosine_similarity(mentee_conv1_1, mentor_conv1_1, mentor_conv1_2, mentee_conv2_1,mentor_conv2_1, mentor_conv2_2,mentee_conv3_1,mentor_conv3_1, mentor_conv3_2, mentor_conv3_3,mentee_conv4_1, mentor_conv4_1, mentor_conv4_2, mentor_conv4_3, mentee_conv5_1, mentor_conv5_1, mentor_conv5_2, mentor_conv5_3)
                    """
                    softmax = softmax_mentee
                    softmax_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(softmax_mentor, softmax_mentee))))
                    mentor_variables_to_restore = []
                    mentor_variables_to_restore = get_mentor_variables_to_restore(mentor_variables_to_restore)
		    loss = vgg16_mentee.loss(labels_placeholder)
                    l1, l2, l3, l4, l5, l6 = rmse_loss(mentor_conv1_2, mentee_conv1_1, mentee_conv2_1,mentor_conv2_2, mentor_conv3_1,mentee_conv3_1,mentor_conv4_3,mentee_conv4_1, mentor_conv5_2,mentee_conv5_1, logits_mentor, logits_mentee)
                    lr = tf.train.exponential_decay(FLAGS.learning_rate,global_step, decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
                    if FLAGS.single_optimizer:
                        train_op = train_op_for_single_optimizer(lr, loss, l1, l2, l3, l4, l5, l6)
                    if FLAGS.multiple_optimizers:
                        train_op0, train_op1, train_op2, train_op3, train_op4, train_op5, train_op6 = train_op_for_multiple_optimizers(lr, loss, l1, l2, l3, l4, l5, l6)

                    #train_op0, train_op1, train_op2, train_op3, train_op4, train_op5, train_op6 = train_op_for_multiple_optimizers(lr, loss, l1, l2, l3, l4, l5, l6)
		    init = tf.initialize_all_variables()
		    sess.run(init)
                    saver = tf.train.Saver(mentor_variables_to_restore)
                    saver.restore(sess, "./summary-log/new_method_teacher_weights_filename")
		eval_correct= evaluation(softmax, labels_placeholder)
                count = 0
		try:
			for i in range(NUM_ITERATIONS):
                                count = count + 1
				feed_dict = fill_feed_dict(data_input_train, images_placeholder,
								labels_placeholder, sess, 'Train', phase_train)
                                if FLAGS.student or FLAGS.teacher:
				    _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                                    if FLAGS.dataset == 'mnist':
                                        batch = mnist.train.next_batch(FLAGS.batch_size)
                                        _, loss_value = sess.run([train_op, loss], feed_dict = {images_placeholder: np.reshape(batch[0], [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, FLAGS.num_channels]), labels_placeholder: batch[1]})

				    if i % 10 == 0:
					print ('Step %d: loss_value = %.20f' % (i, loss_value))

                                if FLAGS.dependent_student and FLAGS.single_optimizer:
                                    
                                    loss_value = calculate_loss_with_single_optimizer(train_op, loss, feed_dict, sess)
				    if i % 10 == 0:
					print ('Step %d: loss_value = %.20f' % (i, loss_value))
                                if FLAGS.dependent_student and FLAGS.multiple_optimizers:

                                    loss_value0, loss_value1, loss_value2, loss_value3, loss_value4, loss_value5, loss_value6 = calculate_loss_with_multiple_optimizers(train_op0, loss, train_op1, l1, train_op2, l2, train_op3, l3, train_op4, l4, train_op5, l5, train_op6, l6, feed_dict, sess)
				    
                                    if i % 10 == 0:
					print ('Step %d: loss_value0 = %.20f' % (i, loss_value0))
					print ('Step %d: loss_value1 = %.20f' % (i, loss_value1))
					print ('Step %d: loss_value2 = %.20f' % (i, loss_value2))
					print ('Step %d: loss_value3 = %.20f' % (i, loss_value3))
					print ('Step %d: loss_value4 = %.20f' % (i, loss_value4))
					print ('Step %d: loss_value5 = %.20f' % (i, loss_value5))
					print ('Step %d: loss_value6 = %.20f' % (i, loss_value6))

					summary_str = sess.run(summary, feed_dict=feed_dict)
					summary_writer.add_summary(summary_str, i)
					summary_writer.flush()

				if (i) %(FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN//FLAGS.batch_size)  == 0 or (i) == NUM_ITERATIONS:
					
                                        checkpoint_file = os.path.join(SUMMARY_LOG_DIR, 'model.ckpt')
                                        if FLAGS.teacher:
                                            saver.save(sess, FLAGS.teacher_weights_filename)

                                        elif FLAGS.student:
                                            saver.save(sess, FLAGS.student_filename)
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
                                            do_eval(sess, 
                                                    eval_correct,
                                                    softmax,
                                                    images_placeholder,
                                                    labels_placeholder,
                                                    mnist, 
                                                    'Validation', phase_train)

                                            print('test accuracy::MNIST')
                                            do_eval(sess, 
                                                    eval_correct,
                                                    softmax,
                                                    images_placeholder,
                                                    labels_placeholder,
                                                    mnist, 
                                                    'Test', phase_train)
                                            
                                        else:
                                            print ("Training Data Eval:")
                                            do_eval(sess, 
                                                    eval_correct,
                                                    softmax,
                                                    images_placeholder,
                                                    labels_placeholder,
                                                    data_input_train, 
                                                    'Train', phase_train)

                                            print ("Test  Data Eval:")
                                            do_eval(sess, 
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
            default = 7316                                    
        )
        parser.add_argument(
            '--num_training_examples',
            type = int,
            default = 7316
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
            default = 'cifar10'
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

        FLAGS, unparsed = parser.parse_known_args()
        tf.app.run(main=main, argv = [sys.argv[0]] + unparsed)


