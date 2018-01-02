# VGG16-TF
TensorFlow implementation of the training version of VGG16

python main.py --student True --train_dataset train_map.txt --test_dataset test_map.txt --validation_dataset validation_map.txt --num_training_examples 45000 --num_testing_examples 10000 --num_validation_examples 5000 --NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN 45000 --num_classes 10 --image_width 28 --image_height 28  --batch_size 128 --learning_rate 0.0000005 --dataset mnist

We are also gonna add Generalized version of Trainable VGG19 in the future.

vgg16.npy can be downloaded from here:ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
vgg16.npz can be downloaded from here: https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
