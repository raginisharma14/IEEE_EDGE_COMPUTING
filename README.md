Modified Knowledge Transfer Technique

Command to execute independent student:
1. python vgg16_main.py 
          --student True 
          --dataset caltech101
          --learning_rate 0.0001
          
2. python vgg16_main.py 
          --student True 
          --dataset cifar10
          --learning_rate 0.0001

Command to execute dependent student:
python main.py
      --student True
      --train_dataset caltech101-train.txt 
      --test_dataset caltech101-test
      --validation_dataset caltech101-validation.txt
      --num_training_examples 45000
      --num_testing_examples 10000
      --num_validation_examples 5000
      --NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN 45000
      --num_classes 10
      --image_width 224 
      --image_height 224
      --batch_size 45
      --learning_rate 0.005
      --dataset caltech101



vgg16.npy can be downloaded from here:ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
vgg16.npz can be downloaded from here: https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
