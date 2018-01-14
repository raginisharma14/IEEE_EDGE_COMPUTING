Modified Knowledge Transfer Technique 

Command to execute independent student:<br />
- python vgg16_main.py <br />
          --student True <br />
          --dataset caltech101 <br />
          --learning_rate 0.0001 <br />
          
- python vgg16_main.py 
          --student True <br />
          --dataset cifar10 <br />
          --learning_rate 0.0001 <br />

Command to execute dependent student:<br />
- python main.py<br />
      --student True<br />
      --train_dataset caltech101-train.txt <br />
      --test_dataset caltech101-test<br />
      --validation_dataset caltech101-validation.txt<br />
      --num_training_examples 45000<br />
      --num_testing_examples 10000<br />
      --num_validation_examples 5000<br />
      --NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN 45000<br />
      --num_classes 10<br />
      --image_width 224 <br />
      --image_height 224<br />
      --batch_size 45<br />
      --learning_rate 0.005<br />
      --dataset caltech101<br />



vgg16.npy can be downloaded from here:ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
vgg16.npz can be downloaded from here: https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
