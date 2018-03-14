# Modified Knowledge Transfer Technique 

A particular teacher layer gives knowledge in the form of representations only to a particular student layer. This knowledge does not affect any other layer of the student network other than the target layer. By doing so, the student can learn more effectively compared to traditional knowledge transfer techniques. In neural networks every layer extracts specific features of the input image. Each layer of the student and teacher network extract different features. Hence, the features from different layers of the teacher network cannot be applied uniformly to all the layers of the student network. 

For a better understanding, assume the first layer of the teacher and student network extract edges of the input image. Second layer extracts objects like circles. Hence, transfering knowledge from the first layer of the teacher to the second layer of the student is not beneficial. Random mapping of the teacher-student layers can sometimes be lucky but not always. I propose to use cosine similarity metric to find the mapping of student-teacher layers. In cosine similarity metric, the outputs of the student-teacher layer pairs are normalized. Dot product of the normalized outputs is calculated. Higher the value of the dot product, greater is the similarity.

## Command to Execute Independent Student:<br />
- python vgg16_main.py 
          --student True <br />
          --dataset caltech101 <br />
          --learning_rate 0.0001 <br />
          
- python vgg16_main.py 
          --student True <br />
          --dataset cifar10 <br />
          --learning_rate 0.0001 <br />

## Command to Execute Dependent Student:<br />
- python vgg16_main.py<br />
      --dependent_student True<br />
      --train_dataset caltech101-train.txt <br />
      --test_dataset caltech101-test<br />
      --num_training_examples 5853<br />
      --num_testing_examples 1829<br />
      --NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN 5853<br />
      --num_classes 102<br />
      --image_width 224 <br />
      --image_height 224<br />
      --batch_size 45<br />
      --learning_rate 0.005<br />
      --dataset caltech101<br />

- python vgg16_main.py<br /> 
      --dependent_student True<br />
      --train_dataset cifar10-train.txt <br />
      --test_dataset cifar10-test<br />
      --num_training_examples 45000<br />
      --num_testing_examples 10000<br />
      --NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN 45000<br />
      --num_classes 10<br />
      --image_width 32 <br />
      --image_height 32<br />
      --batch_size 128<br />
      --learning_rate 0.01<br />
      --top_1_accuracy True<br />
      --dataset cifar10<br />

## Hyperparameters </br>
-Independent Student - batch-size 45; learning rate 0.0001

## Experiments </br>
- [x] Train VGG16 Independent Student on Caltech101
- [ ] Train VGG16 Dependent Student on Caltech101 with new method
- 

## Pre-Trained Weights </br>

vgg16.npy can be downloaded from here [VGG16 Weights](ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy) </br>
vgg16.npz can be downloaded from here [VGG16 Weights](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz)
