# deepfake_video_detection

## NYU Deep Learning Systems Final Project
Autoencoder + Transformer Model that detects artificially generated videos


## Dataset: 
The train/test set is from [Deepfake Detection Challenge Kaggle Data](https://www.kaggle.com/c/deepfake-detection-challenge/data/)

## 1. Train Autoencoder
First, we want to pre-train the face autoencoder before we run the classifier. Once we pass the folder containing the training video data, it will do the random split to create the validation set and save the best model based on the best validation loss in the current directory.
```
python train_autoencoder.py <folder that contains training video data> <number of epochs>
```

## 2. Train Classifier
The model is composed of autoencoder and transformer. This line of code will take the pre-trained autoencoder and run the entire network.
The train folder and test folder arguments as well as other hyperpameter values need to be passed. The training folders that you download from Kaggle dataset (all or part of them) should be placed under `deepfake-detection-challenge/train_folders/`. This is the same for test dataset - `deepfake-detection-challenge/test_folders/`. In our experiment, we used batch size = 10, number of frames = 60, learning rate = 0.1, number of epochs = 20. For more detailed experimental setup, please refer to our paper. 

```
python face_train_classifier.py <path to train folder> <path to test folder> <batch size> <num epochs> <num frames> <learning rate> <path to pre-trained autoencoder model>
```
This will create a csv file with information about training loss, validation loss, train accuracy, validation accuracy, and epoch run time. We used `dfdc_train_part_1` (~10GB) as the training set, and `training sample data` as our test dataset. 

## Evaluation Results
The final accuracy we obtained is 93.6% of training accuracy and 80.8% of test accuracy. 
