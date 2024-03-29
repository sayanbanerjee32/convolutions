# TASI_ERAv2_S9

## Objective

1. Create a new network that has the following architecture 
    - C1-C2-C3-C4-O where `CX` denotes Convolution block and `O` denotes output block
    - No MaxPooling but last layer of each block is a stride 2 convolution instead OR Dilated kernels here instead of MP or strided convolution
    - total RF must be more than 44
    - one of the layers must use Depthwise Separable Convolution
    - one of the layers must use Dilated Convolution
    - use GAP (compulsory):- add FC after GAP to target #of classes (optional)
2. use albumentation library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
3. Achieve 85% accuracy. Total Params to be less than 200k. No limit on Number of epochs

## Dataset - CIFAR10

1. Training data size -50,000, Test data size - 10,000
2. Image shape - 3 x 32 x 32 (colour image - RGB)
3. Number of classes 10 - plane, car, bird, cat, deer, dog, frog, horse, ship, truck

Example images from training set  
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S8/assets/11560595/711aed42-d235-45f3-b7e1-729fbb8a01fe)

## Augmentation Strategies
Following modules of albumentations library is used for image augmentations
1. HorizontalFlip - this flips half of the images horizontally
2. ShiftScaleRotate - this is also applied on half of the images where
    - image is shifted horizontally and vertically in the range of -0.0625, 0.0625 of hight and width
    - image is scaled in the range of 0.9, 1.1 times of the original size
    - Image is rotated within -45 degrees and 45 degrees

Example of augmented images  
![image](https://github.com/sayanbanerjee32/TASI_ERAv2_S9/assets/11560595/f6f30a39-abdd-4abc-9475-a666726deb32)


## Model features
### Performance
    - Number of parameters: 197,152
    - Train set accuracy:
    - Test set Accuracy:
    - Number of Epochs:
    - Total Receptive field:

Train and test loss and accuracy curve do not suggest any overfitting
    
### Use of Stride 2 colvolution

### Use of Depthwise Separable Convolution

### Use of Dilated Convolution
    
### Error analysis

