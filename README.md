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
3. CoarseDropout - This is applied on all the images. However, as the image if first padded to 64x64 and then a hole 16x16 is created and then centre crop of 32x32 is applied dropout will not be available for all the images. The below set of picture proves the same.

Example of augmented images  
![image](https://github.com/sayanbanerjee32/TASI_ERAv2_S9/assets/11560595/f6f30a39-abdd-4abc-9475-a666726deb32)


## Model features
[Final notebook is available here](https://github.com/sayanbanerjee32/TASI_ERAv2_S9/blob/main/S9_stride2_dilated_SayanBanerjee.ipynb) 
### Performance
    - Number of parameters: 197,152
    - Train set accuracy: 84% (100 epochs) - 79% at 41 epochs
    - Test set Accuracy: 88% (100 epochs) - 85% at 41 epochs
    - Total Receptive field: 

Train and test loss and accuracy curve do not suggest any overfitting.  
![image](https://github.com/sayanbanerjee32/TASI_ERAv2_S9/assets/11560595/2b91da7e-e35d-482b-8725-7ace3e873a73)

    
### Use of Stride 2 Convolution
Stride 2 convolution is used for pooling after 1st convolution block.

### Use of Depthwise Separable Convolution
Depthwise separable convolutions are used in each convolution layer inside Convolution block 3 and Convolution block 4

### Use of Dilated Convolution
Dilated convolution is used for pooling after Convolution block 2 and Convolution block 3
    
### Error analysis

There are total 1190 wrong classifications out of 10,000 test images. Top 5 confused cases of prediction
|target|prediction|number of images|
|------|----------|----------------|
|dog|cat|137|
|cat|dog|71|
|bird|deer|45|
|cat|bird|44|
|frog|cat|41|  
  
Example of 9 misclassified images in descending order of loss.  
![image](https://github.com/sayanbanerjee32/TASI_ERAv2_S9/assets/11560595/8bc6a0bb-d6d5-4f42-8988-65dae492d22a)


