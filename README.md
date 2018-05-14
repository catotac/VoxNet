# VoxNet
Classification of ModelNet40 and ModelNet10 dataset using VoxNet

## 3DConvNet
### Setup
```
3D Convolution -> 3D Max Pooling -> Flatten -> Dense(40)
Loss function: Categorical crossentropy
Optimizer: Adadelta
```
### Model
Trained for 500 epochs with a voxel resolution of 32 and augmentation of models:
```
Test loss 2.8196212050666833
Test accuracy: 0.7845421393841167
