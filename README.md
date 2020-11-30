# Survey of Learning Algorithms for Handwritten Digit Recognition in 2D and 3D Space

## Abstract
  The objective of this study was to survey different learning algorithms to correctly classify handwritten digits. The various approaches surveyed are measured in terms of accuracy and efficiency against multiple datasets of handwritten digit in 2D and 3D space. This survey utilized both the MNIST Dataset, a database of 2D digit images, and the 3D MNIST Dataset, a basebase of 3D digit images, to document more robust results. This study surveyed various practical classification methods, such as K-Nearest Neighbors (K-NN) and Support Vector Machines (SVM), as well as more advanced computer vision techniques, such as Convolutional Neural Networks (CNN) and Extreme Learning Machines (ELM). Test data results demonstrated the efficacy of deep neural networks such as CNNs. However, ELMs provided a faster alternative to CNNs with better generalization performance.

## Datasets 

### MNIST (2D Images)
<p align="center">
  <img width="432" height="288" src="https://github.com/brodyu/handwritten-digit-recognition/blob/main/graphics/training_set_images.jpg">
</p>

  The Modified National Institute of Standards and Technology (MNIST) database is a large database of handwritten digit images. MNIST contains 60,000 training images and 10,000 testing images.
  
<p align="center">
  <img width="460" height="300" src="https://github.com/brodyu/handwritten-digit-recognition/blob/main/graphics/countplt.jpg">
</p>
  
  MNIST was created by combining 30,000 images from NIST’s Special Database 3 (SD-3) and 30,000 images from Special Database 1 (SD-1). SD-3 images were collected among Census Bureau employees and SD-1 images were collected among American high-school students. The original NIST images were preprocessed through size-normalization and centered in a 28x28 image. Human performance of digit recognition upon the MNIST dataset hovers around a 2-2.5% test error rate. 
  
  In 2004, the best test error rate reported on the MNIST dataset was 0.42 percent. This was achieved by a team of researchers utilizing a neural classifier Limited Receptive Area (LIRA) for image recognition. Following landmark achievements within AI in 2015, a team of researchers using a single convolutional neural network reduced the test error rate to 0.31 percent. As it stands, the University of Virginia holds the best test error rate of 0.18 percent by simultaneously stacking fully connected, recurrent, and convolutional neural networks together simultaneously.
  
  While the MNIST dataset is traditionally used to in academia to benchmark new learning algorithms, it is gaining popularity among computer vision beginners and practitioners to learn different deep learning techniques on a clean pre-processed dataset. 
  
### 3D MNIST

<p align="center">
  <img width="600" height="500" src="https://github.com/brodyu/handwritten-digit-recognition/blob/main/graphics/3d_image.png">
</p>

The 3D MNIST dataset is a 3-dimensional version of the original MNIST digit dataset. The dataset was created by generating 3D point clouds from the original 2D images in MNIST. Point clouds are a set of points that represents a 3D object with the set of X, Y, and Z coordinates. The 3D MNIST dataset contains 10,000 training objects and 2000 testing objects formatted in H5 file format. 

In order to pull the training and testing data into Python from the H5 file format we must run:
```python
  with h5py.File("./full_dataset_vectors.h5", "r") as dataset:
    x_train = dataset["X_train"][:]
    y_train = dataset["y_train"][:]
    x_test = dataset["X_test"][:] 
    y_test = dataset["y_test"][:]
```

#### Pre Processing

To preprocess our data for training we must first reshape the data into a 3D format with the dimensions (16, 16, 16, 3). Next we convert our target variables to categorical targets ulizing Tensorflow's to_categorical method:

```python
  y_train = to_categorical(y_train, 10).astype(np.int32)
  y_test = to_categorical(y_test, 10).astype(np.int32)
```
