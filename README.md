# ML-7641 Binarized Neural Networks
<!-- >How far can you go on datsets like MNIST with neural nets of 1s and 0s -->

### Fall'23: Project Proposal - Group 49

# Team Members
Members names orderd in Lexicographically ascending order.

| Member Name                      | Student-Id (@gatech.edu) |
|----------------------------------|----------------|
| Gaurang Kamat                    | gkamat8        |
| Oscar Laird                      | olaird3        |
| Somu Bhargava                    | bsomu3         |
| Sri Kamal                        | schillarage3   |
| Tenzin Bhotia                    | tbhotia3       |

# Table of Contents
1. [Introduction](#introduction)
2. [Problem Definition ](#problem-definition)
3. [Methods](#methods)
4. [Dataset](#dataset)
5. [Potential Results and Discussion ](#potential-results-and-discussion)
6. [Timeline](#timeline)
7. [Member Contributions](#member-contributions)
8. [References](#references)

# Introduction 

As Neural Networks have grown increasingly capable, their memory and compute footprint has grown at a commensurate pace. To reduce the time taken to train and to reduce the memory and storage requirements, we propose the use of binarized neural networks. The basic idea is that instead of using a 32-bit floating-point value to represent a single weight, we will use one bit to represent it. We also aim to binarize the input so that we can replace expensive matrix multiplication with logical AND operation. Since we are exploring a novel problem, we wanted to use a standard dataset like MNIST so that we can compare our work with popular benchmarks. 

# Problem Definition 

A standard operation when training a neural network of any kind is to multiply the weights with input and add bias. If we assume a neural network with 784 nodes in a single layer, and a flattened MNIST image of size 784 as input, to store them as float 32 objects, we will use 50,176 bits to represent the data. If we instead use binary inputs and weights, we can cut it down to 1568 bits. That is a 32x reduction in memory usage. In addition, a logical AND operation is much faster than a dot product. This means that a binary model should, in theory, be smaller and faster to train and run. We aim to test this hypothesis by developing several models with and without binary quantization and observe if our hypothesis holds true. 

# Methods 

We will first implement a simple K-means classifier with K set to 10 to get an idea of the data, and log examples where k-means struggles to classify a given example. We will also implement feed forward neural networks, convolutional neural networks, Logistic Regression and Transformers and implement binarized versions (using techniques like quantization and binary connect) in each of the above models. We will then compare model accuracy, size, time taken to train, time to classify an input example and present this information in our final writeup. 

## Exploratory Data Analysis of MNIST

We start with some preliminary visualizations and analysis of the MNIST Data. 
Majorly we would like to understand if a mere k-means classification can help us form good clusters. And if not, then what are the digits that often look like another digit. 

First, we sample 10 images for each label, and visualize them as shown below.
![Sampled Images](images/mnist.png)

Each image is a 28*28 single channel numpy array. To make things simpler for visualizations, we apply dimensionality reduction and bring down each image to a 2 dimensional feature. We use t-SNE (t-distributed Stochastic Neighbor Embedding) for an unsupervised non linear dimensionality reduction.

After fitting to the 100 samples from MNIST, we obtain the following visualization

![t-SNE](images/2dtsne-mnist.png)

### The Eye Test

Before doing anything complex, we proceed with a simple eye test. We clearly observe some distinct clusters being formed. For instance, digits 0, 6, 8, and 9 form clear clusters, and are easily distinguishible. However, digits 1, 7, 3, 4, 5, and 2 are overall quite scattered. Notably, the digit 2 is the most scattered spanning the entire t-SNE componenet 1 axes, and invading clusters of other digits.

This clearly indicates, that while MNIST is considered a simple dataset, it is not as simple to be easily modelled by a non-paramterized clustering mechanism.

### Diving Deeper

We dive deeper into why some of these digits get miss-classified so often. To do that we locate the digits that get missclassified, and subjectively compare their images in juxtapostion with the other digit in cluster. 

We isolate the following local missclassified regions in the t-SNE plot

![t-SNE](images/2dtsne-mnist-labelled.png)

Juxtaposing confusing miss-classified images,

2 vs 7:

<img src="images/miss-2.png" alt="Alt text for image 1" width="300" height="200"/> <img src="images/miss-7.png" alt="Alt text for image 2" width="300" height="200"/>

4 vs 9:

<img src="images/miss-4.png" alt="Alt text for image 1" width="300" height="200"/> <img src="images/miss-9.png" alt="Alt text for image 2" width="300" height="200"/>

1 vs 2:

<img src="images/miss-1.png" alt="Alt text for image 1" width="300" height="200"/> <img src="images/miss-2a.png" alt="Alt text for image 2" width="300" height="200"/>


# Dataset 

We will be using [MNIST](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=mnist) dataset for this project. MNIST is a dataset of 60,000 handwritten digits that is commonly used for training various image classification systems. Data visualized below. 

 ![MNIST](images/mnist.webp)

# Potential Results and Discussion 

The main result that we are expecting at the end is to have a small model which takes less memory than the traditional model and can be used in devices which have a memory constraint especially in Robotics etc. Though there may not be significant reduction in resource consumption during training, the models that we end up in each of the proposed categories (Feed Forward neural networks, convolutional neural networks, Logistic Regression and Transformers) are expected to be smaller than their respective counter parts. 

# Timeline 

### Gantt chart 

Find the [Gantt Chart here](https://gtvault-my.sharepoint.com/:x:/g/personal/bsomu3_gatech_edu/EelUHYYmrTRGlW9DHgme1MUBtKZvp8KfHR6h5FXsjXqcjg?e=aNcJNW). 

# Member Contributions

| TASK TITLE                       | TASK OWNER |
|----------------------------------|------------|
| Introduction & Background        | Sri Kamal  |
| Problem Definition               | Sri Kamal  |
| Methods                          | Sri Kamal  |
| Potential Results & Discussion   | Bhargava   |
| Timeline & Distribution          | Bhargava   |
| Video Recording                  | Oscar      |
| Build the GitHub Page            | Tenzin     |
| Literature Review (References)   | Tenzin     |


# References 

1. Courbariaux, Matthieu, Yoshua Bengio, and Jean-Pierre David. "Binaryconnect: Training deep neural networks with binary weights during propagations." Advances in neural information processing systems 28 (2015). 

2. Kim, Minje, and Paris Smaragdis. "Bitwise neural networks." arXiv preprint arXiv:1601.06071 (2016). 

3. Tu, Zhijun, et al. "Adabin: Improving binary neural networks with adaptive binary sets." European conference on computer vision. Cham: Springer Nature Switzerland, 2022. 

4. Lin, Xiaofan, Cong Zhao, and Wei Pan. "Towards accurate binary convolutional neural network." Advances in neural information processing systems 30 (2017). 

5. Zhang, Dongqing, et al. "Lq-nets: Learned quantization for highly accurate and compact deep neural networks." Proceedings of the European conference on computer vision (ECCV). 2018. 

6. Liu, Zechun, et al. "Bit: Robustly binarized multi-distilled transformer." Advances in neural information processing systems 35 (2022): 14303-14316. 