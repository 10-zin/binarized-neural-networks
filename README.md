# Binarized Neural Networks
### CS - 7641: Machine Learning Project Proposal 
<!-- >How far can you go on datsets like MNIST with neural nets of 1s and 0s -->


# Introduction 

As Neural Networks have grown increasingly capable, their memory and compute footprint has grown at a commensurate pace. To reduce the time taken to train and to reduce the memory and storage requirements, we propose the use of binarized neural networks. The basic idea is that instead of using a 32-bit floating-point value to represent a single weight, we will use one bit to represent it. We also aim to binarize the input so that we can replace expensive matrix multiplication with logical AND operation. Since we are exploring a novel problem, we wanted to use a standard dataset like MNIST so that we can compare our work with popular benchmarks. 

# Problem Definition 

A standard operation when training a neural network of any kind is to multiply the weights with input and add bias. If we assume a neural network with 784 nodes in a single layer, and a flattened MNIST image of size 784 as input, to store them as float 32 objects, we will use 50,176 bits to represent the data. If we instead use binary inputs and weights, we can cut it down to 1568 bits. That is a 32x reduction in memory usage. In addition, a logical AND operation is much faster than a dot product. This means that a binary model should, in theory, be smaller and faster to train and run. We aim to test this hypothesis by developing several models with and without binary quantization and observe if our hypothesis holds true. 

# Methods 

We will first implement a simple K-means classifier with K set to 10 to get an idea of the data, and log examples where k-means struggles to classify a given example. We will also implement feed forward neural networks, convolutional neural networks, Logistic Regression and Transformers and implement binarized versions (using techniques like quantization and binary connect) in each of the above models. We will then compare model accuracy, size, time taken to train, time to classify an input example and present this information in our final writeup. 

# Potential Results and Discussion 

The main result that we are expecting at the end is to have a small model which takes less memory than the traditional model and can be used in devices which have a memory constraint especially in Robotics etc. Though there may not be significant reduction in resource consumption during training, the models that we end up in each of the proposed categories (Feed Forward neural networks, convolutional neural networks, Logistic Regression and Transformers) are expected to be smaller than their respective counter parts. 

# Timeline 

### Gantt chart 

Find the [Gantt Chart here](https://gtvault-my.sharepoint.com/:x:/g/personal/bsomu3_gatech_edu/EelUHYYmrTRGlW9DHgme1MUBtKZvp8KfHR6h5FXsjXqcjg?e=aNcJNW). 

# Contributions

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

 

# Dataset 

We will be using [MNIST](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=mnist) dataset for this project. MNIST is a dataset of 60,000 handwritten digits that is commonly used for training various image classification systems. Data visualized below. 

 ![MNIST](images/mnist.webp)


# References 

1. Courbariaux, Matthieu, Yoshua Bengio, and Jean-Pierre David. "Binaryconnect: Training deep neural networks with binary weights during propagations." Advances in neural information processing systems 28 (2015). 

2. Kim, Minje, and Paris Smaragdis. "Bitwise neural networks." arXiv preprint arXiv:1601.06071 (2016). 

3. Tu, Zhijun, et al. "Adabin: Improving binary neural networks with adaptive binary sets." European conference on computer vision. Cham: Springer Nature Switzerland, 2022. 

4. Lin, Xiaofan, Cong Zhao, and Wei Pan. "Towards accurate binary convolutional neural network." Advances in neural information processing systems 30 (2017). 

5. Zhang, Dongqing, et al. "Lq-nets: Learned quantization for highly accurate and compact deep neural networks." Proceedings of the European conference on computer vision (ECCV). 2018. 

6. Liu, Zechun, et al. "Bit: Robustly binarized multi-distilled transformer." Advances in neural information processing systems 35 (2022): 14303-14316. 