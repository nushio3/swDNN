#The repository contains test code for the paper -- swDNN: a Library for Efficient Deep Learning Applications on Sunway Supercomputer.

swCNNv11~14 are unitest code for forward propagation with four different loop transformations.

swCNNv21    are unitest code for backward propagation.

./src         contains fully-connected layer, convolutional layer, activation functions and padding operations on 64 CPEs of SW26010 processors.

./test        contains test code for a network.

./lib         contains library for blas and support library for memory management and control on 4 core groups.

