###The repository contains test code for the paper -- swDNN: a Library for Efficient Deep Learning Applications on Sunway Supercomputer.


swCNNv11~14 are unitest code for forward propagation with four different loop transformations.

swCNNv21    contains a conplete conv layer implementaion with gradient updating for forward and backward propagation.

./src         contains the fully-connected layer, the *convolutional layer*, the *activation functions* and the *padding operations* on 64 CPEs of SW26010 processors.

./test        contains the test code for GO network.

./lib         contains the library for blas and the support library for memory manageron 4 core groups.

2016.11.14 : support for *float* and *double* data formats

