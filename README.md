# pRNN Compilation

Functional-Filet is stable only from version 0.5.0, any previous version corresponds to the development phase.

However, there are several possible optimizations, in particular on the restructuring of *Torch* tensors in Python which could be done in C++. Here we will try to compile our model with the CPU (testing), then with the GPU.

###### Attribution required : Fabien Furfaro (CC 4.0 BY NC ND SA)

## [Extension C++](https://pytorch.org/tutorials/advanced/cpp_extension.html) :

1 - 