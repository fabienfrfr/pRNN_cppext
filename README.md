# pRNN Compilation

Functional-Filet is stable only from version 0.5.0, any previous version corresponds to the development phase.

However, there are several possible optimizations, in particular on the restructuring of *Torch* tensors in Python which could be done in C++. Here we will try to compile our model with the CPU (testing), then with the GPU.

###### Attribution required : Fabien Furfaro (CC 4.0 BY NC ND SA)

## [Extension C++](https://pytorch.org/tutorials/advanced/cpp_extension.html) :

1 - Copy folder https://github.com/pytorch/extension-cpp/cpp in example folder (it's composed of 5 files).

2 - Launch ```python3 setup.py build```. That create a build folder composed of your lib. 

	- In case of cuda, you need to set a compatible gcc maximum version. Example : for cuda9, it's gcc6 (max). TIPS : compile with docker (see in my AnimateCV github repository) 

3 - Put ```.so``` in main folder (libname_cpython3Xm-x86_64-linux-gnu.so).

4 - Launch extension in python with  ```import libname```.

tips :

	- use JIT to see step of code
	- see basic autograd function (see backward)
	- #include <iostream> for use direct library in Terminal (but need 'main') or C code (see libtorch also).

## Observation JIT step :

For unstacked :

	input -> view -> convolution -> view -> to -> select -> to -> (loop) index -> to -> select -> to (end loop) -> [list output, index] -> cat (input Module) -> t -> addmm -> output

for stacked (time dependant) :

	input -> view -> convolution -> view -> (LOOP) (loop) unsqueeze (end loop) -> [list output] -> cat -> t -> addmm (END LOOP) --> [list output Module] -> cat -> output