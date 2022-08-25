# pRNN Compilation

Functional-Filet is stable only from version 0.5.0, any previous version corresponds to the development phase.

However, there are several possible optimizations, in particular on the restructuring of *Torch* tensors in Python which could be done in C++. Here we will try to compile our model with the CPU (testing), then with the GPU.

###### Attribution required : Fabien Furfaro (CC 4.0 BY NC ND SA)

## Minimal example (https://pytorch.org/cppdocs/installing.html) :

1 - LibTorch CPU (in HOME) :

	wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip

unzip this folder in HOME

2 - Implementation of code & CMake

Create file like :

```
	example-app/
  	   CMakeLists.txt
  	   exampleApp.cpp
```

3 - Build in exampleApp/

	mkdir build
	cd build
	cmake -DCMAKE_PREFIX_PATH=$HOME/libtorch ..
	cmake --build . --config Release

4 - Test your build (in build/) :

	./exampleApp

5 - Using your builded function in Python

	from build import exampleApp as f

	f.main()

Convert python code to LibTorch : https://pytorch.org/tutorials/advanced/cpp_export.html (test)
cpp LibTorch : https://pytorch.org/tutorials/advanced/cpp_frontend.html