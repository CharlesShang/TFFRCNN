#!/usr/bin/env bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC

CUDA_PATH=/usr/local/cuda/

cd roi_pooling_layer

nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

## if you install tf using already-built binary, or gcc version 4.x, uncomment the two lines below
#g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o roi_pooling.so roi_pooling_op.cc \
#	roi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64

# for gcc5-built tf
g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=1 -o roi_pooling.so roi_pooling_op.cc \
	roi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
cd ..


# add building psroi_pooling layer
cd psroi_pooling_layer
nvcc -std=c++11 -c -o psroi_pooling_op.cu.o psroi_pooling_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

g++ -std=c++11 -shared -o psroi_pooling.so psroi_pooling_op.cc \
	psroi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64

## if you install tf using already-built binary, or gcc version 4.x, uncomment the two lines below
#g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o psroi_pooling.so psroi_pooling_op.cc \
#	psroi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64

cd ..