cmake_minimum_required(VERSION 3.0)
project(neural-network)

add_executable(nn_test src/main.cpp src/NeuralNetwork.cpp)
add_executable(nn_mnist_test src/mnist.cpp src/NeuralNetwork.cpp)

target_include_directories(nn_test PUBLIC external/eigen)
target_include_directories(nn_mnist_test PUBLIC external/eigen external/fast-cpp-csv-parser)