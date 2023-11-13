cmake_minimum_required(VERSION 3.0)
project(neural-network)

find_package(OpenMP REQUIRED)

add_compile_definitions(EIGEN_DONT_PARALLELIZE)

set(CMAKE_CXX_STANDARD 20)

add_executable(nn_test src/main.cpp src/NeuralNetwork.cpp)
add_executable(nn_mnist_test src/mnist.cpp src/NeuralNetwork.cpp)

target_compile_features(nn_test PRIVATE cxx_std_20)
target_compile_features(nn_mnist_test PRIVATE cxx_std_20)

target_include_directories(nn_test PUBLIC external/eigen external/json)
target_include_directories(nn_mnist_test PUBLIC external/eigen external/fast-cpp-csv-parser external/json)

target_link_libraries(nn_test OpenMP::OpenMP_CXX)
target_link_libraries(nn_mnist_test OpenMP::OpenMP_CXX)
