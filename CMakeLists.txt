cmake_minimum_required(VERSION 3.0)
project(neural-network)

add_executable(nn_test src/main.cpp)

target_include_directories(nn_test PUBLIC external/eigen)