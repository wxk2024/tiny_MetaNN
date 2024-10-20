cmake_minimum_required(VERSION 3.29)
project(tiny_metann)

set(CMAKE_CXX_STANDARD 20)

include_directories(./)
add_subdirectory(test)
add_executable(tiny_metann main.cpp)
