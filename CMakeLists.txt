cmake_minimum_required(VERSION 3.29)
project(tiny_metann)

set(CMAKE_CXX_STANDARD 20)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.4.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)

include_directories(./)
add_subdirectory(test)
add_executable(tiny_metann main.cpp)
