cmake_minimum_required(VERSION 3.29)

add_executable(test_scalar test_scalar.cpp)
target_link_libraries(test_scalar PRIVATE Catch2::Catch2WithMain)
add_executable(test_general_matrix test_general_matrix.cpp)
target_link_libraries(test_general_matrix PRIVATE Catch2::Catch2WithMain)
